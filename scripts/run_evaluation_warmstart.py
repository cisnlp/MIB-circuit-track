"""
Evaluating circuit faithfulness on MIB-Bench tasks.
"""

from __future__ import annotations

import argparse
import logging
import json
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Sequence

from eap.graph import Graph
from transformer_lens import HookedTransformer

import sys
sys.path.append("/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/wmg7rng/MIB-edge-pruning")
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.evaluation import (
    evaluate_area_under_curve,
    evaluate_area_under_roc,
)
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.model_loading import load_model
from MIB_circuit_track.utils import COL_MAPPING, build_dataset

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional replacement for ``sys.argv``.

    Returns:
        argparse.Namespace: Namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(description="Evaluate MIB-Bench circuits.")

    # General configs
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )

    # Model configs
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory of cached model weights",
    )

    # Task data configs
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="validation"
    )
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--batch_size", type=int, default=20)

    # Circuit configs
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method used to generate the circuit (only needed to infer file name).",
    )
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--circuit-dir", type=str, default="circuits")
    parser.add_argument("--circuit-files", type=str, nargs="+", default=None)
    parser.add_argument("--level", choices=["node", "neuron", "edge"], default="edge")
    parser.add_argument(
        "--ablation",
        choices=["patching", "zero", "mean", "mean-positional", "optimal"],
        default="patching",
    )
    parser.add_argument("--optimal-ablation-path", type=str)

    # Output configs
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--warmstart_scale", type=float, default=1.0)

    return parser.parse_args(argv)


def load_graph_from_file(path: Path) -> Graph:
    """
    Load a circuit graph from path.

    Args:
        path (Path): Path ending in ``.json`` or ``.pt``.

    Returns:
        Graph: Loaded graph object.

    Raises:
        ValueError: If the extension is not recognised.
    """
    if path.suffix == ".json":
        return Graph.from_json(str(path))
    if path.suffix == ".pt":
        return Graph.from_pt(str(path))
    raise ValueError(f"Unsupported graph file extension: {path.suffix}")


def evaluate_graph(
    model: HookedTransformer,
    graph: Graph,
    dataset: HFEAPDataset,
    task: str,
    level: Literal["node", "neuron", "edge"],
    absolute: bool,
    apply_greedy: bool,
    batch_size: int,
) -> Dict[str, object]:
    """
    Evaluate *graph* faithfulness for the given *task*.

    For *interpbench* a stand-alone AUROC is computed; otherwise the classic
    area-under-curve metric is applied.

    Args:
        model (HookedTransformer): Model under analysis.
        graph (Graph): Circuit graph to evaluate.
        dataset (HFEAPDataset): Examples to test on.
        task (str): Task identifier.
        level (str): ``edge`` | ``node`` | ``neuron``.
        absolute (bool): Whether to take absolute values of importance scores.
        apply_greedy (bool): Greedy edge expansion (method-specific).
        batch_size (int): The batch size.

    Returns:
        Dict[str, object]: Metric outputs ready for pickling.
    """
    dataloader = dataset.to_dataloader(batch_size=batch_size)
    metric_fn = get_metric("logit_diff", task, model.tokenizer, model)
    attribution_metric = partial(metric_fn, mean=False, loss=False)

    # Special case – when the model *is* interpbench, ``graph`` is the *reference* graph
    if model.cfg.model_name == "interpbench":
        return evaluate_area_under_roc(graph, graph)

    weighted_edge_counts, area, area_from_1, average, faithfulnesses = (
        evaluate_area_under_curve(
            model,
            graph,
            dataloader,
            attribution_metric,
            level=level,
            absolute=absolute,
            apply_greedy=apply_greedy,
        )
    )

    return {
        "weighted_edge_counts": weighted_edge_counts,
        "area_under": area,
        "area_from_1": area_from_1,
        "average": average,
        "faithfulnesses": faithfulnesses,
    }


def save_results(
    results: Dict[str, object],
    output_dir: Path,
    method_name: str,
    task: str,
    model_name: str,
    split: str,
    absolute: bool,
) -> None:
    """
    Persist *results* to a ``.pkl`` file.

    Args:
        results (Dict[str, object]): Evaluation outputs.
        output_dir (Path): Directory in which to write.
        method_name (str): Short method identifier (used as sub-folder).
        task (str): Task identifier.
        model_name (str): Abbreviated model identifier.
        split (str): Dataset split.
        absolute (bool): Whether absolute importance was used.
    """
    dest_dir = output_dir / method_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{task.replace('_', '-')}_{model_name}_{split}_abs-{absolute}.pkl"
    with open(dest_dir / file_name, "wb") as f:
        pickle.dump(results, f)
    with open(dest_dir / file_name.replace(".pkl", ".json"), "w") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    apply_greedy = args.method in {"information-flow-routes"}

    # External users might supply custom circuit files; map them in order.
    circuit_iter = iter(args.circuit_files) if args.circuit_files else None

    for model_name in args.models:
        model, reference_graph = load_model(model_name, cache_dir=args.cache_dir)

        for task in args.tasks:
            # Skip tasks without column mapping (same logic as original script)
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                logging.warning(
                    f"Task {task.replace('_', '-')} not in COL_MAPPING! Skipping task..."
                )
                continue

            # Locate the circuit file
            if circuit_iter:
                circuit_path = Path(next(circuit_iter))
            else:
                method_stub = f"{args.method}_{args.ablation}_{args.level}"
                circuit_path = (
                    Path(args.circuit_dir)
                    / method_stub
                    / f"{task.replace('_', '-')}_{model_name}"
                    / f"importances_warmstart_{args.warmstart_scale}.json"
                )

            logging.info(f"• Loading circuit from {circuit_path}")

            if reference_graph is None:
                reference_graph = load_graph_from_file(circuit_path)

            # Data & evaluation
            dataset = build_dataset(
                task,
                model.tokenizer,
                model_name,
                args.split,
                num_examples=args.num_examples,
            )
            results = evaluate_graph(
                model,
                reference_graph,
                dataset,
                task=task,
                level=args.level,
                absolute=args.absolute,
                apply_greedy=apply_greedy,
                batch_size=args.batch_size,
            )

            # Save
            method_stub = f"{args.method}_{args.ablation}_{args.level}_warmstart_{args.warmstart_scale}"
            save_results(
                results,
                Path(args.output_dir),
                method_stub,
                task,
                model_name,
                args.split,
                absolute=args.absolute,
            )


if __name__ == "__main__":
    main()
