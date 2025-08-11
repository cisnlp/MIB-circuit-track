"""
Run attribution on one or more Hugging Face transformer models and save the
resulting importance scores in JSON format.

The script supports InterpBench, Qwen-2.5, Gemma-2, Llama-3 and any other model
name listed in ``MODEL_NAME_TO_FULLNAME``.  Attribution can be performed at the
edge, node or neuron level with a choice of several ablation strategies.

Taken and modified from the original MIB-circuit-track repository:
<https://github.com/hannamw/MIB-circuit-track/blob/main/run_attribution.py>
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Sequence

from eap.attribute import attribute
from eap.attribute_node import attribute_node
from eap.graph import Graph
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer  # type: ignore

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
    parser = argparse.ArgumentParser(description="Run MIB circuit attribution.")

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
        "--split", choices=["train", "validation", "test"], default="train"
    )
    parser.add_argument("--num-examples", type=int, default=100)

    # Circuit discovery configs
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--circuit-dir", type=str, default="circuits")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--level", choices=["node", "neuron", "edge"], default="edge")
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument(
        "--ablation",
        choices=["patching", "zero", "mean", "mean-positional", "optimal"],
        default="patching",
    )
    parser.add_argument("--optimal-ablation-path", type=str)

    return parser.parse_args(argv)


def attribute_circuit(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric_fn,
    args: argparse.Namespace,
) -> None:
    """
    Perform attribution according to CLI arguments.

    Args:
        model (HookedTransformer): Model under analysis.
        graph (Graph): Computation graph extracted from the model.
        dataloader (DataLoader): Batched task data.
        metric_fn (Callable): Metric configured for importance attribution.
        args (argparse.Namespace): Parsed command-line options (must include
            ``method``, ``level``, ``ablation``, etc.).
    """
    if args.level == "edge":
        attribute(
            model,
            graph,
            dataloader,
            metric_fn,
            args.method,
            args.ablation,
            ig_steps=args.ig_steps,
            optimal_ablation_path=args.optimal_ablation_path,
            intervention_dataloader=dataloader,
        )  # type: ignore
    else:
        attribute_node(
            model,
            graph,
            dataloader,
            metric_fn,
            args.method,
            args.ablation,
            neuron=args.level == "neuron",
            ig_steps=args.ig_steps,
            optimal_ablation_path=args.optimal_ablation_path,
            intervention_dataloader=dataloader,
        )  # type: ignore

def method_to_path(
    circuit_dir: Path | str,
    method: str,
    ablation: str,
    level: str,
    task: str,
    model_name: str,
):
    """from method name to the corresponding path where the scores are saved.
    
    Args:
        circuit_dir (Path | str): Root directory for all circuits.
        method (str): Attribution method (e.g. ``ig``).
        ablation (str): Ablation strategy identifier.
        level (str): ``edge``, ``node`` or ``neuron``.
        task (str): Internal task identifier.
        model_name (str): Short model identifier.

    Returns:
        Path: the file path
    """
    method_dir = Path(circuit_dir) / f"{method}_{ablation}_{level}"
    save_path = method_dir / f"{task.replace('_', '-')}_{model_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / "importances.json"
    return file_path

def save_graph(
    graph: Graph,
    circuit_dir: Path | str,
    method: str,
    ablation: str,
    level: str,
    task: str,
    model_name: str,
) -> None:
    """
    Persist importance scores to ``importances.json``.

    Args:
        graph (Graph): Graph containing populated importance scores.
        circuit_dir (Path | str): Root directory for all circuits.
        method (str): Attribution method (e.g. ``ig``).
        ablation (str): Ablation strategy identifier.
        level (str): ``edge``, ``node`` or ``neuron``.
        task (str): Internal task identifier.
        model_name (str): Short model identifier.
    """
    file_path = method_to_path(
        circuit_dir=circuit_dir,
        method=method,
        ablation=ablation,
        level=level,
        task=task,
        model_name=model_name,
    )
    graph.to_json(str(file_path))
    LOGGER.info("Saved graph to %s", file_path.resolve())


def run_single_task(
    model: HookedTransformer,
    model_name: str,
    task: str,
    args: argparse.Namespace,
) -> None:
    """
    Execute attribution for a single (model, task) pair.

    Args:
        model (HookedTransformer): Model being analysed.
        model_name (str): Short model identifier.
        task (str): Task identifier.
        args (argparse.Namespace): Parsed CLI options (used for thresholds,
            file paths, etc.).
    """
    key = f"{task.replace('_', '-')}_{model_name}"
    if key not in COL_MAPPING:
        LOGGER.warning("Skipping '%s' - no column mapping defined.", key)
        return

    graph = Graph.from_model(
        model,
        neuron_level=args.level == "neuron",
        node_scores=args.level == "node",
    )

    dataset = build_dataset(
        task,
        model.tokenizer,
        model_name,
        split=args.split,
        num_examples=args.num_examples,
    )
    dataloader = dataset.to_dataloader(batch_size=args.batch_size)

    output_metric = get_metric("logit_diff", task, model.tokenizer, model)
    attribution_metric = partial(output_metric, mean=True, loss=True)

    attribute_circuit(
        model=model,
        graph=graph,
        dataloader=dataloader,
        metric_fn=attribution_metric,
        args=args,
    )

    save_graph(
        graph,
        args.circuit_dir,
        args.method,
        args.ablation,
        args.level,
        task,
        model_name,
    )


def main() -> None:
    """
    Entrypoint for console execution.
    """
    args = parse_args()

    for model_name in args.models:
        model, _ = load_model(model_name, cache_dir=args.cache_dir, device=args.device)
        for task in args.tasks:
            run_single_task(model=model, model_name=model_name, task=task, args=args)


if __name__ == "__main__":
    main()
