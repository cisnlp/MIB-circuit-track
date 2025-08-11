"""
Convert mask into Graph.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from eap.graph import Graph
from transformer_lens import HookedTransformer

from MIB_circuit_track.model_loading import load_reference_graph
from MIB_circuit_track.utils import model_registry

LAMBDAS: List[float] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]


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
    parser = argparse.ArgumentParser(description="Convert mask to graph in UGS run.")

    # General configs
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )
    parser.add_argument(
        "--path", type=Path, required=True, help="Root folder containing UGS outputs."
    )

    # Model configs
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt2-small", "qwen"],
        help="Model family used during the UGS run.",
    )

    # Task data configs
    parser.add_argument(
        "--task", type=str, required=True, help="Task the UGS run targeted."
    )

    # Circuit discovery configs
    parser.add_argument(
        "--ablation", type=str, required=True, help="Ablation used during the UGS run."
    )

    return parser.parse_args(argv)


def sigmoid(t: torch.Tensor) -> torch.Tensor:
    """Vectorised sigmoid implemented with torch for numerical stability."""
    return 1.0 / (1.0 + (-t).exp())


def load_thetas(snapshot_path: Path) -> Tuple[Dict[str, Dict[int, torch.Tensor]], int]:
    """
    Recover *θ* values from a snapshot file.

    Args:
        snapshot_path (Path): ``snapshot.pth`` produced by UGS.

    Returns:
        Tuple[Dict[str, Dict[int, torch.Tensor]], int]:
            *   Mapping ``edge_type -> layer_idx -> θ-tensor``.
            *   Total number of scalar parameters recovered.
    """
    snapshot = torch.load(snapshot_path, map_location="cpu")
    pruner_dict: Dict[str, torch.Tensor] = snapshot["pruner_dict"]  # type: ignore[assignment]

    thetas: Dict[str, Dict[int, torch.Tensor]] = {}
    param_count = 0
    for name, raw_params in pruner_dict.items():
        # e.g. name == "attn-attn.5"
        edge_type, layer_idx_txt = name.rsplit(".", 1)
        layer_idx = int(layer_idx_txt)
        thetas.setdefault(edge_type, {})[layer_idx] = sigmoid(raw_params)
        param_count += raw_params.numel()

    logging.info(f"Loaded {param_count} parameters from {snapshot_path}")
    return thetas, param_count


def convert_thetas_to_edges(
    thetas: Dict[str, Dict[int, torch.Tensor]],
    graph: Graph,
    model: HookedTransformer,
) -> Dict[str, float]:
    """
    Translate *θ* tensors into a flat ``edge_name -> score`` mapping.

    The naming convention exactly mirrors :class:`eap.graph.Graph`:

    * ``a{l}.h{h}<{q|k|v}>``  — attention heads (input/output split by QKV)
    * ``m{l}``                — MLP residual blocks
    * ``input`` / ``logits``  — graph endpoints

    Args:
        thetas (Dict[str, Dict[int, torch.Tensor]]): Output of :func:`load_thetas`.
        graph (Graph): Reference graph used for validation.
        model (HookedTransformer): Model whose config drives edge enumeration.

    Returns:
        Dict[str, float]: All edges with their corresponding scores.

    Raises:
        AssertionError: If any edge present in *graph* is missing, or vice-versa.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    edges: Dict[str, float] = {}

    # 1. From every earlier source to each attention *output* split (q/k/v)
    for dest_layer in range(n_layers):
        for split_idx, split_letter in enumerate("qkv"):
            θ_attn = thetas["attn-attn"][dest_layer][
                split_idx
            ]  # [n_heads, dest_layer, n_heads]
            θ_mlp = thetas["mlp-attn"][dest_layer][split_idx]  # [n_heads, dest_layer+1]

            # attn → attn
            for src_layer in range(dest_layer):
                for src_head in range(n_heads):
                    for dest_head in range(n_heads):
                        edge = f"a{src_layer}.h{src_head}->a{dest_layer}.h{dest_head}<{split_letter}>"
                        edges[edge] = float(θ_attn[dest_head, src_layer, src_head])

            # mlp → attn
            for src_layer in range(dest_layer + 1):
                src_name = "input" if src_layer == 0 else f"m{src_layer - 1}"
                for dest_head in range(n_heads):
                    edge = f"{src_name}->a{dest_layer}.h{dest_head}<{split_letter}>"
                    edges[edge] = float(θ_mlp[dest_head, src_layer])

    # 2. To every MLP block
    for dest_layer in range(n_layers):
        θ_attn = thetas["attn-mlp"][dest_layer]  # [dest_layer+1, n_heads]
        θ_mlp = thetas["mlp-mlp"][dest_layer]  # [dest_layer+1]

        for src_layer in range(dest_layer + 1):
            # attn → mlp
            for src_head in range(n_heads):
                edges[f"a{src_layer}.h{src_head}->m{dest_layer}"] = float(
                    θ_attn[src_layer, src_head]
                )

            # mlp → mlp
            src_name = "input" if src_layer == 0 else f"m{src_layer - 1}"
            edges[f"{src_name}->m{dest_layer}"] = float(θ_mlp[src_layer])

    # 3. To logits
    θ_attn_logits = thetas["attn-mlp"][n_layers]  # [n_layers, n_heads]
    θ_mlp_logits = thetas["mlp-mlp"][n_layers]  # [n_layers+1]

    for src_layer in range(n_layers):
        for src_head in range(n_heads):
            edges[f"a{src_layer}.h{src_head}->logits"] = float(
                θ_attn_logits[src_layer, src_head]
            )

    for src_layer in range(n_layers + 1):
        src_name = "input" if src_layer == 0 else f"m{src_layer - 1}"
        edges[f"{src_name}->logits"] = float(θ_mlp_logits[src_layer])

    # Sanity check against the reference graph
    missing = [e for e in graph.edges if e not in edges]
    excess = [e for e in edges if e not in graph.edges]

    assert not missing, f"Missing edges: {missing}"
    assert not excess, f"Unexpected edges: {excess}"

    logging.info(f"Converted θ-values into {len(edges):,} edges")
    return edges


def build_graph_payload(
    graph: Graph,
    edges: Dict[str, float],
    model: HookedTransformer,
) -> Dict[str, object]:
    """
    Assemble the JSON structure expected by downstream tooling.

    Args:
        graph (Graph): Reference graph (only node names are needed).
        edges (Dict[str, float]): Fully-validated ``edge_name -> score`` mapping.
        model (HookedTransformer): Supplies configuration metadata.

    Returns:
        Dict[str, object]: Serializable graph description.
    """
    edges_json = {
        name: {"score": score, "in_graph": False} for name, score in edges.items()
    }
    nodes_json = {name: {"in_graph": False} for name in graph.nodes.keys()}

    cfg_json = {
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "parallel_attn_mlp": False,
        "d_model": model.cfg.d_model,
    }
    return {"cfg": cfg_json, "edges": edges_json, "nodes": nodes_json}


def process_lambda(
    root: Path,
    task: str,
    ablation: str,
    model_tag: str,
    lam: float,
    model: HookedTransformer,
    graph: Graph,
) -> None:
    """
    Handle a single *λ* value end-to-end.

    1. Load ``snapshot.pth``.
    2. Convert parameters to edge scores.
    3. Persist the resulting ``graph.json``.

    Args:
        root (Path): Root path supplied by the user.
        task (str): Task sub-folder.
        ablation (str): Ablation sub-folder.
        model_tag (str): ``ugs_mib_{model_key}``.
        lam (float): Hyper-parameter value being processed.
        model (HookedTransformer): Reference model.
        graph (Graph): Empty reference graph.
    """
    res_folder = root / task / ablation / model_tag / f"{lam:.0e}"
    snapshot_path = res_folder / "snapshot.pth"
    thetas, _ = load_thetas(snapshot_path)
    edges = convert_thetas_to_edges(thetas, graph, model)
    graph_json = build_graph_payload(graph, edges, model)

    out_path = res_folder / "graph.json"
    out_path.write_text(json.dumps(graph_json))
    logging.info(f"Wrote graph to {out_path}")


def main() -> None:
    args = parse_args()

    model_name = model_registry(args.model)
    model, graph = load_reference_graph(model_name)
    model_tag = f"ugs_mib_{args.model}"

    for lam in LAMBDAS:
        process_lambda(
            root=args.path,
            task=args.task,
            ablation=args.ablation,
            model_tag=model_tag,
            lam=lam,
            model=model,
            graph=graph,
        )


if __name__ == "__main__":
    main()
