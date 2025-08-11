"""
Helper scripts and functions to evaluate edge-pruning.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from eap.graph import Graph
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

import wandb
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.model_loading import load_model
from MIB_circuit_track.utils import build_dataset

LOGGER = logging.getLogger(__name__)


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


def evaluate_cmd_cpr_for_ep(
    model_name: str,
    task: str,
    circuit_root: Optional[str] = None,
    circuit_path: Optional[str] = None,
    method_name: str = "edge-pruning",
    ablation: str = "patching",
    level: str = "edge",
    percentages: Tuple[float, ...] = (
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1,
    ),
    split: str = "validation",
    num_examples: Optional[int] = None,
    batch_size: int = 20,
    cache_dir: Optional[str] = None,
    model: Optional[HookedTransformer] = None,
    dataloader: Optional[DataLoader] = None,
    log_to_wandb: bool = False,
) -> Dict[str, float]:
    """
    Compute CMD/CPR once from the saved EP importances, optionally reusing a preloaded
    HookedTransformer `model` and `dataloader` to avoid per-eval overhead.

    Args:
        model_name (str): The name of the model to consider.
        task (str): The task name.
        circuit_root (Optional[str], optional): The root directory of the circuits. Defaults to None.
        circuit_path (Optional[str], optional): The path to the circuit attribution file. Defaults to None.
        method_name (str, optional): The name of the method. Defaults to "edge-pruning".
        ablation (str, optional): The ablation strategy. Defaults to "patching".
        level (str, optional): The level (node or edge). Defaults to "edge".
        percentages (Tuple[float, ...], optional): The sparsities to evaluate. Defaults to (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1).
        split (str, optional): The data split to consider. Defaults to "validation".
        num_examples (Optional[int], optional): The number of examples to consider. Defaults to None.
        batch_size (int, optional): The batch size for the evaluation. Defaults to 20.
        cache_dir (Optional[str], optional): The cache directory for model loading. Defaults to None.
        model (Optional[HookedTransformer], optional): The pre-loaded model. Defaults to None.
        dataloader (Optional[DataLoader], optional): The pre-loaded dataset. Defaults to None.
        log_to_wandb (bool, optional): Whether to log to wandb. Defaults to False.

    Returns:
        Dict[str, float]: The CPR and CMD metrics.
    """
    if circuit_path is None:
        if circuit_root is None:
            raise ValueError("Provide either circuit_path or circuit_root.")
        method_stub = f"{method_name}_{ablation}_{level}"
        ep_task_dir = Path(circuit_root) / method_stub / f"{task}_{model_name}"
        mib_task_dir = (
            Path(circuit_root) / method_stub / f"{task.replace('_', '-')}_{model_name}"
        )
        if (ep_task_dir / "importances.json").exists():
            circuit_path = str(ep_task_dir / "importances.json")
        else:
            circuit_path = str(mib_task_dir / "importances.json")

    cpath = Path(circuit_path)
    if not cpath.exists():
        raise FileNotFoundError(f"importances.json not found at {cpath}")
    LOGGER.info(f"[CMD/CPR] Using circuit: {cpath}")

    if model is None:
        model, _ = load_model(model_name, cache_dir=cache_dir)

    if dataloader is None:
        dataset = build_dataset(
            task, model.tokenizer, model_name, split, num_examples=num_examples
        )
        dataloader = dataset.to_dataloader(batch_size=batch_size)

    # Attribution metric
    metric_fn = get_metric("logit_diff", task, model.tokenizer, model)
    attribution_metric = lambda *a, **k: metric_fn(*a, mean=False, loss=False, **k)

    graph = load_graph_from_file(cpath)
    _, cpr, _, _, _ = evaluate_area_under_curve(
        model,
        graph,
        dataloader,
        attribution_metric,
        level=level,  # type: ignore
        absolute=False,
        percentages=percentages,
        apply_greedy=(method_name in {"information-flow-routes"}),
    )
    _, _, cmd, _, _ = evaluate_area_under_curve(
        model,
        graph,
        dataloader,
        attribution_metric,
        level=level,  # type: ignore
        absolute=True,
        percentages=percentages,
        apply_greedy=(method_name in {"information-flow-routes"}),
    )

    cpr = float(cpr)
    cmd = float(cmd)
    obj = cmd - cpr

    if log_to_wandb:
        try:
            wandb.log(  # type: ignore
                {
                    "eval_cpr": cpr,
                    "eval_cmd": cmd,
                    "eval_cmd_minus_cpr": obj,
                }
            )
        except Exception:
            pass

    return {"cmd": cmd, "cpr": cpr, "cmd_minus_cpr": obj}
