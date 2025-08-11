"""
Callbacks for edge-pruning (via HF Trainer).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import wandb
from edge_pruning.eval import evaluate_cmd_cpr_for_ep
from edge_pruning.importance_scores.ep_z_attribution import (
    edge_signs_from_z_attribution,
)
from edge_pruning.importance_scores.logalphas_to_scores import EdgeKey, save_edge_scores

LOGGER = logging.getLogger(__name__)


class CMDCPRCallback(TrainerCallback):
    """
    Computes CMD/CPR at evaluation points.
    """

    def __init__(
        self,
        model_name: str,
        base_model,
        task: str,
        circuit_root: str,
        importances_path: str,
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
        edges_attr_signs: Optional[Dict[EdgeKey, int]] = None,
        split: str = "validation",
        num_examples: Optional[int] = None,
        batch_size: int = 20,
        cache_dir: Optional[str] = None,
        method_name: str = "edge-pruning",
        every_n_evals: int = 1,
    ):
        self.model_name = model_name
        self.base_model = base_model
        self.task = task
        self.circuit_root = circuit_root
        self.ablation = ablation
        self.level = level
        self.percentages = percentages
        self.edges_attr_signs = edges_attr_signs
        self.split = split
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.method_name = method_name
        self.every_n_evals = max(1, every_n_evals)
        self._eval_counter = 0

        # Save current EP scores to a canonical path (one file that gets overwritten at each eval)
        self.importances_path = Path(importances_path)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._eval_counter += 1
        if (self._eval_counter % self.every_n_evals) != 0:
            return

        model = kwargs.get("model")
        if model is None:
            LOGGER.warning("[CMD/CPR] No model in callback kwargs; skipping.")
            return

        train_dataloader = kwargs.get("train_dataloader")
        if train_dataloader is None:
            LOGGER.warning(
                "[CMD/CPR] No train_dataloader in callback kwargs; skipping."
            )
            return

        # compute attribution signs
        if self.edges_attr_signs is None:
            edges_attr_signs = edge_signs_from_z_attribution(
                sparse_model=model,
                base_model=self.base_model,
                dataloader=train_dataloader,
                device=model.device,
                reduce_over_examples="mean",
                custom_metric_fn=None,
                include_nodes=False,
            )
        else:
            edges_attr_signs = self.edges_attr_signs

        try:
            save_edge_scores(
                model=model,
                file_path=str(self.importances_path),
                mode="z_score",
                orig_attr_signs=edges_attr_signs,
                overwrite=True,
            )
        except Exception as e:
            LOGGER.exception(f"[CMD/CPR] Failed to save edge scores: {e}")
            return

        # Run CMD/CPR once (reuses the saved file + MIB evaluator)
        try:
            out = evaluate_cmd_cpr_for_ep(
                model_name=self.model_name,
                task=self.task,
                circuit_root=self.circuit_root,
                circuit_path=str(self.importances_path),
                method_name=self.method_name,
                ablation=self.ablation,
                level=self.level,
                percentages=self.percentages,
                split=self.split,
                num_examples=self.num_examples,
                batch_size=self.batch_size,
                cache_dir=self.cache_dir,
                model=None,
                dataloader=None,
                log_to_wandb=False,
            )
        except Exception as e:
            LOGGER.exception(f"[CMD/CPR] Evaluation failed: {e}")
            return

        # Log to W&B and trainer metrics
        metrics = {
            "eval/cmd": out["cmd"],
            "eval/cpr": out["cpr"],
            "eval/cmd_minus_cpr": out["cmd_minus_cpr"],
        }
        try:
            wandb.log(metrics, step=state.global_step)  # type: ignore
        except Exception:
            pass

        # merge into last_log history so HF Trainer prints it
        if hasattr(kwargs, "logs"):
            kwargs["logs"].update(metrics)
        LOGGER.info(
            f"[CMD/CPR] step={state.global_step} CMD={out['cmd']:.4f} "
            f"CPR={out['cpr']:.4f} OBJ={out['cmd_minus_cpr']:.4f}\n"
        )
