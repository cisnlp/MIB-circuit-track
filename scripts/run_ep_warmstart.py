"""
Run edge-pruning on model and task.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence, Tuple, Type, Union

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import wandb

sys.path.append(
    "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/wmg7rng/MIB-edge-pruning"
)
from edge_pruning.data_prep import EdgePruningAdapter, EPDataCollator
from edge_pruning.importance_scores.logalphas_to_scores import save_edge_log_alphas
from edge_pruning.importance_scores.scores_to_logalphas import *
from edge_pruning.modeling.modeling_fpt2 import FPT2LMHeadModel
from edge_pruning.modeling.modeling_fqwen_kv_expansion import FQwen2ForCausalLM
from edge_pruning.modeling.params import SparsityConfig
from edge_pruning.optim import build_optim
from edge_pruning.trainer import EPTrainer
from MIB_circuit_track.utils import (
    COL_MAPPING,
    MODEL_NAME_TO_FULLNAME,
    MODEL_NAME_TO_MODEL_CLASS,
    build_dataset,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


os.environ["WANDB_LOG_MODEL"] = "false"  # or "end", "checkpoint"
os.environ["WANDB_WATCH"] = "false"


TAU = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional replacement for ``sys.argv``.

    Returns:
        argparse.Namespace: Namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(description="Run edge-pruning.")

    # General configs
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Whether to do sweeping via wandb.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        # default="/nfs/gdata/pmondorf/projects/MIB-circuit-track/edge-pruning/results",
        default="results",
        help="Directory to store model checkpoints.",
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
        "--num-examples", type=int, help="Number of training and validation examples."
    )
    parser.add_argument("--max_seq_len", type=int, default=64)

    # Edgpe pruning configs
    parser.add_argument(
        "--circuit-dir",
        type=str,
        default="circuits",
        help="Director to save circuits to.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=3000, help="Maximum number of steps."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument(
        "--ablation",
        choices=["patching", "zero", "mean", "mean-positional", "optimal"],
        default="patching",
    )
    parser.add_argument(
        "--with_embedding_nodes",
        action="store_true",
        help="Whether to also prune embeddings.",
    )
    parser.add_argument(
        "--disable_linear_reg",
        action="store_true",
        help="Disable linear regularization.",
    )
    parser.add_argument(
        "--disable_node_loss", action="store_true", help="Disable node loss."
    )
    parser.add_argument(
        "--stop_layer_loss_if_target",
        action="store_true",
        help="Stop layer loss if target is reached.",
    )

    # Learning rate configs
    parser.add_argument(
        "--edge_learning_rate", type=float, default=0.8, help="Edge learning rate."
    )
    parser.add_argument(
        "--layer_learning_rate", type=float, default=0.8, help="Layer learning rate."
    )
    parser.add_argument(
        "--reg_edge_learning_rate",
        type=float,
        default=0.8,
        help="The learning rate for the edge regularization term.",
    )
    parser.add_argument(
        "--reg_layer_learning_rate",
        type=float,
        default=0.8,
        help="The learning rate for the layer regularization term.",
    )

    # Sparsity configs
    parser.add_argument(
        "--start_edge_sp", type=float, default=0.0, help="Start edge sparsity."
    )
    parser.add_argument(
        "--target_edge_sp", type=float, default=0.99, help="Target edge sparsity."
    )
    parser.add_argument(
        "--start_layer_sp", type=float, default=0.0, help="Start layer sparsity."
    )
    parser.add_argument(
        "--target_layer_sp", type=float, default=0.68, help="Target layer sparsity."
    )
    parser.add_argument(
        "--edge_warmup_steps", type=int, help="Number of edge warmup steps."
    )
    parser.add_argument(
        "--layer_warmup_steps", type=int, help="Number of layer warmup steps."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2500, help="Number of warmup steps."
    )
    parser.add_argument(
        "--warmup_type",
        type=str,
        choices=["linear", "logarithmic"],
        default="linear",
        help="Type of warmup schedule.",
    )

    parser.add_argument(
        "--warmstart_scores_path",
        type=str,
        default=None,
        help="Path to load the pretrained importance scores",
    )
    parser.add_argument(
        "--warmstart_scale",
        type=float,
        default=1.0,
        help="Number to scale up the warmstart importance scores",
    )
    parser.add_argument(
        "--warmstart_target_min_score",
        type=float,
        default=-1.0,
        help="Min. value when scaling up the warmstart importance scores",
    )
    parser.add_argument(
        "--warmstart_target_max_score",
        type=float,
        default=1.0,
        help="Max. value when scaling up the warmstart importance scores",
    )
    parser.add_argument(
        "--warmstart_with_absolute",
        action="store_true",
        help="Whether use the absolute importance scores for warmstart",
    )
    return parser.parse_args(argv)


def load_base_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    with_embedding_nodes: bool = False,
) -> Tuple[Union[FPT2LMHeadModel, FQwen2ForCausalLM], PreTrainedTokenizer]:
    """
    Load a pre-trained Union[FPT2LMHeadModel, FQwen2ForCausalLM] and its corresponding tokenizer.

    Args:
        model_name (str): The name of the model to load.
        cache_dir (Optional[str]): Directory to cache model weights.
        device (str): The device to load the model onto, defaults to 'cuda'.
        with_embedding_nodes (bool): Flag to include embedding nodes in the model.

    Returns:
        Tuple[Union[FPT2LMHeadModel, FQwen2ForCausalLM], PreTrainedTokenizer]: The loaded model and tokenizer.
    """
    pretrained_name = (
        MODEL_NAME_TO_FULLNAME[model_name]
        if model_name in {"qwen2.5", "gemma2", "llama3"}
        else model_name
    )
    pretrained_class: Type[Union[FPT2LMHeadModel, FQwen2ForCausalLM]] = (
        MODEL_NAME_TO_MODEL_CLASS[model_name]
    )
    model = pretrained_class.from_pretrained(
        pretrained_name,
        cache_dir=cache_dir,
        with_embedding_nodes=with_embedding_nodes,
        attn_implementation=(
            "eager"
            if model_name in {"qwen2.5", "gemma2", "llama3"}
            else "flash_attention_2"
        ),
        torch_dtype=(
            torch.bfloat16 if model_name in {"qwen2.5", "gemma2", "llama3"} else None
        ),
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    tokenizer.pad_token = tokenizer.eos_token

    if model_name in {"qwen2.5", "gemma2", "llama3"}:
        tokenizer.padding_side = "right"

    return model, tokenizer


def freeze_all_except_pruning_params(
    circuit_model: Union[FPT2LMHeadModel, FQwen2ForCausalLM]
) -> None:
    """
    Freeze all model parameters except for the pruning-related ones (log_alpha,
    sparsity_lambda). This is used to create a "pruning-only" model for training.

    Args:
        circuit_model (Union[FPT2LMHeadModel, FQwen2ForCausalLM]): The model to freeze.
    """
    for n, p in circuit_model.named_parameters():
        if "log_alpha" in n or "sparsity_lambda" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def eval_fn(eval_pred):
    """
    Evaluation function for the model.

    Args:
        eval_pred (TrainerPrediction): The prediction object returned by the Trainer.

    Returns:
        dict: A dictionary containing the accuracy of the model, the model's edge and layer sparsity, the target edge and layer sparsity, and the KL loss, regularization edge loss, and regularization layer loss.
    """
    (
        predictions,
        target_edge_sparsity,
        target_layer_sparsity,
        model_edge_sparsity,
        model_layer_sparsity,
        reg_edge_loss,
        reg_layer_loss,
        kl_loss,
    ) = eval_pred.predictions
    if len(model_edge_sparsity.shape) > 0:
        model_edge_sparsity = model_edge_sparsity[0].item()
        model_layer_sparsity = model_layer_sparsity[0].item()
        target_edge_sparsity = target_edge_sparsity[0].item()
        target_layer_sparsity = target_layer_sparsity[0].item()
    else:
        model_edge_sparsity = model_edge_sparsity.item()
        model_layer_sparsity = model_layer_sparsity.item()
        target_edge_sparsity = target_edge_sparsity.item()
        target_layer_sparsity = target_layer_sparsity.item()

    labels = eval_pred.label_ids[:, 1:]

    eval_mask = (labels != -100).astype(int)
    predictions = predictions * eval_mask
    labels = labels * eval_mask

    correct = (predictions == labels).all(axis=1)
    accuracy = correct.sum().item() / correct.shape[0]

    kl_loss = kl_loss.mean().item()
    reg_edge_loss = reg_edge_loss.mean().item()
    reg_layer_loss = reg_layer_loss.mean().item()

    sparse_performance = TAU * accuracy + (1 - TAU) * model_edge_sparsity

    return {
        "eval_accuracy": accuracy,
        "model_edge_sparsity": model_edge_sparsity,
        "model_layer_sparsity": model_layer_sparsity,
        "target_edge_sparsity": target_edge_sparsity,
        "target_layer_sparsity": target_layer_sparsity,
        "eval_kl_loss": kl_loss,
        "eval_reg_edge_loss": reg_edge_loss,
        "eval_reg_layer_loss": reg_layer_loss,
        "eval_sparse_performance": sparse_performance,
    }


def run_single_task(
    base_model: Union[FPT2LMHeadModel, FQwen2ForCausalLM],
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    task: str,
    args: argparse.Namespace,
) -> None:
    """
    Execute attribution for a single (model, task) pair.

    Args:
        base_model (Union[FPT2LMHeadModel, FQwen2ForCausalLM]): The base model.
        model_name (str): Short model identifier.
        task (str): Task identifier.
        args (argparse.Namespace): Parsed CLI options (used for thresholds,
            file paths, etc.).
    """
    key = f"{task.replace('_', '-')}_{model_name}"
    if key not in COL_MAPPING:
        LOGGER.warning("Skipping '%s' - no column mapping defined.", key)
        return

    # get dataset
    train_ds = build_dataset(
        task,
        tokenizer,
        model_name,
        split="train",
        num_examples=args.num_examples,
    )
    wrapped_train_ds = EdgePruningAdapter(train_ds, tokenizer=tokenizer)

    eval_ds = build_dataset(
        task,
        tokenizer,
        model_name,
        split="validation",
        num_examples=args.num_examples,
    )
    wrapped_eval_ds = EdgePruningAdapter(eval_ds, tokenizer=tokenizer)

    collator = EPDataCollator(tokenizer=tokenizer, max_length=args.max_seq_len)

    # get circuit model
    pretrained_name = (
        MODEL_NAME_TO_FULLNAME[model_name]
        if model_name in {"qwen2.5", "gemma2", "llama3"}
        else model_name
    )
    pretrained_class: Type[Union[FPT2LMHeadModel, FQwen2ForCausalLM]] = (
        MODEL_NAME_TO_MODEL_CLASS[model_name]
    )
    circuit_model = pretrained_class.from_pretrained(
        pretrained_name,
        cache_dir=args.cache_dir,
        with_embedding_nodes=args.with_embedding_nodes,
        disable_linear_regularization_term=args.disable_linear_reg,
        attn_implementation=(
            "eager"
            if model_name in {"qwen2.5", "gemma2", "llama3"}
            else "flash_attention_2"
        ),
        torch_dtype=(
            torch.bfloat16 if model_name in {"qwen2.5", "gemma2", "llama3"} else None
        ),
    )
    freeze_all_except_pruning_params(circuit_model)

    # optimizer
    optimizers = build_optim(
        model=circuit_model,
        lr_edge=args.edge_learning_rate,
        lr_reg_edge=args.reg_edge_learning_rate,
        lr_layer=args.layer_learning_rate,
        lr_reg_layer=args.reg_layer_learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        disable_node_loss=args.disable_node_loss,
    )

    # trainer
    output_dir = f"{args.output_root}/{model_name}/{task}/steps_{args.max_steps}_warmstart_{args.warmstart_scale}"
    if args.warmstart_target_min_score and args.warmstart_target_max_score:
        output_dir += f"_range_{args.warmstart_target_min_score}_{args.warmstart_target_max_score}"
    wandb.init(  # type: ignore[attr-defined]
        project="mib-edge-pruning",
        entity="pcfgs-edge-pruning",
        name=f"{model_name}_{task}_edge_pruning",
        dir=output_dir,
        config=vars(args),
    )
    model_config_str = f"elr{args.edge_learning_rate}_llr{args.layer_learning_rate}_relr{args.reg_edge_learning_rate}_rllr{args.reg_layer_learning_rate}_tesp{args.target_edge_sp}_tlsp{args.target_layer_sp}_wm{args.warmup_steps}_steps_{args.max_steps}"
    model_output_dir = f"{output_dir}/{model_config_str}"

    sparsity_config = SparsityConfig(
        start_edge_sparsity=args.start_edge_sp,
        target_edge_sparsity=args.target_edge_sp,
        start_layer_sparsity=args.start_layer_sp,
        target_layer_sparsity=args.target_layer_sp,
        num_edge_sparsity_warmup_steps=args.edge_warmup_steps,
        num_layer_sparsity_warmup_steps=args.layer_warmup_steps,
        num_sparsity_warmup_steps=args.warmup_steps,
        warmup_type=args.warmup_type,
        skip_layer_loss_if_higher_sparsity=args.stop_layer_loss_if_target,
    )
    seq2seq_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        max_steps=args.max_steps,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        do_eval=True,
        per_device_eval_batch_size=args.batch_size,
        eval_steps=args.max_steps // 5,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=20,
        report_to=["wandb"],
        save_strategy="no" if args.sweep else "steps",
        predict_with_generate=False,
    )
    trainer = EPTrainer(
        model=circuit_model,
        base_model=base_model,
        data_collator=collator,
        train_dataset=wrapped_train_ds,
        eval_dataset=wrapped_eval_ds,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        sparsity_config=sparsity_config,
        args=seq2seq_args,
    )

    # overwrite logalphas with warmstart scores
    if args.warmstart_scores_path:
        edge_scores = load_edge_attr_scores(
            Path(args.warmstart_scores_path), abs_scores=args.warmstart_with_absolute
        )
        if os.path.exists(
            os.path.join(
                os.path.dirname(args.warmstart_scores_path),
                "statistics/importances_dist.json",
            )
        ):
            with open(
                os.path.join(
                    os.path.dirname(args.warmstart_scores_path),
                    "statistics/importances_dist.json",
                ),
                "r",
            ) as f:
                stats = json.load(f)["overall"]
                min_score, max_score = stats["min"], stats["max"]
        else:
            min_score, max_score = None, None

        set_log_alphas_from_attr_scores(
            circuit_model,
            edge_scores,
            policy="copy",
            scale=args.warmstart_scale,
            target_range=[
                args.warmstart_target_min_score,
                args.warmstart_target_max_score,
            ],
            min_score=min_score,
            max_score=max_score,
        )

    # run ep
    train_result = trainer.train()
    metrics = train_result.metrics

    circuit_dir = f"{args.circuit_dir}/edge-pruning_patching_edge/{task}_{model_name}/{model_config_str}"
    circuit_filepath = (
        f"{circuit_dir}/importances_warmstart_{args.warmstart_scale}.json"
    )
    if args.warmstart_target_min_score and args.warmstart_target_max_score:
        circuit_filepath = (
            f"{circuit_filepath.replace('.json', '')}"
            + f"_range_{args.warmstart_target_min_score}_{args.warmstart_target_max_score}.json"
        )
    save_edge_log_alphas(
        model=trainer.model,
        file_path=f"{circuit_dir}/importances_warmstart_{args.warmstart_scale}.json",
        overwrite=True,
    )

    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def main() -> None:
    """
    Entrypoint for console execution.
    """
    args = parse_args()
    set_seed(args.seed)

    if args.sweep:
        logging.info("Running sweep!")
    else:
        logging.info(f"Edge-pruning parameters:\n{args}")

    for model_name in args.models:
        base_model, base_tokenizer = load_base_model_and_tokenizer(
            model_name=model_name,
            cache_dir=args.cache_dir,
            device=args.device,
            with_embedding_nodes=args.with_embedding_nodes,
        )

        for task in args.tasks:
            run_single_task(
                base_model=base_model,
                tokenizer=base_tokenizer,
                model_name=model_name,
                task=task,
                args=args,
            )


if __name__ == "__main__":
    main()
