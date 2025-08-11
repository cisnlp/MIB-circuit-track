"""
Utility functions and modules.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from transformers import PreTrainedTokenizer

from edge_pruning.modeling.modeling_fpt2 import FPT2LMHeadModel
from edge_pruning.modeling.modeling_fqwen_kv_expansion import FQwen2ForCausalLM
from MIB_circuit_track.dataset import HFEAPDataset

TASKS_TO_HF_NAMES = {
    "ioi": "ioi",
    "mcqa": "copycolors_mcqa",
    "arithmetic_addition": "arithmetic_addition",
    "arithmetic_subtraction": "arithmetic_subtraction",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
}

MODEL_NAME_TO_FULLNAME = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}

MODEL_NAME_TO_MODEL_CLASS: Dict[
    str, Type[Union[FPT2LMHeadModel, FQwen2ForCausalLM]]
] = {
    "gpt2": FPT2LMHeadModel,
    "qwen2.5": FQwen2ForCausalLM,
}

"""
This script will print a table of the following form:
Method      | IOI (GPT) | IOI (QWen) | IOI (Gemma) | IOI (Llama) | MCQA (QWen) | MCQA (Gemma) | MCQA (Llama) | Arithmetic (Llama) | ARC-E (Gemma) | ARC-E (Llama) | ARC-C (Llama)
Random      |
Method 1    |
Method 2    |
...
"""

COL_MAPPING = {
    "ioi_gpt2": 0,
    "ioi_qwen2.5": 1,
    "ioi_gemma2": 2,
    "ioi_llama3": 3,
    "mcqa_qwen2.5": 4,
    "mcqa_gemma2": 5,
    "mcqa_llama3": 6,
    "arithmetic-addition_llama3": 7,
    "arithmetic-subtraction_llama3": 8,
    "arc-easy_gemma2": 9,
    "arc-easy_llama3": 10,
    "arc-challenge_llama3": 11,
    "ioi_interpbench": None,
}


def load_pickle(path: Path) -> Dict[str, Any]:
    """
    Load a ``pickle`` file from *path*.

    Args:
        path (Path): File ending in ``.pkl`` or ``.pt``.

    Returns:
        Dict[str, Any]: Un-pickled Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pickle.UnpicklingError: If the file cannot be un-pickled.
    """
    with path.open("rb") as handle:
        return pickle.load(handle)


def model_registry(model_key: str) -> str:
    """
    Map a user-facing *model_key* to the actual HuggingFace identifier.

    Args:
        model_key (str): Either ``"gpt2-small"`` or ``"qwen"``.

    Returns:
        str: HuggingFace model identifier.

    Raises:
        ValueError: If *model_key* is unknown.
    """
    if model_key == "gpt2-small":
        return "gpt2-small"
    if model_key == "qwen":
        return "Qwen/Qwen2.5-0.5B-Instruct"
    raise ValueError(f"Unknown model key: {model_key!r}")


def build_dataset(
    task: str,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    split: str,
    num_examples: Optional[int] = None,
) -> HFEAPDataset:
    """
    Construct a ``HFEAPDataset`` for a given task.

    Args:
        task (str): Internal task identifier (e.g. ``arc_easy``).
        tokenizer (PreTrainedTokenizer): Tokenizer of the model.
        model_name (str): Abbreviated model identifier.
        split (str): One of ``train``, ``validation`` or ``test``.
        num_examples (Optional[int]): Truncate to num_examples (``None`` -> no truncation).

    Returns:
        HFEAPDataset: The task-specific dataset.
    """
    hf_task_name = f"mib-bench/{TASKS_TO_HF_NAMES[task]}"
    dataset = HFEAPDataset(
        hf_task_name,
        tokenizer,
        split=split,
        task=task,
        model_name=model_name,
        num_examples=num_examples,
    )
    logging.info(
        "Loaded %d examples for task '%s' (%s split).", len(dataset), task, split
    )
    return dataset
