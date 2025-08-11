"""
Utility functions and modules.
"""

import logging

import torch
from torch import nn
from transformers import PreTrainedTokenizerFast


def load_tokenizer_from_checkpoint(model_dir: str) -> PreTrainedTokenizerFast:
    """
    Load the tokenizer from a model checkpoint directory.

    Args:
        model_dir (str): Path to the directory containing the saved model and tokenizer.

    Returns:
        PreTrainedTokenizerFast: The loaded tokenizer.
    """
    logging.info(f"Loading tokenizer from checkpoint: {model_dir}")
    return PreTrainedTokenizerFast.from_pretrained(model_dir)


def clip_log_alpha_(
    tensor: torch.Tensor,
    nan_val: float = 10.0,
    min_val: float = 9.99,
    max_val: float = 10.01,
) -> None:
    """
    Clips the values of a tensor in-place to handle NaNs and ensure they fall within a specified range.

    Args:
        tensor (torch.Tensor): The tensor whose values are to be clipped.
        nan_val (float, optional): The value to replace NaNs, positive infinity, and negative infinity with.
            Defaults to 10.0.
        min_val (float, optional): The minimum allowable value for the tensor. Defaults to 9.99.
        max_val (float, optional): The maximum allowable value for the tensor. Defaults to 10.01.

    Returns:
        None: The operation is performed in-place, and the tensor is modified directly.
    """

    with torch.no_grad():
        tensor.nan_to_num_(nan=nan_val, posinf=max_val, neginf=min_val)


def clip_log_alphas(model: nn.Module):
    """
    In-place clip all log-alpha values in a model.

    Args:
        model (nn.Module): The model containing the log-alpha values to be clipped.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "log_alpha" in name and not param.is_meta:
                clip_log_alpha_(param)
