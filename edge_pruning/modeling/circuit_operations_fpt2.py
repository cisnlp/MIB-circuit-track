"""
Functions and modules to handle circuit.

Code given by *Finding Transformer Circuits with Edge Pruning*  (Adithya Bhaskar et al. 2024).
<https://github.com/princeton-nlp/Edge-Pruning>, and adapted for further circuit processing.

MIT License

Copyright (c) 2024 Adithya Bhaskar, Alexander Wettig, Dan Friedman, and Danqi Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import logging
from typing import List, Optional, Tuple

import torch

from edge_pruning.modeling.modeling_fpt2 import FPT2LMHeadModel


def find_edge_sparsity_threshold(
    model: FPT2LMHeadModel,
    left_bound: float,
    right_bound: float,
    target_sparsiy: torch.Tensor | float,
) -> torch.Tensor | float:
    """
    Perform a binary search to find the edge sparsity threshold that achieves the desired target sparsity.

    Args:
        model (FPT2LMHeadModel): The model on which to set the edge sparsity threshold.
        left_bound (float): The lower bound of the binary search interval.
        right_bound (float): The upper bound of the binary search interval.
        target_sparsiy (float): The desired sparsity level for the model's edges.

    Returns:
        torch.Tensor | float: The identified edge threshold.
    """
    while abs(right_bound - left_bound) > 1e-15:
        threshold = (left_bound + right_bound) / 2
        model.set_edge_threshold_for_deterministic(threshold)
        sparsity = model.get_edge_sparsity()
        if sparsity > target_sparsiy:
            right_bound = threshold
        else:
            left_bound = threshold

    return threshold


def find_node_sparsity_threshold(
    model: FPT2LMHeadModel,
    left_bound: float,
    right_bound: float,
    target_sparsiy: torch.Tensor | float,
) -> torch.Tensor | float:
    """
    Find the node threshold that makes the model have a sparsity of `target_sparsiy`.

    Args:
        model (FPT2LMHeadModel): The model to operate on.
        left_bound (float): The lower bound of the binary search.
        right_bound (float): The upper bound of the binary search.
        target_sparsiy (torch.Tensor): The target sparsity of the model.

    Returns:
        torch.Tensor | float: The identified node threshold.
    """
    while abs(right_bound - left_bound) > 1e-15:
        threshold = (left_bound + right_bound) / 2
        model.set_node_threshold_for_deterministic(threshold)
        sparsity = model.get_node_sparsity()
        if sparsity > target_sparsiy:
            right_bound = threshold
        else:
            left_bound = threshold

    return sparsity


def set_edge_sparsity_threshold(
    model: FPT2LMHeadModel,
    left_bound: float = 0,
    right_bound: float = 1,
    initial_sparsity: Optional[torch.Tensor] = None,
) -> torch.Tensor | float:
    """
    Find the edge threshold that makes the model have a sparsity of `target_sparsiy`.

    Args:
        model (FPT2LMHeadModel): The model to operate on.
        left_bound (float): The lower bound of the binary search.
        right_bound (float): The upper bound of the binary search.
        initial_sparsity (torch.Tensor): The target sparsity of the model.
    """
    if initial_sparsity is None:
        initial_sparsity = model.get_edge_sparsity()

    identified_threshold = find_edge_sparsity_threshold(
        model=model,
        left_bound=left_bound,
        right_bound=right_bound,
        target_sparsiy=initial_sparsity - 0.02,
    )

    return identified_threshold


def set_node_sparsity_threshold(
    model: FPT2LMHeadModel,
    left_bound: float = 0,
    right_bound: float = 1,
    initial_sparsity: Optional[torch.Tensor] = None,
) -> torch.Tensor | float:
    """
    Find the node threshold that makes the model have a sparsity of `target_sparsiy`.

    Args:
        model (FPT2LMHeadModel): The model to operate on.
        left_bound (float): The lower bound of the binary search.
        right_bound (float): The upper bound of the binary search.
        initial_sparsity (torch.Tensor): The target sparsity of the model.
    """
    if initial_sparsity is None:
        initial_sparsity = model.get_node_sparsity()

    identified_threshold = find_node_sparsity_threshold(
        model=model,
        left_bound=left_bound,
        right_bound=right_bound,
        target_sparsiy=initial_sparsity + 0.01,
    )

    return identified_threshold


def obtain_circuit_edges(
    model: FPT2LMHeadModel,
    edge_sparsity: Optional[torch.Tensor],
    node_sparsity: Optional[torch.Tensor],
) -> List[Tuple[str, str]]:
    """
    Obtain the circuit edges from a given model by determining the sparsity thresholds.

    Args:
        model (FPT2LMHeadModel): The model to process for obtaining circuit edges.
        edge_sparsity (Optional[torch.Tensor]): The desired sparsity level for edges.
        node_sparsity (Optional[torch.Tensor]): The desired sparsity level for nodes.

    Returns:
        List[Tuple[str, str]]: A list of edges from the model based on the calculated sparsity thresholds.
    """

    edge_sparsity_threshold = set_edge_sparsity_threshold(
        model=model, left_bound=0.0, right_bound=1.0, initial_sparsity=edge_sparsity
    )
    if isinstance(edge_sparsity_threshold, torch.Tensor):
        edge_sparsity_threshold = edge_sparsity_threshold.item()

    node_sparsity_threshold = set_node_sparsity_threshold(
        model=model, left_bound=0.0, right_bound=1.0, initial_sparsity=node_sparsity
    )
    if isinstance(node_sparsity_threshold, torch.Tensor):
        node_sparsity_threshold = node_sparsity_threshold.item()

    overall_edge_sparsity = model.get_effective_edge_sparsity()
    if isinstance(overall_edge_sparsity, torch.Tensor):
        overall_edge_sparsity = overall_edge_sparsity.item()

    logging.info(f"Node sparsity threshold: {node_sparsity_threshold}")
    logging.info(f"Edge sparsity threshold: {edge_sparsity_threshold}")
    logging.info(f"Overall edge sparsity: {overall_edge_sparsity}")

    edges = model.get_edges()

    return edges
