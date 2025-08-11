"""
Grouping and scheduler utilities for sparsity-aware optimisation.

Code given by *Finding Transformer Circuits with Edge Pruning*  (Adithya Bhaskar et al. 2024).
<https://github.com/princeton-nlp/Edge-Pruning>

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

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from typing import List, Tuple

import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


def _split_params(
    model: nn.Module, disable_node_loss: bool
) -> Tuple[List[nn.Parameter], ...]:
    """
    Split parameters of a model into four groups: edge parameters, regularization strength for edge parameters,
    layer parameters, and regularization strength for layer parameters.

    Args:
        model: Model containing the parameters to be grouped
        disable_node_loss: Whether to disable node loss regularization

    Returns:
        A tuple of four lists: the first contains edge parameters, the second contains regularization strength for
        edge parameters, the third contains layer parameters, and the fourth contains regularization strength for
        layer parameters.
    """
    g_edge, g_reg_edge, g_layer, g_reg_layer = [], [], [], []

    for name, param in model.named_parameters():
        if "write_log_alpha" in name:
            g_layer.append(param)
        elif "read_log_alpha" in name:
            g_edge.append(param)
        elif "sparsity_lambda_edge" in name:
            g_reg_edge.append(param)
        elif "sparsity_lambda_node" in name and not disable_node_loss:
            g_reg_layer.append(param)

    return g_edge, g_reg_edge, g_layer, g_reg_layer


def build_optim(
    model: nn.Module,
    lr_edge: float,
    lr_reg_edge: float,
    lr_layer: float,
    lr_reg_layer: float,
    max_steps: int,
    warmup_steps: int,
    disable_node_loss: bool,
):
    """
    Build AdamW optimizer and scheduler for FPT model.

    Args:
        model: FPT model
        lr_edge: Learning rate for edge parameters
        lr_reg_edge: Learning rate for regularization strength of edge parameters
        lr_layer: Learning rate for layer parameters
        lr_reg_layer: Learning rate for regularization strength of layer parameters
        max_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        disable_node_loss: If True, disable node loss

    Returns:
        Tuple of AdamW optimizer and LinearWarmup scheduler
    """
    g_edge, g_reg_edge, g_layer, g_reg_layer = _split_params(model, disable_node_loss)

    optim = AdamW(
        [
            {"params": g_edge, "lr": lr_edge},
            {"params": g_reg_edge, "lr": lr_reg_edge, "maximize": True},
            {"params": g_layer, "lr": lr_layer},
            {"params": g_reg_layer, "lr": lr_reg_layer, "maximize": True},
        ]
    )

    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return optim, sched
