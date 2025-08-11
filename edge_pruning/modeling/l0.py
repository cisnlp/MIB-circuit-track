"""
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

"""

from typing import Union

import torch

LIMIT_LEFT = -0.1
LIMIT_RIGHT = 1.1
RANGE = LIMIT_RIGHT - LIMIT_LEFT
EPS = 1e-6
TEMPERATURE = 2 / 3
FACTOR = 0.8


def _safe_clamp(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Clamps x to be within (eps, 1-eps), where eps is a small positive value.
    This is useful for avoiding hard zeros/ones when using torch.clamp.

    Args:
        x (torch.Tensor): The input to be clamped.
        eps (float, optional): The small positive value for robustness. Defaults to EPS.

    Returns:
        torch.Tensor: The clamped tensor.
    """
    return torch.clamp(x, eps, 1.0 - eps)


def _safe_logit(p: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Computes the logit of a probability, safely.
    Avoids log(0) and log(1) by clamping to a small positive value.
    This is useful for avoiding hard zeros/ones when using torch.clamp.

    Args:
        p (torch.Tensor): The input probability.
        eps (float, optional): The small positive value for robustness. Defaults to EPS.

    Returns:
        torch.Tensor: The computed logit.
    """
    p = _safe_clamp(p, eps)
    return torch.log(p) - torch.log1p(-p)  # log1p keeps precision for tiny p


def cdf_stretched_concrete(
    x: Union[float, torch.Tensor], log_alpha: torch.Tensor
) -> torch.Tensor:
    """
    Computes the cumulative distribution function (CDF) of the stretched concrete
    distribution for the input x given the log scale parameter log_alpha.

    Args:
        x (Union[float, torch.Tensor]): The input value(s) to compute the CDF for,
            can be a float or a tensor. If a float, it is converted to a tensor.
        log_alpha (torch.Tensor): The logarithm of the scale parameter alpha,
            must be a tensor.

    Returns:
        torch.Tensor: The CDF value(s) as a tensor, clamped within the range
        defined by EPS.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=log_alpha.dtype, device=log_alpha.device)

    x01 = (x - LIMIT_LEFT) / RANGE  # stretch to (-eps,1+eps) -> (0,1)
    logits = TEMPERATURE * _safe_logit(x01) - log_alpha  # T*(logit) - logα
    return _safe_clamp(torch.sigmoid(logits))


def sample_z_from_u(u: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Samples a tensor z from a given uniform random tensor u and a log-scale
    parameter log_alpha, using the stretched concrete distribution.

    Args:
        u (torch.Tensor): A uniform random tensor, with values in the range (0, 1).
        log_alpha (torch.Tensor): The logarithm of the scale parameter alpha.

    Returns:
        torch.Tensor: A tensor z sampled from the stretched concrete distribution,
        with values in the range defined by `LIMIT_LEFT` and `LIMIT_RIGHT`.
    """
    u = _safe_clamp(u)
    inner = (torch.logit(u, EPS) + log_alpha) / TEMPERATURE
    s = torch.sigmoid(inner)
    return RANGE * s + LIMIT_LEFT


def continuous_z_from_log_alpha(log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Computes a continuous mask from a given log-scale parameter log_alpha.
    The result is a tensor in the range [0, 1], where 0 means the edge is
    completely pruned and 1 means the edge is completely kept.

    Args:
        log_alpha (torch.Tensor): The logarithm of the scale parameter alpha.

    Returns:
        torch.Tensor: The continuous mask, with values in the range [0, 1].
    """
    return torch.sigmoid(log_alpha * FACTOR / TEMPERATURE)


def deterministic_z_from_log_alpha(
    log_alpha: torch.Tensor, apply_one: bool = False
) -> torch.Tensor:
    """
    Computes a deterministic binary mask from a given log-scale parameter log_alpha.

    Args:
        log_alpha (torch.Tensor): The logarithm of the scale parameter alpha.
        apply_one (bool, optional): Whether to set non-zero values in the mask to 1.0.
            Defaults to False.

    Returns:
        torch.Tensor: The deterministic binary mask, with values in {0, 1}.
    """
    size = log_alpha.numel()

    p_zero = cdf_stretched_concrete(0.0, log_alpha)  # shape == log_alpha
    expected_zeros = torch.sum(p_zero)  # scalar
    num_zeros = int(torch.round(expected_zeros).clamp(0, size).item())

    # Soft scores in (0,1).  Using clamp keeps back-prop stable in earlier phases
    soft_mask = torch.sigmoid(log_alpha * FACTOR / TEMPERATURE).flatten()

    if num_zeros > 0 and soft_mask.numel() > 0:
        _, idx = torch.topk(soft_mask, k=num_zeros, largest=False)
        soft_mask[idx] = 0.0
        if apply_one:
            soft_mask[soft_mask > 0] = 1.0

    return soft_mask.reshape_as(log_alpha)


def sample_z_from_log_alpha(log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Samples a tensor z from a given log-scale parameter log_alpha, using the stretched concrete distribution.

    Args:
        log_alpha (torch.Tensor): The logarithm of the scale parameter alpha.

    Returns:
        torch.Tensor: A tensor z sampled from the stretched concrete distribution, with values in the range [0, 1].
    """
    u = torch.empty_like(log_alpha).uniform_(EPS, 1.0 - EPS)
    z = sample_z_from_u(u, log_alpha)
    # hard clip to [0,1] – F.hardtanh is overkill here
    return torch.clamp(z, 0.0, 1.0)
