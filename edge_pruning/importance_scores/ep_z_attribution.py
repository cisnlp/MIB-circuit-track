"""
Implement post-hoc z-score (edge) attribution method.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from edge_pruning.importance_scores.logalphas_to_scores import (
    EdgeKey,
    _as_base_model,
    _mk_edge_key,
    reader_idx_to_name_fpt2,
    reader_idx_to_name_fqwen,
    writer_idx_to_name_fpt2,
    writer_idx_to_name_fqwen,
)
from edge_pruning.modeling.modeling_fpt2 import FPT2Model
from edge_pruning.modeling.modeling_fqwen_kv_expansion import FQwen2ForCausalLM

MetricFn = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]


def _is_edge_param(name: str) -> bool:
    """
    Checks if a model parameter is an edge parameter.

    Args:
        name (str): The parameter name.

    Returns:
        bool: True if the parameter is an edge parameter, False otherwise.
    """
    return name.endswith(
        (
            "final_read_log_alphas",
            "q_read_log_alphas",
            "k_read_log_alphas",
            "v_read_log_alphas",
            "mlp_read_log_alphas",
        )
    )


def _is_node_param(name: str) -> bool:
    """
    Checks if a model parameter is a node parameter.

    Args:
        name (str): The parameter name.

    Returns:
        bool: True if the parameter is a node parameter, False otherwise.
    """
    return name.endswith(
        ("attn_write_log_alphas", "mlp_write_log_alphas", "token_write_log_alpha")
    )


def _collect_log_alpha_params(
    model: nn.Module, include_nodes: bool
) -> List[Tuple[str, nn.Parameter]]:
    """
    Collects log_alpha parameters from the model.

    Args:
        model (nn.Module): The model from which to collect log_alpha parameters.
        include_nodes (bool): If True, includes node parameters in the collection.

    Returns:
        List[Tuple[str, nn.Parameter]]: A list of tuples containing the parameter name and
        the parameter itself for each log_alpha parameter that matches the criteria.

    Raises:
        RuntimeError: If no gate parameters are found in the model.
    """
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if "log_alpha" not in name:
            continue
        if _is_edge_param(name) or (include_nodes and _is_node_param(name)):
            if not p.requires_grad:
                p.requires_grad_(True)
            out.append((name, p))
    if not out:
        raise RuntimeError(
            "No gate parameters found. Expected names like '*_read_log_alphas', "
            "'final_read_log_alphas', and (optionally) '*_write_log_alphas'."
        )
    return out


def _kl_slice(
    sparse_logits: torch.Tensor, base_logits: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """
    Compute KL divergence between base model and sparse model logits over a slice.

    Args:
        sparse_logits (torch.Tensor): Logits from the sparse model (unnormalised).
        base_logits (torch.Tensor): Logits from the frozen teacher model (unnormalised).
        start (int): Slice start index (inclusive).
        end (int): Slice end index (exclusive).

    Returns:
        torch.Tensor: Scalar KL divergence.
    """
    s_logp = torch.nn.functional.log_softmax(sparse_logits[start:end], dim=-1)
    t_logp = torch.nn.functional.log_softmax(base_logits[start:end], dim=-1)
    return torch.nn.functional.kl_div(
        s_logp, t_logp, reduction="batchmean", log_target=True
    )


@torch.no_grad()
def _base_forward_for_ep(
    base_model: FQwen2ForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    corr_input_ids: torch.Tensor,
    corr_attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the *base* model on the clean/corrupted inputs to obtain writer states used
    by EP when mixing clean/corrupted along edges.

    Args:
        base_model (FQwen2ForCausalLM): The pre-trained (frozen) model.
        input_ids (torch.Tensor): Input ids for the clean input.
        attention_mask (Optional[torch.Tensor]): Attention mask for the clean input, if applicable.
        corr_input_ids (torch.Tensor): Input ids for the corrupted input.
        corr_attention_mask (Optional[torch.Tensor]): Attention mask for the corrupted input, if applicable.

    Returns:
        A tuple of (clean logits, corrupted writer states).
    """
    base_logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits

    corr_out = base_model(
        input_ids=corr_input_ids,
        attention_mask=corr_attention_mask,
        output_writer_states=True,
    )
    corr_writer_states = corr_out.writer_states
    return base_logits, corr_writer_states


def _sign_default_plus_one(x: torch.Tensor) -> torch.Tensor:
    s = x.sign()
    return s.masked_fill(s == 0, 1).to(torch.int8)


def z_score_attribution(
    sparse_model: FQwen2ForCausalLM,
    base_model: FQwen2ForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    reduce_over_examples: Literal["mean", "sum"] = "mean",
    custom_metric_fn: Optional[MetricFn] = None,
    include_nodes: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute post-hoc edge score attribution by differentiating a scalar metric M w.r.t. gate parameters log_alpha.

    Default M is KL(sparse || base) over the supervised tokens in each example (same as trainer).
    Because z increases monotonically with log_alpha in the (hard-)concrete parameterization, sign(dM/d log_alpha) == sign(dM/d z).

    Args:
        sparse_model (FQwen2ForCausalLM): The sparse (ciruit) model.
        base_model (FQwen2ForCausalLM): The base model to obtain ablation values and ground truth logits.
        dataloader (DataLoader): The dataloader to evaluate on.
        device (torch.device): The device.
        reduce_over_examples (Literal["mean", "sum"], optional): How to reduce over samples. Defaults to "mean".
        custom_metric_fn (Optional[ Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor] ], optional): Optional custom metric. Defaults to None.
        include_nodes (bool, optional): Whether to include entire nodes. Defaults to False.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Dicts with parameter name to gradient & sign of gradient.
    """
    sparse_model = sparse_model.to(device).eval()
    base_model = base_model.to(device).eval()
    sparse_model.set_soft_gates(True)

    # Choose which gate params to attribute
    named_gates: List[Tuple[str, nn.Parameter]] = _collect_log_alpha_params(
        sparse_model, include_nodes=include_nodes
    )

    # Accumulators
    grads_by_param: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, device=device) for n, p in named_gates
    }
    total_examples: int = 0

    for batch in tqdm(dataloader, desc="Z-Score attribution"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        attention_mask = (
            attention_mask.to(device) if attention_mask is not None else None
        )

        corr_input_ids = batch["corr_input_ids"].to(device)
        corr_attention_mask = batch.get("corr_attention_mask")
        corr_attention_mask = (
            corr_attention_mask.to(device) if corr_attention_mask is not None else None
        )

        start_idxes = batch["start_idxes"].to(device)
        end_idxes = batch["end_idxes"].to(device)
        B = int(input_ids.size(0))
        total_examples += B

        # Base model (no grad)
        with torch.no_grad():
            base_logits, corr_writer_states = _base_forward_for_ep(
                base_model=base_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corr_input_ids=corr_input_ids,
                corr_attention_mask=corr_attention_mask,
            )
            # EP sometimes shards writer states; ensure [writers, B, ...]
            if corr_writer_states.size(1) != B:
                try:
                    corr_writer_states = corr_writer_states.view(
                        corr_writer_states.size(0), B, *corr_writer_states.shape[2:]
                    )
                except Exception:
                    corr_writer_states = corr_writer_states.contiguous()

        # Zero grads only on gate params
        for _, p in named_gates:
            if p.grad is not None:
                p.grad.zero_()

        # Sparse model forward (with grad)
        out = sparse_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            corr_x=corr_writer_states,
        )
        sparse_logits: torch.Tensor = out["logits"]

        # Metric over batch for stability)
        per_ex_m: List[torch.Tensor] = []
        for i in range(B):
            s, e = int(start_idxes[i]), int(end_idxes[i])
            if custom_metric_fn is None:
                m_i = _kl_slice(sparse_logits[i], base_logits[i], s, e)
            else:
                m_i = custom_metric_fn(sparse_logits[i], base_logits[i], s, e)
            per_ex_m.append(m_i)
        M = torch.stack(per_ex_m).mean()
        M.backward()

        # Accumulate dM/d log_alpha
        for name, p in named_gates:
            if p.grad is not None:
                grads_by_param[name] += p.grad.detach()

    # Aggregate across dataset
    if reduce_over_examples not in {"mean", "sum"}:
        raise ValueError("reduce_over_examples must be 'mean' or 'sum'")
    if reduce_over_examples == "mean" and total_examples > 0:
        for n in grads_by_param:
            grads_by_param[n] = grads_by_param[n] / float(total_examples)

    signs_by_param: Dict[str, torch.Tensor] = {
        n: _sign_default_plus_one(grads_by_param[n]) for n in grads_by_param
    }

    # Disable soft gates
    sparse_model.set_soft_gates(False)

    # Move to CPU for serialization / downstream processing
    grads_by_param = {n: t.detach().cpu() for n, t in grads_by_param.items()}
    signs_by_param = {n: t.detach().cpu() for n, t in signs_by_param.items()}

    return grads_by_param, signs_by_param


def _param_identity_map(module: nn.Module) -> Dict[nn.Parameter, str]:
    """
    Build a reverse map from Parameter object identity to its qualified name.
    This lets us look up sign tensors by **object**, not by fragile string parsing.
    """
    out: Dict[nn.Parameter, str] = {}
    for name, p in module.named_parameters():
        out[p] = name
    return out


def _get_sign_for_param_obj(
    param_obj: nn.Parameter,
    param_to_name: Dict[nn.Parameter, str],
    signs_by_param: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Return the sign tensor for a specific parameter object (if present), else None.
    """
    name = param_to_name.get(param_obj)
    if name is None:
        return None
    return signs_by_param.get(name)


def merge_input_edge_signs(edge_signs: Dict[EdgeKey, int]) -> Dict[EdgeKey, int]:
    """
    Create explicit 'input->reader' signs by merging any 'tok_embeds->reader',
    'pos_embeds->reader', or 'embeds->reader' signs.

    Rule:
      - If any non-zero sign exists among the sources, use the first non-zero encountered.
      - If sources disagree (e.g., +1 and -1), fall back to +1 (benign default).
      - If all are zero or missing, default to +1 (consistent with your 0→+1 policy).

    Returns:
        A new dict with the original entries plus 'input->reader' keys.
    """
    merged: Dict[EdgeKey, int] = dict(edge_signs)
    by_reader: Dict[str, List[int]] = {}

    for k, s in edge_signs.items():
        if k.startswith(("tok_embeds->", "pos_embeds->", "embeds->")):
            _, reader = k.split("->", 1)
            by_reader.setdefault(reader, []).append(int(s))

    for reader, signs in by_reader.items():
        # prefer a non-zero if available; resolve conflicts to +1
        non_zero = [z for z in signs if z != 0]
        if not non_zero:
            chosen = 1
        else:
            chosen = non_zero[0]
            if any(z != chosen for z in non_zero[1:]):
                chosen = 1
        merged[f"input->{reader}"] = chosen

    return merged


def edge_signs_from_z_attribution(
    sparse_model: FQwen2ForCausalLM,
    base_model: FQwen2ForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    reduce_over_examples: Literal["mean", "sum"] = "mean",
    custom_metric_fn: Optional[MetricFn] = None,
    include_nodes: bool = False,
) -> Dict[EdgeKey, int]:
    """
    Convert per-parameter sign tensors (from z-score attribution) into **edge-level signs**.

    Args:
        model: EP-instrumented model (LM-head wrapper or bare decoder).
        signs_by_param: Mapping {parameter_name: sign_tensor}, where sign_tensor has the
            same shape as the corresponding log_alpha (e.g., [W,H] for q/k/v, [W] for MLP read,
            [W] for final_read).
        expand_qwen_kv: If True and the model is Qwen with GQA, expand compact
            KV-reader edges (e.g., 'aL.hk<k>') into full per-head readers.

    Returns:
        Dict[EdgeKey, int]: {"writer->reader": -1|0|+1} for all edges present in the model.
                            Missing/unsupported params are simply skipped.
    """
    # get z_attribution scores
    _, signs_by_param = z_score_attribution(
        sparse_model=sparse_model,
        base_model=base_model,
        dataloader=dataloader,
        device=device,
        reduce_over_examples=reduce_over_examples,
        custom_metric_fn=custom_metric_fn,
        include_nodes=include_nodes,
    )

    # assign param to edge
    base = _as_base_model(sparse_model)
    is_gpt2 = isinstance(base, FPT2Model)

    n_layer = base.n_layer if hasattr(base, "n_layer") else base.num_layers
    n_head = base.n_head if hasattr(base, "n_head") else base.num_heads
    num_kv_heads = 0 if is_gpt2 else base.num_kv_heads

    with_embed = getattr(base, "with_embedding_nodes", False)
    model_layers = base.h if hasattr(base, "h") else base.layers

    # name helpers
    def w_name(idx: int) -> str:
        if isinstance(base, FPT2Model):
            return writer_idx_to_name_fpt2(idx, n_layer, n_head, with_embed)
        else:
            return writer_idx_to_name_fqwen(idx, n_layer, n_head, with_embed)

    def r_name(idx: int) -> str:
        if isinstance(base, FPT2Model):
            return reader_idx_to_name_fpt2(idx, n_layer, n_head)
        else:
            return reader_idx_to_name_fqwen(idx, n_layer, n_head, num_kv_heads)

    # Build reverse lookup: Parameter object -> name (to fetch sign tensor reliably)
    param_to_name = _param_identity_map(
        sparse_model if isinstance(sparse_model, nn.Module) else base
    )

    edge_signs: Dict[EdgeKey, int] = {}

    # ---- Layer-wise edges ----
    for L, block in enumerate(model_layers):
        # figure out writer limits (only earlier writers can read into this layer)
        attn_limit = getattr(block, "attn_writer_offset", None)
        if attn_limit is None:
            attn_limit = block.attn_writer_idx
        mlp_limit = getattr(block, "mlp_writer_offset", None)
        if mlp_limit is None:
            mlp_limit = block.mlp_writer_idx

        # Signs for q/k/v and MLP-read
        sign_q = _get_sign_for_param_obj(
            block.q_read_log_alphas, param_to_name, signs_by_param
        )
        sign_k = _get_sign_for_param_obj(
            block.k_read_log_alphas, param_to_name, signs_by_param
        )
        sign_v = _get_sign_for_param_obj(
            block.v_read_log_alphas, param_to_name, signs_by_param
        )
        sign_m = _get_sign_for_param_obj(
            block.mlp_read_log_alphas, param_to_name, signs_by_param
        )

        # Reader index bases per arch
        if is_gpt2:
            per_layer = 3 * n_head + 1
            r_base = L * per_layer
            q_heads = k_heads = v_heads = n_head
            k_offset = n_head
            v_offset = 2 * n_head
        else:
            q_heads = sign_q.shape[1] if sign_q is not None else n_head
            k_heads = sign_k.shape[1] if sign_k is not None else n_head
            v_heads = sign_v.shape[1] if sign_v is not None else n_head
            per_layer = q_heads + 2 * k_heads + 1
            r_base = L * per_layer
            k_offset = q_heads
            v_offset = q_heads + k_heads

        # Q/K/V read edges (shape [W, H])
        if sign_q is not None:
            for w in range(attn_limit):
                w_str = w_name(w)
                for h in range(q_heads):
                    reader = r_name(r_base + h)
                    key = _mk_edge_key(w_str, reader)
                    edge_signs[key] = int(sign_q[w, h].item())

        if sign_k is not None:
            for w in range(attn_limit):
                w_str = w_name(w)
                for h in range(k_heads):
                    reader = r_name(r_base + k_offset + h)
                    key = _mk_edge_key(w_str, reader)
                    edge_signs[key] = int(sign_k[w, h].item())

        if sign_v is not None:
            for w in range(attn_limit):
                w_str = w_name(w)
                for h in range(v_heads):
                    reader = r_name(r_base + v_offset + h)
                    key = _mk_edge_key(w_str, reader)
                    edge_signs[key] = int(sign_v[w, h].item())

        # MLP read edges (shape [W])
        if sign_m is not None:
            mlp_reader_idx = (L + 1) * per_layer - 1
            mlp_reader = r_name(mlp_reader_idx)
            for w in range(mlp_limit):
                key = _mk_edge_key(w_name(w), mlp_reader)
                edge_signs[key] = int(sign_m[w].item())

    # ---- Final read edges to logits ----
    final_sign = _get_sign_for_param_obj(
        base.final_read_log_alphas, param_to_name, signs_by_param
    )
    if final_sign is not None:
        num_writers = base.n_writers if hasattr(base, "n_writers") else base.num_writers
        for w in range(num_writers):
            key = _mk_edge_key(w_name(w), "logits")
            edge_signs[key] = int(final_sign[w].item())

    edge_signs = merge_input_edge_signs(edge_signs)

    return edge_signs
