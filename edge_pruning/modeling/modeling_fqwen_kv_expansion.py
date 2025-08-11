"""
Adapts HF transformers code for Qwen-2.5 <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py>
to integrate edge-pruning (Adithya Bhaskar et al. 2024) <https://github.com/princeton-nlp/Edge-Pruning>.

The original HF transformers code is publsihed under the Apache License:

Copyright 2018- The Hugging Face team. All rights reserved.

        Apache License
    Version 2.0, January 2004
http://www.apache.org/licenses/
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import ModelOutput, logging

from edge_pruning.modeling.l0 import (
    continuous_z_from_log_alpha,
    deterministic_z_from_log_alpha,
    sample_z_from_log_alpha,
)
from edge_pruning.modeling.modeling_fllama import (
    num_nodes,
    num_writers,
    writer_idx_to_name,
    writer_name_to_idx,
)

logger = logging.get_logger(__name__)


def readers_per_layer(
    num_heads: int, num_key_value_heads: int, expanded_kv: bool
) -> int:
    """
    Per layer readers = Q block + K block + V block + 1 (MLP).
    With expanded-KV gating, K and V each have num_heads readers.
    """
    kv_size = num_heads if expanded_kv else num_key_value_heads
    return num_heads + 2 * kv_size + 1


def reader_idx_to_name(
    reader_idx: int,
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    expanded_kv: bool = True,
) -> str:
    per_layer = readers_per_layer(num_heads, num_key_value_heads, expanded_kv)
    layer_idx = reader_idx // per_layer
    head_idx = reader_idx % per_layer

    if layer_idx == num_layers:
        return "resid_post"

    kv_size = num_heads if expanded_kv else num_key_value_heads

    # Q range: [0, num_heads)
    if head_idx < num_heads:
        return f"a{layer_idx}.h{head_idx}.q"

    # K range: [num_heads, num_heads + kv_size)
    elif head_idx < num_heads + kv_size:
        return f"a{layer_idx}.h{head_idx - num_heads}.k"

    # V range: [num_heads + kv_size, num_heads + 2*kv_size)
    elif head_idx < num_heads + 2 * kv_size:
        return f"a{layer_idx}.h{head_idx - num_heads - kv_size}.v"

    # MLP is the final slot per layer
    else:
        return f"m{layer_idx}"


def num_readers(config, expanded_kv: bool = True) -> int:
    """
    Readers per layer:
      Q: H
      K: H if expanded_kv else H_kv
      V: H if expanded_kv else H_kv
      MLP: 1
    Plus one final read ("resid_post").
    """
    H = config.num_attention_heads
    H_kv = config.num_key_value_heads
    kv = H if expanded_kv else H_kv

    per_layer = H + 2 * kv + 1
    return config.num_hidden_layers * per_layer + 1


def num_edges(
    config, with_embedding_nodes: bool = False, expanded_kv: bool = True
) -> int:
    """
    Edge counting that matches your gating rules:
      - An attention *writer head* in layer l connects to:
          * this layer's MLP (1)
          * all readers of future layers (num_future_layers * per_layer)
          * the final read (1)
      - The MLP writer in layer l connects to:
          * all readers of future layers (num_future_layers * per_layer)
          * the final read (1)
      - (Optional) an embedding writer connects to all readers + final read.
    """
    H = config.num_attention_heads
    H_kv = config.num_key_value_heads
    L = config.num_hidden_layers
    kv = H if expanded_kv else H_kv
    per_layer = H + 2 * kv + 1

    # Optional embedding writer edges: to all readers + resid_post
    n_edges = (
        num_readers(config, expanded_kv=expanded_kv) if with_embedding_nodes else 0
    )

    for l in range(L):
        num_future_layers = L - l - 1

        # Attn writer heads in this layer
        n_edges += H * (
            1  # this layer's MLP
            + num_future_layers * per_layer  # all future readers
            + 1
        )  # final read

        # MLP writer in this layer
        n_edges += num_future_layers * per_layer + 1  # future readers + final read

    return n_edges


def get_mask(
    log_alpha: torch.Tensor,
    training: bool = False,
    threshold_for_deterministic: Optional[float] = None,
    apply_one: bool = False,
    soft_gate: bool = False,
) -> torch.Tensor:
    """
    Computes a binary mask from a given log-scale parameter log_alpha.

    Args:
        log_alpha (torch.Tensor): The logarithm of the scale parameter alpha.
        training (bool, optional): Whether the model is in training mode. Defaults to False.
        threshold_for_deterministic (float, optional): The threshold for obtaining a binary mask. Defaults to None.
        apply_one (bool, optional): Whether to set non-zero values in the mask to 1. Defaults to False.

    Returns:
        torch.Tensor: The binary mask, with values in {0, 1}.
    """
    if soft_gate:
        return continuous_z_from_log_alpha(log_alpha)
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask


class FQwen2Attention(Qwen2Attention):
    """
    Adaption of Qwen2Attention to allow for edge-pruning. Specifically, we pass hidden states for Q, K, V matrices independently.
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    def _apply_headwise_linear(
        self,
        x_heads: Tensor,
        proj: nn.Linear,
        num_heads: int,
    ) -> Tensor:
        """
        Per-head linear projection with bias.

        Args:
            x_heads (Tensor): The Q,K,V matrices across different heads.
            proj (nn.Linear): The linear projection.
            num_heads (int): The number of heads.

        Returns:
            Tensor: (B, H, S, head_dim)
        """
        # weight: (H, head_dim, hidden_size)  →  permute to (H, hidden_size, head_dim)
        w = proj.weight.view(num_heads, self.head_dim, self.hidden_size).permute(
            0, 2, 1
        )

        # hbsk, hkd  →  bhsd       (k = hidden_size, d = head_dim)
        out = torch.einsum("hbsk,hkd->bhsd", x_heads, w)

        if proj.bias is not None:
            bias = proj.bias.view(num_heads, self.head_dim)  # (H, head_dim)
            out = out + bias[None, :, None, :]  # broadcast B and S

        return out  # (B, H, S, head_dim)

    def _apply_headwise_linear_with_kv_sharing(
        self, x_heads: Tensor, proj: nn.Linear, num_kv_heads: int, num_heads: int
    ) -> Tensor:
        """
        x_heads: (H, B, S, hidden_size). We share weights across KV groups.
        Returns: (B, H, S, head_dim)
        """
        H, B, S, K = x_heads.shape
        group = num_heads // num_kv_heads

        # proj.weight: (num_kv_heads*head_dim, hidden_size)
        w_kv = proj.weight.view(num_kv_heads, self.head_dim, self.hidden_size).permute(
            0, 2, 1
        )  # (H_kv, K, D)
        w = w_kv.repeat_interleave(group, dim=0)[:H]  # (H, K, D)

        out = torch.einsum("hbsk,hkd->bhsd", x_heads, w)

        if proj.bias is not None:
            b = proj.bias.view(num_kv_heads, self.head_dim).repeat_interleave(
                group, dim=0
            )[
                :H
            ]  # (H, D)
            out = out + b[None, :, None, :]  # (B, H, S, D)

        return out

    def _apply_output_linear(
        self,
        x: Tensor,
        proj: nn.Linear,
        num_heads: int,
        chunk_size: int = 4,
    ) -> Tensor:
        """
        Memory-friendly out-projection.

        Returns (H, B, S, hidden_size) — same shape your `attn_write`
        already expects, but built in small head-chunks to avoid a huge
        temporary.
        """
        B, H, S, D = x.shape
        assert H == num_heads and D == self.head_dim

        out = x.new_empty(H, B, S, self.hidden_size)  # destination buffer

        weight = proj.weight.view(self.hidden_size, H, D)  # (M, H, D)
        bias = proj.bias  # may be None

        for start in range(0, H, chunk_size):
            end = min(start + chunk_size, H)

            # slice
            x_chunk = x[:, start:end]  # (B, h', S, D)
            w_chunk = weight[:, start:end, :]  # (M, h', D)

            # bhsd, mhd → hbsm  (out: h', B, S, M)
            proj_chunk = torch.einsum("bhsd,mhd->hbsm", x_chunk, w_chunk)

            if bias is not None:
                proj_chunk += bias.view(1, 1, 1, -1)

            out[start:end] = proj_chunk

        return out.contiguous()  # (H, B, S, hidden_size)

    def forward(
        self,
        q_hidden_states: Tensor,  # (H_q, B, S, D)
        k_hidden_states: Tensor,  # (H_kv, B, S, D)
        v_hidden_states: Tensor,  # (H_kv, B, S, D)
        attention_mask: Optional[Tensor] = None,  # broadcastable
        position_ids: Optional[Tensor] = None,  # (B, S)
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Cache]]:
        _, bsz, q_len, _ = q_hidden_states.shape

        # Per-head linear projections
        query_states = self._apply_headwise_linear(
            q_hidden_states, self.q_proj, self.num_heads
        )
        key_states = self._apply_headwise_linear_with_kv_sharing(
            k_hidden_states, self.k_proj, self.num_key_value_heads, self.num_heads
        )
        value_states = self._apply_headwise_linear_with_kv_sharing(
            v_hidden_states, self.v_proj, self.num_key_value_heads, self.num_heads
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = self._apply_output_linear(
            attn_output, self.o_proj, self.num_heads
        )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": FQwen2Attention,  # other support needs to be implemented still
}


# ============================================================================
#  Decoder Layer with *edge-pruning* read / write logic
# ============================================================================


class FQwen2DecoderLayer(nn.Module):
    """
    One transformer block with *edge-pruning* instrumentation.

    Incoming *hidden-states* tensor shape:
        (W, B, S, D)   - `W == n_writers(config)`
    """

    def __init__(
        self, config: Qwen2Config, layer_idx: int, *, with_embedding_nodes: bool
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.with_embedding_nodes = with_embedding_nodes

        if config.use_sliding_window:
            logging.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        # Core sub-modules
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

        #  Book-keeping for edge-pruning masks
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_writers = num_writers(
            config, with_embedding_nodes=with_embedding_nodes
        )
        self.num_readers = num_readers(config, expanded_kv=True)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.mlp.gate_proj.weight.dtype

        # Index offsets for this layer
        writer_offset = 1 if with_embedding_nodes else 0
        self.attn_writer_idx = writer_offset + layer_idx * (self.num_heads + 1)
        self.attn_reader_idx = layer_idx * (self.num_heads + 2 * self.num_heads + 1)
        self.mlp_writer_idx = writer_offset + (layer_idx + 1) * (self.num_heads + 1) - 1
        self.mlp_reader_idx = (layer_idx + 1) * (
            self.num_heads + 2 * self.num_heads + 1
        ) - 1

        self.q_read_log_alphas = nn.Parameter(
            torch.empty(self.num_writers, self.num_heads, dtype=self._dtype)
        )
        self.k_read_log_alphas = nn.Parameter(
            torch.empty(self.num_writers, self.num_heads, dtype=self._dtype)
        )
        self.v_read_log_alphas = nn.Parameter(
            torch.empty(self.num_writers, self.num_heads, dtype=self._dtype)
        )
        self.attn_write_log_alphas = nn.Parameter(
            torch.empty(self.num_heads, dtype=self._dtype)
        )
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)

        attn_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        attn_read_common_mask[: self.attn_writer_idx] = 1
        attn_read_common_mask = attn_read_common_mask.unsqueeze(1)
        self.register_buffer("attn_read_common_mask", attn_read_common_mask)

        attn_write_common_mask = F.pad(
            torch.eye(self.num_heads, dtype=torch.float32).to(
                self._dtype
            ),  # eye does not support bfloat16
            (
                self.attn_writer_idx,
                self.num_writers - self.attn_writer_idx - self.num_heads,
                0,
                0,
            ),
        )
        self.register_buffer("attn_write_common_mask", attn_write_common_mask)

        self.mlp_read_log_alphas = nn.Parameter(
            torch.empty(self.num_writers, dtype=self._dtype)
        )
        self.mlp_write_log_alphas = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

        mlp_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        mlp_read_common_mask[: self.mlp_writer_idx] = 1
        self.register_buffer("mlp_read_common_mask", mlp_read_common_mask)

        mlp_write_common_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        mlp_write_common_mask[self.mlp_writer_idx, 0] = 1
        self.register_buffer("mlp_write_common_mask", mlp_write_common_mask)

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

    @torch.no_grad()
    def get_edge_masks(self, soft_gate: bool = False):
        z_q = get_mask(
            self.q_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )
        z_q = z_q[: self.attn_writer_idx, :]
        z_k = get_mask(
            self.k_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )
        z_k = z_k[: self.attn_writer_idx, :]
        z_v = get_mask(
            self.v_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )
        z_v = z_v[: self.attn_writer_idx, :]

        z_mlp = get_mask(
            self.mlp_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )
        z_mlp = z_mlp[: self.mlp_writer_idx]

        return (z_q, z_k, z_v, z_mlp)

    @torch.no_grad()
    def get_node_masks(self, soft_gate: bool = False):
        z_attn = get_mask(
            self.attn_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic,
            soft_gate=soft_gate,
        )

        z_mlp = get_mask(
            self.mlp_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic,
            soft_gate=soft_gate,
        ).reshape([])

        return (z_attn, z_mlp)

    @torch.no_grad()
    def set_attn_mask_value(self, from_idx, head_idx, qkv, value):
        if qkv == "q":
            old_value = self.q_read_log_alphas[from_idx, head_idx].detach().item()
            self.q_read_log_alphas[from_idx, head_idx] = value
        elif qkv == "k":
            old_value = self.k_read_log_alphas[from_idx, head_idx].detach().item()
            self.k_read_log_alphas[from_idx, head_idx] = value
        elif qkv == "v":
            old_value = self.v_read_log_alphas[from_idx, head_idx].detach().item()
            self.v_read_log_alphas[from_idx, head_idx] = value
        else:
            raise ValueError(f"Unrecognized qkv {qkv}")
        return old_value

    @torch.no_grad()
    def set_mlp_mask_value(self, from_idx, value):
        old_value = self.mlp_read_log_alphas[from_idx].detach().item()
        self.mlp_read_log_alphas[from_idx] = value
        return old_value

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

    @torch.no_grad()
    def load_attn_log_alphas(self, attn_in_edges):
        # Fill with -10 by default
        self.q_read_log_alphas.data.fill_(-10)
        self.k_read_log_alphas.data.fill_(-10)
        self.v_read_log_alphas.data.fill_(-10)

        for writer_idx, reader in attn_in_edges:
            reader_portions = reader.split(".")
            assert len(reader_portions) == 3, f"Invalid reader format: {reader}"
            layer_idx = int(reader_portions[0][1:])
            head = int(reader_portions[1][1:])
            qkv = reader_portions[2]
            assert layer_idx == self.layer_idx, f"Invalid layer index: {layer_idx}"
            if qkv == "q":
                self.q_read_log_alphas[writer_idx, head] = 10
            elif qkv == "k":
                self.k_read_log_alphas[writer_idx, head] = 10
            elif qkv == "v":
                self.v_read_log_alphas[writer_idx, head] = 10

        # Fill with 10 for node masks, since we don't want any further restraint on edges
        self.attn_write_log_alphas.data.fill_(10)

    @torch.no_grad()
    def load_mlp_log_alphas(self, mlp_in_edges):
        # Fill with -10 by default
        self.mlp_read_log_alphas.data.fill_(-10)

        for writer_idx, reader in mlp_in_edges:
            reader_portions = reader.split(".")
            assert len(reader_portions) == 1, f"Invalid reader format: {reader}"
            layer_idx = int(reader_portions[0][1:])
            assert layer_idx == self.layer_idx, f"Invalid layer index: {layer_idx}"
            self.mlp_read_log_alphas[writer_idx] = 10

        # Fill with 10 for node masks, since we don't want any further restraint on edges
        self.mlp_write_log_alphas.data.fill_(10)

    def attn_read(self, x, corr_x=None, embeds=None, soft_gate: bool = False):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        # embeds, if it exists, is (batch_size, sequence_length, hidden_size)

        q_m = get_mask(
            self.q_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )
        k_m = get_mask(
            self.k_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )
        v_m = get_mask(
            self.v_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )

        q_z = q_m * self.attn_read_common_mask
        k_z = k_m * self.attn_read_common_mask
        v_z = v_m * self.attn_read_common_mask

        x_q = torch.einsum("wbsd,wh->hbsd", x, q_z[:, : self.num_heads])
        x_k = torch.einsum("wbsd,wh->hbsd", x, k_z[:, : self.num_heads])
        x_v = torch.einsum("wbsd,wh->hbsd", x, v_z[:, : self.num_heads])

        if embeds is not None:
            x_q = x_q + embeds.unsqueeze(0)
            x_k = x_k + embeds.unsqueeze(0)
            x_v = x_v + embeds.unsqueeze(0)

        if corr_x is not None:
            x_q = x_q + torch.einsum(
                "wbsd,wh->hbsd", corr_x, (1 - q_m) * self.attn_read_common_mask
            )
            x_k = x_k + torch.einsum(
                "wbsd,wh->hbsd", corr_x, (1 - k_m) * self.attn_read_common_mask
            )
            x_v = x_v + torch.einsum(
                "wbsd,wh->hbsd", corr_x, (1 - v_m) * self.attn_read_common_mask
            )

        z_edges_sum = torch.sum(q_z) + torch.sum(k_z) + torch.sum(v_z)

        return x_q, x_k, x_v, z_edges_sum

    def attn_write(self, residual, x, corr_x=None, soft_gate: bool = False):
        # residual is (writers, batch_size, sequence_length, hidden_size)
        # x is (num_heads, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        z = get_mask(
            self.attn_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic,
            soft_gate=soft_gate,
        ).reshape(-1, 1, 1, 1)
        x = x * z

        if corr_x is not None:
            x = x + corr_x[
                self.attn_writer_idx : self.attn_writer_idx + self.num_heads
            ] * (1 - z)

        x = torch.einsum("nbsd,nw->wbsd", x, self.attn_write_common_mask)

        residual = residual + x
        z_nodes_sum = torch.sum(z)

        return residual, z_nodes_sum

    def mlp_read(self, x, corr_x=None, embeds=None, soft_gate: bool = False):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        # embeds, if it exists, is (batch_size, sequence_length, hidden_size)
        m = get_mask(
            self.mlp_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=soft_gate,
        )

        z = m * self.mlp_read_common_mask
        x_z = torch.einsum("wbsd,w->bsd", x, z)

        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum(
                "wbsd,w->bsd", corr_x, (1 - m) * self.mlp_read_common_mask
            )

        z_edges_sum = torch.sum(z)

        return x_z, z_edges_sum

    def mlp_write(self, residual, x, corr_x=None, soft_gate: bool = False):
        # residual is (writers, batch_size, sequence_length, hidden_size)
        # x is (batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        z = get_mask(
            self.mlp_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic,
            soft_gate=soft_gate,
        ).reshape(1, 1, 1)
        x = x * z

        if corr_x is not None:
            x = x + corr_x[self.mlp_writer_idx] * (1 - z)

        x = torch.einsum("ibsd,wi->wbsd", x.unsqueeze(0), self.mlp_write_common_mask)
        residual = residual + x

        return residual, torch.sum(z)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        corr_x: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        soft_gate: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        # get hidden states per matrix
        q_hidden_states, k_hidden_states, v_hidden_states, z_attn_edges_sum = (
            self.attn_read(
                hidden_states, corr_x=corr_x, embeds=embeds, soft_gate=soft_gate
            )
        )
        q_hidden_states = self.input_layernorm(q_hidden_states)
        k_hidden_states = self.input_layernorm(k_hidden_states)
        v_hidden_states = self.input_layernorm(v_hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            q_hidden_states=q_hidden_states,
            k_hidden_states=k_hidden_states,
            v_hidden_states=v_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        residual, z_attn_nodes_sum = self.attn_write(
            residual, hidden_states, corr_x=corr_x, soft_gate=soft_gate
        )

        # Fully Connected
        hidden_states, z_mlp_edges_sum = self.mlp_read(
            residual, corr_x=corr_x, embeds=embeds, soft_gate=soft_gate
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states, z_mlp_nodes_sum = self.mlp_write(
            residual, hidden_states, corr_x=corr_x, soft_gate=soft_gate
        )

        z_edges_sum = z_attn_edges_sum + z_mlp_edges_sum
        z_nodes_sum = z_attn_nodes_sum + z_mlp_nodes_sum

        outputs = (hidden_states, z_edges_sum, z_nodes_sum)

        if output_attentions:
            outputs += (self_attn_weights,)  # type: ignore

        if use_cache:
            outputs += (present_key_value,)  # type: ignore

        return outputs


@dataclass
class FQwenModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None


class FQwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(
        self,
        config: Qwen2Config,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
        use_soft_gates: bool = False,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_soft_gates = use_soft_gates

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                FQwen2DecoderLayer(
                    config, layer_idx, with_embedding_nodes=with_embedding_nodes
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Edge-pruning params
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_writers = num_writers(
            config, with_embedding_nodes=with_embedding_nodes
        )
        self.num_readers = num_readers(config, expanded_kv=True)
        self.num_layers = config.num_hidden_layers
        self.num_edges = num_edges(
            config, with_embedding_nodes=with_embedding_nodes, expanded_kv=True
        )
        self.num_nodes = num_nodes(config, with_embedding_nodes=with_embedding_nodes)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.norm.weight.dtype
        self.with_embedding_nodes = with_embedding_nodes

        if self.with_embedding_nodes:
            self.token_write_log_alpha = nn.Parameter(
                torch.tensor([0.0], dtype=self._dtype)
            )
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)

            token_write_mask = torch.zeros(self.num_writers, dtype=self._dtype)
            token_write_mask[0] = 1
            self.register_buffer("token_write_mask", token_write_mask)

        self.final_read_log_alphas = nn.Parameter(
            torch.empty(self.num_writers, dtype=self._dtype)
        )
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)

        if disable_linear_regularization_term:
            self.sparsity_lambda_edges_1 = torch.tensor([0.0], dtype=self._dtype)
            self.sparsity_lambda_nodes_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_edges_1 = nn.Parameter(
                torch.tensor([0.0], dtype=self._dtype)
            )
            self.sparsity_lambda_nodes_1 = nn.Parameter(
                torch.tensor([0.0], dtype=self._dtype)
            )
        self.sparsity_lambda_edges_2 = nn.Parameter(
            torch.tensor([0.0], dtype=self._dtype)
        )
        self.sparsity_lambda_nodes_2 = nn.Parameter(
            torch.tensor([0.0], dtype=self._dtype)
        )

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_soft_gates(self, flag: bool = True):
        """Enable/disable continuous (differentiable) gates globally."""
        self.use_soft_gates = flag

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
        for layer in self.layers:
            layer.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic
        for layer in self.layers:
            layer.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

    @torch.no_grad()
    def get_edge_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_edge_masks(soft_gate=self.use_soft_gates))
        z_final = get_mask(
            self.final_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=self.use_soft_gates,
        )
        masks.append((z_final,))
        return masks

    @torch.no_grad()
    def get_node_masks(self):
        masks = []
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha,
                training=self.training,
                threshold_for_deterministic=self.node_threshold_for_deterministic,
                soft_gate=self.use_soft_gates,
            ).reshape([])
            masks.append((z_tokens,))
        for layer in self.layers:
            masks.append(layer.get_node_masks(soft_gate=self.use_soft_gates))
        return masks

    @torch.no_grad()
    def get_edge_sparsity(self):
        edge_masks = self.get_edge_masks()

        def process(mask):
            return torch.sum(mask), torch.numel(mask)

        s, n = 0, 0
        for layer_id in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[layer_id][i])
                s += s_
                n += n_

        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_

        s /= 1 if n == 0 else n
        return 1 - s

    @torch.no_grad()
    def get_node_sparsity(self):
        node_masks = self.get_node_masks()

        def process(mask):
            return torch.sum(mask), torch.numel(mask)

        s, n = 0, 0
        if self.with_embedding_nodes:
            s_, n_ = process(node_masks[0][0])
            s += s_
            n += n_
            offset = 1
        else:
            offset = 0
        for layer_id in range(len(self.layers)):
            for i in range(2):
                s_, n_ = process(node_masks[layer_id + offset][i])
                s += s_
                n += n_

        s /= 1 if n == 0 else n
        return 1 - s

    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()

        full_node_mask = torch.cat(
            [mask.reshape(-1) for group in node_masks for mask in group], dim=0
        )

        def process(mask):
            mask = mask * full_node_mask[: mask.shape[0]].reshape(
                -1, *([1] * (mask.ndim - 1))
            )
            return torch.sum(mask), torch.numel(mask)

        s, n = 0, 0
        for layer_id in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[layer_id][i])
                s += s_
                n += n_

        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_

        s /= 1 if n == 0 else n
        return 1 - s

    @torch.no_grad()
    def get_edges(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()

        allowed_writers = []
        edges = []

        if self.with_embedding_nodes:
            if node_masks[0][0] == 1:
                allowed_writers.append(0)
            offset = 1
            layer_offset = 1
        else:
            offset = 0
            layer_offset = 0

        for layer_id in range(self.num_layers):
            attn_writers = node_masks[layer_id + layer_offset][0]
            for i in range(self.num_heads):
                if attn_writers[i] == 1:
                    allowed_writers.append(offset + layer_id * (1 + self.num_heads) + i)
            mlp_writers = node_masks[layer_id + layer_offset][1]
            if mlp_writers == 1:
                allowed_writers.append(
                    offset + (layer_id + 1) * (1 + self.num_heads) - 1
                )

            attn_q_edges, attn_k_edges, attn_v_edges, mlp_edges = edge_masks[layer_id]
            for from_idx in range(attn_q_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                for head_no in range(attn_q_edges.shape[1]):
                    if attn_q_edges[from_idx, head_no] == 1:
                        to_idx = (
                            layer_id * (1 + self.num_heads + 2 * self.num_heads)
                            + head_no
                        )
                        edges.append(
                            (
                                writer_idx_to_name(
                                    from_idx,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    with_embedding_nodes=self.with_embedding_nodes,
                                ),
                                reader_idx_to_name(
                                    to_idx,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    num_key_value_heads=self.num_kv_heads,
                                    expanded_kv=True,
                                ),
                            )
                        )
                for head_no in range(attn_k_edges.shape[1]):
                    if attn_k_edges[from_idx, head_no] == 1:
                        to_idx = (
                            layer_id * (1 + self.num_heads + 2 * self.num_heads)
                            + self.num_heads
                            + head_no
                        )
                        edges.append(
                            (
                                writer_idx_to_name(
                                    from_idx,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    with_embedding_nodes=self.with_embedding_nodes,
                                ),
                                reader_idx_to_name(
                                    to_idx,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    num_key_value_heads=self.num_kv_heads,
                                    expanded_kv=True,
                                ),
                            )
                        )
                for head_no in range(attn_v_edges.shape[1]):
                    if attn_v_edges[from_idx, head_no] == 1:
                        to_idx = (
                            layer_id * (1 + self.num_heads + 2 * self.num_heads)
                            + self.num_heads
                            + self.num_heads
                            + head_no
                        )
                        edges.append(
                            (
                                writer_idx_to_name(
                                    from_idx,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    with_embedding_nodes=self.with_embedding_nodes,
                                ),
                                reader_idx_to_name(
                                    to_idx,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    num_key_value_heads=self.num_kv_heads,
                                    expanded_kv=True,
                                ),
                            )
                        )
            for from_idx in range(mlp_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                if mlp_edges[from_idx] == 1:
                    to_idx = (layer_id + 1) * (
                        1 + self.num_heads + 2 * self.num_heads
                    ) - 1
                    edges.append(
                        (
                            writer_idx_to_name(
                                from_idx,
                                num_layers=self.num_layers,
                                num_heads=self.num_heads,
                                with_embedding_nodes=self.with_embedding_nodes,
                            ),
                            reader_idx_to_name(
                                to_idx,
                                num_layers=self.num_layers,
                                num_heads=self.num_heads,
                                num_key_value_heads=self.num_kv_heads,
                                expanded_kv=True,
                            ),
                        )
                    )
        final_read_mask = edge_masks[self.num_layers][0]
        for from_idx in range(self.num_writers):
            if (from_idx in allowed_writers) and (final_read_mask[from_idx] == 1):
                edges.append(
                    (
                        writer_idx_to_name(
                            from_idx,
                            num_layers=self.num_layers,
                            num_heads=self.num_heads,
                            with_embedding_nodes=self.with_embedding_nodes,
                        ),
                        "resid_post",
                    )
                )
        return edges

    @torch.no_grad()
    def add_or_remove_edge(self, from_node, to_node, remove=False, value=None):
        if value is None:
            value = -10 if remove else 10
        from_idx = writer_name_to_idx(
            from_node,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            with_embedding_nodes=self.with_embedding_nodes,
        )
        if to_node == "resid_post":
            old_value = self.final_read_log_alphas[from_idx].detach().item()
            self.final_read_log_alphas[from_idx] = value
        elif to_node.startswith("m"):
            layer_idx = int(to_node[1:])
            old_value = self.layers[layer_idx].set_mlp_mask_value(from_idx, value)
        else:
            parts = to_node.split(".")
            layer_idx = int(parts[0][1:])
            head_idx = int(parts[1][1:])
            qkv = parts[2]
            old_value = self.layers[layer_idx].set_attn_mask_value(
                from_idx, head_idx, qkv, value
            )
        return old_value

    @torch.no_grad()
    def reset_all_log_alphas(self):
        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)
        for layer in self.layers:
            layer.reset_all_log_alphas()
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.sparsity_lambda_edges_1.data.zero_()
        self.sparsity_lambda_nodes_1.data.zero_()

    @torch.no_grad()
    def load_resid_post_log_alphas(self, edges):
        # Fill with -10 by default
        self.final_read_log_alphas.data.fill_(-10)

        for writer_idx, reader in edges:
            assert reader == "resid_post", f"Invalid reader format: {reader}"
            self.final_read_log_alphas[writer_idx] = 10

        # Fill with 10 for node masks, since we don't want any further restraint on edges
        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.fill_(10)

    @torch.no_grad()
    def load_all_log_alphas(self, edges):
        layer_attn_in_edges = [[] for _ in range(self.num_layers)]
        layer_mlp_in_edges = [[] for _ in range(self.num_layers)]
        resid_post_edges = []
        for edge in edges:
            writer, reader = edge
            writer_idx = writer_name_to_idx(
                writer,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                with_embedding_nodes=self.with_embedding_nodes,
            )
            if reader == "resid_post":
                resid_post_edges.append((writer_idx, reader))
            elif reader.startswith("m"):
                layer_idx = int(reader[1:])
                layer_mlp_in_edges[layer_idx].append((writer_idx, reader))
            elif reader.startswith("a"):
                layer_idx = int(reader[1 : reader.find(".")])
                layer_attn_in_edges[layer_idx].append((writer_idx, reader))
            else:
                raise ValueError(f"Invalid reader format: {reader}")
        for layer_idx, attn_in_edges in enumerate(layer_attn_in_edges):
            self.layers[layer_idx].load_attn_log_alphas(attn_in_edges)
        for layer_idx, mlp_in_edges in enumerate(layer_mlp_in_edges):
            self.layers[layer_idx].load_mlp_log_alphas(mlp_in_edges)
        self.load_resid_post_log_alphas(resid_post_edges)

    def read(self, x, corr_x=None, embeds=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        # embeds, if it exists, is (batch_size, sequence_length, hidden_size)
        z = get_mask(
            self.final_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic,
            soft_gate=self.use_soft_gates,
        )
        x_z = torch.einsum("wbsd,w->bsd", x, z)

        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1 - z))

        z_edges_sum = torch.sum(z)

        return x_z, z_edges_sum

    def write(self, tok_embeds, corr_x=None):
        # tok_embeds is (batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha,
                training=self.training,
                threshold_for_deterministic=self.node_threshold_for_deterministic,
                soft_gate=self.use_soft_gates,
            ).reshape(1, 1, 1)
            tok_embeds = tok_embeds * z_tokens
            if corr_x is not None:
                tok_embeds = tok_embeds + corr_x[0] * (1 - z_tokens)

            # hidden_states = tok_embeds.unsqueeze(0) * self.token_write_mask.reshape(-1, 1, 1, 1)
            hidden_states = tok_embeds.detach().unsqueeze(
                0
            ) * self.token_write_mask.reshape(-1, 1, 1, 1)
            z_nodes_sum = torch.sum(z_tokens)

            return hidden_states, None, z_nodes_sum
        else:
            hidden_states = torch.zeros(
                self.num_writers,
                *tok_embeds.shape,
                dtype=tok_embeds.dtype,
                device=tok_embeds.device,
            )
            z_nodes_sum = 0
            return hidden_states, tok_embeds, z_nodes_sum

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x=None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FQwenModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        # hidden_states = inputs_embeds

        # embed positions
        hidden_states, embeds, z_nodes_sum = self.write(inputs_embeds, corr_x=corr_x)
        z_edges_sum = 0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    corr_x,
                    embeds,
                    self.use_soft_gates,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    corr_x=corr_x,
                    embeds=embeds,
                    soft_gate=self.use_soft_gates,
                )

            hidden_states, z_layer_edges_sum, z_layer_nodes_sum = (
                layer_outputs[0],
                layer_outputs[1],
                layer_outputs[2],
            )
            z_edges_sum = z_edges_sum + z_layer_edges_sum
            z_nodes_sum = z_nodes_sum + z_layer_nodes_sum

            if use_cache:
                next_decoder_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        if output_writer_states:
            writer_states = hidden_states
        else:
            writer_states = None

        hidden_states, z_final_edges_sum = self.read(
            hidden_states, corr_x=corr_x, embeds=embeds
        )
        z_edges_sum = z_edges_sum + z_final_edges_sum
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        model_edge_sparsity = 1 - (z_edges_sum / self.num_edges)
        model_node_sparsity = 1 - (z_nodes_sum / self.num_nodes)

        if target_edge_sparsity is None:
            edge_loss = None
        else:
            edge_loss = (
                self.sparsity_lambda_edges_1.reshape([])
                * (model_edge_sparsity - target_edge_sparsity)
                + self.sparsity_lambda_edges_2.reshape([])
                * (model_edge_sparsity - target_edge_sparsity) ** 2
            )

        if target_node_sparsity is None:
            node_loss = None
        else:
            node_loss = (
                self.sparsity_lambda_nodes_1.reshape([])
                * (model_node_sparsity - target_node_sparsity)
                + self.sparsity_lambda_nodes_2.reshape([])
                * (model_node_sparsity - target_node_sparsity) ** 2
            )

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if target_edge_sparsity is not None:
            target_edge_sparsity = torch.tensor(
                target_edge_sparsity,
                device=model_edge_sparsity.device,
                dtype=model_edge_sparsity.dtype,
            )
        if target_node_sparsity is not None:
            target_node_sparsity = torch.tensor(
                target_node_sparsity,
                device=model_node_sparsity.device,
                dtype=model_node_sparsity.dtype,
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    writer_states,
                    target_edge_sparsity,
                    target_node_sparsity,
                    model_edge_sparsity,
                    model_node_sparsity,
                    edge_loss,
                    node_loss,
                ]
                if v is not None
            )
        return FQwenModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            writer_states=writer_states,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            model_edge_sparsity=model_edge_sparsity,
            model_node_sparsity=model_node_sparsity,
            edge_loss=edge_loss,
            node_loss=node_loss,
        )


@dataclass
class FQwenForCausalLMOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None


class FQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
        use_soft_gates: bool = False,
    ):
        super().__init__(config)
        self.model = FQwen2Model(
            config,
            with_embedding_nodes=with_embedding_nodes,
            disable_linear_regularization_term=disable_linear_regularization_term,
            use_soft_gates=use_soft_gates,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_soft_gates(self, flag: bool = True):
        self.model.set_soft_gates(flag)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.model.set_edge_threshold_for_deterministic(
            edge_threshold_for_deterministic
        )

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.model.set_node_threshold_for_deterministic(
            node_threshold_for_deterministic
        )

    @torch.no_grad()
    def get_edge_masks(self):
        return self.model.get_edge_masks()

    @torch.no_grad()
    def get_node_masks(self):
        return self.model.get_node_masks()

    @torch.no_grad()
    def get_edge_sparsity(self):
        return self.model.get_edge_sparsity()

    @torch.no_grad()
    def get_node_sparsity(self):
        return self.model.get_node_sparsity()

    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        return self.model.get_effective_edge_sparsity()

    @torch.no_grad()
    def get_edges(self):
        return self.model.get_edges()

    @torch.no_grad()
    def add_or_remove_edge(self, from_node, to_node, remove=False, value=None):
        return self.model.add_or_remove_edge(
            from_node, to_node, remove=remove, value=value
        )

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.model.reset_all_log_alphas()

    @torch.no_grad()
    def load_all_log_alphas(self, edges):
        self.model.load_all_log_alphas(edges)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x=None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FQwenForCausalLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            corr_x=corr_x,
            output_writer_states=output_writer_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FQwenForCausalLMOutput(
            lm_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            writer_states=outputs.writer_states,
            target_edge_sparsity=outputs.target_edge_sparsity,
            target_node_sparsity=outputs.target_node_sparsity,
            model_edge_sparsity=outputs.model_edge_sparsity,
            model_node_sparsity=outputs.model_node_sparsity,
            edge_loss=outputs.edge_loss,
            node_loss=outputs.node_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
