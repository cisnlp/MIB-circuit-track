"""
Utility helpers to export and reload the continuous log_alpha values that
parameter-ize every edge in a trained Edge-Pruning model.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import torch

from edge_pruning.modeling.l0 import continuous_z_from_log_alpha
from edge_pruning.modeling.modeling_fpt2 import FPT2LMHeadModel, FPT2Model
from edge_pruning.modeling.modeling_fpt2 import (
    reader_idx_to_name as reader_idx_to_name_fpt2,
)
from edge_pruning.modeling.modeling_fpt2 import (
    writer_idx_to_name as writer_idx_to_name_fpt2,
)
from edge_pruning.modeling.modeling_fqwen_kv_expansion import (
    FQwen2ForCausalLM,
    FQwen2Model,
)
from edge_pruning.modeling.modeling_fqwen_kv_expansion import (
    reader_idx_to_name as reader_idx_to_name_fqwen,
)
from edge_pruning.modeling.modeling_fqwen_kv_expansion import (
    writer_idx_to_name as writer_idx_to_name_fqwen,
)

EdgeKey = str  # e.g. "a0.h0->a0.h1<q>"
EdgeInfo = Dict[str, float | bool]  # {"score": 8.7, "in_graph": False}


def _as_base_model(
    model: FPT2Model | FPT2LMHeadModel | FQwen2ForCausalLM | FQwen2Model,
) -> FPT2Model | FQwen2Model:
    """
    Always return the bare FPT2Model.

    Args:
        model (FPT2Model | FPT2LMHeadModel): The initial model that was passed.

    Raises:
        TypeError: If an unsupported model type was passed.

    Returns:
        FPT2Model: The unwrapped base model.
    """
    if isinstance(model, FPT2LMHeadModel):
        return model.transformer
    if isinstance(model, FPT2Model):
        return model
    if isinstance(model, FQwen2ForCausalLM):
        return model.model
    if isinstance(model, FQwen2Model):
        return model
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def _union_log_alpha(a: float, b: float) -> float:
    """
    Compute the log-odds of the union of two Bernoulli random variables.

    Args:
        a (float): The log-odds of the first Bernoulli variable.
        b (float): The log-odds of the second Bernoulli variable.

    Returns:
        float: The log-odds of the union of the two variables.
    """
    pa, pb = torch.sigmoid(torch.tensor([a, b])).tolist()
    p = pa + pb - pa * pb  # union of Bernoulli probs
    p = min(max(p, 1e-7), 1.0 - 1e-7)  # numerical safety
    return math.log(p / (1.0 - p))


def _union_prob(p_list: List[float]) -> float:
    """
    Compute the probability of the union of multiple independent events.

    Args:
        p_list (List[float]): The list of probabilities of each event happening.

    Returns:
        float: The probability of at least one event happening.
    """
    p_keep = 0.0
    for p in p_list:
        p_keep = p_keep + (1.0 - p_keep) * p
    return p_keep


def _mk_edge_key(writer: str, reader: str) -> EdgeKey:
    """
    Create a string edge key from writer and reader node names.

    If the reader name ends with ".q", ".k", or ".v", it is rewritten to
    "<reader name> <suffix>" to disambiguate multi-head attention.

    Args:
        writer (str): The writer node name.
        reader (str): The reader node name.

    Returns:
        EdgeKey: The constructed edge key string.
    """
    if reader.endswith((".q", ".k", ".v")):
        reader, suffix = reader.rsplit(".", 1)
        reader = f"{reader}<{suffix}>"
    return f"{writer}->{reader}"


def _merge_embedding_edges(
    edges: Dict[EdgeKey, EdgeInfo], mode: Literal["log_alpha", "z_score"] = "z_score"
) -> Dict[EdgeKey, EdgeInfo]:
    """
    Merge log_alpha values for edges connecting input embeddings to layer nodes.

    For every node connected to either the token embeddings or the position
    embeddings, this function groups the edges by their target node and
    computes the log-odds of the union of the corresponding Bernoulli variables.

    Args:
        edges (Dict[EdgeKey, EdgeInfo]): The original edge dictionary.

    Returns:
        Dict[EdgeKey, EdgeInfo]: The modified edge dictionary with input
            embedding edges merged according to the union of their
            corresponding Bernoulli variables.
    """
    merged: Dict[EdgeKey, EdgeInfo] = {}
    stash: defaultdict[str, list[float]] = defaultdict(list)

    for key, info in edges.items():
        if (
            key.startswith("tok_embeds->")
            or key.startswith("pos_embeds->")
            or key.startswith("embeds->")
        ):
            reader = key.split("->", 1)[1]
            stash[reader].append(info["score"])
        else:
            merged[key] = info  # keep
    # build the combined edges
    for reader, scores in stash.items():
        if mode == "log_alpha":
            combined = scores[0] if len(scores) == 1 else _union_log_alpha(*scores[:2])
        elif mode == "z_score":
            combined = scores[0] if len(scores) == 1 else _union_prob(scores[:2])
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Should be either 'log_alpha' or 'z_score'!"
            )

        merged[f"input->{reader}"] = {"score": combined, "in_graph": False}

    return merged


def _extract_edge_scores(
    model: FPT2Model | FQwen2Model, mode: Literal["log_alpha", "z_score"] = "z_score"
) -> Dict[EdgeKey, EdgeInfo]:
    """
    Traverse model once and collect every edge's z-values.
    All in_graph flags are set to False (place-holders).

    Args:
        model (FPT2Model | FQwen2Model): The model to consider.

    Returns:
        Dict[EdgeKey, EdgeInfo]: Mapping edge_key → {"score": float, "in_graph": False}.
    """
    edges: Dict[EdgeKey, EdgeInfo] = {}

    is_gpt2 = isinstance(model, FPT2Model)

    n_layer = model.n_layer if hasattr(model, "n_layer") else model.num_layers
    n_head = model.n_head if hasattr(model, "n_head") else model.num_heads
    num_kv_heads = 0 if is_gpt2 else model.num_kv_heads

    with_embed = model.with_embedding_nodes

    def w_name(idx: int) -> str:
        if isinstance(model, FPT2Model):
            return writer_idx_to_name_fpt2(idx, n_layer, n_head, with_embed)
        elif isinstance(model, FQwen2Model):
            return writer_idx_to_name_fqwen(idx, n_layer, n_head, with_embed)
        else:
            raise ValueError(f"Model type not supported: {model}!")

    def r_name(idx: int, n_layer: int, n_heads: int, num_kv_heads: int = 0) -> str:
        if isinstance(model, FPT2Model):
            return reader_idx_to_name_fpt2(idx, n_layer, n_heads)
        elif isinstance(model, FQwen2Model):
            return reader_idx_to_name_fqwen(idx, n_layer, n_heads, num_kv_heads)
        else:
            raise ValueError(f"Model type not supported: {model}!")

    model_layers = model.h if hasattr(model, "h") else model.layers

    with torch.no_grad():
        for L, block in enumerate(model_layers):
            if mode == "log_alpha":
                q_edge_score = block.q_read_log_alphas.detach().cpu()
                k_edge_score = block.k_read_log_alphas.detach().cpu()
                v_edge_score = block.v_read_log_alphas.detach().cpu()
                m_edge_score = block.mlp_read_log_alphas.detach().cpu()
            elif mode == "z_score":
                q_edge_score = (
                    continuous_z_from_log_alpha(block.q_read_log_alphas).detach().cpu()
                )
                k_edge_score = (
                    continuous_z_from_log_alpha(block.k_read_log_alphas).detach().cpu()
                )
                v_edge_score = (
                    continuous_z_from_log_alpha(block.v_read_log_alphas).detach().cpu()
                )
                m_edge_score = (
                    continuous_z_from_log_alpha(block.mlp_read_log_alphas)
                    .detach()
                    .cpu()
                )
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Should be either 'log_alpha' or 'z_score'!"
                )

            attn_limit = (
                block.attn_writer_offset
                if hasattr(block, "attn_writer_offset")
                else block.attn_writer_idx
            )
            mlp_limit = (
                block.mlp_writer_offset
                if hasattr(block, "mlp_writer_offset")
                else block.mlp_writer_idx
            )

            if is_gpt2:
                per_layer = 3 * n_head + 1
                r_base = L * per_layer
                q_heads = k_heads = v_heads = n_head
                k_offset = n_head
                v_offset = 2 * n_head
            else:
                q_heads = q_edge_score.shape[1]
                k_heads = k_edge_score.shape[1]
                v_heads = v_edge_score.shape[1]
                per_layer = q_heads + 2 * k_heads + 1
                r_base = L * per_layer
                k_offset = q_heads
                v_offset = q_heads + k_heads

            # Q / K / V
            for w in range(attn_limit):
                w_str = w_name(w)

                for h in range(q_heads):
                    r_str = r_name(r_base + h, n_layer, n_head, num_kv_heads)
                    edges[_mk_edge_key(w_str, r_str)] = {
                        "score": float(q_edge_score[w, h]),
                        "in_graph": False,
                    }

                for h in range(k_heads):
                    r_str = r_name(r_base + k_offset + h, n_layer, n_head, num_kv_heads)
                    edges[_mk_edge_key(w_str, r_str)] = {
                        "score": float(k_edge_score[w, h]),
                        "in_graph": False,
                    }

                for h in range(v_heads):
                    r_str = r_name(r_base + v_offset + h, n_layer, n_head, num_kv_heads)
                    edges[_mk_edge_key(w_str, r_str)] = {
                        "score": float(v_edge_score[w, h]),
                        "in_graph": False,
                    }

            # MLP reads
            mlp_reader_idx = (L + 1) * per_layer - 1
            mlp_reader = r_name(mlp_reader_idx, n_layer, n_head, num_kv_heads)

            for w in range(mlp_limit):
                key = _mk_edge_key(w_name(w), mlp_reader)
                edges[key] = {
                    "score": float(m_edge_score[w]),
                    "in_graph": False,
                }

        # final read
        if mode == "log_alpha":
            final_edge_score = model.final_read_log_alphas.detach().cpu()
        elif mode == "z_score":
            final_edge_score = (
                continuous_z_from_log_alpha(model.final_read_log_alphas).detach().cpu()
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Should be either 'log_alpha' or 'z_score'!"
            )

        num_writers = (
            model.n_writers if hasattr(model, "n_writers") else model.num_writers
        )
        for w in range(num_writers):
            key = _mk_edge_key(w_name(w), "logits")
            edges[key] = {
                "score": float(final_edge_score[w]),
                "in_graph": False,
            }

    return edges


def _expand_qwen_kv_edges_to_full_heads(
    edges: Dict[str, Dict[str, float | bool]],
    num_heads: int,
    num_kv_heads: int,
    mode: Literal["copy", "both"] = "copy",
) -> Dict[str, Dict[str, float | bool]]:
    """
    For Qwen GQA: expand reader names like 'aL.hk<k>' / 'aL.hk<v>' where k in [0..num_kv_heads-1]
    into full attention-head indices 'aL.hH<k|v>' for H in the KV group.

    mode:
      - "copy":    return only full-head edges (drop the compact KV edges)
      - "both":    keep the original KV edges and add the full-head duplicates

    We simply *copy* the score to all repeated heads (that's the usual convention).
    """
    if num_kv_heads <= 0 or num_kv_heads == num_heads:
        return edges  # nothing to expand (no GQA or MHA == KV)

    group = num_heads // num_kv_heads
    if group * num_kv_heads != num_heads:
        raise ValueError(
            f"Inconsistent GQA: num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )

    new_edges = {} if mode == "copy" else dict(edges)
    for key, info in list(edges.items()):
        try:
            writer, reader = key.split("->", 1)
        except ValueError:
            continue

        # Readers like "aL.hX<q/k/v>"
        if not reader.startswith("a"):
            continue
        if not (reader.endswith("<k>") or reader.endswith("<v>")):
            continue

        # Parse "aL.hX" and suffix "<k>" / "<v>"
        base, suffix = reader.rsplit("<", 1)
        suffix = "<" + suffix
        a, h = base.split(".")
        if not h.startswith("h"):
            continue
        kv_idx = int(h[1:])

        # Only expand compact KV indices (0..num_kv_heads-1)
        if kv_idx >= num_kv_heads:
            if mode == "copy":
                new_edges[key] = info
            continue

        H_start = kv_idx * group
        for H in range(H_start, H_start + group):
            new_reader = f"{a}.h{H}{suffix}"
            new_key = f"{writer}->{new_reader}"
            new_edges[new_key] = {
                "score": info["score"],
                "in_graph": info.get("in_graph", False),
            }

    return new_edges


def _all_nodes(model: Union[FPT2Model, FQwen2Model]) -> Dict[str, Dict[str, bool]]:
    """
    Return a dictionary of all nodes in the model, with their in_graph flags initialized to False.

    Args:
        model (Union[FPT2Model, FQwen2Model]): The model to consider.

    Returns:
        Dict[str, Dict[str, bool]]: Mapping node name → {"in_graph": False}.
    """
    n_layer = model.n_layer if hasattr(model, "n_layer") else model.num_layers
    n_head = model.n_head if hasattr(model, "n_head") else model.num_heads
    nodes: Dict[str, Dict[str, bool]] = {"input": {"in_graph": False}}

    for L in range(n_layer):
        for H in range(n_head):
            nodes[f"a{L}.h{H}"] = {"in_graph": False}
        nodes[f"m{L}"] = {"in_graph": False}
    nodes["logits"] = {"in_graph": False}

    return nodes


def _apply_signs(
    edges: Dict[EdgeKey, EdgeInfo],
    attr_signs: Optional[Dict[EdgeKey, int]],
) -> Dict[EdgeKey, EdgeInfo]:
    """
    Multiply edge scores by corresponding signs from attribution data.

    Args:
        edges (Dict[EdgeKey, float]): Edge scores to be modified.
        attr_signs (Optional[Dict[EdgeKey, int]]): Signs from attribution data,
            where attr_signs[k] = 1 if edge k is positive, -1 if negative, 0 if
            not present in attribution data.

    Returns:
        Dict[EdgeKey, float]: Modified edge scores.
    """
    if attr_signs is None:
        return edges
    out: Dict[EdgeKey, EdgeInfo] = {}
    for k, info in edges.items():
        s = attr_signs[k]
        out[k] = {
            "score": float(s * info["score"]),
            "in_graph": info.get("in_graph", False),
        }
    return out


def save_edge_scores(
    model: FPT2Model | FPT2LMHeadModel,
    file_path: Union[str, Path],
    mode: Literal["log_alpha", "z_score"] = "z_score",
    orig_attr_signs: Optional[Dict[EdgeKey, int]] = None,
    overwrite: bool = False,
) -> None:
    """
    Persist all edge log-alphas to a JSON file.

    Args:
        model (FPT2Model | FPT2LMHeadModel): The model to consider.
        file_path (Union[str, Path]): The file path to write to.
        overwrite (bool): Whether to overwrite an existing file. Defaults to False.

    Raises:
        FileExistsError: If the file already exists and `overwrite=False`.
    """
    path = Path(file_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists - pass `overwrite=True`")
    path.parent.mkdir(parents=True, exist_ok=True)

    base = _as_base_model(model)

    # get nodes and edges
    nodes = _all_nodes(base)
    edges = _extract_edge_scores(base, mode=mode)
    if base.with_embedding_nodes:
        edges = _merge_embedding_edges(edges, mode=mode)

    # apply signs
    edges = _apply_signs(edges, orig_attr_signs)

    n_layers = base.n_layer if hasattr(base, "n_layer") else base.num_layers
    n_heads = base.n_head if hasattr(base, "n_head") else base.num_heads

    circuit = {
        "cfg": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "parallel_attn_mlp": False,
            "d_model": base.config.hidden_size,
        },
        "nodes": nodes,
        "edges": edges,
    }

    path.write_text(json.dumps(circuit, indent=2))
    logging.info(f"✓ saved {len(circuit['edges']):,} edges to {path}")


def _sign(x: float) -> int:
    """
    Return the sign of the input float x as an integer.

    Args:
        x (float): The input float.

    Returns:
        int: The sign of the float.
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def load_attr_signs(path: Union[str, Path]) -> Dict[EdgeKey, int]:
    """
    Load an attribution JSON file (with the same edge-key format) and
    return a simple map: edge_key -> sign ∈ {-1, 0, +1}.
    Expected format:
      {
        "edges": {
          "<writer->reader>": {"score": <float>, ...},
          ...
        }
      }
    """
    data = json.loads(Path(path).read_text())
    if "edges" not in data:
        raise KeyError("`edges` field missing in attribution file")
    out: Dict[EdgeKey, int] = {}
    for k, v in data["edges"].items():
        out[k] = _sign(float(v["score"]))
    return out


def load_edge_log_alphas(
    file_path: Union[str, Path],
) -> Dict[EdgeKey, float]:
    """
    Load the edge log-alphas from a JSON file.

    Args:
        file_path (Union[str, Path]): The file path to read from.

    Returns:
        Dict[EdgeKey, float]: A dictionary mapping edge names to their corresponding log-alphas.
    """
    data = json.loads(Path(file_path).read_text())
    return {k: float(v["score"]) for k, v in data["edges"].items()}
