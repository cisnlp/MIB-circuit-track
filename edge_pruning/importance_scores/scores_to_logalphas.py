"""
Functions and helpers to load edge-attribution scores from file and assign them to log_alpha values for edge-pruning.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch

from edge_pruning.modeling.l0 import EPS, LIMIT_LEFT, LIMIT_RIGHT, TEMPERATURE
from edge_pruning.modeling.modeling_fpt2 import FPT2LMHeadModel
from edge_pruning.modeling.modeling_fqwen_kv_expansion import FQwen2Model

LOGGER = logging.getLogger(__name__)

EdgeKey = str
EdgeScoreMap = Dict[Tuple[str, str], float]


# -----------------------------
# Hard-Concrete parameters
# -----------------------------


@dataclass
class HardConcreteParams:
    temperature: float = TEMPERATURE
    limit_left: float = LIMIT_LEFT
    limit_right: float = LIMIT_RIGHT
    clip_log_alpha: float = 10.0
    eps: float = EPS


@dataclass
class WarmstartOptions:
    mode: Literal["absolute", "signed"] = "absolute"
    epsilon_mix: float = 0.0
    start_edge_sp: float = 0.90
    per_layer: bool = True
    per_layer_overrides: Optional[Mapping[Optional[int], float]] = (
        None  # e.g., {0:0.85, 31:0.85}
    )
    tau_init: float = 0.20
    protected_reader_prefixes: Tuple[str, ...] = (
        "a0.",
        "m0.",
    )
    boost_delta: float = 0.0


# -----------------------------
# Parsing helpers
# -----------------------------


def edgekey_split(edge_key: EdgeKey) -> Tuple[str, str]:
    """
    Split an edge key (e.g., 'a0.h1->a1.h3<q>') into (writer, reader_raw).

    Args:
        edge_key (EdgeKey): The key of the edge to consider.

    Raises:
        ValueError: If the key has an unexpected format.

    Returns:
        Tuple[str, str]: The split key.
    """
    try:
        src, dst = edge_key.split("->", 1)
    except ValueError:
        raise ValueError(f"Malformed edge key: {edge_key}")
    return src, dst


def node_name_to_reader_name(node_name: str) -> str:
    """
    Convert node format from file into model reader name.

    Examples
    --------
    'a0.h1<q>' -> 'a0.h1.q'
    'm3'       -> 'm3'
    'logits'   -> 'resid_post'

    Args:
        node_name (str): The name of the node in file format.

    Returns:
        str: The converted node name.
    """
    if node_name == "logits":
        return "resid_post"
    if "<" in node_name and node_name.endswith(">"):
        base, suf = node_name[:-1].split("<", 1)
        return f"{base}.{suf}"
    return node_name


def parse_edge_scores_from_json(edge_scores: Dict[str, Dict[str, Any]]) -> EdgeScoreMap:
    """
    Low-level loader that *does not* expand 'input'. Returns (writer, reader) tuples.

    Args:
        edge_scores (Dict[str, Dict[str, Any]]): The edge-score dict.

    Returns:
        EdgeScoreMap: The converted edge-score map.
    """
    m: EdgeScoreMap = {}
    for ek, info in edge_scores.items():
        w_raw, r_raw = edgekey_split(ek)
        r = node_name_to_reader_name(r_raw)
        w = node_name_to_reader_name(w_raw)
        m[(w, r)] = float(info["score"])
    return m


def load_edge_attr_scores(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the edge attribution scores from a JSON file.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping edge names (writer.reader) to
            dictionaries containing the "score" and "in_graph" fields.
    """
    with path.open() as fh:
        payload = json.load(fh)
    if "edges" not in payload:
        raise KeyError(f"`edges` field missing in {path}")
    return payload["edges"]


# -----------------------------
# Score refinements
# -----------------------------


def union_inverse(log_alpha_union: float) -> float:
    """
    Given the log-odds for the *union* of two independent symmetric Bernoulli
    switches (p_union = 1 - (1-p_each)^2), recover the per-edge log-odds.

    Solve: p_each = 1 - sqrt(1 - p_union)

    Args:
        log_alpha_union (float): The initial log_alpha values.

    Returns:
        float: The transformed, splitted values.
    """
    p_u = torch.sigmoid(torch.tensor(log_alpha_union)).item()
    p_each = 1.0 - math.sqrt(max(0.0, 1.0 - p_u))
    p_each = min(max(p_each, 1e-7), 1 - 1e-7)
    return math.log(p_each / (1.0 - p_each))


def expand_input_edges(
    in_scores: EdgeScoreMap,
    policy: Literal["copy", "split", "union_inv"] = "copy",
    isgpt: bool = False,
) -> EdgeScoreMap:
    """
    Expand 'input' -> X edges into both 'tok_embeds' and 'pos_embeds' edges.

    policy:
        'copy'         same score to both
        'split'        half to each (linear heuristic)
        'union-inv'    treat input score as union prob; invert to per-embed

    Args:
        in_scores (EdgeScoreMap): The edge scores.
        policy (Literal, optional): The policy how to split. Defaults to "copy".
        isgpt (bool, optional): Whether model is GPT-2. Defaults to False.

    Raises:
        ValueError: If an unknown policy is chosen.

    Returns:
        EdgeScoreMap: The edge score map with inputs accounted for.
    """
    out: EdgeScoreMap = {}
    for (w, r), s in in_scores.items():
        if w != "input":
            out[(w, r)] = s
            continue
        if isgpt:
            if policy == "copy":
                tok_s = pos_s = s
            elif policy == "split":
                tok_s = pos_s = 0.5 * s
            elif policy == "union_inv":
                tok_s = pos_s = union_inverse(s)
            else:
                raise ValueError(f"Unknown input-expand policy: {policy}")
            out[("tok_embeds", r)] = tok_s
            out[("pos_embeds", r)] = pos_s
        else:
            out[("embeds", r)] = s
    return out


def _compress_qwen_kv_fullhead_scores(
    scores: EdgeScoreMap,
    num_heads: int,
    num_kv_heads: int,
    reduce: Literal["maxabs", "mean", "sum"] = "maxabs",
) -> EdgeScoreMap:
    """
    For Qwen GQA: collapse full-head K/V readers 'aL.hH.<k/v>' (H in [0..num_heads-1])
    into KV-group readers 'aL.hK.<k/v>' where K in [0..num_kv_heads-1].

    Args:
        scores (EdgeScoreMap): The edge scores non-collapsed.
        num_heads (int): The number of Q heads.
        num_kv_heads (int): The number of K-V heads.
        reduce (Literal["maxabs", "mean", "sum"]): The strategy to reduce. Defaults to "maxabs".

    Returns:
        EdgeScoreMap: The edge score map with K/V full-head readers collapsed.
    """
    if num_kv_heads <= 0 or num_kv_heads == num_heads:
        return scores
    group = num_heads // num_kv_heads
    if group * num_kv_heads != num_heads:
        raise ValueError(
            f"Inconsistent GQA configuration: num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )
    out: EdgeScoreMap = {}
    kv_bins: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (w, r), s in scores.items():
        if not (r.startswith("a") and r.count(".") == 2):
            out[(w, r)] = s
            continue
        a, h, qkv = r.split(".")
        if qkv == "q":
            out[(w, r)] = s
            continue
        if qkv not in ("k", "v"):
            out[(w, r)] = s
            continue
        try:
            H = int(h[1:])
        except ValueError:
            out[(w, r)] = s
            continue
        if H < num_kv_heads:
            kv_bins[(w, r)].append(s)
        else:
            K = H // group
            if K >= num_kv_heads:
                continue
            r_compact = f"{a}.h{K}.{qkv}"
            kv_bins[(w, r_compact)].append(s)
    for key, vals in kv_bins.items():
        if reduce == "maxabs":
            v = max(vals, key=lambda x: abs(x))
        elif reduce == "mean":
            v = float(sum(vals) / len(vals))
        elif reduce == "sum":
            v = float(sum(vals))
        else:
            raise ValueError(f"Unknown reducer: {reduce}")
        out[key] = v
    return out


def _extract_layer_id(reader: str) -> Optional[int]:
    """
    Extract the layer ID from a reader string.

    Args:
        reader (str): The reader string, expected to start with 'a' for
                      attention layers or 'm' for MLP layers, followed
                      by the layer number and other components.

    Returns:
        Optional[int]: The extracted layer index, or None if the reader
                       string is not in the expected format.
    """
    if reader.startswith("a"):
        try:
            return int(reader.split(".")[0][1:])
        except Exception:
            return None
    if reader.startswith("m"):
        try:
            return int(reader.split(".")[0][1:])
        except Exception:
            return None
    return None


def group_keys_by_layer(
    keys: Sequence[Tuple[str, str]]
) -> Dict[Optional[int], List[int]]:
    """
    Group the given list of keys by layer ID.

    Args:
        keys: A list of tuples, where the first element of each tuple is the writer node
              and the second element is the reader node.

    Returns:
        A dictionary where the keys are the layer IDs (or None if the reader string does not
        contain a layer number) and the values are lists of indices into the original list
        of keys.
    """
    groups: Dict[Optional[int], List[int]] = defaultdict(list)
    for idx, (_, reader) in enumerate(keys):
        groups[_extract_layer_id(reader)].append(idx)
    return groups


# -----------------------------
# Rank / calibration utilities
# -----------------------------


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """
    Convert a 1D tensor to [0,1] ranks (0 = smallest, 1 = largest).
    Stable, ties get averaged.

    Args:
        values (torch.Tensor): The tensor to rank.

    Returns:
        torch.Tensor: The ranked tensor, with values in [0, 1].
    """
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(values), dtype=torch.float, device=values.device)
    return ranks / max(1, len(values) - 1)


def _pkeep_to_logalpha(p_keep: torch.Tensor, hc: HardConcreteParams) -> torch.Tensor:
    """
    Approximate inversion: map desired keep probability to a log_alpha
    so that the stretched-sigmoid gate expectation matches p_keep.

    Args:
        p_keep (torch.Tensor): Tensor of `p_keep` values in [0,1].
        hc (HardConcreteParams): Parameters for the hard-concrete distribution.

    Returns:
        torch.Tensor: The log-alpha values, clamped to [-clip_log_alpha, clip_log_alpha].
    """
    p_tilde = ((p_keep - hc.limit_left) / (hc.limit_right - hc.limit_left)).clamp(
        hc.eps, 1 - hc.eps
    )
    logit = torch.log(p_tilde) - torch.log1p(-p_tilde)
    log_alpha = hc.temperature * logit
    return log_alpha.clamp(-hc.clip_log_alpha, hc.clip_log_alpha)


def ranks_to_log_alpha_target_keep_expectation(
    q: torch.Tensor,
    target_sp: float,
    tau: float,
    hc: HardConcreteParams,
    iters: int = 40,
) -> torch.Tensor:
    """
    Choose bias b so mean(sigmoid((q - b)/tau)) == target_keep = 1 - target_sp,
    then invert to log_alpha using expected-keep mapping.

    Args:
        q (torch.Tensor): Tensor of `q` values in [0,1].
        target_sp (float): Target sparsity level in [0,1].
        tau (float): Temperature parameter for the sigmoid function.
        hc (HardConcreteParams): Parameters for the hard-concrete distribution.
        iters (int, optional): Number of bisection iterations to find the bias.
            Defaults to 40.

    Returns:
        torch.Tensor: The log-alpha values, clamped to [-clip_log_alpha, clip_log_alpha].
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    target_sp = float(min(max(target_sp, 0.0), 1.0))
    target_keep = 1.0 - target_sp

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p_keep = torch.sigmoid((q - mid) / tau)
        m = p_keep.mean().item()
        if m > target_keep:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    p_keep = torch.sigmoid((q - b) / tau)
    return _pkeep_to_logalpha(p_keep, hc=hc)


def scores_to_log_alphas(
    edge_scores: EdgeScoreMap,
    opts: WarmstartOptions,
    hc_params: HardConcreteParams = HardConcreteParams(),
) -> EdgeScoreMap:
    """
    Map edge scores -> log_alpha via:

      (a) pick salience according to mode
      (b) rank to [0,1]
      (c) turn ranks into keep probabilities that match target sparsity
      (d) invert hard-concrete to get log_alpha

    Args:
        edge_scores (EdgeScoreMap): A dictionary mapping edges to their scores.
        opts (WarmstartOptions): Options for the warm-starting process.
        hc_params (HardConcreteParams): Parameters for the hard-concrete distribution.

    Returns:
        EdgeScoreMap: A dictionary mapping edges to their corresponding log_alpha values.
    """
    keys: List[Tuple[str, str]] = list(edge_scores.keys())
    s = torch.tensor([edge_scores[k] for k in keys], dtype=torch.float)

    # score transform
    if opts.mode == "absolute":
        v = s.abs()
    elif opts.mode == "signed":
        v = s
    else:
        raise ValueError(f"Unknown mode: {opts.mode}")

    # ε-mix to soften equalities near the threshold (very helpful at high sparsity)
    if opts.epsilon_mix > 0.0:
        eps = float(min(max(opts.epsilon_mix, 0.0), 0.2))
        u = torch.rand_like(v)
        v = (1 - eps) * v + eps * u

    # per-layer groups
    groups = (
        group_keys_by_layer(keys) if opts.per_layer else {None: list(range(len(keys)))}
    )

    log_alpha = torch.empty_like(v)
    # use a higher init tau for softer gates; we don't mutate hc_params.temperature (used for inverse),
    # we control softness via the calibration sigmoid's tau.
    tau_eff = float(opts.tau_init)

    for gid, idxs in groups.items():
        if len(idxs) == 0:
            continue
        v_g = v[idxs]
        q_g = _rank01(v_g)
        # choose sparsity for this group
        target_sp = opts.start_edge_sp
        if opts.per_layer_overrides and gid in opts.per_layer_overrides:
            target_sp = float(opts.per_layer_overrides[gid])
        la_g = ranks_to_log_alpha_target_keep_expectation(
            q_g, target_sp=target_sp, tau=tau_eff, hc=hc_params
        )
        log_alpha[idxs] = la_g

    log_alpha_np = log_alpha.cpu().numpy()

    # optional protection boost (reader-side prefix match)
    if opts.boost_delta != 0.0 and opts.protected_reader_prefixes:
        for i, (_, reader) in enumerate(keys):
            if any(reader.startswith(pref) for pref in opts.protected_reader_prefixes):
                log_alpha_np[i] += float(opts.boost_delta)

    return {k: float(a) for k, a in zip(keys, log_alpha_np)}


@torch.no_grad()
def apply_log_alphas_to_model(
    model: Union[FPT2LMHeadModel, FQwen2Model],
    log_alpha_map: EdgeScoreMap,
) -> None:
    """
    Applies log_alpha values to a model for each edge specified in the log_alpha_map.

    Args:
        model (Union[FPT2LMHeadModel, FQwen2Model]): The model to update with log_alpha values.
        log_alpha_map (EdgeScoreMap): A mapping of (writer, reader) tuples to log_alpha values.

    Raises:
        ValueError: If an error occurs while assigning a log_alpha value to an edge.
    """
    for (writer, reader), log_alpha in log_alpha_map.items():
        try:
            model.add_or_remove_edge(writer, reader, value=log_alpha)
        except Exception as exc:
            raise ValueError(
                f"Error assigning log_alpha to {writer}->{reader}: {exc}"
            ) from exc


def warmstart_from_attr_scores(
    circuit_model: Union[FPT2LMHeadModel, FQwen2Model],
    raw_attr_score_dict: Dict[str, Dict[str, Any]],
    policy: Literal["copy", "split", "union_inv"] = "copy",
    qwen_kv_reduce: Literal["maxabs", "mean", "sum"] = "maxabs",
    opts: WarmstartOptions = WarmstartOptions(),
    hc_params: HardConcreteParams = HardConcreteParams(),
) -> None:
    """
    Warm-start a circuit model using edge-attribution scores from a raw JSON dictionary.

    Args:
        circuit_model (Union[FPT2LMHeadModel, FQwen2Model]): The model to warm-start.
        raw_attr_score_dict (Dict[str, Dict[str, Any]]): The raw edge attribution scores.
        policy (Literal["copy", "split", "union_inv"], optional): Strategy to expand 'input' edges.
            Defaults to "copy".
        qwen_kv_reduce (Literal["maxabs", "mean", "sum"], optional): Reduction operation to apply to QWEN KV groups.
            Defaults to "maxabs".
        opts (WarmstartOptions, optional): Options for the warm-starting process. Defaults to an instance of WarmstartOptions.
        hc_params (HardConcreteParams, optional): Parameters for the hard-concrete distribution. Defaults to an instance of HardConcreteParams.
    """
    score_map = parse_edge_scores_from_json(raw_attr_score_dict)
    refined_score_map = expand_input_edges(
        score_map, policy=policy, isgpt=isinstance(circuit_model, FPT2LMHeadModel)
    )

    la_map = scores_to_log_alphas(refined_score_map, opts=opts, hc_params=hc_params)
    apply_log_alphas_to_model(circuit_model, la_map)


if __name__ == "__main__":
    attr_file_path = Path("circuits/ioi-eap-ig.json")

    model = FPT2LMHeadModel.from_pretrained(
        "gpt2",
        with_embedding_nodes=True,
        disable_linear_regularization_term=True,
        attn_implementation="eager",
        torch_dtype=None,
        cache_dir="/nfs/gdata/llms/hf-models",
    ).to("cuda")
    model.eval()

    ws_opts = WarmstartOptions(
        mode="absolute",
        epsilon_mix=0.03,
        start_edge_sp=0.9,
        per_layer=True,
        per_layer_overrides=None,
        tau_init=0.25,
        protected_reader_prefixes=("a0.", "m0.", "resid_post"),
        boost_delta=0.15,
    )

    logging.info("Reading edge-attribution scores from %s", attr_file_path)
    edges_raw_scores = load_edge_attr_scores(attr_file_path)

    before = {
        "blocks": [
            {
                "q": blk.q_read_log_alphas.detach().clone(),
                "k": blk.k_read_log_alphas.detach().clone(),
                "v": blk.v_read_log_alphas.detach().clone(),
                "mlp": blk.mlp_read_log_alphas.detach().clone(),
            }
            for blk in model.transformer.h
        ],
        "final": model.transformer.final_read_log_alphas.detach().clone(),
    }

    warmstart_from_attr_scores(
        circuit_model=model,
        raw_attr_score_dict=edges_raw_scores,
        policy=("copy"),
        qwen_kv_reduce="maxabs",
        opts=ws_opts,
    )

    def count_diff(t0, t1, eps=1e-6):
        return (t0 - t1).abs().gt(eps).sum().item()

    for L, blk in enumerate(model.transformer.h):
        diffs = {
            n: count_diff(before["blocks"][L][n], getattr(blk, f"{n}_read_log_alphas"))
            for n in ("q", "k", "v", "mlp")
        }
        if any(diffs.values()):
            print(f"layer {L:2d}", diffs)

    final_changed = count_diff(before["final"], model.transformer.final_read_log_alphas)
    print("final read :", final_changed)
