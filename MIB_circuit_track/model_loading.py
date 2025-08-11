"""
Functions and modules to load models.
"""

import logging
import pickle
from typing import Optional, Tuple

import torch
from eap.graph import Graph
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig  # type: ignore

from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME


def load_interpbench_model(device: str = "cuda") -> HookedTransformer:
    """
    Load the InterpBench language model from the Hugging Face Hub.

    Args:
        device (str): Torch device on which the model weights will be placed.

    Returns:
        HookedTransformer: The fully initialised InterpBench model.
    """
    cfg_file = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    weights_file = hf_hub_download(
        "mib-bench/interpbench",
        subfolder="ioi_all_splits",
        filename="ll_model_100_100_80.pth",
    )

    cfg_obj = pickle.load(open(cfg_file, "rb"))
    cfg = (
        HookedTransformerConfig.from_dict(cfg_obj)
        if isinstance(cfg_obj, dict)
        else cfg_obj
    )
    cfg.device = device

    # InterpBench evaluation settings
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    logging.info("Loaded InterpBench model on %s.", device)
    return model


def load_model(
    model_name: str, cache_dir: str | None = None, device: str = "cuda"
) -> Tuple[HookedTransformer, Optional[Graph]]:
    """
    Load a benchmark model (either InterpBench or a conventional HF model).

    Args:
        model_name (str): Short identifier used in MIB benchmark config.
        cache_dir (str): The cache directory to load the model to.
        device (str): Torch device used for model weights.

    Returns:
        HookedTransformer: Model ready for attribution experiments.
    """
    if model_name == "interpbench":
        model = load_interpbench_model(device)
        reference_path = hf_hub_download(
            "mib-bench/interpbench", filename="interpbench_graph.json"
        )
        reference = Graph.from_json(reference_path)
    else:
        pretrained_name = MODEL_NAME_TO_FULLNAME[model_name]
        model = HookedTransformer.from_pretrained(
            pretrained_name,
            attn_implementation=(
                "eager"
                if model_name in {"qwen2.5", "gemma2", "llama3"}
                else "flash_attention_2"
            ),
            torch_dtype=(
                torch.bfloat16
                if model_name in {"qwen2.5", "gemma2", "llama3"}
                else torch.float16
            ),
            cache_dir=cache_dir,
            device=device,
        )
        reference = None

    # Global attribution-friendly settings
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    logging.info(
        "Loaded model '%s' (%s parameters).", model_name, f"{model._parameters}"
    )
    return model, reference


def load_reference_graph(
    model_name: str, cache_dir: str | None = None, device: str = "cuda"
) -> Tuple[HookedTransformer, Graph]:
    """
    Initialise a *HookedTransformer* and its corresponding empty :class:`eap.graph.Graph`.

    Internally tweaks several configuration flags so that every required activation
    is exposed to the graph builder.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        Tuple[HookedTransformer, Graph]: The model and an empty graph skeleton.
    """
    model = HookedTransformer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device=device,
    )
    # Ensure all activations appear as *nodes* in Graph
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    graph = Graph.from_model(model)
    logging.info(
        f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
    )
    logging.info(f"Graph has {graph.real_edge_mask.sum()} real edges")
    return model, graph
