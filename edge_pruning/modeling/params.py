from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SparsityConfig:
    # Edge (weight / attention-head) sparsity
    start_edge_sparsity: float = 0.0
    target_edge_sparsity: float = 0.0

    # Node / layer sparsity
    start_layer_sparsity: float = 0.0
    target_layer_sparsity: float = 0.0

    # Warm-up lengths
    num_edge_sparsity_warmup_steps: Optional[int] = None
    num_layer_sparsity_warmup_steps: Optional[int] = None
    num_sparsity_warmup_steps: int = 0

    # Schedule shape
    warmup_type: Literal["linear", "logarithmic"] = "linear"

    # Misc
    skip_layer_loss_if_higher_sparsity: bool = False
