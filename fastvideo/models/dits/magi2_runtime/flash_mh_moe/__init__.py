# Vendored flash_mh_moe inference subset.
# Routing/sorting: pure PyTorch.
# Forward GEMM: fused Triton kernel with deterministic scatter support.
from .fwd import flash_mh_moe_fwd
from .route import compute_topk_probs_and_indices, flash_mh_moe_global_sort

__all__ = [
    "compute_topk_probs_and_indices",
    "flash_mh_moe_global_sort",
    "flash_mh_moe_fwd",
]
