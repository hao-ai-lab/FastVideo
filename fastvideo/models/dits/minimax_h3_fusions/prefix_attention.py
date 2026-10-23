# SPDX-License-Identifier: Apache-2.0
"""Optional exact query-tile specialization for the measured GB10 Omni prefix."""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _native():
    try:
        import _fastvideo_h3_prefix
    except ImportError:
        return None
    return _fastvideo_h3_prefix


def try_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
    """BHSD operands; keep Torch SDPA outside the validated eager configuration."""
    if (torch.is_grad_enabled() or torch.compiler.is_compiling()
            or q.shape != (1, 56, 15360, 128) or k.shape != (1, 56, 29678, 128)
            or v.shape != k.shape or not q.is_cuda or k.device != q.device or v.device != q.device
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1
            or torch.cuda.get_device_capability(q.device) != (12, 1)
            or torch.version.git_version != '7661cd9c6b841b62b7f411aa52ec51f05457263b'):
        return None
    native = _native()
    return None if native is None else native.forward(q, k, v)
