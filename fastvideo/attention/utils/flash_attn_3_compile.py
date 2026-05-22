"""torch.library.custom_op wrapper around flash_attn_3 so inductor can graph
through the FA3 call (mirrors the FA4 wrapper in flash_attn_cute.py).

Inference-only: this wrapper returns the output tensor only and does not
register an autograd implementation. Under torch.compile, a backward pass
against this op will fail. The softmax_lse returned by flash_attn_3 is
discarded; bring it back (and add register_autograd / setup_context like
flash_attn_cute.py does) if training/grad use is needed.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    from flash_attn_interface import flash_attn_func as _raw_flash_attn_3_func
else:
    raise ImportError("flash_attn_3 is only available on CUDA devices; this error must be handled by the caller")


@torch.library.custom_op(
    "fastvideo::_flash_attn_3_forward",
    mutates_args=(),
    device_types="cuda",
)
def _flash_attn_3_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    out = _raw_flash_attn_3_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.library.register_fake("fastvideo::_flash_attn_3_forward")
def _flash_attn_3_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    del k, softmax_scale, causal
    return q.new_empty(q.shape[:-1] + (v.shape[-1], ))


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    return torch.ops.fastvideo._flash_attn_3_forward(q, k, v, softmax_scale, causal)
