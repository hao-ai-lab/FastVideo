"""torch.library.custom_op wrapper around flash_attn_3 so inductor can graph
through the FA3 call (mirrors the FA4 wrapper in flash_attn_cute.py).

The custom op has no autograd (softmax_lse discarded), so `flash_attn_func`
routes grad-requiring inputs to raw FA3 (training) and the no-grad inference
path through the custom op, where torch.compile can graph through it.
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
    # No autograd on the custom op → route grad-requiring inputs (training) to raw FA3.
    if q.requires_grad or k.requires_grad or v.requires_grad:
        out = _raw_flash_attn_3_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
        return out[0] if isinstance(out, tuple) else out
    return torch.ops.fastvideo._flash_attn_3_forward(q, k, v, softmax_scale, causal)
