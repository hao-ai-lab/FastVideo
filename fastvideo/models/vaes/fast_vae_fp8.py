# SPDX-License-Identifier: Apache-2.0
"""Per-tensor FP8 (e4m3) linears for the fast 24-layer MiniMax-H3 video decoder.

Approximate tier, opt-in through ``FASTVIDEO_H3_FAST_VAE_FP8``. On sm_121 the fast decoder's bf16
GEMMs run at ~82-101 TFLOPS; the same shapes run at 154-199 TFLOPS as per-tensor ``torch._scaled_mm``
(measured, pass 22). The weight is quantized once (per-tensor absmax) and kept as a frozen E4M3
buffer; the activation is quantized per call. The conv/embedder/proj_out stay at their original
precision (the decoder runs them in fp32), so only the block attention/FFN linears change.
"""
from __future__ import annotations

import torch
from torch import nn

FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


class FP8Linear(nn.Module):
    """Drop-in for ``nn.Linear``: frozen per-tensor E4M3 weight + dynamic per-tensor activation."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        out_features, in_features = linear.weight.shape
        weight = linear.weight.detach()
        amax = weight.abs().amax().float().clamp(min=1e-12)
        scale = amax / FP8_MAX
        self.register_buffer("weight_fp8", (weight / scale.to(weight.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE),
                             persistent=False)
        self.register_buffer("weight_scale", scale.view(1), persistent=False)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone(), persistent=False)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        amax = x2d.abs().amax().float().clamp(min=1e-12)
        scale = amax / FP8_MAX
        x_fp8 = (x2d / scale.to(x2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        out = torch._scaled_mm(x_fp8,
                               self.weight_fp8.t(),
                               scale_a=scale.view(1),
                               scale_b=self.weight_scale,
                               out_dtype=torch.bfloat16)
        if isinstance(out, tuple):
            out = out[0]
        if self.bias is not None:
            out = out + self.bias
        return out.view(*original_shape[:-1], self.out_features)


def convert_linears_to_fp8(module: nn.Module, skip_names: tuple[str, ...] = ()) -> int:
    """Recursively replace ``nn.Linear`` children with :class:`FP8Linear`; returns the number replaced."""
    replaced = 0
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, FP8Linear(child))
            replaced += 1
        else:
            replaced += convert_linears_to_fp8(child, skip_names)
    return replaced


__all__ = ["FP8Linear", "convert_linears_to_fp8"]
