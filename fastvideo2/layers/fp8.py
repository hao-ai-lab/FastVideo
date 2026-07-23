"""Dynamic FP8 (e4m3) linear quantization — vendored from fastvideo-main.

Provenance: hao-ai-lab/FastVideo fastvideo/layers/quantization/fp8_config.py
@ e3f47dc2de2d1fa0c68c5839a0a41ed25b04a953 (per-tensor granularity only; the
released FastWan-QAD recipe). The math is reproduced verbatim because it IS
the artifact's compute contract: weights are quantized in place after loading
(absmax per-tensor scale), activations are quantized dynamically per call,
and the GEMM runs ``torch._scaled_mm`` with bf16 output on sm89+ (bf16
dequant fallback elsewhere). Bit-exactness vs fastvideo-main is gated by
``wan21/gates`` anchors, never assumed.

Import stays torch-free (torch inside functions) per the layers convention.
"""
from __future__ import annotations

from typing import Any

FP8_MAX = 448.0                      # torch.finfo(float8_e4m3fn).max
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def quantize_tensorwise(x_2d: Any) -> tuple[Any, Any]:
    """(x_fp8 [M, K], scale [1] fp32) — main's ``_quantize_tensorwise`` verbatim:
    absmax in fp32, scale clamped, division performed in the INPUT dtype (the
    scale is cast down before dividing — that rounding is part of the contract).
    """
    import torch
    x_absmax = x_2d.abs().amax().float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, x_scale.view(1)


def _supports_fp8_compute() -> bool:
    import torch
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)


class _Torch:
    """Lazy import shim so the module imports torch-free."""
    def __getattr__(self, name: str) -> Any:
        import torch
        return getattr(torch, name)


def make_fp8_linear_class() -> type:
    """Build the FP8Linear nn.Module class (deferred so importing this module
    needs no torch)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FP8Linear(nn.Module):
        """A quantized replacement for one nn.Linear: fp8 weight + fp32 scale
        buffers, dynamic per-tensor activation quant, ``_scaled_mm`` GEMM.
        Mirrors main's ``FP8QuantizeMethod.apply`` + ``convert_model_to_fp8``.
        """

        def __init__(self, linear: nn.Linear):
            super().__init__()
            with torch.no_grad():
                w = linear.weight.detach()
                # main quantizes AFTER the checkpoint was cast to the load
                # dtype (bf16), so the fp8 codes depend on that chain.
                w_absmax = w.abs().amax().nan_to_num().to(torch.float32)
                w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE).view(1)
                w_fp8 = (w / w_scale.to(w.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            self.register_buffer("_fp8_weight", w_fp8.contiguous(), persistent=False)
            self.register_buffer("_fp8_weight_scale", w_scale.to(torch.float32), persistent=False)
            if linear.bias is not None:
                self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
            else:
                self.bias = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out_dim = self._fp8_weight.shape[0]
            original_shape = x.shape
            if not _supports_fp8_compute():
                # main's ``_apply_dequant`` fallback (pre-sm89 / CPU) — also
                # what CPU tests exercise.
                w_scale = self._fp8_weight_scale.to(x.dtype)
                weight = self._fp8_weight.to(x.dtype) * w_scale.unsqueeze(1)
                return F.linear(x, weight, self.bias).view(*original_shape[:-1], out_dim)
            assert x.dtype in (torch.bfloat16, torch.float16), (
                f"only bf16/fp16 inputs to fp8 linear, got {x.dtype}")
            x_fp8, x_scale = quantize_tensorwise(x.reshape(-1, x.shape[-1]))
            out = torch._scaled_mm(
                x_fp8,
                self._fp8_weight.t(),
                scale_a=x_scale,
                scale_b=self._fp8_weight_scale,
                out_dtype=torch.bfloat16,
            )
            if isinstance(out, tuple):
                out = out[0]
            if self.bias is not None:
                out = out + self.bias
            return out.view(*original_shape[:-1], out_dim)

    return FP8Linear


def quantize_fp8_(model: Any, linear_names: list[str]) -> Any:
    """Swap the named nn.Linear submodules for FP8Linear, in place.

    ``linear_names`` are exact module paths (e.g. ``blocks.0.attn1.to_q``) —
    the caller enumerates them so there is no fuzzy suffix matching to audit.
    """
    FP8Linear = make_fp8_linear_class()
    for name in linear_names:
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        old = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        new = FP8Linear(old)
        if leaf.isdigit():
            parent[int(leaf)] = new
        else:
            setattr(parent, leaf, new)
    return model
