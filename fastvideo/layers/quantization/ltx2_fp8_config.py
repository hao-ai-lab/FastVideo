# SPDX-License-Identifier: Apache-2.0
"""LTX-2 FP8 quantization (``torch._scaled_mm``-backed).

A direct FP8 counterpart to :mod:`fastvideo.layers.quantization.nvfp4_config`.
It reuses the same LTX-2 layer set, the same ``base``/``refine`` stage
profiles, and the same ``quantize_input`` / ``apply(pre_quantized=...)``
protocol the LTX-2 attention forward relies on — only the two numeric
kernels differ:

* **quantize** — ``float8_e4m3fn`` cast with an absmax scale (no FlashInfer,
  no block-scale swizzle, no global scale factor).
* **matmul** — :func:`torch._scaled_mm` instead of FlashInfer ``mm_fp4``.

Granularity is **Level 2**: per-output-channel weight scales (computed once
at load) and dynamic per-token activation scales (recomputed each forward).
Both scales live on the non-contraction dimensions, so ``_scaled_mm`` applies
them as a row/column rescale of the FP8 GEMM result — no custom kernel
required. This is the standard LLM FP8 recipe.

Unlike NVFP4 (Blackwell-only), the FP8 ``_scaled_mm`` path runs on Ada
(sm89), Hopper (sm90), and Blackwell (sm100). On pre-sm89 GPUs the method
falls back to a bf16 dequant matmul so the model still runs (without the FP8
speedup).
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
# Floor for scales so an all-zero tile cannot produce a divide-by-zero.
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

# Cross-modal AV projections only the refine stage exercises. Mirrors
# nvfp4_config so the base stage keeps these dense instead of paying the
# quantize tax for layers it never touches.
_LTX2_REFINE_ONLY_SUFFIXES = (
    ".audio_to_video_attn.to_q",
    ".video_to_audio_attn.to_k",
    ".video_to_audio_attn.to_v",
)


def _is_ltx2_refine_only_prefix(prefix: str) -> bool:
    return any(prefix.endswith(suffix) for suffix in _LTX2_REFINE_ONLY_SUFFIXES)


def _get_ltx2_stage_profile(default: str = "refine") -> str:
    """Read the active LTX-2 stage profile from the forward context.

    Shares the ``ltx2_fp4_stage_profile`` batch flag set by the denoising
    stage (the flag is stage-generic despite the ``fp4`` name). Falls back to
    ``default`` whenever the context is unavailable so the op stays safe in
    eager tests outside the streaming server.
    """
    try:
        from fastvideo.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        forward_batch = getattr(forward_ctx, "forward_batch", None)
        if forward_batch is None:
            return default
        extra = getattr(forward_batch, "extra", None)
        if not isinstance(extra, dict):
            return default
        profile = extra.get("ltx2_fp4_stage_profile", default)
        if profile in ("base", "refine"):
            return profile
        return default
    except Exception:
        return default


def _supports_fp8_compute() -> bool:
    """Whether the active device has an FP8 ``_scaled_mm`` path (sm89+)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)


def _quantize_tensorwise(
        x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor dynamic FP8 quantization (one scalar scale).

    Returns ``(x_fp8 [M, K], x_scale [1] float32)`` for the tensorwise (fast)
    ``torch._scaled_mm`` path. Division stays in the input dtype to avoid an
    fp32 upcast of the whole activation.
    """
    x_absmax = x_2d.abs().amax().float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX,
                                                  FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale.view(1)


def _quantize_rowwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row) dynamic FP8 quantization.

    Returns ``(x_fp8 [M, K], x_scale [M, 1] float32)`` ready for the
    ``scale_a`` argument of :func:`torch._scaled_mm` rowwise scaling.
    """
    x_absmax = x_2d.abs().amax(dim=-1, keepdim=True).float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX,
                                                  FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale


class LTX2FP8QuantizeMethod(QuantizeMethodBase):
    """FP8 linear method.

    ``granularity="tensor"`` (default): per-tensor weight + per-tensor dynamic
    activation scales — the fast tensorwise ``torch._scaled_mm`` path that
    reliably beats bf16. ``granularity="channel"``: per-output-channel weight +
    per-token activation scales (rowwise) — higher accuracy but the slower
    ``_scaled_mm`` path; use for quality comparisons.
    """

    def __init__(self, layer_prefix: str = "", granularity: str = "tensor"):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.granularity = granularity
        self._is_refine_only_layer = _is_ltx2_refine_only_prefix(layer_prefix)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Allocate the bf16/fp16 placeholder; convert_model_to_fp8 materializes
        # the FP8 weight + per-channel scale buffers after weights are loaded.
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def quantize_input(
            self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Pre-quantize an activation once for reuse across q/k/v projections.

        Returns a 3-tuple to match the shape the LTX-2 attention forward
        threads through ``apply(pre_quantized=...)``; the third slot is unused
        for FP8 (NVFP4 carries a global scale factor there).
        """
        assert x.dtype in (torch.bfloat16, torch.float16), (
            f"only allow bf16/fp16 inputs to fp8 linear, got {x.dtype}")
        x_2d = x.view(-1, x.shape[-1])
        if self.granularity == "channel":
            x_fp8, x_scale = _quantize_rowwise(x_2d)
        else:
            x_fp8, x_scale = _quantize_tensorwise(x_2d)
        return x_fp8, x_scale, None

    def wants_prequantized_input(self) -> bool:
        if not self._is_refine_only_layer:
            return True
        return _get_ltx2_stage_profile(default="refine") != "base"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quantized: tuple[torch.Tensor, torch.Tensor, Any] | None = None,
    ) -> torch.Tensor:
        out_dim = layer.weight.shape[0]
        original_shape = x.shape

        # Keep refine-only layers dense during stage-1 denoising so the base
        # path never pays the quantize tax for layers it doesn't touch.
        stage_profile = _get_ltx2_stage_profile(default="refine")
        if self._is_refine_only_layer and stage_profile == "base":
            out = F.linear(x, layer.weight, bias)
            return out.view(*original_shape[:-1], out_dim)

        if not _supports_fp8_compute():
            return self._apply_dequant(layer, x, bias)

        if pre_quantized is not None:
            x_fp8, x_scale, _ = pre_quantized
            if x_fp8.dim() > 2:
                x_fp8 = x_fp8.reshape(-1, x_fp8.shape[-1])
            if x_scale.dim() > 2:
                x_scale = x_scale.reshape(-1, x_scale.shape[-1])
        elif self.granularity == "channel":
            x_fp8, x_scale = _quantize_rowwise(x.reshape(-1, x.shape[-1]))
        else:
            x_fp8, x_scale = _quantize_tensorwise(x.reshape(-1, x.shape[-1]))

        w_fp8 = layer._fp8_weight  # [N, K] row-major (== col-major after .t())
        w_scale = layer._fp8_weight_scale  # scalar [1] (tensor) or [N] (channel)

        # channel: scale_a per-token [M,1], scale_b per-channel [1,N] (rowwise).
        # tensor: scalar scales -> fast cuBLASLt FP8 path.
        scale_b = w_scale.view(1, -1) if self.granularity == "channel" else w_scale

        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=x_scale,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
        if isinstance(out, tuple):
            out = out[0]
        if bias is not None:
            out = out + bias
        return out.view(*original_shape[:-1], out_dim)

    def _apply_dequant(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """bf16 fallback for pre-sm89 GPUs (no FP8 ``_scaled_mm``)."""
        out_dim = layer.weight.shape[0]
        original_shape = x.shape
        w_fp8 = getattr(layer, "_fp8_weight", None)
        if w_fp8 is None:
            out = F.linear(x, layer.weight, bias)
            return out.view(*original_shape[:-1], out_dim)
        w_scale = layer._fp8_weight_scale.to(x.dtype)
        weight = w_fp8.to(x.dtype) * w_scale.unsqueeze(1)
        out = F.linear(x, weight, bias)
        return out.view(*original_shape[:-1], out_dim)


class LTX2FP8Config(QuantizationConfig):
    """LTX-2-specific FP8 (e4m3) quantization configuration.

    Per-output-channel weight scales + dynamic per-token activation scales,
    executed with :func:`torch._scaled_mm`. Covers the same LTX-2 linear
    subset as :class:`NVFP4Config`; hardcodes the layer paths here exactly as
    NVFP4 does. When a second model wants FP8, lift the layer-path list into a
    config field instead.
    """

    def __init__(self, layer_profile: str = "refine", granularity: str = "tensor"):
        super().__init__()
        self.layer_profile = layer_profile
        if granularity not in ("tensor", "channel"):
            raise ValueError(
                "granularity must be 'tensor' (per-tensor, fast) or 'channel' "
                f"(per-channel weight + per-token activation), got {granularity}")
        # 'tensor': scalar scales -> fast tensorwise _scaled_mm (recommended).
        # 'channel': rowwise scales -> higher accuracy, slower _scaled_mm.
        self.granularity = granularity

    def get_name(self):
        return "ltx2_fp8"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 89

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LTX2FP8Config:
        return cls(
            layer_profile=config.get("layer_profile", "refine"),
            granularity=config.get("granularity", "tensor"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        fp8_layers = [[
            f"ltx2.blocks.{i}.attn1.to_q",
            f"ltx2.blocks.{i}.attn1.to_k",
            f"ltx2.blocks.{i}.attn1.to_v",
            f"ltx2.blocks.{i}.attn1.to_out",
            f"ltx2.blocks.{i}.attn2.to_q",
            f"ltx2.blocks.{i}.attn2.to_out",
            f"ltx2.blocks.{i}.audio_to_video_attn.to_q",
            f"ltx2.blocks.{i}.audio_to_video_attn.to_out",
            f"ltx2.blocks.{i}.video_to_audio_attn.to_k",
            f"ltx2.blocks.{i}.video_to_audio_attn.to_v",
            f"ltx2.blocks.{i}.ffn.fc_in",
            f"ltx2.blocks.{i}.ffn.fc_out",
        ] for i in range(48)]
        fp8_layers.append([
            "ltx2.adaln_single.linear",
        ])
        if isinstance(layer, LinearBase) and any(prefix in layer_names for layer_names in fp8_layers):
            return LTX2FP8QuantizeMethod(layer_prefix=prefix, granularity=self.granularity)
        return None


def convert_model_to_fp8(model: torch.nn.Module) -> None:
    """Materialize per-channel FP8 weight buffers from loaded bf16 weights.

    Mirrors :func:`convert_model_to_nvfp4`: walks the module tree and, for
    every layer carrying an :class:`LTX2FP8QuantizeMethod`, computes a
    per-output-channel absmax scale and stores the ``float8_e4m3fn`` weight +
    ``float32`` scale as non-persistent buffers.
    """
    from torch.distributed.tensor import DTensor  # type: ignore

    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if not isinstance(qm, LTX2FP8QuantizeMethod):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None:
            continue
        weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
        w = weight_local.float()
        if getattr(qm, "granularity", "tensor") == "channel":
            # Per-output-channel (per-row of [N, K]) absmax scale -> [N].
            w_absmax = w.abs().amax(dim=1).nan_to_num()
            w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
            w_fp8 = (w / w_scale.unsqueeze(1)).clamp(-FP8_MAX,
                                                     FP8_MAX).to(FP8_DTYPE)
        else:
            # Per-tensor absmax scale -> scalar [1] (fast tensorwise path).
            w_absmax = w.abs().amax().nan_to_num()
            w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE).view(1)
            w_fp8 = (w / w_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        mod.register_buffer("_fp8_weight", w_fp8.contiguous(), persistent=False)
        mod.register_buffer(
            "_fp8_weight_scale",
            w_scale.to(torch.float32),
            persistent=False,
        )


__all__ = [
    "LTX2FP8Config",
    "LTX2FP8QuantizeMethod",
    "convert_model_to_fp8",
]
