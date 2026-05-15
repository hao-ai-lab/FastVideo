# SPDX-License-Identifier: Apache-2.0
"""LTX-2 NVFP4 quantization (FlashInfer-backed).

NVFP4 is NVIDIA's block-scaled FP4 format (e2m1 mantissa, fp32 alpha,
``layout_128x4`` scale layout, group size 16) — distinct from
generic FP4 / OCP-FP4 / MX-FP4. We name the public surface ``NVFP4``
explicitly so downstream callers don't conflate it with other FP4
variants that may land later (e.g. AMD's MX-FP4 or vendor-neutral
e3m0).

Upstreamed from ``FastVideo-internal`` so consumers that load LTX-2
weights with NVFP4 quantization can drive the public package
end-to-end.

`flashinfer` is imported lazily inside the call paths that need it.
This keeps ``import fastvideo`` cheap on hosts where flashinfer is
not installed; only the actual NVFP4 quantize / matmul ops fail at
use time, with a clear error.
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


def _require_flashinfer() -> tuple[Any, Any, Any]:
    """Lazy flashinfer import — raised at use time, not import time.

    Returns the bound ``(SfLayout, mm_fp4, nvfp4_quantize)`` triple from
    flashinfer. Raises ``ImportError`` with an actionable hint if the
    package is not available.
    """
    try:
        from flashinfer import (  # type: ignore[import-not-found]
            SfLayout, mm_fp4, nvfp4_quantize,
        )
    except ImportError as exc:  # pragma: no cover - depends on host env
        raise ImportError("NVFP4 quantization requires flashinfer. "
                          "Install with `pip install flashinfer-python`.") from exc
    return SfLayout, mm_fp4, nvfp4_quantize


_LTX2_REFINE_ONLY_SUFFIXES = (
    ".audio_to_video_attn.to_q",
    ".video_to_audio_attn.to_k",
    ".video_to_audio_attn.to_v",
)


def _is_ltx2_refine_only_prefix(prefix: str) -> bool:
    return any(prefix.endswith(suffix) for suffix in _LTX2_REFINE_ONLY_SUFFIXES)


def _get_ltx2_fp4_stage_profile(default: str = "refine") -> str:
    """Read the active stage profile from the forward context.

    Streaming inference flips between ``base`` and ``refine`` between
    segments; the FP4 layer set differs across the two. Falls back to
    ``default`` whenever the context is not available — this keeps the
    op safe to run outside the streaming server (e.g. during eager
    tests).
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


_OPS_REGISTERED = False


def _register_ops_once() -> None:
    """Register the fastvideo_fp4 torch ops on first import that needs
    them. Each op binds to flashinfer at call time; this just sets up
    the dispatcher entries."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    @torch.library.custom_op(
        "fastvideo_fp4::nvfp4_quantize",
        mutates_args=(),
        device_types="cuda",
    )
    def _nvfp4_quantize_op(
        x: torch.Tensor,
        global_sf: torch.Tensor,
        sf_layout: int,
        do_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        SfLayout, _, nvfp4_quantize = _require_flashinfer()
        return nvfp4_quantize(x, global_sf, sfLayout=SfLayout(sf_layout), do_shuffle=do_shuffle)

    @_nvfp4_quantize_op.register_fake
    def _nvfp4_quantize_op_fake(
        x: torch.Tensor,
        global_sf: torch.Tensor,
        sf_layout: int,
        do_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del global_sf, sf_layout, do_shuffle
        m, k = x.shape
        quantized = torch.empty((m, (k + 1) // 2), device=x.device, dtype=torch.uint8)
        scales = torch.empty((m, (k + 15) // 16), device=x.device, dtype=torch.uint8)
        return quantized, scales

    @torch.library.custom_op(
        "fastvideo_fp4::mm_fp4",
        mutates_args=(),
        device_types="cuda",
    )
    def _mm_fp4_op(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
        out: torch.Tensor | None = None,
        block_size: int = 16,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        _, mm_fp4, _ = _require_flashinfer()
        if a.dtype == torch.float4_e2m1fn_x2:
            a = a.view(torch.uint8) if a.is_contiguous() else a.contiguous().view(torch.uint8)
        if b.dtype == torch.float4_e2m1fn_x2:
            b = b.view(torch.uint8) if b.is_contiguous() else b.contiguous().view(torch.uint8)

        return mm_fp4(
            a,
            b,
            a_scale,
            b_scale,
            alpha,
            out_dtype,
            out,
            block_size=block_size,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
        )

    @_mm_fp4_op.register_fake
    def _mm_fp4_op_fake(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
        out: torch.Tensor | None = None,
        block_size: int = 16,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        del a_scale, b_scale, alpha, block_size, use_8x4_sf_layout, backend
        del use_nvfp4
        if out is not None:
            return out
        out_shape = (*a.shape[:-1], b.shape[1])
        return torch.empty(out_shape, device=a.device, dtype=out_dtype)

    _OPS_REGISTERED = True


def _nvfp4_quantize(
    x: torch.Tensor,
    global_sf: Any,
    *,
    sfLayout: Any,
    do_shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _register_ops_once()
    SfLayout, _, _ = _require_flashinfer()
    if isinstance(sfLayout, SfLayout):
        sf_layout = sfLayout.value
    elif hasattr(sfLayout, "value"):
        sf_layout = int(sfLayout.value)
    else:
        sf_layout = int(sfLayout)
    if not torch.is_tensor(global_sf):
        global_sf = torch.tensor(global_sf, device=x.device, dtype=torch.float32)
    elif global_sf.device != x.device:
        global_sf = global_sf.to(device=x.device)
    if sf_layout == SfLayout.layout_linear.value:
        x_for_quant = x
        logical_rows = x.shape[0]
    else:
        # Sequence-parallel can feed either logical rows or row-padded
        # rows. Normalize to the kernel tile shape for swizzled layouts
        # so both paths share a stable quantization contract.
        row_tile = 8 if sf_layout == SfLayout.layout_8x4.value else 128
        logical_rows = x.shape[0]
        pad_rows = (-logical_rows) % row_tile
        x_for_quant = F.pad(x, (0, 0, 0, pad_rows))

    quantized, scales = torch.ops.fastvideo_fp4.nvfp4_quantize(x_for_quant, global_sf, sf_layout, do_shuffle)
    if sf_layout != SfLayout.layout_linear.value:
        quantized = quantized.narrow(0, 0, logical_rows)
    return quantized, scales


def _mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: Any,
    out_dtype: torch.dtype,
    out: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    _register_ops_once()
    block_size = kwargs.pop("block_size", 16)
    use_8x4_sf_layout = kwargs.pop("use_8x4_sf_layout", False)
    backend = kwargs.pop("backend", "auto")
    use_nvfp4 = kwargs.pop("use_nvfp4", True)
    if kwargs:
        raise TypeError(f"Unsupported kwargs for _mm_fp4: {sorted(kwargs)}")
    if alpha is not None and not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=a.device, dtype=torch.float32)
    return torch.ops.fastvideo_fp4.mm_fp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_8x4_sf_layout,
        backend,
        use_nvfp4,
    )


class NVFP4QuantizeMethod(QuantizeMethodBase):

    def __init__(self, layer_prefix: str = ""):
        super().__init__()
        self.weight_fp4 = None
        self.weight_scale = None
        self.x_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        self.layer_prefix = layer_prefix
        self._is_refine_only_layer = _is_ltx2_refine_only_prefix(layer_prefix)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        SfLayout, _, _ = _require_flashinfer()
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16, (
            f"only allow bf16/fp16 inputs to fp4 linear, got {x.dtype}")
        x_2d = x.view(-1, x.shape[-1])
        x_fp4, x_scale = _nvfp4_quantize(
            x_2d,
            self.x_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        return x_fp4, x_scale, self.x_global_sf

    def wants_prequantized_input(self) -> bool:
        if not self._is_refine_only_layer:
            return True
        stage_profile = _get_ltx2_fp4_stage_profile(default="refine")
        return stage_profile != "base"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        SfLayout, _, _ = _require_flashinfer()
        out_dim = layer.weight.shape[0]
        original_shape = x.shape

        # Stage-aware profile: keep refine-only FP4 layers in dense mode
        # during stage-1 denoising so the base path doesn't pay the
        # quantize/dequantize tax for layers it never touches.
        stage_profile = _get_ltx2_fp4_stage_profile(default="refine")
        if self._is_refine_only_layer and stage_profile == "base":
            out = (F.linear(x, layer.weight, bias) if torch.cuda.is_available() or bias is None else F.linear(
                x, layer.weight, bias.to(x.dtype)))
            return out.view(*original_shape[:-1], out_dim)
        if pre_quantized is not None:
            x_fp4, x_scale, x_global_sf = pre_quantized
            # FlashInfer fused norm+quant APIs may return 3D tensors for
            # 3D inputs. mm_fp4 only accepts 2D tensors, so flatten
            # batch/sequence dims here.
            if x_fp4.dim() > 2:
                x_fp4 = x_fp4.view(-1, x_fp4.shape[-1])
            if x_scale.dim() > 2:
                x_scale = x_scale.view(-1, x_scale.shape[-1])
        else:
            assert x.dtype == torch.bfloat16 or x.dtype == torch.float16, (
                f"only allow bf16/fp16 inputs to fp4 linear, got {x.dtype}")
            x = x.view(-1, x.shape[-1])
            x_global_sf = self.x_global_sf
            x_fp4, x_scale = _nvfp4_quantize(
                x,
                x_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )

        weight_fp4 = layer._nvfp4_weight
        weight_scale = layer._nvfp4_weight_scale
        weight_global_sf = layer._weight_global_sf

        if hasattr(layer, "_nvfp4_alpha"):
            alpha = layer._nvfp4_alpha / x_global_sf
        else:
            alpha = 1.0 / (x_global_sf * weight_global_sf)

        out = _mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_scale,
            weight_scale.T,
            alpha,
            torch.bfloat16,
            None,
            backend='auto',
        )

        if bias is not None:
            out = out + bias
        out = out.view(*original_shape[:-1], out_dim)
        return out


class NVFP4Config(QuantizationConfig):
    """LTX-2-specific NVFP4 quantization configuration.

    NVFP4 is NVIDIA's block-scaled FP4 (e2m1 mantissa, fp32 alpha,
    ``layout_128x4`` scale layout, group size 16). Today this class
    hardcodes the LTX-2 layer paths it covers. When a second model
    wants NVFP4, lift the layer-path list into a config field
    instead of hardcoding it here.
    """

    def __init__(self, layer_profile: str = "refine"):
        super().__init__()
        # ``base``: stage-1 set (no attn2.to_out, no cross-modal AV
        # projections). ``refine``: full stage-2 set.
        self.layer_profile = layer_profile

    def get_name(self):
        return "nvfp4"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NVFP4Config:
        return cls(layer_profile=config.get("layer_profile", "refine"))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        # Use the superset at build/load time, then switch active subset
        # dynamically in NVFP4QuantizeMethod.apply based on stage profile.
        fp4_layers = [[
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
        fp4_layers.append([
            "ltx2.adaln_single.linear",
        ])
        if isinstance(layer, LinearBase) and any(prefix in layer_names for layer_names in fp4_layers):
            return NVFP4QuantizeMethod(layer_prefix=prefix)
        return None


def convert_model_to_nvfp4(model: torch.nn.Module) -> None:
    SfLayout, _, _ = _require_flashinfer()
    from torch.distributed.tensor import DTensor  # type: ignore

    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, NVFP4QuantizeMethod):
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
            weight_global_sf = (448 * 6) / weight_local.float().abs().nan_to_num().max()
            fp4_w, fp4_s = _nvfp4_quantize(
                weight_local,
                weight_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            weight_global_sf_t = torch.as_tensor(
                weight_global_sf,
                device=weight_local.device,
                dtype=torch.float32,
            )
            mod.register_buffer("_nvfp4_weight", fp4_w, persistent=False)
            mod.register_buffer("_nvfp4_weight_scale", fp4_s, persistent=False)
            mod.register_buffer(
                "_weight_global_sf",
                weight_global_sf_t.to(dtype=torch.bfloat16),
                persistent=False,
            )
            mod.register_buffer(
                "_nvfp4_alpha",
                (1.0 / weight_global_sf_t).to(dtype=torch.float32),
                persistent=False,
            )


__all__ = [
    "NVFP4Config",
    "NVFP4QuantizeMethod",
    "convert_model_to_nvfp4",
]
