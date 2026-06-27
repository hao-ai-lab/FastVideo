# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization-aware (QAD) linear method, inference path.

Quantizes every targeted linear's weight to NVFP4 once at load time and
runs each forward as a registered flashinfer-backed FP4 matmul. The
original fp16/bf16 weight is *popped* immediately after quantization so
the half-precision copy does not keep occupying GPU memory — that's
what lets a Wan-2.1 pipeline stay fully resident on a single GPU
without any CPU offloading.

The quantize / matmul custom ops are owned by
:mod:`fastvideo.layers.quantization.nvfp4_config` and registered under
the ``fastvideo_fp4::`` namespace. We reuse them here for two reasons:

1. Re-registering the same op name in a second module would raise.
2. The registered ops have ``register_fake`` shape/dtype kernels, which
   is what makes the inference pipeline's per-block ``torch.compile``
   trace through without graph breaks. Calling raw flashinfer functions
   (the old behavior of this file, plus a ``@torch.compile`` on
   ``apply``) graph-breaks at every quantize and every matmul.

For QAT *training*, see ``nvfp4_qat_train_config`` which keeps the
weight trainable and fake-quantizes on the fly via a straight-through
estimator.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import torch
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.layers.quantization.nvfp4_config import (
    _mm_fp4,
    _nvfp4_quantize,
    _require_flashinfer,
)
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)

# Wan-style attention + FFN projection layers. Matched as substrings of the
# layer prefix (e.g. "blocks.0.attn1.to_q" contains "to_q").
DEFAULT_FP4_LAYERS = (
    "ffn.fc_in",
    "ffn.fc_out",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
)


def _layout_128x4() -> Any:
    SfLayout, _, _ = _require_flashinfer()
    return SfLayout.layout_128x4


class NVFP4QATQuantizeMethod(QuantizeMethodBase):
    """Inference-only NVFP4 linear method with weight popping.

    The dense ``weight`` parameter is materialized at load time only so
    that :func:`convert_model_to_fp4` can read it once; the loader then
    removes it via ``mod._parameters.pop('weight')``. From that point
    forward, ``apply`` reads only ``_fp4_weight`` / ``_fp4_weight_scale``
    / ``_weight_global_sf``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight_fp4 = None
        self.weight_scale = None
        # Static input global scale factor. Matches the FastVideo-Quantization
        # production path; recomputing it per-call via a ``.max()`` reduction
        # (the previous behavior) adds a sync point, costs a kernel launch,
        # and produces a data-dependent value that prevents CUDA-graph
        # capture under ``torch.compile(mode='reduce-overhead')``.
        self.x_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # ``_fp4_weight`` carries the (out, in/2) packed fp4 weight, so its
        # row count is the output dim even after the dense weight is popped.
        out_dim = layer._fp4_weight.shape[0]
        original_shape = x.shape

        assert x.dtype in (torch.bfloat16, torch.float16), (f"only allow bf16/fp16 inputs to fp4 linear, got {x.dtype}")
        x = x.view(-1, x.shape[-1])
        x_global_sf = self.x_global_sf
        x_fp4, x_scale = _nvfp4_quantize(
            x,
            x_global_sf,
            sfLayout=_layout_128x4(),
            do_shuffle=False,
        )

        weight_fp4 = layer._fp4_weight
        weight_scale = layer._fp4_weight_scale
        weight_global_sf = layer._weight_global_sf

        out = _mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_scale,
            weight_scale.T,
            1.0 / (x_global_sf * weight_global_sf),
            torch.bfloat16,
            None,
            backend="cutlass",
        )

        if bias is not None:
            out = out + bias
        out = out.view(*original_shape[:-1], out_dim)
        return out


class NVFP4QATConfig(QuantizationConfig):
    """NVFP4 (Wan-style) linear quantization, inference.

    Args:
        target_layers: Substrings matched against each linear layer's
            prefix. A layer is quantized if any substring is contained in
            its prefix. Defaults to the standard Wan attention + FFN
            projections (:data:`DEFAULT_FP4_LAYERS`).
    """

    def __init__(self, target_layers: tuple[str, ...] | None = None) -> None:
        super().__init__()
        self.target_layers = (tuple(target_layers) if target_layers else DEFAULT_FP4_LAYERS)

    def get_name(self) -> str:
        return "nvfp4_qat"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NVFP4QATConfig:
        target_layers = config.get("target_layers")
        if target_layers is not None:
            target_layers = tuple(target_layers)
        return cls(target_layers=target_layers)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase
        if isinstance(layer, LinearBase) and any(name in prefix for name in self.target_layers):
            return NVFP4QATQuantizeMethod()
        return None


def convert_model_to_fp4(model: torch.nn.Module) -> None:
    """Prequantize every FP4-tagged linear and drop its dense weight.

    Walks the module tree, and for each layer whose ``quant_method`` is
    an :class:`NVFP4QATQuantizeMethod`, computes the NVFP4 packed weight
    / scale / global-scale buffers, then pops the original fp16/bf16
    ``weight`` parameter so it no longer occupies GPU memory.
    """
    SfLayout, _, _ = _require_flashinfer()
    from torch.distributed.tensor import DTensor  # type: ignore

    with torch.no_grad():
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if not isinstance(qm, NVFP4QATQuantizeMethod):
                continue

            weight = getattr(mod, "weight", None)
            if weight is None:
                continue

            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]

            # Only the reduced scalar needs fp32; avoid a full fp32 copy.
            weight_absmax = (weight_local.detach().abs().nan_to_num().amax().to(dtype=torch.float32))
            weight_global_sf = (448 * 6) / weight_absmax
            fp4_w, fp4_s = _nvfp4_quantize(
                weight_local,
                weight_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            mod.register_buffer("_fp4_weight", fp4_w, persistent=False)
            mod.register_buffer("_fp4_weight_scale", fp4_s, persistent=False)
            mod.register_buffer(
                "_weight_global_sf",
                weight_global_sf.to(dtype=torch.bfloat16),
                persistent=False,
            )

            # Drop the dense weight as soon as the fp4 buffers are installed
            # so it cannot keep occupying GPU memory.
            removed_weight = mod._parameters.pop("weight", None)
            if removed_weight is not None:
                removed_weight.grad = None
            del removed_weight, weight, weight_local, weight_absmax
    gc.collect()
    torch.cuda.empty_cache()


__all__ = [
    "NVFP4QATConfig",
    "NVFP4QATQuantizeMethod",
    "convert_model_to_fp4",
    "DEFAULT_FP4_LAYERS",
]
