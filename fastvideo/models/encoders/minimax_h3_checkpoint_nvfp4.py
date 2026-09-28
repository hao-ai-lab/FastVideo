# SPDX-License-Identifier: Apache-2.0
"""Serialized NVFP4 execution for the MiniMax-H3 Qwen3-VL encoder.

The bf16 conditioner is 48 GB resident and sets the single-GB10 peak once the
DiT runs in FP8. A checkpoint written by
``scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py``
stores every ``language_model.layers.*`` linear as three tensors and no
``weight``::

    <prefix>.weight_packed        uint8   [out, in // 2]   two E2M1 values per byte
    <prefix>.weight_scale         uint8   [out, in // 16]  one E4M3 scale per 16 values,
                                                           FlashInfer 128x4 swizzled layout
    <prefix>.weight_global_scale  float32 [1]              (448 * 6) / amax(|W|)

The bytes are exactly what ``flashinfer.nvfp4_quantize(W, global_scale,
sfLayout=SfLayout.layout_128x4)`` returns, so loading them reproduces the state
``nvfp4_config.convert_model_to_nvfp4`` builds at runtime without ever
materializing the bf16 weight. Activations are quantized per call with a unit
global scale and multiplied with ``flashinfer.mm_fp4``.

The checkpoint selects this path through ``config.json``; every field is
required so a checkpoint from another exporter cannot pass by omission::

    "quantization_config": {"quant_method": "nvfp4", "activation_scheme": "dynamic",
                            "fmt": "e2m1", "group_size": 16, "scale_fmt": "e4m3",
                            "scale_layout": "128x4",
                            "modules_to_not_convert": ["model.visual", "lm_head"]}

Whole projection kinds may stay bf16 by listing their suffix in
``modules_to_not_convert`` (for example ``"mlp.down_proj"``); those linears
keep a plain ``weight`` and the unquantized method. Only the seven language
projection names are accepted there, so a typo fails at config time.

Single GPU only: packed columns and swizzled scale rows cannot be narrowed per
tensor-parallel rank without repacking. Missing tensors are reported by the
loader's strict check before the post-load hook runs; the hook adds only the
content checks a copied tensor can still fail.
"""

from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from fastvideo.distributed import get_tp_world_size
from fastvideo.layers.linear import LinearBase, LinearMethodBase
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.layers.quantization.nvfp4_config import (
    _coerce_fp4_input_dtype,
    _mm_fp4,
    _nvfp4_quantize,
    _register_ops_once,
    _require_flashinfer,
)
from fastvideo.models.utils import set_weight_attrs

NVFP4_GROUP_SIZE = 16
NVFP4_SCALE_LAYOUT = "128x4"
# FlashInfer's 128x4 layout tiles scales in 128 rows by 4 scale columns. Keeping
# every weight a whole number of tiles means the serialized scale tensor is
# exactly [out, in // 16] with no padding to describe. Every Qwen3-VL language
# linear satisfies this (out in {1024, 5120, 8192, 25600}, in in {5120, 8192, 25600}).
NVFP4_ROW_TILE = 128
NVFP4_COLUMN_MULTIPLE = 4 * NVFP4_GROUP_SIZE
# The linears of one Qwen3-VL language layer, as the checkpoint and the model name them.
LANGUAGE_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
# The three tensors a quantized linear stores, as parameter names on the layer
# and as checkpoint key suffixes. The converter imports these so both sides
# spell them once.
NVFP4_TENSOR_SUFFIXES = ("weight_packed", "weight_scale", "weight_global_scale")
_REQUIRED_METADATA_KEYS = ("activation_scheme", "fmt", "group_size", "scale_fmt", "scale_layout",
                           "modules_to_not_convert")
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# An E4M3 byte of 0xFF is NaN, so a scale tile that is still all 0xFF after
# loading is one the checkpoint never filled. The loader copies whole tensors
# or asserts, so the first 128x4 tile speaks for the tensor.
_UNLOADED_SCALE_BYTE = 0xFF
_SCALE_TILE_BYTES = NVFP4_ROW_TILE * 4


def validate_nvfp4_geometry(output_size: int, input_size: int) -> None:
    if output_size % NVFP4_ROW_TILE:
        raise ValueError(f"MiniMax-H3 serialized NVFP4 requires output_size divisible by {NVFP4_ROW_TILE}, "
                         f"got {output_size}")
    if input_size % NVFP4_COLUMN_MULTIPLE:
        raise ValueError(f"MiniMax-H3 serialized NVFP4 requires input_size divisible by {NVFP4_COLUMN_MULTIPLE}, "
                         f"got {input_size}")


def nvfp4_packed_weight_shape(output_size: int, input_size: int) -> tuple[int, int]:
    return output_size, input_size // 2


def nvfp4_scale_shape(output_size: int, input_size: int) -> tuple[int, int]:
    return output_size, input_size // NVFP4_GROUP_SIZE


def nvfp4_weight_global_scale(weight: torch.Tensor) -> torch.Tensor:
    """The per-tensor scale ``convert_model_to_nvfp4`` uses: E4M3 max times E2M1 max over amax.

    Unlike the runtime converter this refuses NaN or infinite weights instead of
    mapping them to zero, because a checkpoint written from them would be wrong
    forever.
    """
    amax = weight.float().abs().max()
    if not torch.isfinite(amax) or amax <= 0:
        raise ValueError("MiniMax-H3 NVFP4 global scale needs a finite, non-zero weight amax")
    scale = ((_E4M3_MAX * _E2M1_MAX) / amax).to(torch.float32)
    if not torch.isfinite(scale):
        raise ValueError(f"MiniMax-H3 NVFP4 global scale overflowed float32 for weight amax {amax.item():.3e}")
    return scale


def serialized_nvfp4_quantization_config(
    *,
    keep_bf16: tuple[str, ...] | list[str] = (),
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``config.json`` ``quantization_config`` the converter writes and ``from_config`` accepts.

    ``keep_bf16`` lists projection kinds from ``LANGUAGE_PROJECTIONS`` that stay
    bf16 in every language layer; they are appended to ``modules_to_not_convert``.
    """
    unknown = sorted(set(keep_bf16) - set(LANGUAGE_PROJECTIONS))
    if unknown:
        raise ValueError(f"keep_bf16 names unknown projection kinds {unknown}; choose from {LANGUAGE_PROJECTIONS}")
    config: dict[str, Any] = {
        "quant_method": "nvfp4",
        "activation_scheme": "dynamic",
        "fmt": "e2m1",
        "group_size": NVFP4_GROUP_SIZE,
        "scale_fmt": "e4m3",
        "scale_layout": NVFP4_SCALE_LAYOUT,
        "modules_to_not_convert": ["model.visual", "lm_head", *keep_bf16],
    }
    if producer:
        config["producer"] = dict(producer)
    return config


def _quantize_activation_nvfp4(x_2d: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one bf16 activation the way ``NVFP4QuantizeMethod`` does: unit global scale, 128x4 layout.

    The scale tensor is returned as uint8 bytes whatever dtype FlashInfer labels
    it with, so it always matches the uint8 weight scales the checkpoint stores.
    """
    sf_layout, _, _ = _require_flashinfer()
    x_fp4, x_scale = _nvfp4_quantize(x_2d, global_scale, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
    if x_scale.dtype != torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    return x_fp4, x_scale


def _nvfp4_linear(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """``x @ W.T`` on packed operands. Mirrors ``NVFP4QuantizeMethod.apply``: the weight is
    handed to ``mm_fp4`` transposed, the output is bf16, the backend is FlashInfer's choice."""
    return _mm_fp4(
        x_fp4,
        weight_packed.t(),
        x_scale,
        weight_scale.t(),
        alpha,
        torch.bfloat16,
        None,
        backend="auto",
    )


class MiniMaxH3SerializedNVFP4Config(QuantizationConfig):
    """Serialized 16-group NVFP4 contract for the H3 text encoder.

    The group size and scale layout are fixed by the loader's parameter shapes,
    so the only state is which projection kinds the checkpoint kept in bf16.
    """

    def __init__(self, bf16_projections: tuple[str, ...] = ()) -> None:
        super().__init__()
        unknown = sorted(set(bf16_projections) - set(LANGUAGE_PROJECTIONS))
        if unknown:
            raise ValueError(f"MiniMax-H3 serialized NVFP4 cannot keep unknown projection kinds {unknown} in bf16; "
                             f"choose from {LANGUAGE_PROJECTIONS}")
        # Projection kinds the checkpoint kept in bf16 in every language layer,
        # e.g. ("mlp.down_proj",). Those linears load a plain weight.
        self.bf16_projections = tuple(bf16_projections)

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MiniMaxH3SerializedNVFP4Config":
        quant_method = str(config.get("quant_method", "")).lower()
        if quant_method != "nvfp4":
            raise ValueError(f"MiniMax-H3 serialized NVFP4 config got quant_method {quant_method!r}")
        missing = [key for key in _REQUIRED_METADATA_KEYS if key not in config]
        if missing:
            raise ValueError("MiniMax-H3 serialized NVFP4 requires explicit quantization_config fields; "
                             f"missing {missing}")
        if str(config["activation_scheme"]).lower() != "dynamic":
            raise ValueError("MiniMax-H3 serialized NVFP4 requires dynamic activation quantization")
        if str(config["fmt"]).lower() not in ("e2m1", "float4_e2m1fn", "nvfp4"):
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires E2M1 weights, got {config['fmt']!r}")
        if str(config["scale_fmt"]).lower() not in ("e4m3", "float8_e4m3fn"):
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires E4M3 block scales, got {config['scale_fmt']!r}")
        group_size = config["group_size"]
        if isinstance(group_size, bool) or group_size != NVFP4_GROUP_SIZE:
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires group_size={NVFP4_GROUP_SIZE}, got {group_size!r}")
        if config["scale_layout"] != NVFP4_SCALE_LAYOUT:
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires scale_layout={NVFP4_SCALE_LAYOUT!r}, "
                             f"got {config['scale_layout']!r}")
        ignored_layers = config["modules_to_not_convert"]
        if not isinstance(ignored_layers, list | tuple):
            raise ValueError("MiniMax-H3 serialized NVFP4 modules_to_not_convert must be a sequence")
        bf16_projections: list[str] = []
        saw_visual = False
        for name in ignored_layers:
            if not isinstance(name, str) or not name:
                raise ValueError("MiniMax-H3 serialized NVFP4 modules_to_not_convert entries must be non-empty "
                                 f"strings, got {name!r}")
            if "visual" in name:
                saw_visual = True
            elif name == "lm_head" or name.endswith(".lm_head"):
                continue
            elif name in LANGUAGE_PROJECTIONS:
                bf16_projections.append(name)
            else:
                # A whole projection kind may stay bf16 across every language
                # layer; a single layer, a typo, or an HF-style full path is not
                # a contract this loader supports.
                raise ValueError("MiniMax-H3 serialized NVFP4 keeps bf16 per projection kind only; "
                                 f"modules_to_not_convert entry {name!r} is not one of {LANGUAGE_PROJECTIONS}")
        if not saw_visual:
            raise ValueError("MiniMax-H3 serialized NVFP4 requires the vision stack to be listed in "
                             "modules_to_not_convert")
        return cls(tuple(bf16_projections))

    def validate_runtime(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(f"MiniMax-H3 serialized NVFP4 requires a CUDA device; got {device.type!r}")
        capability = torch.cuda.get_device_capability(device)
        capability_number = capability[0] * 10 + capability[1]
        if capability_number < self.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized NVFP4 requires GPU capability "
                               f"sm{self.get_min_capability()} or newer, got sm{capability_number}")
        if capability[0] not in (10, 12):
            raise RuntimeError("MiniMax-H3 serialized NVFP4 runs FlashInfer's Blackwell FP4 GEMM; "
                               f"got unsupported sm{capability_number}")
        if get_tp_world_size() > 1:
            raise NotImplementedError("MiniMax-H3 serialized NVFP4 supports a single GPU: packed FP4 columns and "
                                      "128x4 swizzled scale rows cannot be narrowed per tensor-parallel rank")
        sf_layout, _, _ = _require_flashinfer()
        if not hasattr(sf_layout, "layout_128x4"):
            raise RuntimeError("The installed flashinfer has no SfLayout.layout_128x4; MiniMax-H3 serialized NVFP4 "
                               "checkpoints store their scales in that layout")
        _register_ops_once()

    def is_kept_bf16(self, prefix: str) -> bool:
        return any(prefix == name or prefix.endswith("." + name) for name in self.bf16_projections)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if not isinstance(layer, LinearBase) or ".language_model.layers." not in prefix:
            return None
        if self.is_kept_bf16(prefix):
            return None
        return MiniMaxH3SerializedNVFP4LinearMethod()


def _strict_dtype_loader(base_loader):
    """Wrap a layer's weight loader so a checkpoint tensor must carry the parameter's exact dtype."""
    if base_loader is None:
        raise ValueError("MiniMax-H3 serialized NVFP4 linears need the layer's weight_loader")

    def load(param: torch.Tensor, loaded_weight: torch.Tensor, *args: Any, **kwargs: Any):
        if loaded_weight.dtype != param.dtype:
            raise ValueError("Serialized MiniMax-H3 NVFP4 tensors must be stored as their declared dtype; "
                             f"got {loaded_weight.dtype} for a {param.dtype} parameter of shape {tuple(param.shape)}")
        return base_loader(param, loaded_weight, *args, **kwargs)

    return load


class MiniMaxH3SerializedNVFP4LinearMethod(LinearMethodBase):
    """Execute serialized NVFP4 weights without re-quantizing them."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        validate_nvfp4_geometry(output_size_per_partition, input_size_per_partition)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # No input_dim/output_dim: these tensors are not shardable, and without
        # the attributes the weight loaders copy them whole. The copy is a numeric
        # cast, so a checkpoint that stored scales as float8 bytes would be
        # silently converted value by value; refuse any dtype but the declared one.
        loader_attrs = {"weight_loader": _strict_dtype_loader(extra_weight_attrs.get("weight_loader"))}
        packed_name, scale_name, global_scale_name = NVFP4_TENSOR_SUFFIXES
        weight_packed = Parameter(
            torch.zeros(nvfp4_packed_weight_shape(output_size_per_partition, input_size_per_partition),
                        dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight_packed, loader_attrs)
        layer.register_parameter(packed_name, weight_packed)

        weight_scale = Parameter(
            torch.full(nvfp4_scale_shape(output_size_per_partition, input_size_per_partition),
                       _UNLOADED_SCALE_BYTE,
                       dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, loader_attrs)
        layer.register_parameter(scale_name, weight_scale)

        weight_global_scale = Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(weight_global_scale, loader_attrs)
        layer.register_parameter(global_scale_name, weight_global_scale)
        # No bf16 weight ever exists on this layer; ``None`` keeps ``layer.weight``
        # readable for code that inspects it, matching the purged NVFP4 path.
        layer.register_parameter("weight", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject a layer the checkpoint did not fill and derive the GEMM multiplier.

        Shapes and dtypes are fixed by ``create_weights`` and enforced by the
        weight loader's copy, and the loader reports missing tensors by name
        before this runs; what remains is content a copied tensor can still get
        wrong: a non-positive global scale, or a scale tile left at its 0xFF
        initializer by a direct caller that bypassed the loader.
        """
        weight_scale = layer.weight_scale
        global_scale = float(layer.weight_global_scale.item())
        if not (global_scale > 0) or global_scale == float("inf"):
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_global_scale was not loaded: "
                             f"it must be a finite positive value, got {global_scale}")
        if bool((weight_scale.view(-1)[:_SCALE_TILE_BYTES] == _UNLOADED_SCALE_BYTE).all()):
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_scale was not loaded: its first tile is still 0xFF")
        # ``mm_fp4`` folds both global scales into one multiplier. Activations use a
        # unit global scale, so the multiplier is the inverse weight global scale.
        device = weight_scale.device
        layer.register_buffer("_nvfp4_alpha", torch.tensor(1.0 / global_scale, dtype=torch.float32, device=device),
                              persistent=False)
        layer.register_buffer("_nvfp4_x_global_scale", torch.ones((), dtype=torch.float32, device=device),
                              persistent=False)

    @staticmethod
    def _apply_finalized(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        x = _coerce_fp4_input_dtype(x)
        original_shape = x.shape
        if x.numel() == 0:
            # An empty prompt has nothing to quantize; the FP4 kernels are not defined for zero rows.
            return x.new_zeros(*original_shape[:-1], layer.output_size_per_partition, dtype=torch.bfloat16)
        x_fp4, x_scale = _quantize_activation_nvfp4(x.reshape(-1, original_shape[-1]), layer._nvfp4_x_global_scale)
        output = _nvfp4_linear(x_fp4, x_scale, layer.weight_packed, layer.weight_scale, layer._nvfp4_alpha)
        if bias is not None:
            # The GEMM emits bf16; keep it that way whatever dtype the bias was built in.
            output = output + bias.to(output.dtype)
        return output.view(*original_shape[:-1], output.shape[-1])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("MiniMax-H3 serialized NVFP4 execution requires CUDA")
        if getattr(layer, "_nvfp4_alpha", None) is None:
            raise RuntimeError("MiniMax-H3 serialized NVFP4 linear was not finalized: "
                               "process_weights_after_loading has not run")
        return self._apply_finalized(layer, x, bias)


__all__ = [
    "LANGUAGE_PROJECTIONS",
    "NVFP4_TENSOR_SUFFIXES",
    "MiniMaxH3SerializedNVFP4Config",
    "MiniMaxH3SerializedNVFP4LinearMethod",
    "NVFP4_GROUP_SIZE",
    "NVFP4_SCALE_LAYOUT",
    "nvfp4_packed_weight_shape",
    "nvfp4_scale_shape",
    "nvfp4_weight_global_scale",
    "serialized_nvfp4_quantization_config",
    "validate_nvfp4_geometry",
]
