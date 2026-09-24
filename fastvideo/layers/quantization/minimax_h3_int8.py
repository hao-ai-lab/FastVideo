# SPDX-License-Identifier: Apache-2.0
"""Serialized per-row int8 execution for the MiniMax-H3 transformer.

The rank-16 FastH3 transformer is 44 GB in bf16, and every quantized path on
``main`` either covers part of the blocks or keeps the bf16 copy alive while it
runs. A checkpoint written by
``scripts/checkpoint_conversion/convert_minimax_h3_transformer_int8.py`` stores
the seven large linears of every main transformer block as two tensors::

    <prefix>.weight        int8     [out, in]   round(W / scale) clamped to [-127, 127]
    <prefix>.weight_scale  float32  [out]       amax(|W[row]|) / 127

The token refiner, the AdaLN factors, the patch and context projections and
the output head stay in their released dtypes. Activations are quantized per
row at call time with the same symmetric rule and multiplied with
``torch._int_mm``, so the path needs an int8 tensor core and nothing else: no
FlashInfer, no Blackwell. The int32 product is rescaled by both scales and
returned in the activation dtype; rows are processed in chunks so the int32
intermediate never exceeds a few hundred megabytes.

The checkpoint selects this path through ``transformer/config.json``; every
field is required so a checkpoint from another exporter cannot pass by
omission::

    "quantization_config": {"quant_method": "int8", "activation_scheme": "dynamic",
                            "weight_granularity": "row", "activation_granularity": "row",
                            "linears": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
                                        "attn.to_gate_compress", "ff.fc_in", "ff.fc_out"]}

Single GPU only: FSDP inference rejects quantized transformers before the
weights load. The layerwise offload hook pins and streams ``param.data``
regardless of dtype, so it works on these parameters unchanged.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from fastvideo.layers.linear import LinearBase, LinearMethodBase
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.models.utils import set_weight_attrs

INT8_QUANT_METHOD = "int8"
INT8_TENSOR_SUFFIXES = ("weight", "weight_scale")
# The seven linears of every main transformer block, in FastVideo-native names.
# ``attn.to_gate_compress`` exists only in VSA-trained checkpoints; a checkpoint
# without it is loaded by a dense attention backend that never builds the layer.
TRANSFORMER_LINEARS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "attn.to_gate_compress",
    "ff.fc_in",
    "ff.fc_out",
)
INT8_ROW_CHUNK = 4096
# ``torch._int_mm`` on CUDA refuses fewer than 17 rows; pad tiny batches up to this.
_INT8_MIN_ROWS = 32
_REQUIRED_METADATA_KEYS = ("quant_method", "activation_scheme", "weight_granularity", "activation_granularity",
                           "linears")
_BLOCK_LINEAR = re.compile(r"(?:^|\.)transformer_blocks\.\d+\.(?P<name>" +
                           "|".join(re.escape(name) for name in TRANSFORMER_LINEARS) + r")$")


def is_minimax_h3_int8_linear_prefix(prefix: str) -> bool:
    """Return whether *prefix* names one of the seven quantized linears of a main block."""
    return _BLOCK_LINEAR.search(prefix) is not None


def validate_int8_geometry(output_size: int, input_size: int) -> None:
    """``torch._int_mm`` wants both GEMM dimensions to be multiples of 8."""
    if output_size % 8 or input_size % 8:
        raise ValueError(
            f"MiniMax-H3 serialized int8 needs out and in sizes divisible by 8, got {output_size}x{input_size}")


def serialized_int8_quantization_config(producer: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``quantization_config`` block the converter writes and ``from_config`` requires."""
    config: dict[str, Any] = {
        "quant_method": INT8_QUANT_METHOD,
        "activation_scheme": "dynamic",
        "weight_granularity": "row",
        "activation_granularity": "row",
        "linears": list(TRANSFORMER_LINEARS),
    }
    if producer:
        config["producer"] = dict(producer)
    return config


def quantize_rows_int8(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8: returns ``(q int8 [rows, cols], scale float32 [rows])``.

    Used for weights in the converter and for activations at call time, so
    both sides of the GEMM follow one rule. Computed in float32.
    """
    if matrix.dim() != 2:
        raise ValueError(f"quantize_rows_int8 expects a 2-D tensor, got shape {tuple(matrix.shape)}")
    values = matrix.to(torch.float32)
    scale = values.abs().amax(dim=1).div_(127.0).clamp_min_(torch.finfo(torch.float32).tiny)
    quantized = torch.round(values / scale.unsqueeze(1)).clamp_(-127.0, 127.0).to(torch.int8)
    return quantized, scale


def _int8_matmul(x_int8: torch.Tensor, weight_int8: torch.Tensor) -> torch.Tensor:
    """``x_int8 [m, in] @ weight_int8 [out, in].T`` accumulated in int32."""
    if x_int8.device.type == "cuda":
        rows = x_int8.shape[0]
        if rows < _INT8_MIN_ROWS:
            padded = torch.zeros((_INT8_MIN_ROWS, x_int8.shape[1]), dtype=torch.int8, device=x_int8.device)
            padded[:rows] = x_int8
            return torch._int_mm(padded, weight_int8.t())[:rows]
        return torch._int_mm(x_int8, weight_int8.t())
    # CPU has no int8 tensor-core kernel; an exact int32 product keeps tests and
    # the converter's report honest about what the CUDA path computes.
    return torch.mm(x_int8.to(torch.int32), weight_int8.to(torch.int32).t())


def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    row_chunk: int = INT8_ROW_CHUNK,
) -> torch.Tensor:
    """``x [m, in] @ W.T`` with per-row dynamic int8 activations against per-row int8 weights.

    Rows are quantized and multiplied in chunks of ``row_chunk`` so the int32 and
    float32 intermediates stay bounded whatever the sequence length.
    """
    if x.dim() != 2:
        raise ValueError(f"int8_linear expects a 2-D activation, got shape {tuple(x.shape)}")
    if weight_int8.dtype != torch.int8:
        raise ValueError(f"int8_linear expects an int8 weight, got {weight_int8.dtype}")
    if x.shape[1] != weight_int8.shape[1]:
        raise ValueError(f"int8_linear got activation width {x.shape[1]} for a weight of width {weight_int8.shape[1]}")
    output = torch.empty((x.shape[0], weight_int8.shape[0]), dtype=out_dtype, device=x.device)
    scale_row = weight_scale.to(torch.float32).unsqueeze(0)
    for start in range(0, x.shape[0], row_chunk):
        chunk = x[start:start + row_chunk]
        x_int8, x_scale = quantize_rows_int8(chunk)
        accumulated = _int8_matmul(x_int8, weight_int8)
        output[start:start + row_chunk] = (accumulated.to(torch.float32) * x_scale.unsqueeze(1) *
                                           scale_row).to(out_dtype)
    return output


def transformer_quantization_config_from_metadata(metadata: Any) -> QuantizationConfig | None:
    """Pick the serialized transformer scheme a ``config.json`` block names, or ``None`` when absent."""
    if not metadata:
        return None
    if not isinstance(metadata, dict):
        raise ValueError(f"transformer quantization_config must be a mapping, got {type(metadata).__name__}")
    quant_method = str(metadata.get("quant_method", "")).lower()
    if quant_method == INT8_QUANT_METHOD:
        return MiniMaxH3SerializedInt8Config.from_config(metadata)
    raise ValueError(f"Unsupported serialized transformer quantization {quant_method!r}; "
                     f"this checkpoint format supports {INT8_QUANT_METHOD!r}")


class MiniMaxH3SerializedInt8Config(QuantizationConfig):
    """Serialized per-row int8 contract for the H3 transformer blocks."""

    @classmethod
    def get_name(cls) -> str:
        return INT8_QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # int8 tensor cores: Turing and newer.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MiniMaxH3SerializedInt8Config:
        quant_method = str(config.get("quant_method", "")).lower()
        if quant_method != INT8_QUANT_METHOD:
            raise ValueError(f"MiniMax-H3 serialized int8 config got quant_method {quant_method!r}")
        missing = [key for key in _REQUIRED_METADATA_KEYS if key not in config]
        if missing:
            raise ValueError("MiniMax-H3 serialized int8 requires explicit quantization_config fields; "
                             f"missing {missing}")
        expected = {
            "activation_scheme": "dynamic",
            "weight_granularity": "row",
            "activation_granularity": "row",
        }
        for key, value in expected.items():
            if str(config[key]).lower() != value:
                raise ValueError(f"MiniMax-H3 serialized int8 expects {key}={value!r}, got {config[key]!r}")
        linears = config["linears"]
        if not isinstance(linears, list) or tuple(linears) != TRANSFORMER_LINEARS:
            raise ValueError("MiniMax-H3 serialized int8 quantizes exactly the seven block linears "
                             f"{list(TRANSFORMER_LINEARS)}; the checkpoint declares {linears!r}")
        return cls()

    def get_quant_method(self, layer: nn.Module, prefix: str) -> LinearMethodBase | None:
        if isinstance(layer, LinearBase) and is_minimax_h3_int8_linear_prefix(prefix):
            return MiniMaxH3SerializedInt8LinearMethod()
        return None

    def validate_runtime(self, device: torch.device) -> None:
        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            if major * 10 + minor < self.get_min_capability():
                raise RuntimeError(f"MiniMax-H3 serialized int8 needs an int8 tensor core, compute capability "
                                   f"{self.get_min_capability() / 10:.1f}+; got {major}.{minor}")


def _strict_dtype_loader(base_loader):
    """Wrap a layer's weight loader so a checkpoint tensor must carry the parameter's exact dtype."""
    if base_loader is None:
        raise ValueError("MiniMax-H3 serialized int8 linears need the layer's weight_loader")

    def load(param: torch.Tensor, loaded_weight: torch.Tensor, *args: Any, **kwargs: Any):
        if loaded_weight.dtype != param.dtype:
            raise ValueError("Serialized MiniMax-H3 int8 tensors must be stored as their declared dtype; "
                             f"got {loaded_weight.dtype} for a {param.dtype} parameter of shape {tuple(param.shape)}")
        return base_loader(param, loaded_weight, *args, **kwargs)

    return load


class MiniMaxH3SerializedInt8LinearMethod(LinearMethodBase):
    """Execute serialized int8 weights without a bf16 copy ever existing."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        validate_int8_geometry(output_size_per_partition, input_size_per_partition)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer._int8_finalized = False

        # No input_dim/output_dim: nothing here is shardable. The loader assigns
        # checkpoint tensors straight into these parameters, so the shapes below
        # and the loader's dtype check are the whole contract.
        loader_attrs = {"weight_loader": _strict_dtype_loader(extra_weight_attrs.get("weight_loader"))}
        weight = Parameter(torch.zeros((output_size_per_partition, input_size_per_partition), dtype=torch.int8),
                           requires_grad=False)
        set_weight_attrs(weight, loader_attrs)
        layer.register_parameter("weight", weight)
        weight_scale = Parameter(torch.zeros(output_size_per_partition, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(weight_scale, loader_attrs)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject a layer the checkpoint did not fill.

        Missing tensors are reported by name by the loader before this runs;
        what remains is content a copied tensor can still get wrong.
        """
        weight = layer.weight
        weight_scale = layer.weight_scale
        if weight.dtype != torch.int8:
            raise ValueError(f"{layer.prefix}: serialized int8 weight must be int8, got {weight.dtype}")
        if weight_scale.dtype != torch.float32:
            raise ValueError(f"{layer.prefix}: serialized int8 weight_scale must be float32, got {weight_scale.dtype}")
        if tuple(weight_scale.shape) != (weight.shape[0], ):
            raise ValueError(f"{layer.prefix}: weight_scale shape {tuple(weight_scale.shape)} does not match "
                             f"{weight.shape[0]} output rows")
        finite = torch.isfinite(weight_scale).all()
        positive = (weight_scale > 0).all()
        if not bool(finite) or not bool(positive):
            raise ValueError(f"{layer.prefix}: serialized int8 weight_scale must be finite and positive; "
                             "the checkpoint did not fill this layer")
        layer._int8_finalized = True

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not getattr(layer, "_int8_finalized", False):
            raise RuntimeError(f"{layer.prefix}: serialized int8 weights were not finalized after loading")
        out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        if x_2d.shape[0] == 0:
            return x.new_empty((*original_shape[:-1], layer.weight.shape[0]), dtype=out_dtype)
        output = int8_linear(x_2d, layer.weight, layer.weight_scale, out_dtype=out_dtype)
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output.view(*original_shape[:-1], layer.weight.shape[0])


def has_serialized_int8_linears(model: nn.Module) -> bool:
    """Whether any linear in *model* executes serialized int8 weights."""
    return any(
        isinstance(getattr(module, "quant_method", None), MiniMaxH3SerializedInt8LinearMethod)
        for module in model.modules())


def reject_lora_on_serialized_int8(model: nn.Module, lora_path: str) -> None:
    """Refuse an adapter against serialized int8 weights.

    Both merge paths assume a float base weight. The whole-parameter path adds
    the delta to the int8 codes without dequantizing by ``weight_scale``, which
    silently produces a wrong weight; the low-rank path casts the adapter to the
    base dtype and then multiplies in place, which raises on an integer tensor.
    Merging correctly means dequantize, merge, requantize, and that changes the
    scales the checkpoint shipped, so it is refused rather than approximated.
    """
    if not has_serialized_int8_linears(model):
        return
    raise NotImplementedError(
        f"A LoRA adapter ({lora_path}) cannot be applied to a serialized int8 MiniMax-H3 transformer: merging "
        "into int8 codes would need a dequantize, merge and requantize round trip that rewrites the checkpoint's "
        "row scales. Merge the adapter into the bf16 checkpoint first, then convert that with "
        "scripts/checkpoint_conversion/convert_minimax_h3_transformer_int8.py.")


def finalize_serialized_int8_model(model: nn.Module) -> int:
    """Run the post-load check on every serialized int8 linear; return how many were validated."""
    validated = 0
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, MiniMaxH3SerializedInt8LinearMethod):
            quant_method.process_weights_after_loading(module)
            validated += 1
    return validated


__all__ = [
    "INT8_QUANT_METHOD",
    "INT8_TENSOR_SUFFIXES",
    "TRANSFORMER_LINEARS",
    "MiniMaxH3SerializedInt8Config",
    "MiniMaxH3SerializedInt8LinearMethod",
    "finalize_serialized_int8_model",
    "has_serialized_int8_linears",
    "int8_linear",
    "is_minimax_h3_int8_linear_prefix",
    "reject_lora_on_serialized_int8",
    "quantize_rows_int8",
    "serialized_int8_quantization_config",
    "transformer_quantization_config_from_metadata",
    "validate_int8_geometry",
]
