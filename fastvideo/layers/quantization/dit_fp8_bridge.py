"""Inject FP8 parameters + forward hooks into plain nn.Linear modules.

Required because DiT models use nn.Linear (not LinearBase), so the standard
get_quant_method() path does not apply.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fastvideo.layers.quantization.utils.quant_utils import is_layer_skipped
from fastvideo.logger import init_logger
from fastvideo.models.utils import set_weight_attrs

if TYPE_CHECKING:
    from fastvideo.layers.linear import LinearMethodBase
    from fastvideo.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn

_SAFETENSORS_DTYPE_MAP: dict[str, torch.dtype] = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def scan_fp8_modules(weight_paths: list[str]) -> set[str]:
    """Return module prefixes whose weight is FP8, by reading safetensors headers only."""
    fp8_prefixes: set[str] = set()
    for path in weight_paths:
        with open(path, "rb") as f:
            (header_size, ) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_size).decode("utf-8"))
        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                continue
            dtype_str = info.get("dtype", "")
            if tensor_name.endswith(
                    ".weight") and dtype_str in _SAFETENSORS_DTYPE_MAP:
                prefix = tensor_name[:-len(".weight")]
                fp8_prefixes.add(prefix)
    return fp8_prefixes


def _make_fp8_forward(mod: nn.Linear, method: LinearMethodBase):

    def _forward(x: torch.Tensor) -> torch.Tensor:
        return method.apply(mod, x, mod.bias)

    return _forward


def prepare_model_for_fp8(
    model: nn.Module,
    fp8_module_prefixes: set[str],
    output_dtype: torch.dtype,
    quant_config: QuantizationConfig | None = None,
) -> None:
    """Replace weight placeholders with FP8 params and register scales for offline FP8."""
    from fastvideo.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    if quant_config is None or not isinstance(quant_config, Fp8Config):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )

    _method = Fp8LinearMethod(quant_config)
    use_static_input_scale = quant_config.activation_scheme == "static"
    ignored_layers = quant_config.ignored_layers
    min_layer_size = quant_config.min_layer_size
    injected = 0
    skipped_ignored = 0
    skipped_small = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        stripped = name.removeprefix("model.")
        if stripped not in fp8_module_prefixes and name not in fp8_module_prefixes:
            continue

        if ignored_layers and is_layer_skipped(name, ignored_layers):
            skipped_ignored += 1
            logger.debug("Skipping FP8 for %s (in ignored_layers)", name)
            continue

        out_features, in_features = module.weight.shape
        layer_size = in_features * out_features
        if min_layer_size > 0 and layer_size < min_layer_size:
            skipped_small += 1
            logger.debug(
                "Skipping FP8 for %s (size %d < min %d)",
                name,
                layer_size,
                min_layer_size,
            )
            continue

        device = module.weight.device

        fp8_weight = nn.Parameter(
            torch.empty(out_features,
                        in_features,
                        dtype=_FP8_DTYPE,
                        device=device),
            requires_grad=False,
        )
        set_weight_attrs(fp8_weight, {
            "input_dim": 1,
            "output_dim": 0,
            "output_dtype": output_dtype,
        })
        module.weight = fp8_weight
        module._fp8_output_dtype = output_dtype

        module.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty((), dtype=torch.float32, device=device),
                         requires_grad=False),
        )
        if use_static_input_scale:
            module.register_parameter(
                "input_scale",
                nn.Parameter(torch.empty((), dtype=torch.float32,
                                         device=device),
                             requires_grad=False),
            )

        module.forward = _make_fp8_forward(module,
                                           _method)  # type: ignore[assignment]
        injected += 1

    logger.info(
        "Prepared %d nn.Linear modules for FP8 compute "
        "(out of %d FP8 prefixes in checkpoint, "
        "skipped %d by ignored_layers, %d by min_layer_size)",
        injected,
        len(fp8_module_prefixes),
        skipped_ignored,
        skipped_small,
    )


def prepare_model_for_online_fp8(
    model: nn.Module,
    output_dtype: torch.dtype,
    quant_config: QuantizationConfig | None = None,
) -> None:
    """Attach FP8 forward hooks; weights stay bf16 and are quantized lazily on first forward."""
    from fastvideo.layers.quantization.fp8 import (
        Fp8Config,
        Fp8OnlineLinearMethod,
    )

    if quant_config is None or not isinstance(quant_config, Fp8Config):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
        )

    _method = Fp8OnlineLinearMethod(quant_config)
    ignored_layers = quant_config.ignored_layers
    min_layer_size = quant_config.min_layer_size
    injected = 0
    skipped_ignored = 0
    skipped_small = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if ignored_layers and is_layer_skipped(name, ignored_layers):
            skipped_ignored += 1
            logger.debug("Skipping online FP8 for %s (in ignored_layers)", name)
            continue

        out_features, in_features = module.weight.shape
        layer_size = in_features * out_features
        if min_layer_size > 0 and layer_size < min_layer_size:
            skipped_small += 1
            logger.debug(
                "Skipping online FP8 for %s (size %d < min %d)",
                name,
                layer_size,
                min_layer_size,
            )
            continue

        module.orig_dtype = output_dtype
        module._fp8_output_dtype = output_dtype
        module.forward = _make_fp8_forward(module,
                                           _method)  # type: ignore[assignment]
        injected += 1

    logger.info(
        "Prepared %d nn.Linear modules for online FP8 quantization "
        "(bf16 weights will be quantized on first forward, "
        "skipped %d by ignored_layers, %d by min_layer_size)",
        injected,
        skipped_ignored,
        skipped_small,
    )
