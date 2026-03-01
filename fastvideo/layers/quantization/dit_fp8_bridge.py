"""Inject FP8 parameters + forward hooks into plain nn.Linear modules.

Required because DiT models use nn.Linear (not LinearBase), so the standard
get_quant_method() path does not apply.
"""

from __future__ import annotations

import json
import struct

import torch
import torch.nn as nn

from fastvideo.logger import init_logger
from fastvideo.models.utils import set_weight_attrs

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


def prepare_model_for_fp8(
    model: nn.Module,
    fp8_module_prefixes: set[str],
    output_dtype: torch.dtype,
) -> None:
    """Must be called after model construction on meta device, before weight loading."""
    from fastvideo.layers.quantization.absmax_fp8 import AbsMaxFP8LinearMethod

    _method = AbsMaxFP8LinearMethod()
    injected = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        stripped = name.removeprefix("model.")
        if stripped not in fp8_module_prefixes and name not in fp8_module_prefixes:
            continue

        out_features, in_features = module.weight.shape
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
        module.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty((), dtype=torch.float32, device=device),
                         requires_grad=False),
        )

        def _make_fp8_forward(mod: nn.Linear, method: AbsMaxFP8LinearMethod):

            def _forward(x: torch.Tensor) -> torch.Tensor:
                return method.apply(mod, x, mod.bias)

            return _forward

        module.forward = _make_fp8_forward(module,
                                           _method)  # type: ignore[assignment]
        injected += 1

    logger.info(
        "Prepared %d nn.Linear modules for FP8 compute (out of %d FP8 prefixes in checkpoint)",
        injected,
        len(fp8_module_prefixes),
    )
