import json
import struct
from typing import Literal

import torch

from fastvideo.layers.quantization.utils.quant_utils import FP8_DTYPE
from fastvideo.logger import init_logger

logger = init_logger(__name__)

FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

_USE_FP8_COMPUTE: bool | None = None

_SAFETENSORS_FP8_DTYPE_STRINGS = {"F8_E4M3", "F8_E5M2"}


def supports_fp8_compute() -> bool:
    global _USE_FP8_COMPUTE
    if _USE_FP8_COMPUTE is None:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _USE_FP8_COMPUTE = cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)
        else:
            _USE_FP8_COMPUTE = False
    return _USE_FP8_COMPUTE


def quantize_input_dynamic(
    x: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x.abs().amax().float()
    scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


def quantize_input_static(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def is_fp8_dtype(dtype: torch.dtype | str) -> bool:
    if isinstance(dtype, str):
        return dtype in _SAFETENSORS_FP8_DTYPE_STRINGS
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def detect_fp8_from_safetensors(
    safetensors_paths: list[str],
) -> tuple[bool, Literal["static", "dynamic"]]:
    """Detect FP8 weights from safetensors headers. Returns (is_fp8, activation_scheme)."""
    has_fp8_weights = False
    has_input_scale = False

    for path in safetensors_paths:
        try:
            with open(path, "rb") as f:
                (header_size, ) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_size).decode("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read safetensors header from %s: %s",
                           path, e)
            continue

        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                continue
            dtype_str = info.get("dtype", "")

            if tensor_name.endswith(".weight") and is_fp8_dtype(dtype_str):
                has_fp8_weights = True

            if tensor_name.endswith(".input_scale"):
                has_input_scale = True

            if has_fp8_weights and has_input_scale:
                break

        if has_fp8_weights and has_input_scale:
            break

    activation_scheme: Literal["static",
                               "dynamic"] = ("static"
                                             if has_input_scale else "dynamic")

    if has_fp8_weights:
        logger.info(
            "Auto-detected FP8 checkpoint (activation_scheme=%s) from %d "
            "safetensors file(s)",
            activation_scheme,
            len(safetensors_paths),
        )

    return has_fp8_weights, activation_scheme
