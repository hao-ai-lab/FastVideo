import torch

from fastvideo.layers.quantization.utils.quant_utils import FP8_DTYPE

FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

_USE_FP8_COMPUTE: bool | None = None


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


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
