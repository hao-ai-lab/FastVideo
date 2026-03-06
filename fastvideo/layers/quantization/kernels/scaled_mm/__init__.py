# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py
import torch

from fastvideo.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from fastvideo.layers.quantization.kernels.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from fastvideo.layers.quantization.utils.quant_utils import QuantKey
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Priority order: first match wins
_POSSIBLE_FP8_KERNELS: list[type[FP8ScaledMMLinearKernel]] = [
    PerTensorTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
]


def choose_fp8_kernel(
    config: FP8ScaledMMLinearLayerConfig,
    compute_capability: int | None = None,
) -> type[FP8ScaledMMLinearKernel]:
    if compute_capability is None and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        compute_capability = cap[0] * 10 + cap[1]

    failure_reasons: list[str] = []
    for kernel_cls in _POSSIBLE_FP8_KERNELS:
        is_supported, reason = kernel_cls.is_supported(compute_capability)
        if not is_supported:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        can_implement, reason = kernel_cls.can_implement(config)
        if not can_implement:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        return kernel_cls

    raise ValueError(
        "No FP8 scaled_mm kernel can implement the given config.\n" +
        "\n".join(failure_reasons))


def init_fp8_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel:
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        out_dtype=out_dtype,
    )

    kernel_type = choose_fp8_kernel(config)

    if module_name:
        logger.info(
            "Selected %s for %s",
            kernel_type.__name__,
            module_name,
        )

    return kernel_type(
        config,
        layer_param_names=[
            "weight", "weight_scale", "input_scale", "input_scale_ub"
        ],
    )


__all__ = [
    "FP8ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "choose_fp8_kernel",
    "init_fp8_kernel",
]
