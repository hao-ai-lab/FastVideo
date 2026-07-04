from typing import Literal, get_args

from v2._vendor.layers.quantization.base_config import QuantizationConfig

QuantizationMethods = Literal[None, "AbsMaxFP8", "FP8", "NVFP4", "nvfp4_qat"]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from v2._vendor.layers.quantization import register_quantization_config
        >>> from v2._vendor.layers.quantization import get_quantization_config
        >>> from v2._vendor.layers.quantization.base_config import QuantizationConfig
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(f"The quantization method `{quantization}` is already exists.")
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError("The quantization config must be a subclass of "
                             "`QuantizationConfig`.")
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]
    if quantization == "AbsMaxFP8":
        from .absmax_fp8 import AbsMaxFP8Config
        return AbsMaxFP8Config
    if quantization == "FP8":
        from .fp8_config import FP8Config
        return FP8Config
    if quantization == "NVFP4":
        from .nvfp4_config import NVFP4Config
        return NVFP4Config
    if quantization == "nvfp4_qat":
        from .nvfp4_qat_config import NVFP4QATConfig
        return NVFP4QATConfig
    raise ValueError(f"Invalid quantization method: {quantization}")


all = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
