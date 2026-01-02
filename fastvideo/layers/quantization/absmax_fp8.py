from typing import Any
import torch
from fastvideo.layers.linear import (
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from fastvideo.layers.quantization import QuantizationMethods
from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.models.utils import set_weight_attrs
import torch.nn as nn


class AbsMaxFP8Config(QuantizationConfig):
    """
    Config class for absmax float8_e4m3fn quantization.
    Currently only support per-tensor quantization.
    """

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        return cls()

    def get_name(self) -> QuantizationMethods:
        return "AbsMaxFP8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return AbsMaxFP8LinearMethod()
        return None


class AbsMaxFP8Parameter(nn.Parameter):

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        share_id: str | None = None,
    ) -> None:
        assert share_id is None, (
            "AbsMaxFP8Parameter does not support share_id in weight_loader.")
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}")
        param.data.copy_(loaded_weight)


class AbsMaxFP8MergedParameter(nn.Parameter):

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        share_id: str | None = None,
    ) -> None:
        output_partition_sizes: list[int] = self.output_partition_sizes
        match share_id:
            case "q":
                start_idx = 0
                end_idx = output_partition_sizes[0]
            case "k":
                start_idx = output_partition_sizes[0]
                end_idx = start_idx + output_partition_sizes[1]
            case "v":
                start_idx = (output_partition_sizes[0] +
                             output_partition_sizes[1])
                end_idx = start_idx + output_partition_sizes[2]
            case _:
                raise ValueError(
                    f"AbsMaxFP8Parameter only supports share_id in ['q', 'k', 'v'] for QKVParallelLinear now, got {share_id}."
                )
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert loaded_weight.numel() == 1
        # fill in the corresponding partition by repeating the val
        param.data[start_idx:end_idx].fill_(loaded_weight.item())


class AbsMaxFP8LinearMethod(LinearMethodBase):
    """Linear method with AbsMax FP8 quantization."""

    @staticmethod
    def _convert_scale(scale: Any) -> torch.nn.Parameter:
        if scale is None:
            scale = torch.tensor([1.0], dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([scale], dtype=torch.float32)
        if scale.dtype != torch.float32:
            raise NotImplementedError("Only float32 scale is supported")
        return AbsMaxFP8Parameter(scale, requires_grad=False)

    @staticmethod
    def _merged_placeholder(
        output_partition_sizes: list[int], ) -> torch.nn.Parameter:
        scale = torch.ones(
            sum(output_partition_sizes),
            dtype=torch.float32,
        )
        para = AbsMaxFP8MergedParameter(
            scale,
            False,
        )
        set_weight_attrs(
            para,
            {
                "output_partition_sizes": output_partition_sizes,
            },
        )
        return para

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
        assert params_dtype in [
            torch.bfloat16, torch.float16
        ], (f"AbsMaxFP8LinearMethod only supports bfloat16 or float16 original dtype, got {params_dtype}."
            )
        weight = nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        if isinstance(layer, QKVParallelLinear):
            scale_weight = self._merged_placeholder(output_partition_sizes, )
            scale_input = self._merged_placeholder(output_partition_sizes, )
        elif isinstance(layer, MergedColumnParallelLinear):
            raise NotImplementedError(
                "AbsMaxFP8LinearMethod does not support MergedColumnParallelLinear yet."
            )
        else:
            scale_weight = self._convert_scale(
                extra_weight_attrs.get("scale_weight"))
            scale_input = self._convert_scale(
                extra_weight_attrs.get("scale_input"))

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        layer.register_parameter("scale_weight", scale_weight)
        layer.register_parameter("scale_input", scale_input)
        set_weight_attrs(
            weight,
            {
                "output_dtype": params_dtype,
            },
        )
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_quant = layer.weight
        output_dtype: torch.dtype = weight_quant.output_dtype
        scale_weight: torch.Tensor = layer.scale_weight.data.to(output_dtype)
        scale_input: torch.Tensor = layer.scale_input.data.to(output_dtype)
        weight_output_type = weight_quant.to(dtype=output_dtype)
        weight_final = weight_output_type * scale_weight.unsqueeze(1)
        return (nn.functional.linear(
            x, weight_final, bias=bias).to(dtype=output_dtype) * scale_input)
