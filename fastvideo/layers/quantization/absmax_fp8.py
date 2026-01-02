from typing import Any
import torch
from fastvideo.distributed.parallel_state import get_tp_world_size
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
        _share_id: str | None = None,
    ) -> None:
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
        share_id: str | int | None = None,
    ) -> None:
        output_partition_sizes: list[int] = self.output_partition_sizes
        if share_id is None:
            share_id = 0
        if isinstance(share_id, str) and share_id in ["q", "k", "v"]:
            # QKVParallelLinear case
            share_idx = ["q", "k", "v"].index(share_id)
            start_idx = sum(output_partition_sizes[:share_idx])
            end_idx = start_idx + output_partition_sizes[share_idx]
        elif isinstance(share_id, int):
            # MergedColumnParallelLinear case
            tp_size = get_tp_world_size()
            if tp_size > 1:
                # TODO: support this case
                raise NotImplementedError(
                    "AbsMaxFP8MergedParameter with integer share_id is not supported in tensor parallelism greater than 1 yet."
                )
            start_idx = sum(output_partition_sizes[:share_id])
            end_idx = start_idx + output_partition_sizes[share_id]
        else:
            raise ValueError(
                f"AbsMaxFP8MergedParameter requires share_id to be ['q', 'k', 'v'] or int, got {share_id}."
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
        if isinstance(layer, QKVParallelLinear | MergedColumnParallelLinear):
            scale_weight = self._merged_placeholder(output_partition_sizes, )
        else:
            scale_weight = self._convert_scale(
                extra_weight_attrs.get("scale_weight"))
        scale_input = self._convert_scale(extra_weight_attrs.get("scale_input"))

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
        x_final = x.to(dtype=output_dtype) * scale_input
        # if (
        #     torch.isnan(x_final).any()
        #     or torch.isinf(x_final).any()
        #     or torch.isnan(weight_final).any()
        #     or torch.isinf(weight_final).any()
        # ):
        #     raise ValueError(
        #         "NaN or Inf detected in AbsMaxFP8LinearMethod inputs."
        #     )
        res = nn.functional.linear(x_final, weight_final,
                                   bias=bias).to(dtype=output_dtype)
        # res_nan = torch.isnan(res)
        # res_inf = torch.isinf(res)
        # if res_nan.any():
        #     raise ValueError("NaN detected in AbsMaxFP8LinearMethod output.")
        # if res_inf.any():
        #     raise ValueError("Inf detected in AbsMaxFP8LinearMethod output.")
        return res
