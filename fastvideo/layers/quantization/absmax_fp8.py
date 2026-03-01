from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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
from fastvideo.layers.quantization.fp8_utils import (
    FP8_MAX,
    quantize_input_dynamic,
    quantize_input_static,
    supports_fp8_compute,
)
from fastvideo.layers.quantization.utils.quant_utils import FP8_DTYPE
from fastvideo.logger import init_logger
from fastvideo.models.utils import set_weight_attrs

# Re-export bridge functions so existing imports continue to work.
from fastvideo.layers.quantization.dit_fp8_bridge import (  # noqa: F401
    prepare_model_for_fp8, scan_fp8_modules,
)

logger = init_logger(__name__)

# Re-export private names used by test_absmax_fp8.py
_FP8_DTYPE = FP8_DTYPE
_FP8_MAX = FP8_MAX
_supports_fp8_compute = supports_fp8_compute
_quantize_input_dynamic = quantize_input_dynamic
_quantize_input_static = quantize_input_static


class AbsMaxFP8Config(QuantizationConfig):
    """Config class for absmax float8_e4m3fn quantization."""

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig:
        quant_method = config.get("quant_method")
        if quant_method is not None and quant_method != "AbsMaxFP8":
            raise ValueError(
                "AbsMaxFP8Config received incompatible quant_method="
                f"{quant_method}.")
        return cls()

    def get_name(self) -> QuantizationMethods:
        return "AbsMaxFP8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

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
            share_idx = ["q", "k", "v"].index(share_id)
            start_idx = sum(output_partition_sizes[:share_idx])
            end_idx = start_idx + output_partition_sizes[share_idx]
        elif isinstance(share_id, int):
            tp_size = get_tp_world_size()
            if tp_size > 1:
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
            torch.bfloat16, torch.float16, torch.float32
        ], (f"AbsMaxFP8LinearMethod only supports bfloat16, float16, or float32 original dtype, got {params_dtype}."
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
            weight_scale = self._merged_placeholder(output_partition_sizes, )
        else:
            weight_scale = self._convert_scale(
                extra_weight_attrs.get("weight_scale"))
        input_scale = self._convert_scale(extra_weight_attrs.get("input_scale"))

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("input_scale", input_scale)
        set_weight_attrs(
            weight,
            {
                "output_dtype": params_dtype,
            },
        )
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Prepare weight for torch._scaled_mm.

        cuBLASLt requires A=row-major, B=column-major.  A contiguous
        [out, in] tensor viewed via .t() is [in, out] in column-major
        layout -- exactly what cuBLASLt needs.  We store the .t() *view*
        (no copy) so _apply_fp8 can pass it directly.
        """
        if not supports_fp8_compute():
            return
        w_col_major = layer.weight.data.t()
        output_dtype = getattr(
            layer.weight,
            "output_dtype",
            getattr(layer, "_fp8_output_dtype", None),
        )
        if output_dtype is None:
            raise AttributeError(
                f"Cannot determine output_dtype for {layer}. "
                "Ensure prepare_model_for_fp8 or create_weights was called.")
        layer.weight = nn.Parameter(w_col_major, requires_grad=False)
        set_weight_attrs(layer.weight, {"output_dtype": output_dtype})
        layer._fp8_weight_transposed = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not supports_fp8_compute():
            return self._apply_dequant(layer, x, bias)

        if not getattr(layer, "_fp8_weight_transposed", False):
            self.process_weights_after_loading(layer)

        return self._apply_fp8(layer, x, bias)

    _fp8_logged: bool = False

    def _apply_fp8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not AbsMaxFP8LinearMethod._fp8_logged:
            AbsMaxFP8LinearMethod._fp8_logged = True
            logger.info(
                "FP8 compute active: using torch._scaled_mm on %s "
                "(input dtype=%s, weight dtype=%s)",
                type(layer).__name__,
                x.dtype,
                layer.weight.dtype,
            )
        w_fp8 = layer.weight
        output_dtype: torch.dtype = getattr(
            w_fp8,
            "output_dtype",
            getattr(layer, "_fp8_output_dtype", torch.bfloat16),
        )
        w_scale = layer.weight_scale.data
        x_scale_static = layer.input_scale.data
        out_features = w_fp8.shape[1]

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        if x_scale_static.numel() == 1 and x_scale_static.item() != 1.0:
            x_fp8 = quantize_input_static(x_2d, x_scale_static)
            a_scale = x_scale_static.float()
        else:
            x_fp8, a_scale = quantize_input_dynamic(x_2d)

        if w_scale.numel() == 1:
            out = torch._scaled_mm(
                x_fp8,
                w_fp8,
                scale_a=a_scale,
                scale_b=w_scale.float(),
                out_dtype=output_dtype,
            )
            if isinstance(out, tuple):
                out = out[0]
            if bias is not None:
                out = out + bias
        else:
            ones = torch.ones(1, dtype=torch.float32, device=x_fp8.device)
            out = torch._scaled_mm(
                x_fp8,
                w_fp8,
                scale_a=ones,
                scale_b=ones,
                out_dtype=torch.float32,
            )
            if isinstance(out, tuple):
                out = out[0]
            out = out * a_scale * w_scale.float().unsqueeze(0)
            if bias is not None:
                out = out + bias
            out = out.to(output_dtype)

        return out.reshape(*original_shape[:-1], out_features)

    def _apply_dequant(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_quant = layer.weight
        output_dtype: torch.dtype = getattr(
            weight_quant,
            "output_dtype",
            getattr(layer, "_fp8_output_dtype", torch.bfloat16),
        )
        weight_scale: torch.Tensor = layer.weight_scale.data.to(output_dtype)
        input_scale: torch.Tensor = layer.input_scale.data.to(output_dtype)
        if weight_scale.dim() == 0:
            weight_scale = weight_scale.unsqueeze(0)
        if input_scale.dim() == 0:
            input_scale = input_scale.unsqueeze(0)
        weight_output_type = weight_quant.to(dtype=output_dtype)
        weight_final = weight_output_type * weight_scale.unsqueeze(1)
        x_final = x.to(dtype=output_dtype) * input_scale

        return nn.functional.linear(x_final, weight_final,
                                    bias=bias).to(dtype=output_dtype)
