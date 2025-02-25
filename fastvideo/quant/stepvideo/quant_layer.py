from typing import Optional
import torch
from torch.nn.parameter import Parameter

import vllm
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8Config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_fp8_linear
from vllm.model_executor.utils import set_weight_attrs


class FP8Linear(LinearBase):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config = None,
                 prefix: str = ""):
        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         None,
                         prefix=prefix)

        self.quant_method = Fp8LinearQuantizer(quant_config)
        assert self.quant_method is not None
        self.quant_method.create_weights(self,
                                         self.input_size, [self.output_size],
                                         self.input_size,
                                         self.output_size,
                                         self.params_dtype,
                                         weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)


    def weight_loader(self, param, loaded_weight: torch.Tensor):
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s

class Fp8LinearQuantizer(Fp8LinearMethod):
    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        # Note: lazy import to avoid triton import error.
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            apply_w8a8_block_fp8_linear)
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )

        if not x.is_contiguous():
            x = x.contiguous()

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=self.cutlass_fp8_supported)
        