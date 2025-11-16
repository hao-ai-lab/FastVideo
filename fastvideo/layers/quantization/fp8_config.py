# # SPDX-License-Identifier: Apache-2.0
# from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
# import torch
# from torch.nn.parameter import Parameter

# from typing import Any, Tuple
# import deep_gemm
# from deep_gemm import ceil_div, get_mn_major_tma_aligned_tensor
# from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
# from fastvideo.models.utils import set_weight_attrs

# class FP8QuantizeMethod(QuantizeMethodBase):
#     def __init__(self):
#         super().__init__()
#         self.weight_fp8 = None
#         self.weight_scale = None

#     def create_weights(self, layer: torch.nn.Module,
#                        input_size_per_partition: int,
#                        output_partition_sizes: list[int], input_size: int,
#                        output_size: int, params_dtype: torch.dtype,
#                        **extra_weight_attrs):
#         """Create weights for a linear layer. Note the corrected signature to match LinearMethodBase."""
#         weight = Parameter(torch.empty(
#             sum(output_partition_sizes),
#             input_size_per_partition,
#             dtype=params_dtype,
#         ),
#         requires_grad=False)
#         set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
#         layer.register_parameter("weight", weight)
#         set_weight_attrs(weight, extra_weight_attrs)

#     @torch.compile
#     def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
#         """Apply FP8 quantized computation."""
#         out_dim = layer.weight.shape[0]
#         # Need contiguous tensors for collectives.
#         assert x.dtype == torch.bfloat16 or x.dtype == torch.float16 or x.dtype == torch.float32, f"only allow bf16/fp16/fp32 inputs to fp8 linear, got {x.dtype}"
#         x_fp8, x_scale = per_token_cast_to_fp8(x.view(-1, x.shape[-1]), use_ue8m0=False)
#         x_scale = get_mn_major_tma_aligned_tensor(x_scale)
#         weight_fp8 = layer._fp8_weight
#         weight_scale = layer._fp8_weight_scale
        
#         original_shape = x.shape
#         out = torch.zeros((x_fp8.shape[0], out_dim), device=x.device, dtype=x.dtype)
#         deep_gemm.fp8_gemm_nt(
#             (x_fp8, x_scale),
#             (weight_fp8, weight_scale),
#             out,
#             disable_ue8m0_cast=False
#         )   
            
#         if bias is not None:
#             if bias.device != out.device or bias.dtype != out.dtype:
#                 bias = bias.to(device=out.device, dtype=out.dtype)
#             out = out + bias
        
#         if len(original_shape) == 3:
#             out = out.view(original_shape[0], original_shape[1], out_dim)
        
#         return out
        

# class FP8Config(QuantizationConfig):
#     def __init__(self):
#         super().__init__()

#     def get_name(self):
#         return "fp8"
    
#     def get_supported_act_dtypes(self):
#         return [torch.bfloat16, torch.float16, torch.float32]
    
#     @classmethod
#     def get_min_capability(cls):
#         return 90
    
#     @staticmethod
#     def get_config_filenames():
#         return []

#     @classmethod
#     def from_config(cls, config: dict[str, Any]) -> "FP8Config":
#         return cls()
    
#     def get_quant_method(self, layer: torch.nn.Module, prefix: str):
#         from fastvideo.layers.linear import LinearBase
#         fp8_layers = ["ffn.fc_in", "ffn.fc_out", "to_q", "to_k", "to_v", "to_out"]
#         if isinstance(layer, LinearBase) and any(layer_name in prefix for layer_name in fp8_layers):
#             return FP8QuantizeMethod()
#         return None

# @torch.compile
# def convert_model_to_fp8(model: torch.nn.Module):
#     from torch.distributed.tensor import DTensor  # type: ignore
#     for mod in model.modules():
#         qm = getattr(mod, "quant_method", None)
#         if isinstance(qm, FP8QuantizeMethod):
#             weight = getattr(mod, "weight", None)
#             if weight is None:
#                 continue
#             if isinstance(weight, DTensor):  # type: ignore
#                 weight_local = weight.to_local()
#             else:
#                 weight_local = weight
#             fp8_w, fp8_s = per_block_cast_to_fp8(weight_local, use_ue8m0=False)
#             mod.register_buffer("_fp8_weight", fp8_w, persistent=False)
#             mod.register_buffer("_fp8_weight_scale", fp8_s, persistent=False)