# SPDX-License-Identifier: Apache-2.0
"""Serialized block-FP8 execution for the MiniMax-H3 Qwen3-VL encoder."""

from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from fastvideo.distributed import get_tp_world_size
from fastvideo.layers.linear import LinearBase, LinearMethodBase
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.layers.quantization.fp8_config import FP8_DTYPE
from fastvideo.models.utils import set_weight_attrs


class MiniMaxH3SerializedFP8Config(QuantizationConfig):
    """Serialized 128x128 block-FP8 contract for the H3 text encoder."""

    def __init__(self, weight_block_size: tuple[int, int]) -> None:
        super().__init__()
        if weight_block_size != (128, 128):
            raise ValueError("MiniMax-H3 serialized FP8 requires weight_block_size=[128, 128], "
                             f"got {list(weight_block_size)}")
        self.weight_block_size = weight_block_size
        self.is_checkpoint_fp8_serialized = True
        self.activation_scheme = "dynamic"

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MiniMaxH3SerializedFP8Config":
        quant_method = str(config.get("quant_method", "")).lower()
        if quant_method != "fp8":
            raise ValueError(f"MiniMax-H3 only supports serialized FP8 text-encoder checkpoints, got {quant_method!r}")
        if str(config.get("activation_scheme", "")).lower() != "dynamic":
            raise ValueError("MiniMax-H3 serialized FP8 requires dynamic activation quantization")
        if str(config.get("fmt", "e4m3")).lower() not in ("e4m3", "float8_e4m3fn"):
            raise ValueError(f"MiniMax-H3 serialized FP8 requires E4M3 weights, got {config.get('fmt')!r}")
        block_size = config.get("weight_block_size")
        if not isinstance(block_size, list | tuple) or len(block_size) != 2:
            raise ValueError("MiniMax-H3 serialized FP8 requires a two-dimensional weight_block_size")
        ignored_layers = config.get("modules_to_not_convert", config.get("ignored_layers", []))
        if not isinstance(ignored_layers, list | tuple):
            raise ValueError("MiniMax-H3 serialized FP8 modules_to_not_convert must be a sequence")
        language_exclusions = [
            name for name in ignored_layers
            if isinstance(name, str) and (name.startswith("language_model.") or ".language_model." in name)
        ]
        if language_exclusions:
            raise ValueError("MiniMax-H3 does not support partially quantized language stacks; "
                             f"ignored language layers: {language_exclusions[:3]}")
        if not any(isinstance(name, str) and "visual" in name for name in ignored_layers):
            raise ValueError("MiniMax-H3 serialized FP8 requires the vision stack to be listed in "
                             "modules_to_not_convert")
        return cls((int(block_size[0]), int(block_size[1])))

    def validate_runtime(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 requires a CUDA device; "
                               f"got {device.type!r}")
        capability = torch.cuda.get_device_capability(device)
        capability_number = capability[0] * 10 + capability[1]
        if capability_number < self.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 requires GPU capability "
                               f"sm{self.get_min_capability()} or newer, got sm{capability_number}")
        if capability[0] not in (10, 12):
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 currently adapts SGLang's Blackwell "
                               f"FlashInfer path; got unsupported sm{capability_number}")
        _require_sglang_per_token_group_fp8_quantization()
        _get_flashinfer_groupwise_fp8_gemm()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase) and ".language_model.layers." in prefix:
            return MiniMaxH3SerializedFP8LinearMethod(self.weight_block_size)
        return None


# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Adapted from SGLang's per-token-group quantization kernels and Blackwell
# FlashInfer dispatch at commit f99c62063c7dcfcd06784b885dc08cb52cf23865:
# https://github.com/sgl-project/sglang/blob/f99c62063c7dcfcd06784b885dc08cb52cf23865/python/sglang/kernels/ops/quantization/fp8_kernel.py
# https://github.com/sgl-project/sglang/blob/f99c62063c7dcfcd06784b885dc08cb52cf23865/python/sglang/srt/layers/quantization/fp8_utils.py
if triton is not None:

    @triton.jit
    def _h3_per_token_group_quant_fp8_row_major(
        input_ptr,
        output_ptr,
        scale_ptr,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        BLOCK: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        input_ptr += group_id.to(tl.int64) * group_size
        output_ptr += group_id.to(tl.int64) * group_size
        scale_ptr += group_id

        offsets = tl.arange(0, BLOCK)
        mask = offsets < group_size
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(values)), eps)
        scale = absmax / fp8_max
        quantized = tl.clamp(values / scale, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)

        tl.store(output_ptr + offsets, quantized, mask=mask)
        tl.store(scale_ptr, scale)

    @triton.jit
    def _h3_per_token_group_quant_fp8_column_major(
        input_ptr,
        output_ptr,
        scale_ptr,
        group_size,
        input_columns,
        scale_column_stride,
        eps,
        fp8_min,
        fp8_max,
        BLOCK: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        input_ptr += group_id.to(tl.int64) * group_size
        output_ptr += group_id.to(tl.int64) * group_size

        groups_per_row = input_columns // group_size
        scale_column = group_id % groups_per_row
        scale_row = group_id // groups_per_row
        scale_ptr += scale_column * scale_column_stride + scale_row

        offsets = tl.arange(0, BLOCK)
        mask = offsets < group_size
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(values)), eps)
        scale = absmax / fp8_max
        quantized = tl.clamp(values / scale, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)

        tl.store(output_ptr + offsets, quantized, mask=mask)
        tl.store(scale_ptr, scale)
else:
    _h3_per_token_group_quant_fp8_row_major = None
    _h3_per_token_group_quant_fp8_column_major = None


def _require_sglang_per_token_group_fp8_quantization() -> None:
    if (triton is None or _h3_per_token_group_quant_fp8_row_major is None
            or _h3_per_token_group_quant_fp8_column_major is None):
        raise RuntimeError(
            "MiniMax-H3 serialized blockwise FP8 requires Triton for SGLang-compatible "
            "per-token-group activation quantization")


def _sglang_per_token_group_quant_fp8(
    input_tensor: torch.Tensor,
    group_size: int,
    *,
    column_major_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang-compatible dynamic FP8 quantization for contiguous 2-D activations."""
    _require_sglang_per_token_group_fp8_quantization()
    if input_tensor.ndim != 2:
        raise ValueError(f"per-token-group FP8 quantization expects 2-D input, got {input_tensor.ndim}-D")
    if not input_tensor.is_contiguous():
        raise ValueError("per-token-group FP8 quantization requires contiguous input")
    if input_tensor.shape[-1] % group_size:
        raise ValueError(f"activation width {input_tensor.shape[-1]} is not divisible by group_size={group_size}")

    quantized = torch.empty_like(input_tensor, dtype=FP8_DTYPE)
    rows, columns = input_tensor.shape
    groups_per_row = columns // group_size
    if column_major_scales:
        scales = torch.empty(
            (groups_per_row, rows),
            device=input_tensor.device,
            dtype=torch.float32,
        ).permute(1, 0)
    else:
        scales = torch.empty(
            (rows, groups_per_row),
            device=input_tensor.device,
            dtype=torch.float32,
        )

    if rows:
        num_groups = input_tensor.numel() // group_size
        block = triton.next_power_of_2(group_size)
        num_warps = min(max(block // 256, 1), 8)
        if column_major_scales:
            _h3_per_token_group_quant_fp8_column_major[(num_groups,)](
                input_tensor,
                quantized,
                scales,
                group_size,
                columns,
                scales.stride(1),
                1e-10,
                -448.0,
                448.0,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _h3_per_token_group_quant_fp8_row_major[(num_groups,)](
                input_tensor,
                quantized,
                scales,
                group_size,
                1e-10,
                -448.0,
                448.0,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )
    return quantized, scales


def _get_flashinfer_groupwise_fp8_gemm():
    try:
        from flashinfer.gemm import gemm_fp8_nt_groupwise
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "MiniMax-H3 serialized blockwise FP8 requires "
            "flashinfer.gemm.gemm_fp8_nt_groupwise (validated with flashinfer-python==0.6.8). "
            "FastVideo will not re-quantize this checkpoint to tensorwise FP8.") from error
    return gemm_fp8_nt_groupwise


def _get_flashinfer_groupwise_backend(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 12:
        return "cutlass"
    if capability[0] == 10:
        return "trtllm"
    capability_number = capability[0] * 10 + capability[1]
    raise RuntimeError(f"FlashInfer groupwise FP8 requires a Blackwell GPU, got sm{capability_number}")


def _flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    block_size: tuple[int, int],
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    input_2d = input_tensor.view(-1, input_tensor.shape[-1])
    output_shape = [*input_tensor.shape[:-1], weight.shape[0]]
    backend = _get_flashinfer_groupwise_backend(input_tensor.device)
    if input_2d.dtype != torch.bfloat16:
        raise RuntimeError("MiniMax-H3 FlashInfer groupwise FP8 requires BF16 activations; "
                           f"got {input_2d.dtype}. The SGLang FP16 Triton GEMM fallback is not enabled for H3.")
    if backend == "trtllm" and input_2d.shape[1] < 256:
        raise RuntimeError("MiniMax-H3 FlashInfer TRTLLM groupwise FP8 requires K >= 256; "
                           f"got K={input_2d.shape[1]}. The SGLang Triton GEMM fallback is not enabled for H3.")

    gemm_fp8_nt_groupwise = _get_flashinfer_groupwise_fp8_gemm()
    block_n, block_k = block_size
    q_input, x_scale = _sglang_per_token_group_quant_fp8(
        input_2d,
        block_k,
        column_major_scales=(backend == "trtllm"),
    )
    if backend == "cutlass":
        m, k = input_2d.shape
        n = weight.shape[0]
        if x_scale.shape == (m, k // block_k):
            x_scale = x_scale.transpose(-1, -2).contiguous()
        if weight_scale.shape == (n // block_n, k // block_k):
            weight_scale = weight_scale.transpose(-1, -2).contiguous()
        # FlashInfer documents that ``m`` "should be padded to a multiple of 4
        # before calling this function", and the CUTLASS groupwise module
        # rejects unaligned m at dispatch (``cutlass gemm.can_implement
        # failed``). Prompt token counts are arbitrary, so pad the quantized
        # activation rows and their per-token scales with zeros, then slice
        # the padded rows off the GEMM output. The trtllm (sm100) route
        # tolerates any m and stays unpadded.
        padded_m = (m + 3) // 4 * 4
        if padded_m != m:
            padded_q_input = q_input.new_zeros((padded_m, q_input.shape[1]))
            padded_q_input[:m] = q_input
            q_input = padded_q_input
            padded_x_scale = x_scale.new_zeros((x_scale.shape[0], padded_m))
            padded_x_scale[:, :m] = x_scale
            x_scale = padded_x_scale
        expected_x_scale_shape = (k // block_k, padded_m)
        expected_weight_scale_shape = (k // block_k, n // block_n)
        if x_scale.shape != expected_x_scale_shape or weight_scale.shape != expected_weight_scale_shape:
            raise RuntimeError("FlashInfer CUTLASS block-FP8 scale layout mismatch: "
                               f"x_scale={tuple(x_scale.shape)}, weight_scale={tuple(weight_scale.shape)}, "
                               f"expected={expected_x_scale_shape}/{expected_weight_scale_shape}")
        if x_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
            raise RuntimeError("FlashInfer CUTLASS block-FP8 scales must be float32")
        output = gemm_fp8_nt_groupwise(
            q_input,
            weight,
            x_scale.contiguous(),
            weight_scale.contiguous(),
            out_dtype=input_2d.dtype,
            backend="cutlass",
            scale_major_mode="MN",
        )
        if padded_m != m:
            output = output[:m]
    else:
        expected_x_scale_shape = (input_2d.shape[0], input_2d.shape[1] // block_k)
        expected_weight_scale_shape = (weight.shape[0] // block_n, weight.shape[1] // block_k)
        if x_scale.shape != expected_x_scale_shape or x_scale.stride(0) != 1:
            raise RuntimeError("FlashInfer TRTLLM block-FP8 activation scale layout mismatch: "
                               f"shape={tuple(x_scale.shape)}, stride={x_scale.stride()}, "
                               f"expected column-major {expected_x_scale_shape}")
        if weight_scale.shape != expected_weight_scale_shape:
            raise RuntimeError("FlashInfer TRTLLM block-FP8 weight scale layout mismatch: "
                               f"shape={tuple(weight_scale.shape)}, expected={expected_weight_scale_shape}")
        output = gemm_fp8_nt_groupwise(
            q_input,
            weight,
            x_scale,
            weight_scale,
            out_dtype=input_2d.dtype,
            backend="trtllm",
        )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


class MiniMaxH3SerializedFP8LinearMethod(LinearMethodBase):
    """Execute serialized 128x128 block-FP8 weights without re-quantizing them."""

    def __init__(self, weight_block_size: tuple[int, int]) -> None:
        super().__init__()
        self.weight_block_size = weight_block_size

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
        output_size_per_partition = sum(output_partition_sizes)
        block_n, block_k = self.weight_block_size
        tp_size = get_tp_world_size()
        if tp_size > 1 and input_size // input_size_per_partition == tp_size:
            if input_size_per_partition % block_k:
                raise ValueError(f"Weight input_size_per_partition={input_size_per_partition} is not divisible "
                                 f"by block_k={block_k}")
        if tp_size > 1 and output_size // output_size_per_partition == tp_size:
            for output_partition_size in output_partition_sizes:
                if output_partition_size % block_n:
                    raise ValueError(f"Weight output_partition_size={output_partition_size} is not divisible "
                                     f"by block_n={block_n}")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = Parameter(
            torch.empty(output_size_per_partition, input_size_per_partition, dtype=FP8_DTYPE),
            requires_grad=False,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })
        layer.register_parameter("weight", weight)

        scale = Parameter(
            torch.empty((output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(scale, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })
        scale.data.fill_(torch.finfo(torch.float32).min)
        layer.register_parameter("weight_scale_inv", scale)
        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = getattr(layer, "weight", None)
        block_scales = getattr(layer, "weight_scale_inv", None)
        if weight is None or block_scales is None:
            raise ValueError("Serialized MiniMax-H3 FP8 linear is missing weight or weight_scale_inv")
        if weight.dtype != FP8_DTYPE:
            raise ValueError(f"Serialized MiniMax-H3 FP8 weight must be {FP8_DTYPE}, got {weight.dtype}")
        if block_scales.dtype != torch.float32:
            raise ValueError("Serialized MiniMax-H3 FP8 weight_scale_inv must be float32, "
                             f"got {block_scales.dtype}")

        block_n, block_k = self.weight_block_size
        output_size, input_size = weight.shape
        if output_size % block_n or input_size % block_k:
            raise ValueError("Serialized MiniMax-H3 FP8 weight dimensions must be divisible by the 128x128 block size; "
                             f"got {tuple(weight.shape)}")
        expected_scale_shape = (output_size // block_n, input_size // block_k)
        if tuple(block_scales.shape) != expected_scale_shape:
            raise ValueError("Serialized MiniMax-H3 FP8 scale shape mismatch: "
                             f"expected {expected_scale_shape}, got {tuple(block_scales.shape)}")
        if not bool(torch.isfinite(block_scales).all()) or bool((block_scales <= 0).any()):
            raise ValueError("Serialized MiniMax-H3 FP8 weight_scale_inv must contain finite positive values")
        layer.weight.data = weight.data
        layer.weight_scale_inv.data = block_scales.data

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 execution requires CUDA")

        capability = torch.cuda.get_device_capability(x.device)
        capability_number = capability[0] * 10 + capability[1]
        if capability_number < MiniMaxH3SerializedFP8Config.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 requires GPU capability "
                               f"sm{MiniMaxH3SerializedFP8Config.get_min_capability()} or newer, "
                               f"got sm{capability_number}")

        if not x.is_contiguous():
            x = x.contiguous()
        return _flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
            x,
            layer.weight,
            self.weight_block_size,
            layer.weight_scale_inv,
            bias,
        )


__all__ = [
    "MiniMaxH3SerializedFP8Config",
    "MiniMaxH3SerializedFP8LinearMethod",
]
