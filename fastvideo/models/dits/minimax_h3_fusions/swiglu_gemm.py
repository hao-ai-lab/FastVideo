# SPDX-License-Identifier: Apache-2.0
"""Exact eager GB10 FFN-up GEMM, BF16 SwiGLU, and NVFP4 output preparation.

Only immutable weights are cached. Their value/gate columns are paired in
memory so each GEMM tile can produce complete SwiGLU quantization groups.
The checkpoint layout and files are unchanged. Both original BF16 rounding
boundaries precede the original block-16 scale/code conversion.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _backend():
    try:
        import _fastvideo_h3_swiglu_gemm
    except ImportError:
        return None
    return _fastvideo_h3_swiglu_gemm


def pair_weight_columns(weight: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair value/gate rows of packed weights and their 128x4-swizzled scales."""
    n, k = weight.shape[0], weight.shape[1] * 2
    values = torch.arange(n // 2, device=weight.device)
    permutation = torch.stack((values, values + n // 2), dim=1).reshape(-1)
    paired_weight = weight[permutation].contiguous().T
    plain_scales = scales.reshape(n // 128, k // 64, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(n, k // 16)
    paired_scales = (plain_scales[permutation].reshape(n // 128, 4, 32, k // 64, 4)
                     .permute(0, 3, 2, 1, 4).contiguous().reshape_as(scales).T)
    return paired_weight, paired_scales


def _weight_key(weight: torch.Tensor, scales: torch.Tensor) -> tuple:
    # Packed inference buffers may have no version counter. Their storage is
    # immutable; replacement or migration changes the identity/storage key.
    return tuple((id(t), t.data_ptr(), t.device, None if t.is_inference() else t._version)
                 for t in (weight, scales))


def try_swiglu_gemm(ff, hidden_states: torch.Tensor, pre_quantized) -> torch.Tensor | None:
    """Return None outside the measured T2AV/Omni single/pair-Spark NVFP4 shapes."""
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod, _mm_fp4_row_chunks

    if (pre_quantized is None or torch.is_grad_enabled() or torch.compiler.is_compiling()
            or not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16
            or torch.cuda.get_device_capability(hidden_states.device) != (12, 1)
            or type(getattr(ff.fc_in, 'quant_method', None)) is not NVFP4QuantizeMethod
            or type(getattr(ff.fc_out, 'quant_method', None)) is not NVFP4QuantizeMethod
            or ff.fc_in.bias is not None):
        return None
    a, a_scale, input_scale = pre_quantized
    a = a.reshape(-1, a.shape[-1])
    a_scale = a_scale.reshape(-1, a_scale.shape[-1])
    weight = ff.fc_in._nvfp4_weight
    weight_scales = ff.fc_in._nvfp4_weight_scale
    if (a.shape not in ((14859, 2688), (7430, 2688), (29678, 2688)) or weight.shape != (28672, 2688)
            or not ff.fc_out.quant_method.wants_prequantized_input()):
        return None
    backend = _backend()
    if backend is None:
        return None
    method = ff.fc_out.quant_method
    output_scale = method._static_activation_global_sf()
    if output_scale is None:
        output_scale = getattr(ff.fc_out, '_nvfp4_input_global_sf', None)
    if output_scale is None:
        if method._dynamic_activation_scale():
            return None
        output_scale = method.x_global_sf
    key = _weight_key(weight, weight_scales)
    if getattr(ff.fc_in, '_h3_swiglu_weight_key', None) != key:
        paired_weight, paired_scales = pair_weight_columns(weight, weight_scales)
        ff.fc_in.register_buffer('_h3_swiglu_paired_weight', paired_weight, persistent=False)
        ff.fc_in.register_buffer('_h3_swiglu_paired_scales', paired_scales, persistent=False)
        ff.fc_in._h3_swiglu_weight_key = key
    paired_weight = ff.fc_in._h3_swiglu_paired_weight
    paired_scales = ff.fc_in._h3_swiglu_paired_scales
    if hasattr(ff.fc_in, '_nvfp4_alpha'):
        alpha = ff.fc_in._nvfp4_alpha / input_scale
    else:
        alpha = 1.0 / (input_scale * ff.fc_in._weight_global_sf)
    rows, width = a.shape[0], weight.shape[0] // 2
    q = torch.empty((rows, width // 2), device=a.device, dtype=torch.uint8)
    scales = torch.empty(((rows + 127) // 128 * 128, width // 16), device=a.device, dtype=torch.uint8)
    start = 0
    parts = 8 if rows == 29678 else (2 if rows == 7430 else 4)
    for chunk, chunk_scale in _mm_fp4_row_chunks(a, a_scale, parts):
        end = start + chunk.shape[0]
        backend.run(chunk, paired_weight, chunk_scale, paired_scales, alpha, output_scale,
                    q[start:end], scales[start:(end + 127) // 128 * 128])
        start = end
    # apply(pre_quantized=...) reads only hidden_states' leading shape.
    return method.apply(ff.fc_out, hidden_states, ff.fc_out.bias, pre_quantized=(q, scales, output_scale))
