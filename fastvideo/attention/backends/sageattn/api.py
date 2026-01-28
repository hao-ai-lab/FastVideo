"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda

# Centralized block size configuration for sageattn_blackwell kernels
# These should match the values in fastvideo/attention/backends/sageattn/blackwell/block_config.h
BLOCK_M = 128  # Block size for M dimension (query sequence length)
BLOCK_N = 128  # Block size for N dimension (key/value sequence length)


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)
    
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = BLOCK_M
    num_groups = L // GROUP_SIZE
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True, enable_smoothing_q: bool = False, enable_smoothing_k: bool = True):

    def pad_to_block_size(x):
        L = x.size(2)
        pad_len = (BLOCK_M - L % BLOCK_M) % BLOCK_M
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    if enable_smoothing_k:
        k -= k.mean(dim=-2, keepdim=True)  
    q, k, v = map(lambda x: pad_to_block_size(x), [q, k, v])
    if per_block_mean and enable_smoothing_q:
        q, qm = triton_group_mean(q)
    elif enable_smoothing_q:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    if enable_smoothing_q:
        delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    else:  # used to disable q smoothing
        B, H, L, D = q.shape
        delta_s = torch.zeros((B, H, L // BLOCK_M, k.shape[2]), device=q.device, dtype=torch.float32)
        
    return q, k, v, delta_s

def scale_and_quant_fp4(x: torch.Tensor, two_level: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 with block-scaled quantization.
    
    Args:
        x: Input tensor [B, H, N, D]
        two_level: If True, use two-level quantization with per-row first-level scale
                   and per-16-element second-level scale. If False, use standard
                   single-level per-16-element quantization.
    
    Returns:
        packed_fp4: Quantized FP4 values [B, H, N, D // 2]
        fp8_scale: Per-16-element scale factors [B, H, N, D // 16]
        fp8_scale_row: Per-row scale factors [B, H, N, 1] (only meaningful if two_level=True,
                       otherwise returns ones)
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    
    if two_level:
        fp8_scale_row = torch.empty((B, H, N, 1), device=x.device, dtype=torch.float8_e4m3fn)
        fp4quant_cuda.scaled_fp4_quant_two_level(x, packed_fp4, fp8_scale, fp8_scale_row, 1)
    else:
        fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
        # Return dummy scale_row (all ones represented as FP8)
        fp8_scale_row = torch.ones((B, H, N, 1), device=x.device, dtype=torch.float8_e4m3fn)
    
    return packed_fp4, fp8_scale, fp8_scale_row

def scale_and_quant_fp4_permute(x: torch.Tensor, two_level: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 with permutation for K tensor.
    
    Args:
        x: Input tensor [B, H, N, D]
        two_level: If True, use two-level quantization
    
    Returns:
        packed_fp4: Quantized FP4 values [B, H, N, D // 2]
        fp8_scale: Per-16-element scale factors [B, H, N, D // 16]
        fp8_scale_row: Per-row scale factors [B, H, N, 1]
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    
    if two_level:
        fp8_scale_row = torch.empty((B, H, N, 1), device=x.device, dtype=torch.float8_e4m3fn)
        fp4quant_cuda.scaled_fp4_quant_permute_two_level(x, packed_fp4, fp8_scale, fp8_scale_row, 1)
    else:
        fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
        fp8_scale_row = torch.ones((B, H, N, 1), device=x.device, dtype=torch.float8_e4m3fn)
    
    return packed_fp4, fp8_scale, fp8_scale_row

def scale_and_quant_fp4_transpose(x: torch.Tensor, two_level: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 with transpose for V tensor.
    
    Args:
        x: Input tensor [B, H, N, D]
        two_level: If True, use two-level quantization
    
    Returns:
        packed_fp4: Quantized FP4 values [B, H, D, N // 2]
        fp8_scale: Per-16-element scale factors [B, H, D, N // 16]
        fp8_scale_row: Per-row (head_dim element) scale factors [B, H, D, 1]
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    
    if two_level:
        fp8_scale_row = torch.empty((B, H, D, 1), device=x.device, dtype=torch.float8_e4m3fn)
        fp4quant_cuda.scaled_fp4_quant_trans_two_level(x, packed_fp4, fp8_scale, fp8_scale_row, 1)
    else:
        fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
        fp8_scale_row = torch.ones((B, H, D, 1), device=x.device, dtype=torch.float8_e4m3fn)
    
    return packed_fp4, fp8_scale, fp8_scale_row

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True,
                         single_level_p_quant: bool = False,
                         two_level_qkv_quant: bool = False,
                         q_scale_row: torch.Tensor = None,
                         k_scale_row: torch.Tensor = None,
                         v_scale_row: torch.Tensor = None
                        ):
    """
    Block-scaled FP4 attention kernel.
    
    Args:
        qlist: Tuple of (packed_fp4_q, fp8_scale_q)
        klist: Tuple of (packed_fp4_k, fp8_scale_k)
        vlist: Tuple of (packed_fp4_v, fp8_scale_v)
        delta_s: Delta scores for Q smoothing
        KL: Key sequence length (unpadded)
        is_causal: Whether to use causal masking
        per_block_mean: Whether to use per-block mean for Q smoothing
        is_bf16: Whether output is bfloat16
        single_level_p_quant: P quantization mode (single vs two-level)
        two_level_qkv_quant: If True, QKV uses two-level quantization (row_scale tensors must be provided)
        q_scale_row: Per-row scale for Q [B, H, N, 1] (used when two_level_qkv_quant=True)
        k_scale_row: Per-row scale for K [B, H, N, 1] (used when two_level_qkv_quant=True)
        v_scale_row: Per-row scale for V [B, H, D, 1] (used when two_level_qkv_quant=True)
    
    Returns:
        Output tensor and softmax LSE
    """
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    
    # Pass row scales to kernel for two-level QKV quantization
    # The kernel will use these to properly rescale during the attention computation
    return fp4attn_cuda.fwd(
        qlist[0], klist[0], vlist[0], 
        qlist[1], klist[1], vlist[1], 
        delta_s, KL, None, softmax_scale, 
        is_causal, per_block_mean, is_bf16, 
        single_level_p_quant, two_level_qkv_quant,
        q_scale_row, k_scale_row, v_scale_row
    )


def sageattn_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, single_level_p_quant = True, two_level_qkv_quant = False, **kwargs):
    """
    SageAttention3 Blackwell kernel for FP4 attention.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        v: Value tensor [B, H, L, D]
        attn_mask: Attention mask (not used)
        is_causal: Whether to use causal masking
        per_block_mean: Whether to use per-block mean for Q smoothing
        single_level_p_quant: If True, use single-level quantization for P: s_P2, P̂_2 = φ(P̃) directly
                              (standard per-block FP4 quantization like V, no s_P1).
                              If False (default), use two-level quantization:
                              s_P1 = rowmax(P̃)/(448×6), then s_P2, P̂_2 = φ(P̃/s_P1).
        two_level_qkv_quant: If True, use two-level quantization for Q, K, V:
                              - Level 1: Per-row scale s_row = absmax_row(x)/(448×6)
                              - Level 2: Per-16-element scale s_16 = absmax_block(x/s_row)/6
                              This provides better dynamic range similar to P quantization.
                              Note: absmax is used (not max) because Q, K, V can have negative values.
                              If False (default), use standard single-level per-16-element quantization.
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Output tensor [B, H, L, D]
    """
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    
    # Quantize Q, K, V with optional two-level quantization
    q_packed, q_scale, q_scale_row = scale_and_quant_fp4(q, two_level=two_level_qkv_quant)
    k_packed, k_scale, k_scale_row = scale_and_quant_fp4_permute(k, two_level=two_level_qkv_quant)
    v_packed, v_scale, v_scale_row = scale_and_quant_fp4_transpose(v, two_level=two_level_qkv_quant)
    
    qlist_from_cuda = (q_packed, q_scale)
    klist_from_cuda = (k_packed, k_scale)
    vlist_from_cuda = (v_packed, v_scale)
    
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda, 
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16,
        single_level_p_quant,
        two_level_qkv_quant,
        q_scale_row,
        k_scale_row,
        v_scale_row
    )[0][:, :, :QL, :].contiguous()
    return o_fp4