# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing qat_attn
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import triton
try:
    # Use local version first (avoids issues with installed package)
    from fastvideo.attention.backends.sageattn import sageattn_blackwell
except ImportError:
    # Fall back to installed package if local version not available
    try:
        from sageattn import sageattn_blackwell
    except ImportError:
        sageattn_blackwell = None

from fastvideo.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.logger import init_logger
from test_quantization_precision import get_fake_quant
from qat_attn import attention
from qat_attn import _attn_bwd_preprocess, _attn_bwd_dq_cross, _attn_bwd_dkdv_cross, _attn_bwd
# from fastvideo.attention.backends.sageattn.api import triton_group_mean
logger = init_logger(__name__)

from math import sqrt
try:
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward, _wrapped_flash_attn_backward
except ImportError:
    _wrapped_flash_attn_forward = None
    _wrapped_flash_attn_backward = None

logger = init_logger(__name__)
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from quant_utils import fake_quantize_q, fake_quantize_kv

def fp_16_qk(q, k):
    qk = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.shape[-1], device=q.device, dtype=q.dtype))
    p = torch.softmax(qk, dim=-1)
    return p


def matmul_3d_4bit(a, b):
    H, M, K1 = a.shape
    H, N, K2 = b.shape
    assert K1 == K2
    padded_K = (K2 + 127) // 128 * 128
    a = torch.nn.functional.pad(a, (0, padded_K - K2))
    b = torch.nn.functional.pad(b, (0, padded_K - K2))
    out = torch.empty((H, M, N), device=a.device, dtype=a.dtype)
    one = torch.ones(1, device=a.device, dtype=torch.float32)
    for h in range(H):
        q_fp4, q_inv_s = nvfp4_quantize(a[h], one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        k_fp4, k_inv_s = nvfp4_quantize(b[h], one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        mm_fp4(
            q_fp4, k_fp4.T, q_inv_s, k_inv_s.T, one,
            a.dtype, out[h],
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
        )
    return out


class _Matmul3d4bitFWD16bitBWD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scale=None):
        ctx.save_for_backward(a, b)
        if scale is not None:
            a = a / scale
            return matmul_3d_4bit(a, b) * scale
        else:
            return matmul_3d_4bit(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_a = grad_out.matmul(b)
        grad_b = grad_out.transpose(1, 2).matmul(a)
        # None for the three extra forward args
        return grad_a, grad_b, None
    
    
class _Matmul3d4bitFWD16bitBWD_with_smoothing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        ctx.save_for_backward(q, k)
        k = k - k.mean(dim=-2, keepdim=True)  # [H, L, D]
        qm = q.mean(dim=-2, keepdim=True)  # [H, 1, D]
        q = q - qm  # [B, H, L, D]
        delta_s = torch.matmul(qm, k.transpose(-2, -1)).contiguous()  # [H, 1, L]
        return matmul_3d_4bit(q, k) + delta_s

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_a = grad_out.matmul(b)
        grad_b = grad_out.transpose(1, 2).matmul(a)
        # None for the three extra forward args
        return grad_a, grad_b, None

def attn_forward_4bit_fwd_16bit_bwd(q, k, v, is_causal=False, per_block_mean=False):
    assert is_causal == False
    assert per_block_mean == False
    q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    p = sage_attn_qk_torch(q, k).to(q.dtype)
    row_max_p = p.max(dim=-1, keepdim=True).values
    scale_row = row_max_p / (448 * 6)
    out = _Matmul3d4bitFWD16bitBWD.apply(p.squeeze(0), v.squeeze(0).transpose(1, 2).contiguous(), scale_row.squeeze(0)).unsqueeze(0)
    return out.transpose(1, 2).contiguous()

def preprocess(q, k, v):
    return get_fake_quant(q), get_fake_quant(k), get_fake_quant(v)

def qat_attn_qk_torch(fq_q, fq_k):
    """
    Inputs:  q/k/v in [B, H, L, D]
    Returns: out in  [B, H, L, D]
    """
    B, H, qL, D = fq_q.shape
    B, H, kL, D = fq_k.shape
    assert B == 1
    q, k = fq_q.squeeze(0), fq_k.squeeze(0)
    qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.shape[-1], device=q.device, dtype=q.dtype))
    qk = qk / torch.sqrt(torch.tensor(D, device=q.device, dtype=q.dtype))
    p = torch.softmax(qk, dim=-1)
    return p.unsqueeze(0)

def qat_attn_backward_16bit(fq_q_BLHD, fq_k_BLHD, fq_v_BLHD, high_prec_o_BLHD, grad_out_BLHD, is_causal=False):
    assert is_causal == False
    fq_q, fq_k, fq_v, high_prec_o, do = fq_q_BLHD.transpose(1 ,2).contiguous(), fq_k_BLHD.transpose(1 ,2).contiguous(), fq_v_BLHD.transpose(1 ,2).contiguous(), high_prec_o_BLHD.transpose(1 ,2).contiguous(), grad_out_BLHD.transpose(1 ,2).contiguous()
    B, H, _, D = fq_q.shape
    p = qat_attn_qk_torch(fq_q, fq_k).to(do.dtype)
    fq_p = get_fake_quant(p).unsqueeze(0)
    dV = fq_p.transpose(-2, -1) @ do  # [B,H,L,D]

    dP = do @ fq_v.transpose(-2, -1)  # [B,H,L,L]

    do_dot_o = (do * high_prec_o).sum(dim=-1, keepdim=True)  # [B,H,L,1]
    dS = p * (dP - do_dot_o)  # [B,H,L,L]


    dQ = (dS @ fq_k) / torch.sqrt(torch.tensor(D, device=fq_q.device, dtype=fq_q.dtype))            # [B,H,L,D]
    dK = (dS.transpose(-2, -1) @ fq_q) / torch.sqrt(torch.tensor(D, device=fq_q.device, dtype=fq_q.dtype))  # [B,H,L,D]

    # grads w.r.t inputs, None for non-tensor inputs
    return dQ.transpose(1, 2).contiguous(), dK.transpose(1, 2).contiguous(), dV.transpose(1, 2).contiguous(), None, None, None

USE_SMOOTHING = False
def sage_attn_qk_torch(q, k, per_block_mean=False):
    """
    Inputs:  q/k/v in [B, H, L, D]
    Returns: out in  [B, H, L, D]
    """
    B, H, qL, D = q.shape
    B, H, kL, D = k.shape
    assert B == 1
    q, k = q.squeeze(0), k.squeeze(0)
    if USE_SMOOTHING:
        qk = _Matmul3d4bitFWD16bitBWD_with_smoothing.apply(q, k)
    else:
        qk = _Matmul3d4bitFWD16bitBWD.apply(q, k)
    qk = qk / torch.sqrt(torch.tensor(D, device=q.device, dtype=q.dtype))
    p = torch.softmax(qk, dim=-1)
    return p.unsqueeze(0)

def attn_backward_16bit(q_BLHD, k_BLHD, v_BLHD, out_BLHD, grad_out_BLHD, is_causal=False):
    assert is_causal == False
    q, k, v, o, do = q_BLHD.transpose(1 ,2).contiguous(), k_BLHD.transpose(1 ,2).contiguous(), v_BLHD.transpose(1 ,2).contiguous(), out_BLHD.transpose(1 ,2).contiguous(), grad_out_BLHD.transpose(1 ,2).contiguous()
    B, H, _, D = q.shape
    p = sage_attn_qk_torch(q, k, per_block_mean=False).to(do.dtype)
    dV = p.transpose(-2, -1) @ do  # [B,H,L,D]

    dP = do @ v.transpose(-2, -1)  # [B,H,L,L]

    # do_dot_o = (do.float() * o.float()).sum(dim=-1, keepdim=True).to(dP.dtype)  # [B,H,L,1]
    # do_dot_o = (do * o).sum(dim=-1, keepdim=True)  # [B,H,L,1]
    dp_dot_p = (dP * p).sum(dim=-1, keepdim=True)  # [B,H,L,1]
    dS = p * (dP - dp_dot_p)  # [B,H,L,L]


    dQ = (dS @ k) / torch.sqrt(torch.tensor(D, device=q.device, dtype=q.dtype))            # [B,H,L,D]
    dK = (dS.transpose(-2, -1) @ q) / torch.sqrt(torch.tensor(D, device=q.device, dtype=q.dtype))  # [B,H,L,D]

    # grads w.r.t inputs, None for non-tensor inputs
    return dQ.transpose(1, 2).contiguous(), dK.transpose(1, 2).contiguous(), dV.transpose(1, 2).contiguous(), None, None, None
    
def attn_forward_4bit(q_BHLD, k_BHLD, v_BHLD, is_causal=False):
    assert is_causal == False
    p_bf16 = sage_attn_qk_torch(q_BHLD, k_BHLD, per_block_mean=False).to(q_BHLD.dtype)
    row_max_p = p_bf16.max(dim=-1, keepdim=True).values
    scale_row = row_max_p / (448 * 6)
    p = p_bf16 / scale_row
    out = matmul_3d_4bit(p.squeeze(0), v_BHLD.squeeze(0).transpose(1, 2).contiguous()).unsqueeze(0)
    out = out * scale_row
    return out, p_bf16
    
    
def flash_attn_backward(q_BLHD, k_BLHD, v_BLHD, out_BLHD, grad_out_BLHD, is_causal=False):
    D = q_BLHD.shape[-1]
    softmax_scale = 1.0 / sqrt(D)

    # FA forward only to get softmax_lse (rng_state is also returned)
    _, softmax_lse, _S_dmask, rng_state = _wrapped_flash_attn_forward(
        q_BLHD, k_BLHD, v_BLHD,
        0.0,               # dropout_p
        softmax_scale,     # softmax_scale
        is_causal,         # causal
        -1, -1,            # window_size_left/right
        0.0,               # softcap
        None,              # alibi_slopes
        False              # return_softmax
    )

    # Allocate grads and call FA backward using the SAGE output
    dq_BLHD = torch.empty_like(q_BLHD)
    dk_BLHD = torch.empty_like(k_BLHD)
    dv_BLHD = torch.empty_like(v_BLHD)

    _wrapped_flash_attn_backward(
        grad_out_BLHD,          # dout (BLHD, 16-bit)
        q_BLHD, k_BLHD, v_BLHD,
        out_BLHD,    # use SAGE forward output here
        softmax_lse,
        dq_BLHD, dk_BLHD, dv_BLHD,
        0.0,               # dropout_p
        softmax_scale,     # softmax_scale
        is_causal,         # causal
        -1, -1,            # window_size_left/right
        0.0,               # softcap
        None,              # alibi_slopes
        False,             # deterministic
        rng_state=rng_state,
    )
    return dq_BLHD, dk_BLHD, dv_BLHD, None, None

USE_TORCH_4BIT_FWD = False
USE_TORCH_16BIT_BWD = False
USE_QAT_ATTN = False
class _SageAttnBlackwellWith16bitBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_BLHD, k_BLHD, v_BLHD, is_causal=False, per_block_mean=False):
        """
        Inputs:  q/k/v in [B, L, H, D]
        Returns: out in  [B, L, H, D]
        """
        # Save originals for backward (we'll recompute FA in 16-bit there)
        ctx.is_causal = bool(is_causal)

        # Convert to BHLD for your sageattn forward
        q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
        k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
        v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()

        with torch.no_grad():
            if USE_QAT_ATTN:
                fq_q, fq_k, fq_v = preprocess(q_BHLD, k_BHLD, v_BHLD)
                # Compute attention probabilities and output
                p = qat_attn_qk_torch(fq_q, fq_k)  # [B, H, L, L] where B=1
                fq_p = get_fake_quant(p)  # [B, H, L, L] - preserves shape
                out_BHLD = fq_p @ fq_v
                high_prec_o = p @ fq_v
                # Save fake quantized tensors and high precision output for backward
                ctx.save_for_backward(fq_q, fq_k, fq_v, high_prec_o)
                out_BLHD = out_BHLD.permute(0, 2, 1, 3).contiguous()
            elif USE_TORCH_4BIT_FWD:
                out_BHLD, p = attn_forward_4bit(q_BHLD, k_BHLD, v_BHLD, is_causal)
                # Convert to BLHD for saving
                out_BLHD = out_BHLD.permute(0, 2, 1, 3).contiguous()
                ctx.save_for_backward(q_BLHD, k_BLHD, v_BLHD, out_BLHD)
            else:
                out_BHLD = sageattn_blackwell(
                    q_BHLD, k_BHLD, v_BHLD,
                    attn_mask=None,
                    is_causal=is_causal,
                    per_block_mean=per_block_mean,
                )
                # Convert to BLHD for saving
                out_BLHD = out_BHLD.permute(0, 2, 1, 3).contiguous()
                ctx.save_for_backward(q_BLHD, k_BLHD, v_BLHD, out_BLHD)

        return out_BLHD

    @staticmethod
    def backward(ctx, grad_out_BLHD):
        is_causal = ctx.is_causal
        if USE_QAT_ATTN:
            fq_q, fq_k, fq_v, high_prec_o = ctx.saved_tensors
            # Convert from BHLD (saved format) to BLHD (expected by qat_attn_backward_16bit)
            fq_q_BLHD = fq_q.permute(0, 2, 1, 3).contiguous()
            fq_k_BLHD = fq_k.permute(0, 2, 1, 3).contiguous()
            fq_v_BLHD = fq_v.permute(0, 2, 1, 3).contiguous()
            high_prec_o_BLHD = high_prec_o.permute(0, 2, 1, 3).contiguous()
            # qat_attn_backward_16bit expects BLHD format and returns BLHD format
            grad_q_BLHD, grad_k_BLHD, grad_v_BLHD, _, _, _ = qat_attn_backward_16bit(
                fq_q_BLHD, fq_k_BLHD, fq_v_BLHD, high_prec_o_BLHD, grad_out_BLHD, is_causal
            )
            return grad_q_BLHD, grad_k_BLHD, grad_v_BLHD, None, None
        else:
            q_BLHD, k_BLHD, v_BLHD, out_BLHD = ctx.saved_tensors
            if USE_TORCH_16BIT_BWD:
                return attn_backward_16bit(q_BLHD, k_BLHD, v_BLHD, out_BLHD, grad_out_BLHD, is_causal)
            else:
                return flash_attn_backward(q_BLHD, k_BLHD, v_BLHD, out_BLHD, grad_out_BLHD, is_causal)



def sageattn_blackwell_with_16bit_bwd(q_BLHD, k_BLHD, v_BLHD, is_causal=False, per_block_mean=False):
    """
    Forward: uses sageattn_blackwell under the hood.
    Backward: recomputes FlashAttention fwd+bwd in 16bit directly.
    """
    return _SageAttnBlackwellWith16bitBwd.apply(q_BLHD, k_BLHD, v_BLHD, is_causal, per_block_mean)

class _SageAttnBlackwellWithTritonBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_BHLD, k_BHLD, v_BHLD, softmax_scale, softmax_lse, is_causal=False, per_block_mean=True, use_global_sf=True):
        k_mean = k_BHLD.mean(dim=(0, 1, 2), keepdim=True).view(-1)

        out_BHLD = sageattn_blackwell(
            q_BHLD, k_BHLD, v_BHLD,
            attn_mask=None,
            is_causal=is_causal,
            per_block_mean=per_block_mean,
        )
        
        ctx.HEAD_DIM = q_BHLD.shape[-1]
        ctx.sm_scale = softmax_scale  # Use sm_scale to match backward access
        ctx.causal = is_causal
        ctx.IS_QAT = True
        ctx.k_mean = k_mean
        ctx.use_global_sf = use_global_sf
        ctx.save_for_backward(q_BHLD, k_BHLD, v_BHLD, out_BHLD, softmax_lse)
        return out_BHLD

    @staticmethod
    def backward(ctx, do):
        # note this assumes that we do everything in algorithm 1 from SageAttention3 except adding delta s
        q_BHLD, k_BHLD, v_BHLD, o_for_bwd_BHLD, M_BHLD = ctx.saved_tensors
        do = do.contiguous()
        dq_BHLD = torch.empty_like(q_BHLD)
        dk_BHLD = torch.empty_like(k_BHLD)
        dv_BHLD = torch.empty_like(v_BHLD)
        BATCH, N_HEAD, N_CTX_Q = q_BHLD.shape[:3]
        N_CTX_KV = k_BHLD.shape[2]
        assert k_BHLD.shape[2] == v_BHLD.shape[2], "k and v must have the same sequence length"
        PRE_BLOCK = 128
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 64, 64, 64
        BLK_SLICE_FACTOR = 1
        pre_grid = ((N_CTX_Q + PRE_BLOCK - 1) // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M_BHLD)
        _attn_bwd_preprocess[pre_grid](
            o_for_bwd_BHLD, do,
            delta,
            BATCH, N_HEAD, N_CTX_Q,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )

        fake_q = torch.empty_like(q_BHLD)
        fake_k = torch.empty_like(k_BHLD)
        fake_v = torch.empty_like(v_BHLD)

        grid_1 = (triton.cdiv(q_BHLD.shape[2], BLOCK_M1), q_BHLD.shape[0] * q_BHLD.shape[1], 1)
        grid_2 = (triton.cdiv(k_BHLD.shape[2], BLOCK_N2), q_BHLD.shape[0] * q_BHLD.shape[1], 1)

        fake_quantize_q[grid_1](
            q_BHLD, fake_q,
            q_BHLD.stride(0), q_BHLD.stride(1),
            q_BHLD.stride(2), q_BHLD.stride(3),
            fake_q.stride(0), fake_q.stride(1),
            fake_q.stride(2), fake_q.stride(3),
            N_HEAD, N_CTX_Q,
            BLOCK_M=BLOCK_M1, HEAD_DIM=ctx.HEAD_DIM, use_global_sf=ctx.use_global_sf
        )

        fake_quantize_kv[grid_2](
            k_BHLD, v_BHLD, fake_k, fake_v,
            k_BHLD.stride(0), k_BHLD.stride(1),
            k_BHLD.stride(2), k_BHLD.stride(3),
            fake_k.stride(0), fake_k.stride(1),
            fake_k.stride(2), fake_k.stride(3),
            N_HEAD, N_CTX_KV,
            BLOCK_N=BLOCK_N2, HEAD_DIM=ctx.HEAD_DIM, use_global_sf=ctx.use_global_sf
        )

        # Use fake quantized tensors for gradient computation (QAT)
        # NOTE: The reassignment below only changes local variable references.
        # The gradients (dq_BHLD, dk_BHLD, dv_BHLD) computed using fake_q/fake_k/fake_v
        # will still correctly backpropagate to the original q_BHLD/k_BHLD/v_BHLD tensors
        # from ctx.saved_tensors, because those original tensors remain in the computation graph.
        q_BHLD = fake_q
        k_BHLD = fake_k
        v_BHLD = fake_v

        # NOTE: K is NOT pre-scaled here - scaling is applied AFTER qk dot product in kernels
        # This improves precision by avoiding rounding errors in K before the dot product
        arg_k = k_BHLD

        if N_CTX_Q == N_CTX_KV:
            # Use existing kernel for self-attention (same sequence lengths)
            grid = ((N_CTX_KV + BLOCK_N1 - 1) // BLOCK_N1, 1, BATCH * N_HEAD)
            # Q_MEAN is required by the kernel signature but only used if SMOOTH_Q=True
            # Since we're not using SMOOTH_Q, create a dummy tensor with the same shape as q_BHLD
            q_mean = torch.empty_like(q_BHLD)
            _attn_bwd[grid](
                q_BHLD, arg_k, v_BHLD, ctx.sm_scale, do, dq_BHLD, dk_BHLD, dv_BHLD,
                M_BHLD, delta, q_mean,
                q_BHLD.stride(0), q_BHLD.stride(1), q_BHLD.stride(2), q_BHLD.stride(3),
                N_HEAD, N_CTX_KV,
                ctx.k_mean,
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
                BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
                HEAD_DIM=ctx.HEAD_DIM,
                CAUSAL=ctx.causal,
                IS_QAT=ctx.IS_QAT,
                SMOOTH_K=True,
                two_level_quant_P=True,
                fake_quant_P=True,
                use_global_sf=ctx.use_global_sf,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES
            )
        else:
            # Use separate kernels for cross-attention (different sequence lengths)
            grid_dq = ((N_CTX_Q + BLOCK_M2 - 1) // BLOCK_M2, 1, BATCH * N_HEAD)
            _attn_bwd_dq_cross[grid_dq](
                q_BHLD, arg_k, v_BHLD, ctx.sm_scale, do, dq_BHLD, M_BHLD, delta,
                q_BHLD.stride(0), k_BHLD.stride(0), q_BHLD.stride(1), k_BHLD.stride(1), q_BHLD.stride(2), k_BHLD.stride(2), q_BHLD.stride(3), k_BHLD.stride(3),
                N_HEAD, N_CTX_Q, N_CTX_KV,
                ctx.k_mean,
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
                HEAD_DIM=ctx.HEAD_DIM,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
                SMOOTH_K=True,
            )
            grid_dkdv = ((N_CTX_KV + BLOCK_N1 - 1) // BLOCK_N1, 1, BATCH * N_HEAD)
            # Q_MEAN is required by the kernel signature but only used if SMOOTH_Q=True
            # Since we're not using SMOOTH_Q, create a dummy tensor with the same shape as q_BHLD
            q_mean = torch.empty_like(q_BHLD)
            _attn_bwd_dkdv_cross[grid_dkdv](
                q_BHLD, arg_k, v_BHLD, ctx.sm_scale, do, dk_BHLD, dv_BHLD, M_BHLD, delta, q_mean,
                q_BHLD.stride(0), k_BHLD.stride(0), q_BHLD.stride(1), k_BHLD.stride(1), q_BHLD.stride(2), k_BHLD.stride(2), q_BHLD.stride(3), k_BHLD.stride(3),
                N_HEAD, N_CTX_Q, N_CTX_KV,
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
                HEAD_DIM=ctx.HEAD_DIM,
                IS_QAT=ctx.IS_QAT,
                two_level_quant_P=True,
                fake_quant_P=True,
                use_global_sf=ctx.use_global_sf,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES
            )
        
        return dq_BHLD, dk_BHLD, dv_BHLD, None, None, None, None, None 

def sageattn_blackwell_with_triton_bwd(q_BLHD, k_BLHD, v_BLHD, is_causal=False, per_block_mean=True, use_global_sf=False):
    softmax_scale = 1.0 / sqrt(q_BLHD.shape[-1])
    # FA forward only to get softmax_lse (which is shape BHL)
    # Use no_grad to prevent autograd from tracking this call
    with torch.no_grad():
        _, softmax_lse, _S_dmask, rng_state = _wrapped_flash_attn_forward(
            q_BLHD, k_BLHD, v_BLHD,
            0.0,               # dropout_p
            softmax_scale,     # softmax_scale
            is_causal,         # causal
            -1, -1,            # window_size_left/right
            0.0,               # softcap
            None,              # alibi_slopes
            False              # return_softmax
        )
    
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_BHLD = _SageAttnBlackwellWithTritonBwd.apply(q_BHLD, k_BHLD, v_BHLD, softmax_scale, softmax_lse, is_causal, per_block_mean, use_global_sf)
    return out_BHLD.permute(0, 2, 1, 3).contiguous()

def qat_attn(q_BLHD, k_BLHD, v_BLHD, is_causal=False, use_global_sf=True):
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    o_BHLD = attention(q_BHLD, k_BHLD, v_BHLD, is_causal, 1.0 / sqrt(q_BLHD.shape[-1]), use_global_sf=use_global_sf)
    return o_BHLD.permute(0, 2, 1, 3).contiguous()


class SageAttention3Backend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_THREE"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SageAttention3Impl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # output = sageattn_blackwell_with_16bit_bwd(query, key, value, is_causal=self.causal)
        output = sageattn_blackwell_with_triton_bwd(query, key, value, is_causal=self.causal)
        # output = attn_forward_4bit_fwd_16bit_bwd(query, key, value)
        # output = qat_attn(query, key, value, is_causal=self.causal)
        return output
