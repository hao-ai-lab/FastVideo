# SPDX-License-Identifier: Apache-2.0

from curses import KEY_B2
import torch
try:
    from sageattn import sageattn_blackwell
except ImportError:
    sageattn_blackwell = None

from fastvideo.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

from math import sqrt
try:
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward, _wrapped_flash_attn_backward
except ImportError:
    _wrapped_flash_attn_forward = None
    _wrapped_flash_attn_backward = None

logger = init_logger(__name__)
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

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


def sage_attn_qk_torch(q, k, per_block_mean=False):
    """
    Inputs:  q/k/v in [B, H, L, D]
    Returns: out in  [B, H, L, D]
    """
    B, H, qL, D = q.shape
    B, H, kL, D = k.shape
    assert B == 1
    q, k = q.squeeze(0), k.squeeze(0)
    if True:
        qk = _Matmul3d4bitFWD16bitBWD_with_smoothing.apply(q, k)
    else:
        k = k - k.mean(dim=-2, keepdim=True)  # [H, L, D]
        qm = q.mean(dim=-2, keepdim=True)  # [H, 1, D]
        q = q - qm  # [B, H, L, D]
        delta_s = torch.matmul(qm, k.transpose(-2, -1)).contiguous()  # [H, 1, L]
        qk = _Matmul3d4bitFWD16bitBWD.apply(q, k)
        qk = qk + delta_s 
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

USE_TORCH_4BIT_FWD = True
USE_TORCH_16BIT_BWD = True
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
            if USE_TORCH_4BIT_FWD:
                out_BHLD, p = attn_forward_4bit(q_BHLD, k_BHLD, v_BHLD, is_causal)
            else:
                out_BHLD = sageattn_blackwell(
                    q_BHLD, k_BHLD, v_BHLD,
                    attn_mask=None,
                    is_causal=is_causal,
                    per_block_mean=per_block_mean,
                )

        # Back to BLHD; keep for FA bwd
        out_BLHD = out_BHLD.permute(0, 2, 1, 3).contiguous()
        ctx.save_for_backward(q_BLHD, k_BLHD, v_BLHD, out_BLHD)

        return out_BLHD

    @staticmethod
    def backward(ctx, grad_out_BLHD):
        q_BLHD, k_BLHD, v_BLHD, out_BLHD = ctx.saved_tensors
        is_causal = ctx.is_causal

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
        output = sageattn_blackwell_with_16bit_bwd(query, key, value, is_causal=self.causal)
        # output = attn_forward_4bit_fwd_16bit_bwd(query, key, value)
        return output
