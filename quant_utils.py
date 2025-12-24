import triton
import triton.language as tl
from nvfp4_utils import _compute_quant_and_scale, _compute_dequant

@triton.jit
def fake_quantize(src_tensor, valid_src_mask, BLOCK_SIZE_OUT_DIM: tl.constexpr, 
                    BLOCK_SIZE_QUANT_DIM: tl.constexpr, 
                    dst_dtype: tl.constexpr,
                    two_level_quant_P: tl.constexpr = False):
    high_prec_src_tensor = src_tensor
    src_tensor, src_scale, src_s_dec = _compute_quant_and_scale(src_tensor=src_tensor, 
                                                                valid_src_mask=valid_src_mask,
                                                                use_global_sf=True,
                                                                two_level_quant_P=two_level_quant_P)
    src_tensor = _compute_dequant(mx_tensor=src_tensor, 
                                  scale=src_scale, 
                                  s_dec=src_s_dec, 
                                  BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM, 
                                  BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM, 
                                  dst_dtype=dst_dtype)
    return src_tensor, high_prec_src_tensor.to(src_tensor.dtype)

@triton.jit
def fake_quantize_q(Q, fake_Q, stride_z_q, stride_h_q, 
                    stride_tok_q, stride_d_q,
                    fake_stride_z_q, fake_stride_h_q,
                    fake_stride_tok_q, fake_stride_d_q,
                    H, N_CTX_Q,
                    BLOCK_M: tl.constexpr,
                    HEAD_DIM: tl.constexpr):
    bhid = tl.program_id(1)
    adj_q = (stride_h_q * (bhid % H) + stride_z_q * (bhid // H))
    fake_adj_q = (fake_stride_h_q * (bhid % H) + fake_stride_z_q * (bhid // H))
    Q += adj_q
    fake_Q += fake_adj_q

    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    q_valid = offs_m < N_CTX_Q
    q = tl.load(Q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None], other=0.0)
    q, _ = fake_quantize(src_tensor=q, valid_src_mask=q_valid[:, None], BLOCK_SIZE_OUT_DIM=BLOCK_M, BLOCK_SIZE_QUANT_DIM=HEAD_DIM, dst_dtype=q.dtype)
    tl.store(fake_Q + offs_m[:, None] * fake_stride_tok_q + offs_k[None, :] * fake_stride_d_q, q, mask=q_valid[:, None])

@triton.jit
def fake_quantize_kv(K, V, fake_K, fake_V, stride_z_kv, stride_h_kv, 
                    stride_tok_kv, stride_d_kv,
                    fake_stride_z_kv, fake_stride_h_kv,
                    fake_stride_tok_kv, fake_stride_d_kv,
                    H, N_CTX_KV,
                    BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr):
    bhid = tl.program_id(1)
    adj_kv = (stride_h_kv * (bhid % H) + stride_z_kv * (bhid // H))
    fake_adj_kv = (fake_stride_h_kv * (bhid % H) + fake_stride_z_kv * (bhid // H))
    K += adj_kv
    V += adj_kv
    fake_K += fake_adj_kv
    fake_V += fake_adj_kv

    pid = tl.program_id(0)
    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    kv_valid = offs_n < N_CTX_KV
    k_block = tl.load(K + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None], other=0.0)
    v_block = tl.load(V + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None], other=0.0)
    k, _ = fake_quantize(src_tensor=k_block, valid_src_mask=kv_valid[:, None], BLOCK_SIZE_OUT_DIM=BLOCK_N, BLOCK_SIZE_QUANT_DIM=HEAD_DIM, dst_dtype=k_block.dtype)
    v, _ = fake_quantize(src_tensor=v_block, valid_src_mask=kv_valid[:, None], BLOCK_SIZE_OUT_DIM=BLOCK_N, BLOCK_SIZE_QUANT_DIM=HEAD_DIM, dst_dtype=v_block.dtype)
    tl.store(fake_K + offs_n[:, None] * fake_stride_tok_kv + offs_k[None, :] * fake_stride_d_kv, k, mask=kv_valid[:, None])
    tl.store(fake_V + offs_n[:, None] * fake_stride_tok_kv + offs_k[None, :] * fake_stride_d_kv, v, mask=kv_valid[:, None])
