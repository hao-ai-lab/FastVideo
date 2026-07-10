# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .quant_utils import fake_quantize_q, fake_quantize_kv, fake_quantize
# from attn_qat_infer.api import triton_group_mean


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _mul_alpha(acc, alpha, BM: tl.constexpr, BN: tl.constexpr):
    acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
    acc0 = acc0 * alpha[:, None]
    acc1 = acc1 * alpha[:, None]
    acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
    return acc


@triton.jit
def _attn_fwd_inner(acc, high_prec_acc, l_i, m_i, q, q_valid,
                    desc_k, desc_v,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr,
                    IS_QAT: tl.constexpr,
                    fake_quant_P: tl.constexpr = True,
                    two_level_quant_P: tl.constexpr = False,
                    use_global_sf_P: tl.constexpr = True):
    # range of values handled by this stage (kv blocks)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo  # offset from the start of the current batch-head combination
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_valid = (start_n + offs_n) < N_CTX
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0])
        k = tl.where(kv_valid[:, None], k, 0.0)
        qk = tl.dot(q, tl.trans(k))
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask & kv_valid[None, :], 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = tl.where(kv_valid[None, :], qk, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        if IS_QAT:  
            p, high_prec_p = fake_quantize(
                src_tensor=p,
                valid_src_mask=tl.full(shape=p.shape, value=1.0, dtype=p.dtype) == 1.0,
                BLOCK_SIZE_OUT_DIM=BLOCK_M,
                BLOCK_SIZE_QUANT_DIM=BLOCK_N,
                dst_dtype=dtype,
                two_level_quant_P=two_level_quant_P,
                use_global_sf=use_global_sf_P
            )
            l_ij = tl.sum(high_prec_p, 1)
        else:
            l_ij = tl.sum(p, 1)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc = _mul_alpha(acc, alpha, BM, BN)
            if IS_QAT:
                high_prec_acc = _mul_alpha(high_prec_acc, alpha, BM, BN)
        else:
            acc = acc * alpha[:, None]
            if IS_QAT:
                high_prec_acc = high_prec_acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
            v = tl.where(kv_valid[:, None], v, 0.0)
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v.to(dtype), acc)
        if IS_QAT:
            high_prec_acc = tl.dot(high_prec_p, v, high_prec_acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    if IS_QAT:
        return acc, high_prec_acc, l_i, m_i
    else:
        return acc, acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [1, BLOCK_M, HEAD_DIM]
    if "desc_high_prec_o" in nargs and isinstance(nargs["desc_high_prec_o"], TensorDescriptor):
        nargs["desc_high_prec_o"].block_shape = [1, BLOCK_M, HEAD_DIM]


NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX_Q = kwargs["N_CTX_Q"]
    N_CTX_KV = kwargs["N_CTX_KV"]
    return [
        conf for conf in configs
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX_Q
        and conf.kwargs.get("BLOCK_N", 0) <= N_CTX_KV
        and conf.kwargs.get("BLOCK_N", 0) % conf.kwargs.get("BLOCK_M", 0) == 0
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# @triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX_Q", "N_CTX_KV", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#                  prune_configs_by={'early_config_prune': prune_invalid_configs}, cache_results=True)
@triton.jit
def _attn_fwd(sm_scale, M,
              Z, H, desc_q, desc_k, desc_v, desc_o, desc_high_prec_o, N_CTX_Q, N_CTX_KV,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              FP8_OUTPUT: tl.constexpr,
              STAGE: tl.constexpr,
              warp_specialize: tl.constexpr,
              IS_HOPPER: tl.constexpr,
              IS_QAT: tl.constexpr,
              fake_quant_P: tl.constexpr = True,
              two_level_quant_P: tl.constexpr = False,
              use_global_sf_P: tl.constexpr = True,
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.bfloat16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H  # since it's Z then H in the shape
    off_h = off_hz % H

    y_dim_q = Z * H * N_CTX_Q
    y_dim_kv = Z * H * N_CTX_KV
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim_q, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim_kv], strides=[N_CTX_KV, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim_kv, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim_kv, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(
        desc_o, shape=[Z * H, N_CTX_Q, HEAD_DIM],
        strides=[N_CTX_Q * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, BLOCK_M, HEAD_DIM]
    )
    if IS_QAT:
        desc_high_prec_o = _maybe_make_tensor_desc(
            desc_high_prec_o, shape=[Z * H, N_CTX_Q, HEAD_DIM],
            strides=[N_CTX_Q * HEAD_DIM, HEAD_DIM, 1],
            block_shape=[1, BLOCK_M, HEAD_DIM]
        )

    offset_y_q = off_z * (N_CTX_Q * H) + off_h * N_CTX_Q  # offset for query tensor
    offset_y_kv = off_z * (N_CTX_KV * H) + off_h * N_CTX_KV  # offset for key/value tensors
    qo_offset_y = offset_y_q + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # for kv blocks
    q_valid = offs_m < N_CTX_Q
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    if IS_QAT:
        high_prec_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    else:
        # dummy value
        high_prec_acc = acc
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])  # load from start of q block and start of the entire head dimension
    q = tl.where(q_valid[:, None], q, 0.0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, high_prec_acc, l_i, m_i = _attn_fwd_inner(
            acc, high_prec_acc, l_i, m_i, q, q_valid,
            desc_k, desc_v,
            offset_y_kv, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE, offs_m, offs_n, N_CTX_KV,
            warp_specialize, IS_HOPPER, IS_QAT, fake_quant_P, two_level_quant_P, use_global_sf_P
        )
    # stage 2: on-band
    if STAGE & 2:
        acc, high_prec_acc, l_i, m_i = _attn_fwd_inner(
            acc, high_prec_acc, l_i, m_i, q, q_valid,
            desc_k, desc_v,
            offset_y_kv, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, N_CTX_KV,
            warp_specialize, IS_HOPPER, IS_QAT, fake_quant_P, two_level_quant_P, use_global_sf_P
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    if IS_QAT:
        high_prec_acc = high_prec_acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=q_valid)
    desc_o.store([off_hz, start_m * BLOCK_M, 0], acc[None, :, :])
    if IS_QAT:
        desc_high_prec_o.store([off_hz, start_m * BLOCK_M, 0], high_prec_acc[None, :, :])


@triton.jit
def _attn_bwd_preprocess(O, DO,
                         Delta,
                         Z, H, N_CTX,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    valid = off_m < N_CTX
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=valid[:, None])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=valid[:, None]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=valid)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,
                   Q, k, v, qk_scale,
                   DO,
                   M, D, Q_MEAN,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,
                   H, N_CTX, BLOCK_M1: tl.constexpr,
                   BLOCK_N1: tl.constexpr,
                   HEAD_DIM: tl.constexpr,
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,
                   MASK: tl.constexpr,
                   IS_QAT: tl.constexpr,
                   two_level_quant_P: tl.constexpr = False,
                   fake_quant_P: tl.constexpr = True,
                   SMOOTH_Q: tl.constexpr = False,
                   use_global_sf_P: tl.constexpr = True,
                   warp_specialize: tl.constexpr = False):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    if SMOOTH_Q:
        q_m_ptrs = Q_MEAN + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        q_valid = offs_m < N_CTX
        q = tl.load(q_ptrs, mask=q_valid[:, None])
        if SMOOTH_Q:
            q_m = tl.load(q_m_ptrs, mask=q_valid[:, None])
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=q_valid)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        # Autoregressive masking - apply BEFORE exp2 to match forward pass behavior
        if MASK:
            mask = (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(mask, qk, -1.0e6)
        p = tl.math.exp2(qk - m[:, None])
        do = tl.load(do_ptrs, mask=q_valid[:, None])
        # Compute dV.
        p_quant = p
        if IS_QAT and fake_quant_P:
            p_quant, _ = fake_quantize(
                src_tensor=p,
                valid_src_mask=tl.full(shape=p.shape, value=1.0, dtype=p.dtype) == 1.0,
                BLOCK_SIZE_OUT_DIM=BLOCK_M1,
                BLOCK_SIZE_QUANT_DIM=BLOCK_N1,
                dst_dtype=p.dtype,
                two_level_quant_P=two_level_quant_P,
                use_global_sf=use_global_sf_P
            )
        dv += tl.dot(tl.trans(p_quant.to(tl.bfloat16)), do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m, mask=q_valid)
        # Compute dP and dS.
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        dk += tl.dot(tl.trans(ds), q)
        if SMOOTH_Q:
            dk += tl.sum(ds, axis=1, keep_dims=True) * q_m
        # Increment pointers.
        curr_m += step_m
        q_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        if SMOOTH_Q:
            q_m_ptrs += step_m * stride_tok
    return dk, dv


# The main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,
                 do, m, D, qk_scale,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,
                 H, N_CTX,
                 K_MEAN,
                 BLOCK_M2: tl.constexpr,
                 BLOCK_N2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,
                 MASK: tl.constexpr,
                 SMOOTH_K: tl.constexpr,
                 warp_specialize: tl.constexpr = False):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    k_ptrs = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptrs = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    q_valid = offs_m < N_CTX
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=q_valid)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2

    if SMOOTH_K:
        k_m = tl.load(K_MEAN + offs_k)

    for blk_idx in range(num_steps):
        # bounds checking for kv block (dynamic)
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        kv_valid = offs_n < N_CTX
        k = tl.load(k_ptrs, mask=kv_valid[:, None])
        v = tl.load(v_ptrs, mask=kv_valid[:, None])

        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        # Autoregressive masking - apply BEFORE exp2 to match forward pass behavior
        if MASK:
            mask = (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(mask, qk, -1.0e6)
        p = tl.math.exp2(qk - m)
        # Compute dP and dS.
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because k was pre-scaled.
        dq += tl.dot(ds.to(tl.bfloat16), k)

        if SMOOTH_K:
            dq += tl.sum(ds, axis=1, keep_dims=True) * k_m[None, :]
        # Increment pointers.
        curr_n += step_n
        k_ptrs += step_n * stride_tok
        v_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _compute_cross_attn_pointer_offsets(bhid, H, N_CTX_Q, stride_z_q, stride_z_kv, stride_h_q, stride_h_kv):
    """Helper function to compute pointer offsets for cross-attention backward kernels."""
    off_chz = (bhid * N_CTX_Q).to(tl.int64)
    adj_q = (stride_h_q * (bhid % H) + stride_z_q * (bhid // H))
    adj_kv = (stride_h_kv * (bhid % H) + stride_z_kv * (bhid // H))
    return off_chz, adj_q, adj_kv


@triton.jit
def _attn_bwd_dq_cross(Q, K, V, sm_scale,
                       DO, DQ,
                       M, D,
                       stride_z_q, stride_z_kv, stride_h_q,
                       stride_h_kv, stride_tok_q, stride_tok_kv, stride_d_q, stride_d_kv,
                       H, N_CTX_Q, N_CTX_KV,
                       K_MEAN,
                       BLOCK_M2: tl.constexpr,
                       BLOCK_N2: tl.constexpr,
                       HEAD_DIM: tl.constexpr,
                       SMOOTH_K: tl.constexpr,
                       warp_specialize: tl.constexpr = False):
    # Apply scale AFTER dot product for better precision
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    qk_scale = sm_scale * RCP_LN2

    bhid = tl.program_id(2)
    off_chz, adj_q, adj_kv = _compute_cross_attn_pointer_offsets(bhid, H, N_CTX_Q, stride_z_q, stride_z_kv, stride_h_q, stride_h_kv)

    Q += adj_q
    K += adj_kv
    V += adj_kv
    DO += adj_q
    DQ += adj_q
    M += off_chz
    D += off_chz

    pid = tl.program_id(0)
    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM)

    q_valid = offs_m < N_CTX_Q
    q = tl.load(Q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None])
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None])
    m = tl.load(M + offs_m, mask=q_valid)[:, None]

    Di = tl.load(D + offs_m, mask=q_valid)
    num_steps = (N_CTX_KV + BLOCK_N2 - 1) // BLOCK_N2
    if SMOOTH_K:
        k_m = tl.load(K_MEAN + offs_k)
    for step in range(num_steps):
        start_n = step * BLOCK_N2
        offs_n = start_n + tl.arange(0, BLOCK_N2)
        kv_valid = offs_n < N_CTX_KV

        k = tl.load(K + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None])
        v = tl.load(V + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None])

        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - m)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        dq += tl.dot(ds, k)

        if SMOOTH_K:
            dq += tl.sum(ds, axis=1, keep_dims=True) * k_m[None, :]

    # NOTE: dq is scaled by sm_scale since K is not pre-scaled
    dq *= sm_scale
    dq_ptrs = DQ + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q
    tl.store(dq_ptrs, dq, mask=q_valid[:, None])


@triton.jit
def _attn_bwd_dkdv_cross(Q, K, V, sm_scale,
                         DO, DK, DV,
                         M, D, Q_MEAN,
                         stride_z_q, stride_z_kv, stride_h_q, stride_h_kv, 
                         stride_tok_q, stride_tok_kv, stride_d_q, stride_d_kv,
                         H, N_CTX_Q, N_CTX_KV,
                         BLOCK_M1: tl.constexpr,
                         BLOCK_N1: tl.constexpr,
                         HEAD_DIM: tl.constexpr,
                         IS_QAT: tl.constexpr,
                         two_level_quant_P: tl.constexpr = False,
                         fake_quant_P: tl.constexpr = True,
                         SMOOTH_Q: tl.constexpr = False,
                         use_global_sf_P: tl.constexpr = True,
                         warp_specialize: tl.constexpr = False
                         ):
    # Apply scale AFTER dot product for better precision
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    qk_scale = sm_scale * RCP_LN2

    bhid = tl.program_id(2)
    off_chz, adj_q, adj_kv = _compute_cross_attn_pointer_offsets(bhid, H, N_CTX_Q, stride_z_q, stride_z_kv, stride_h_q, stride_h_kv)

    Q += adj_q
    K += adj_kv
    V += adj_kv
    DO += adj_q
    DK += adj_kv
    DV += adj_kv
    M += off_chz
    D += off_chz

    if SMOOTH_Q:
        Q_MEAN += adj_q

    pid = tl.program_id(0)
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    kv_valid = offs_n < N_CTX_KV
    k_block = tl.load(K + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None])
    v_block = tl.load(V + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None])

    num_q_steps = (N_CTX_Q + BLOCK_M1 - 1) // BLOCK_M1
    for step in range(num_q_steps):
        start_m = step * BLOCK_M1
        offs_m = start_m + tl.arange(0, BLOCK_M1)
        q_valid = offs_m < N_CTX_Q

        q = tl.load(Q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None])
        do = tl.load(DO + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None])
        m = tl.load(M + offs_m, mask=q_valid)

        qk = tl.dot(q, tl.trans(k_block))
        # Apply scale AFTER dot product (matches forward pass, better precision)
        qk = qk * qk_scale
        p = tl.math.exp2(qk - m[:, None])
        p_quant = p
        if IS_QAT and fake_quant_P:
            p_quant, _ = fake_quantize(
                src_tensor=p,
                valid_src_mask=tl.full(shape=p.shape, value=1.0, dtype=p.dtype) == 1.0,
                BLOCK_SIZE_OUT_DIM=BLOCK_M1,
                BLOCK_SIZE_QUANT_DIM=BLOCK_N1,
                dst_dtype=p.dtype,
                two_level_quant_P=two_level_quant_P,
                use_global_sf=use_global_sf_P
            )
        dv += tl.dot(tl.trans(p_quant.to(tl.bfloat16)), do)

        dp = tl.dot(do, tl.trans(v_block))
        Di = tl.load(D + offs_m, mask=q_valid)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        dk += tl.dot(tl.trans(ds), q)

        if SMOOTH_Q:
            q_m = tl.load(Q_MEAN + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None])
            dk += tl.sum(ds, axis=1, keep_dims=True) * q_m

    dv_ptrs = DV + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv
    tl.store(dv_ptrs, dv, mask=kv_valid[:, None])

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv
    tl.store(dk_ptrs, dk, mask=kv_valid[:, None])


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,
              DO,
              DQ, DK, DV,
              M, D, Q_MEAN,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,
              H, N_CTX,
              K_MEAN,
              BLOCK_M1: tl.constexpr,
              BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr,
              BLOCK_N2: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              CAUSAL: tl.constexpr,
              IS_QAT: tl.constexpr,
              SMOOTH_K: tl.constexpr,
              two_level_quant_P: tl.constexpr = False,
              fake_quant_P: tl.constexpr = True,
              SMOOTH_Q: tl.constexpr = False,
              use_global_sf_P: tl.constexpr = True,
              warp_specialize: tl.constexpr = False):

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    if SMOOTH_Q:
        Q_MEAN += adj

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    kv_valid = offs_n < N_CTX

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=kv_valid[:, None])
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=kv_valid[:, None])

    # qk_scale is expressed in base-2 domain because we use exp2/log2.
    # Two equivalent ways to form qk_scale:
    # - **Post-scale (default)**: qk = dot(q, k^T) * (sm_scale / ln2)
    # - **Pre-scale (Flash-aligned)**: k is pre-multiplied by (sm_scale / ln2) and qk_scale = 1
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    qk_scale = sm_scale * RCP_LN2

    # For causal attention, process diagonal block with masking, then rest without
    # For non-causal attention, process all blocks without masking
    if CAUSAL:
        num_steps = BLOCK_N1 // MASK_BLOCK_M1

        dk, dv = _attn_bwd_dkdv(
            dk, dv,
            Q, k, v, qk_scale,
            DO,
            M, D, Q_MEAN,
            stride_tok, stride_d,
            H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps,
            MASK=True,
            IS_QAT=IS_QAT,
            two_level_quant_P=two_level_quant_P,
            fake_quant_P=fake_quant_P,
            SMOOTH_Q=SMOOTH_Q,
            use_global_sf_P=use_global_sf_P,
            warp_specialize=warp_specialize
        )

        start_m += num_steps * MASK_BLOCK_M1
        num_steps = (N_CTX - start_m + BLOCK_M1 - 1) // BLOCK_M1
    else:
        # For non-causal, start from 0 and process all Q blocks
        start_m = 0
        num_steps = (N_CTX + BLOCK_M1 - 1) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, qk_scale,
        DO,
        M, D, Q_MEAN,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        MASK=False,
        IS_QAT=IS_QAT,
        two_level_quant_P=two_level_quant_P,
        fake_quant_P=fake_quant_P,
        SMOOTH_Q=SMOOTH_Q,
        use_global_sf_P=use_global_sf_P,
        warp_specialize=warp_specialize
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv, mask=kv_valid[:, None])

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk, mask=kv_valid[:, None])

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q_valid = offs_m < N_CTX

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=q_valid[:, None])

    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=q_valid[:, None])

    m = tl.load(M + offs_m, mask=q_valid)[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    if CAUSAL:
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q, K, V,
            do, m, D, qk_scale,
            stride_tok, stride_d,
            H, N_CTX,
            K_MEAN,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
            MASK=True,
            SMOOTH_K=SMOOTH_K,
            warp_specialize=warp_specialize,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2: process KV blocks from 0 to end_n (before diagonal), ceiling division to include remainder
        num_steps = (end_n + BLOCK_N2 - 1) // BLOCK_N2
        start_n = 0
    else:
        # For non-causal, process all KV blocks from 0 to N_CTX (ceiling division to include remainder)
        num_steps = (N_CTX + BLOCK_N2 - 1) // BLOCK_N2
        start_n = 0
    # stage 2
    dq = _attn_bwd_dq(
        dq, q, K, V,
        do, m, D, qk_scale,
        stride_tok, stride_d,
        H, N_CTX,
        K_MEAN,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps,
        MASK=False,
        SMOOTH_K=SMOOTH_K,
        warp_specialize=warp_specialize,
    )
    # Write back dQ.
    # NOTE: dq is scaled by sm_scale since K is not pre-scaled (unlike original which used LN2)
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq, mask=q_valid[:, None])


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        causal,
        sm_scale,
        use_qat_qkv_backward=True,
        smooth_k=True,
        warp_specialize=True,
        IS_QAT=True,
        two_level_quant_P=True,
        fake_quant_P=True,
        use_high_prec_o=False,
        smooth_q=False,
        use_global_sf_P=True,
        use_global_sf_QKV=True,
    ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # Support different sequence lengths for q and k/v (needed for cross attention)
        N_CTX_Q = q.shape[2]  # Query sequence length
        N_CTX_KV = k.shape[2]  # Key/Value sequence length (may differ from query)
        assert k.shape[2] == v.shape[2], "k and v must have the same sequence length"

        # smoothing k from SageAttn
        ctx.k_mean = None
        if smooth_k:
            k_mean = k.mean(dim=(0, 1, 2), keepdim=True)
            k = k - k_mean
            ctx.k_mean = k_mean.view(-1)
        
        q_orig, k_orig, v_orig = None, None, None
        if not use_qat_qkv_backward:
            q_orig, k_orig, v_orig = q, k, v
        ctx.q_orig, ctx.k_orig, ctx.v_orig = q_orig, k_orig, v_orig

        o = torch.empty_like(q)
        if IS_QAT:
            high_prec_o = torch.empty_like(q)
        else:
            # Initialize to a dummy value
            high_prec_o = o
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)  # (Z, H, N_CTX_Q)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim_q = q.shape[0] * q.shape[1] * q.shape[2]
            y_dim_kv = k.shape[0] * k.shape[1] * k.shape[2]

            dummy_block = [1, 1]
            # (Z, H, N_CTX_Q, HEAD_DIM_K) -> (Z*H*N_CTX_Q, HEAD_DIM_K)
            desc_q = TensorDescriptor(q, shape=[y_dim_q, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(
                    v, shape=[HEAD_DIM_K, y_dim_kv], strides=[k.shape[2], 1],
                    block_shape=dummy_block
                )
            else:
                desc_v = TensorDescriptor(
                    v, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block
                )
            desc_k = TensorDescriptor(k, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            # Use 3D descriptor for output to handle N_CTX_Q boundaries correctly
            dummy_block3d = [1, 1, 1]
            ZH = q.shape[0] * q.shape[1]
            desc_o = TensorDescriptor(
                o, shape=[ZH, N_CTX_Q, HEAD_DIM_K],
                strides=[N_CTX_Q * HEAD_DIM_K, HEAD_DIM_K, 1],
                block_shape=dummy_block3d
            )
            if IS_QAT:
                desc_high_prec_o = TensorDescriptor(
                    high_prec_o, shape=[ZH, N_CTX_Q, HEAD_DIM_K],
                    strides=[N_CTX_Q * HEAD_DIM_K, HEAD_DIM_K, 1],
                    block_shape=dummy_block3d
                )
            else:
                desc_high_prec_o = desc_o  # Use regular output descriptor when not in QAT mode
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o
            if IS_QAT:
                desc_high_prec_o = high_prec_o
            else:
                desc_high_prec_o = o  # Use regular output when not in QAT mode

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            # (ceil(N_CTX / BLOCK_M), Z * H, 1)
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        BLOCK_M, BLOCK_N = 32, 32
        if IS_QAT:
            fake_q = torch.empty_like(q)
            fake_k = torch.empty_like(k)
            fake_v = torch.empty_like(v)

            # override desc_q, desc_k, desc_v with fake_q, fake_k, fake_v
            if supports_host_descriptor() and not (is_hopper() and warp_specialize):
                # (Z, H, N_CTX_Q, HEAD_DIM_K) -> (Z*H*N_CTX_Q, HEAD_DIM_K)
                desc_q = TensorDescriptor(fake_q, shape=[y_dim_q, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_k = TensorDescriptor(fake_k, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_v = TensorDescriptor(fake_v, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            else:
                desc_q = fake_q
                desc_k = fake_k
                desc_v = fake_v

            H = q.shape[1]
            grid_1 = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
            grid_2 = (triton.cdiv(k.shape[2], BLOCK_N), q.shape[0] * q.shape[1], 1)

            fake_quantize_q[grid_1](
                q, fake_q,
                q.stride(0), q.stride(1),
                q.stride(2), q.stride(3),
                fake_q.stride(0), fake_q.stride(1),
                fake_q.stride(2), fake_q.stride(3),
                H, N_CTX_Q,
                BLOCK_M=BLOCK_M, HEAD_DIM=HEAD_DIM_K,
                use_global_sf=use_global_sf_QKV,
            )
            fake_quantize_kv[grid_2](
                k, v, fake_k, fake_v,
                k.stride(0), k.stride(1),
                k.stride(2), k.stride(3),
                fake_k.stride(0), fake_k.stride(1),
                fake_k.stride(2), fake_k.stride(3),
                H, N_CTX_KV,
                BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K, 
                use_global_sf=use_global_sf_QKV,
            )

        # Apply pre-hook to set block shapes on tensor descriptors
        _host_descriptor_pre_hook({
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "HEAD_DIM": HEAD_DIM_K,
            "desc_q": desc_q,
            "desc_k": desc_k,
            "desc_v": desc_v,
            "desc_o": desc_o,
            "desc_high_prec_o": desc_high_prec_o,
            "FP8_OUTPUT": q.dtype == torch.float8_e5m2,
        })

        _attn_fwd[grid](
            sm_scale, M,
            q.shape[0], q.shape[1],
            desc_q, desc_k, desc_v, desc_o, desc_high_prec_o,
            N_CTX_Q=N_CTX_Q,
            N_CTX_KV=N_CTX_KV,
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=is_hopper(),
            IS_QAT=IS_QAT,
            fake_quant_P=fake_quant_P,
            two_level_quant_P=two_level_quant_P,
            use_global_sf_P=use_global_sf_P,
            num_warps=4,
            num_stages=2,
            **extra_kern_args
        )
        o_for_bwd = high_prec_o if IS_QAT and use_high_prec_o else o

        if IS_QAT:
            q = fake_q
            k = fake_k
            v = fake_v

        ctx.save_for_backward(q, k, v, o_for_bwd, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.IS_QAT = IS_QAT
        ctx.use_qat_qkv_backward = use_qat_qkv_backward
        ctx.smooth_k = smooth_k
        ctx.two_level_quant_P = two_level_quant_P
        ctx.fake_quant_P = fake_quant_P
        ctx.smooth_q = smooth_q
        ctx.use_global_sf_P = use_global_sf_P
        ctx.warp_specialize = warp_specialize
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o_for_bwd, M = ctx.saved_tensors
        do = do.contiguous()
        assert do.is_contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX_Q = q.shape[:3]
        N_CTX_KV = k.shape[2]
        assert k.shape[2] == v.shape[2], "k and v must have the same sequence length"
        PRE_BLOCK = 128
        NUM_STAGES = 3
        NUM_WARPS = 4
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32
        if not ctx.use_qat_qkv_backward:
            q = ctx.q_orig
            k = ctx.k_orig
            v = ctx.v_orig
        BLK_SLICE_FACTOR = 1 if ctx.IS_QAT else 2  # must be 1 for QAT
        # NOTE: K is NOT pre-scaled here - scaling is applied AFTER qk dot product in kernels
        # This improves precision by avoiding rounding errors in K before the dot product
        arg_k = k
        pre_grid = ((N_CTX_Q + PRE_BLOCK - 1) // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o_for_bwd, do,
            delta,
            BATCH, N_HEAD, N_CTX_Q,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )

        q_m = None
        if ctx.smooth_q:
            # _, q_m = triton_group_mean(q)
            q_m = q_m.repeat_interleave(q.shape[2] // q_m.shape[2], dim=2)  # B,H,L,D

        if N_CTX_Q == N_CTX_KV:
            # Use existing kernel for self-attention (same sequence lengths)
            grid = ((N_CTX_KV + BLOCK_N1 - 1) // BLOCK_N1, 1, BATCH * N_HEAD)
            _attn_bwd[grid](
                q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,
                M, delta, q_m,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                N_HEAD, N_CTX_KV,
                ctx.k_mean,
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
                BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
                HEAD_DIM=ctx.HEAD_DIM,
                CAUSAL=ctx.causal,
                IS_QAT=ctx.IS_QAT,
                SMOOTH_K=ctx.smooth_k,
                two_level_quant_P=ctx.two_level_quant_P,
            fake_quant_P=ctx.fake_quant_P,
            SMOOTH_Q=ctx.smooth_q,
            use_global_sf_P=ctx.use_global_sf_P,
            warp_specialize=ctx.warp_specialize,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES
            )
        else:
            # Use separate kernels for cross-attention (different sequence lengths)
            grid_dq = ((N_CTX_Q + BLOCK_M2 - 1) // BLOCK_M2, 1, BATCH * N_HEAD)
            _attn_bwd_dq_cross[grid_dq](
                q, arg_k, v, ctx.sm_scale, do, dq, M, delta,
                q.stride(0), k.stride(0), q.stride(1), k.stride(1), q.stride(2), k.stride(2), q.stride(3), k.stride(3),
                N_HEAD, N_CTX_Q, N_CTX_KV,
                ctx.k_mean,
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
                HEAD_DIM=ctx.HEAD_DIM,
                SMOOTH_K=ctx.smooth_k,
                warp_specialize=ctx.warp_specialize,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            grid_dkdv = ((N_CTX_KV + BLOCK_N1 - 1) // BLOCK_N1, 1, BATCH * N_HEAD)
            _attn_bwd_dkdv_cross[grid_dkdv](
                q, arg_k, v, ctx.sm_scale, do, dk, dv, M, delta, q_m,
                q.stride(0), k.stride(0), q.stride(1), k.stride(1), q.stride(2), k.stride(2), q.stride(3), k.stride(3),
                N_HEAD, N_CTX_Q, N_CTX_KV,
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
                HEAD_DIM=ctx.HEAD_DIM,
                IS_QAT=ctx.IS_QAT,
                two_level_quant_P=ctx.two_level_quant_P,
                fake_quant_P=ctx.fake_quant_P,
                SMOOTH_Q=ctx.smooth_q,
                use_global_sf_P=ctx.use_global_sf_P,
                warp_specialize=ctx.warp_specialize,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


attention = _attention.apply
