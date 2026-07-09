# SPDX-License-Identifier: Apache-2.0
"""Fused block-causal attention with attention sinks, a rolling local window
and the relativistic sink RoPE correction.

Implements the full-sequence training attention of the causal Wan / RobotWM
DiTs (see ``fastvideo/models/dits/_causal_train_attention.py`` for the
semantics and the pure-PyTorch reference):

- ``blockwise`` layout: every query attends the sink frames plus the trailing
  rolling window up to its own frame-block end;
- ``teacher_forcing`` layout (``[clean | noisy]``): clean rows are blockwise
  over the clean half, noisy rows attend their own noisy block plus the
  windowed clean context of strictly-previous blocks.

The relativistic sink correction repositions sink keys per query block
(``delta = max(0, block_end - local_attn_size)`` frames), matching the
re-indexed RoPE phases of the streaming KV cache - something FlexAttention
cannot express. It is applied on the query side (``score =
<rope(k, f+delta), q> = <rope(k, f), R(-delta) q>``): the rotated queries are
precomputed once on the host and used for sink columns only, in both forward
and backward.

Everything is masked structurally: query tiles only iterate the sink /
window / self-block key ranges, and key tiles only iterate the query blocks
that can reach them, so fully-hidden tiles are never visited and no BlockMask
(or 128-padding) is needed.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only without triton
    HAS_TRITON = False

if HAS_TRITON:

    LOG2E = tl.constexpr(1.4426950408889634)

    # Autotuned tile configs. BN choices must divide 128 so the sink/window
    # boundary used by the split dK/dV backward (128-aligned) lands exactly on
    # a tile edge for every config.
    _FWD_CONFIGS = [
        triton.Config({
            "BM": 128,
            "BN": 64
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BM": 128,
            "BN": 128
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BM": 128,
            "BN": 64
        }, num_warps=8, num_stages=3),
        triton.Config({
            "BM": 64,
            "BN": 64
        }, num_warps=4, num_stages=3),
    ]
    _DQ_CONFIGS = [
        triton.Config({
            "BM": 64,
            "BN": 64
        }, num_warps=4, num_stages=2),
        triton.Config({
            "BM": 128,
            "BN": 64
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BM": 64,
            "BN": 128
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BM": 64,
            "BN": 64
        }, num_warps=8, num_stages=3),
    ]
    _DKV_CONFIGS = [
        triton.Config({
            "BM": 64,
            "BN": 64
        }, num_warps=4, num_stages=2),
        triton.Config({
            "BM": 128,
            "BN": 64
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BM": 64,
            "BN": 128
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BM": 64,
            "BN": 64
        }, num_warps=8, num_stages=3),
    ]
    # The constexpr flags must be part of the key: each (IS_TF, HAS_DELTA, ...)
    # specialization compiles to a different kernel with different shared
    # memory needs, so a config tuned for one may not even launch for another.
    _AUTOTUNE_KEY = ["L", "S_TOK", "NFB_TOK", "D", "IS_TF", "NO_WINDOW", "HAS_DELTA"]

    @triton.jit
    def _row_geometry(
        r,
        L,
        TOKENS_PER_HALF,
        NFB_TOK,
        ROLLING_TOK,
        IS_TF: tl.constexpr,
        NO_WINDOW: tl.constexpr,
    ):
        """Per-row (half, block, block-end, visible-end, window-start) in tokens."""
        r_c = tl.minimum(r, L - 1)
        if IS_TF:
            half = r_c // TOKENS_PER_HALF
            pos = r_c % TOKENS_PER_HALF
        else:
            half = tl.zeros_like(r_c)
            pos = r_c
        block = pos // NFB_TOK
        e_tok = (block + 1) * NFB_TOK
        # constexpr branches are pruned at trace time; keep them as statements
        if IS_TF:  # noqa: SIM108
            end_row = tl.where(half == 0, e_tok, e_tok - NFB_TOK)
        else:
            end_row = e_tok
        if NO_WINDOW:  # noqa: SIM108
            win_lo = tl.zeros_like(e_tok)
        else:
            win_lo = tl.maximum(e_tok - ROLLING_TOK, 0)
        return half, block, e_tok, end_row, win_lo

    @triton.jit
    def _rotate_grad_plus_delta(g, dc, ds, BM: tl.constexpr, D: tl.constexpr):
        """Transpose of the host-side R(-delta) query rotation (chain rule for
        dq): the forward maps pairs by [[dc_e, ds_e], [-ds_o, dc_o]].

        out[2i]   =  g[2i]   * dc[2i]   - g[2i+1] * ds[2i+1]
        out[2i+1] =  g[2i+1] * dc[2i+1] + g[2i]   * ds[2i]
        """
        g_p = tl.reshape(g, (BM, D // 2, 2))
        dc_p = tl.reshape(dc, (BM, D // 2, 2))
        ds_p = tl.reshape(ds, (BM, D // 2, 2))
        g_e, g_o = tl.split(g_p)
        dc_e, dc_o = tl.split(dc_p)
        ds_e, ds_o = tl.split(ds_p)
        out_e = g_e * dc_e - g_o * ds_o
        out_o = g_o * dc_o + g_e * ds_e
        return tl.reshape(tl.join(out_e, out_o), (BM, D))

    @triton.jit
    def _rotate_query_minus_delta(q, dc, ds, BM: tl.constexpr, D: tl.constexpr):
        """R(-delta) on roped queries, GPT-J interleaved pairs, full-dim tables.

        out[2i]   =  q[2i]   * dc[2i]   + q[2i+1] * ds[2i]
        out[2i+1] =  q[2i+1] * dc[2i+1] - q[2i]   * ds[2i+1]
        """
        q_p = tl.reshape(q, (BM, D // 2, 2))
        dc_p = tl.reshape(dc, (BM, D // 2, 2))
        ds_p = tl.reshape(ds, (BM, D // 2, 2))
        q_e, q_o = tl.split(q_p)
        dc_e, dc_o = tl.split(dc_p)
        ds_e, ds_o = tl.split(ds_p)
        out_e = q_e * dc_e + q_o * ds_e
        out_o = q_o * dc_o - q_e * ds_o
        return tl.reshape(tl.join(out_e, out_o), (BM, D))

    @triton.jit
    def _rotate_queries_kernel(
        Q,
        QS,
        DC,
        DS,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_dcb,
        stride_dcd,
        H,
        L,
        TOKENS_PER_HALF,
        NFB_TOK,
        BM: tl.constexpr,
        D: tl.constexpr,
    ):
        """One-pass builder of the sink-rotated queries (replaces the eager
        multi-kernel host chain)."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        r = pid_m * BM + tl.arange(0, BM)
        rmask = r < L
        d_offs = tl.arange(0, D)
        block = (tl.minimum(r, L - 1) % TOKENS_PER_HALF) // NFB_TOK
        q = tl.load(Q + b * stride_qb + h * stride_qh + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                    mask=rmask[:, None],
                    other=0.0)
        dc = tl.load(DC + block[:, None] * stride_dcb + d_offs[None, :] * stride_dcd, mask=rmask[:, None], other=1.0)
        ds = tl.load(DS + block[:, None] * stride_dcb + d_offs[None, :] * stride_dcd, mask=rmask[:, None], other=0.0)
        out = _rotate_query_minus_delta(q.to(tl.float32), dc, ds, BM, D)
        tl.store(QS + b * stride_qb + h * stride_qh + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                 out.to(QS.dtype.element_ty),
                 mask=rmask[:, None])

    @triton.autotune(configs=_FWD_CONFIGS, key=_AUTOTUNE_KEY)
    @triton.jit
    def _fwd_kernel(
        Q,
        QS,
        K,
        V,
        Out,
        Lse,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        H,
        L,
        TOKENS_PER_HALF,
        NFB_TOK,
        ROLLING_TOK,
        S_TOK,
        sm_scale,
        IS_TF: tl.constexpr,
        NO_WINDOW: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        q_base = Q + b * stride_qb + h * stride_qh
        qs_base = QS + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        o_base = Out + b * stride_ob + h * stride_oh

        r = pid_m * BM + tl.arange(0, BM)
        rmask = r < L
        d_offs = tl.arange(0, D)

        half, block, e_tok, end_row, win_lo = _row_geometry(r, L, TOKENS_PER_HALF, NFB_TOK, ROLLING_TOK, IS_TF,
                                                            NO_WINDOW)

        q = tl.load(q_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd, mask=rmask[:, None], other=0.0)
        if HAS_DELTA:
            q_sink = tl.load(qs_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                             mask=rmask[:, None],
                             other=0.0)
        else:
            q_sink = q

        acc = tl.zeros((BM, D), dtype=tl.float32)
        m_i = tl.full((BM, ), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BM, ), dtype=tl.float32)
        qk_scale = sm_scale * LOG2E

        # Phase 1: sink columns [0, S_TOK). Visible iff c < end_row; queries
        # use the delta-rotated variant (identity when the window has not
        # scrolled past the sink yet).
        for n0 in range(0, S_TOK, BN):
            c = n0 + tl.arange(0, BN)
            kt = tl.load(k_base + c[None, :] * stride_kl + d_offs[:, None] * stride_kd,
                         mask=(c < S_TOK)[None, :],
                         other=0.0)
            vt = tl.load(v_base + c[:, None] * stride_vl + d_offs[None, :] * stride_vd,
                         mask=(c < S_TOK)[:, None],
                         other=0.0)
            qk = tl.dot(q_sink, kt) * qk_scale
            vis = rmask[:, None] & (c[None, :] < S_TOK) & (c[None, :] < end_row[:, None])
            qk = tl.where(vis, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # Rows with no visible key yet keep m == -inf; anchor the exp2
            # at 0 for them so alpha/p stay 0 instead of exp2(-inf + inf).
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(qk - m_safe[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        # Phase 2: rolling-window columns [max(S_TOK, min win_lo), max end_row).
        lo2_rows = tl.where(rmask, tl.maximum(win_lo, S_TOK), L)
        lo2 = tl.min(lo2_rows)
        lo2 = (lo2 // BN) * BN
        hi2 = tl.max(tl.where(rmask, end_row, 0))
        for n0 in range(lo2, hi2, BN):
            c = n0 + tl.arange(0, BN)
            kt = tl.load(k_base + c[None, :] * stride_kl + d_offs[:, None] * stride_kd,
                         mask=(c < hi2)[None, :],
                         other=0.0)
            vt = tl.load(v_base + c[:, None] * stride_vl + d_offs[None, :] * stride_vd,
                         mask=(c < hi2)[:, None],
                         other=0.0)
            qk = tl.dot(q, kt) * qk_scale
            vis = (rmask[:, None] & (c[None, :] >= S_TOK) & (c[None, :] >= win_lo[:, None]) &
                   (c[None, :] < end_row[:, None]))
            qk = tl.where(vis, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(qk - m_safe[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        # Phase 3 (teacher forcing): a noisy row's own noisy block.
        if IS_TF:
            has_noisy = tl.max(tl.where(rmask, half, 0)) == 1
            if has_noisy:
                lo3_rows = tl.where(rmask & (half == 1), e_tok - NFB_TOK, L)
                lo3 = TOKENS_PER_HALF + (tl.min(lo3_rows) // BN) * BN
                hi3 = TOKENS_PER_HALF + tl.max(tl.where(rmask & (half == 1), e_tok, 0))
                for n0 in range(lo3, hi3, BN):
                    c = n0 + tl.arange(0, BN)
                    kt = tl.load(k_base + c[None, :] * stride_kl + d_offs[:, None] * stride_kd,
                                 mask=(c < L)[None, :],
                                 other=0.0)
                    vt = tl.load(v_base + c[:, None] * stride_vl + d_offs[None, :] * stride_vd,
                                 mask=(c < L)[:, None],
                                 other=0.0)
                    qk = tl.dot(q, kt) * qk_scale
                    c_pos = c[None, :] - TOKENS_PER_HALF
                    vis = (rmask[:, None] & (half[:, None] == 1) & (c_pos >= (e_tok - NFB_TOK)[:, None]) &
                           (c_pos < e_tok[:, None]))
                    qk = tl.where(vis, qk, float("-inf"))
                    m_new = tl.maximum(m_i, tl.max(qk, 1))
                    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                    alpha = tl.math.exp2(m_i - m_safe)
                    p = tl.math.exp2(qk - m_safe[:, None])
                    acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
                    l_i = l_i * alpha + tl.sum(p, 1)
                    m_i = m_new

        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out = acc / l_safe[:, None]
        tl.store(o_base + r[:, None] * stride_ol + d_offs[None, :] * stride_od,
                 out.to(Out.dtype.element_ty),
                 mask=rmask[:, None])
        lse = m_i + tl.math.log2(l_safe)
        tl.store(Lse + pid_bh * L + r, lse, mask=rmask)

    @triton.jit
    def _bwd_preprocess(
        Out,
        GradOut,
        Delta,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        stride_dob,
        stride_doh,
        stride_dol,
        stride_dod,
        H,
        L,
        BM: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        r = pid_m * BM + tl.arange(0, BM)
        rmask = r < L
        d_offs = tl.arange(0, D)
        o = tl.load(Out + b * stride_ob + h * stride_oh + r[:, None] * stride_ol + d_offs[None, :] * stride_od,
                    mask=rmask[:, None],
                    other=0.0)
        do = tl.load(GradOut + b * stride_dob + h * stride_doh + r[:, None] * stride_dol + d_offs[None, :] * stride_dod,
                     mask=rmask[:, None],
                     other=0.0)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        tl.store(Delta + pid_bh * L + r, delta, mask=rmask)

    @triton.autotune(configs=_DQ_CONFIGS, key=_AUTOTUNE_KEY)
    @triton.jit
    def _bwd_dq_kernel(
        Q,
        QS,
        K,
        V,
        DC,
        DS,
        GradOut,
        Lse,
        Delta,
        DQ,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_dol,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dql,
        stride_dqd,
        stride_dcb,
        stride_dcd,
        H,
        L,
        TOKENS_PER_HALF,
        NFB_TOK,
        ROLLING_TOK,
        S_TOK,
        sm_scale,
        IS_TF: tl.constexpr,
        NO_WINDOW: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        q_base = Q + b * stride_qb + h * stride_qh
        qs_base = QS + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        do_base = GradOut + b * stride_dob + h * stride_doh

        r = pid_m * BM + tl.arange(0, BM)
        rmask = r < L
        d_offs = tl.arange(0, D)

        half, block, e_tok, end_row, win_lo = _row_geometry(r, L, TOKENS_PER_HALF, NFB_TOK, ROLLING_TOK, IS_TF,
                                                            NO_WINDOW)

        q = tl.load(q_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd, mask=rmask[:, None], other=0.0)
        do = tl.load(do_base + r[:, None] * stride_dol + d_offs[None, :] * stride_dod, mask=rmask[:, None], other=0.0)
        lse = tl.load(Lse + pid_bh * L + r, mask=rmask, other=0.0)
        dlt = tl.load(Delta + pid_bh * L + r, mask=rmask, other=0.0)

        if HAS_DELTA:
            q_sink = tl.load(qs_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                             mask=rmask[:, None],
                             other=0.0)
        else:
            q_sink = q

        dq_acc = tl.zeros((BM, D), dtype=tl.float32)
        dq_sink_acc = tl.zeros((BM, D), dtype=tl.float32)
        qk_scale = sm_scale * LOG2E

        # Phase 1: sink columns (rotated-query space; rotated back at the end).
        for n0 in range(0, S_TOK, BN):
            c = n0 + tl.arange(0, BN)
            kt = tl.load(k_base + c[None, :] * stride_kl + d_offs[:, None] * stride_kd,
                         mask=(c < S_TOK)[None, :],
                         other=0.0)
            vt = tl.load(v_base + c[None, :] * stride_vl + d_offs[:, None] * stride_vd,
                         mask=(c < S_TOK)[None, :],
                         other=0.0)
            qk = tl.dot(q_sink, kt) * qk_scale
            vis = rmask[:, None] & (c[None, :] < S_TOK) & (c[None, :] < end_row[:, None])
            p = tl.where(vis, tl.math.exp2(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, vt).to(tl.float32)
            dsc = p * (dp - dlt[:, None]) * sm_scale
            dq_sink_acc += tl.dot(dsc.to(kt.dtype), tl.trans(kt))

        # Phase 2: rolling-window columns.
        lo2_rows = tl.where(rmask, tl.maximum(win_lo, S_TOK), L)
        lo2 = (tl.min(lo2_rows) // BN) * BN
        hi2 = tl.max(tl.where(rmask, end_row, 0))
        for n0 in range(lo2, hi2, BN):
            c = n0 + tl.arange(0, BN)
            kt = tl.load(k_base + c[None, :] * stride_kl + d_offs[:, None] * stride_kd,
                         mask=(c < hi2)[None, :],
                         other=0.0)
            vt = tl.load(v_base + c[None, :] * stride_vl + d_offs[:, None] * stride_vd,
                         mask=(c < hi2)[None, :],
                         other=0.0)
            qk = tl.dot(q, kt) * qk_scale
            vis = (rmask[:, None] & (c[None, :] >= S_TOK) & (c[None, :] >= win_lo[:, None]) &
                   (c[None, :] < end_row[:, None]))
            p = tl.where(vis, tl.math.exp2(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, vt).to(tl.float32)
            dsc = p * (dp - dlt[:, None]) * sm_scale
            dq_acc += tl.dot(dsc.to(kt.dtype), tl.trans(kt))

        # Phase 3 (teacher forcing): own noisy block.
        if IS_TF:
            has_noisy = tl.max(tl.where(rmask, half, 0)) == 1
            if has_noisy:
                lo3_rows = tl.where(rmask & (half == 1), e_tok - NFB_TOK, L)
                lo3 = TOKENS_PER_HALF + (tl.min(lo3_rows) // BN) * BN
                hi3 = TOKENS_PER_HALF + tl.max(tl.where(rmask & (half == 1), e_tok, 0))
                for n0 in range(lo3, hi3, BN):
                    c = n0 + tl.arange(0, BN)
                    kt = tl.load(k_base + c[None, :] * stride_kl + d_offs[:, None] * stride_kd,
                                 mask=(c < L)[None, :],
                                 other=0.0)
                    vt = tl.load(v_base + c[None, :] * stride_vl + d_offs[:, None] * stride_vd,
                                 mask=(c < L)[None, :],
                                 other=0.0)
                    qk = tl.dot(q, kt) * qk_scale
                    c_pos = c[None, :] - TOKENS_PER_HALF
                    vis = (rmask[:, None] & (half[:, None] == 1) & (c_pos >= (e_tok - NFB_TOK)[:, None]) &
                           (c_pos < e_tok[:, None]))
                    p = tl.where(vis, tl.math.exp2(qk - lse[:, None]), 0.0)
                    dp = tl.dot(do, vt).to(tl.float32)
                    dsc = p * (dp - dlt[:, None]) * sm_scale
                    dq_acc += tl.dot(dsc.to(kt.dtype), tl.trans(kt))

        if HAS_DELTA:
            dc = tl.load(DC + block[:, None] * stride_dcb + d_offs[None, :] * stride_dcd,
                         mask=rmask[:, None],
                         other=1.0)
            ds = tl.load(DS + block[:, None] * stride_dcb + d_offs[None, :] * stride_dcd,
                         mask=rmask[:, None],
                         other=0.0)
            dq_acc += _rotate_grad_plus_delta(dq_sink_acc, dc, ds, BM, D)
        else:
            dq_acc += dq_sink_acc

        tl.store(DQ + b * stride_dqb + h * stride_dqh + r[:, None] * stride_dql + d_offs[None, :] * stride_dqd,
                 dq_acc.to(DQ.dtype.element_ty),
                 mask=rmask[:, None])

    @triton.jit
    def _dkv_accumulate(
        dk_acc,
        dv_acc,
        lo,
        hi,
        c,
        cmask,
        c_half,
        c_block,
        kt,
        vt,
        q_base,
        qs_base,
        do_base,
        Lse,
        Delta,
        stride_ql,
        stride_qd,
        stride_dol,
        stride_dod,
        pid_bh,
        L,
        TOKENS_PER_HALF,
        NFB_TOK,
        ROLLING_TOK,
        S_TOK,
        sm_scale,
        IS_TF: tl.constexpr,
        NO_WINDOW: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
    ):
        """Accumulate dK/dV contributions from query tiles in [lo, hi)."""
        d_offs = tl.arange(0, D)
        qk_scale = sm_scale * LOG2E
        in_sink_col = c[None, :] < S_TOK
        for m0 in range(lo, hi, BM):
            r = m0 + tl.arange(0, BM)
            rmask = r < L
            half, block, e_tok, end_row, win_lo = _row_geometry(r, L, TOKENS_PER_HALF, NFB_TOK, ROLLING_TOK, IS_TF,
                                                                NO_WINDOW)

            vis_sink = in_sink_col & (c[None, :] < end_row[:, None])
            if IS_TF:
                vis_win = ((c_half[None, :] == 0) & (~in_sink_col) & (c[None, :] >= win_lo[:, None]) &
                           (c[None, :] < end_row[:, None]))
                vis_self = ((c_half[None, :] == 1) & (half[:, None] == 1) & (c_block[None, :] == block[:, None]))
                vis_plain = vis_win | vis_self
            else:
                vis_plain = ((~in_sink_col) & (c[None, :] >= win_lo[:, None]) & (c[None, :] < end_row[:, None]))
            valid = rmask[:, None] & cmask[None, :]
            vis_any = (vis_sink | vis_plain) & valid
            if tl.max(vis_any.to(tl.int32)) == 1:
                do = tl.load(do_base + r[:, None] * stride_dol + d_offs[None, :] * stride_dod,
                             mask=rmask[:, None],
                             other=0.0)
                lse = tl.load(Lse + pid_bh * L + r, mask=rmask, other=0.0)
                dlt = tl.load(Delta + pid_bh * L + r, mask=rmask, other=0.0)
                dp = tl.dot(do, tl.trans(vt)).to(tl.float32)

                if HAS_DELTA:
                    # Sink columns score against the rotated queries, other
                    # columns against the plain ones; a tile straddling the
                    # sink boundary needs both products.
                    touches_sink = tl.min(tl.where(cmask, c, S_TOK)) < S_TOK
                    p_total = tl.zeros((BM, BN), dtype=tl.float32)
                    if touches_sink:
                        qs = tl.load(qs_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                                     mask=rmask[:, None],
                                     other=0.0)
                        qk_s = tl.dot(qs, tl.trans(kt)) * qk_scale
                        p_s = tl.where(vis_sink & valid, tl.math.exp2(qk_s - lse[:, None]), 0.0)
                        dsc_s = p_s * (dp - dlt[:, None]) * sm_scale
                        dk_acc += tl.dot(tl.trans(dsc_s.to(qs.dtype)), qs)
                        p_total += p_s
                    touches_plain = tl.max(tl.where(cmask, c, 0)) >= S_TOK
                    if touches_plain:
                        q = tl.load(q_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                                    mask=rmask[:, None],
                                    other=0.0)
                        qk_p = tl.dot(q, tl.trans(kt)) * qk_scale
                        p_p = tl.where(vis_plain & valid, tl.math.exp2(qk_p - lse[:, None]), 0.0)
                        dsc_p = p_p * (dp - dlt[:, None]) * sm_scale
                        dk_acc += tl.dot(tl.trans(dsc_p.to(q.dtype)), q)
                        p_total += p_p
                    dv_acc += tl.dot(tl.trans(p_total.to(do.dtype)), do)
                else:
                    q = tl.load(q_base + r[:, None] * stride_ql + d_offs[None, :] * stride_qd,
                                mask=rmask[:, None],
                                other=0.0)
                    qk = tl.dot(q, tl.trans(kt)) * qk_scale
                    p = tl.where(vis_any, tl.math.exp2(qk - lse[:, None]), 0.0)
                    dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
                    dsc = p * (dp - dlt[:, None]) * sm_scale
                    dk_acc += tl.dot(tl.trans(dsc.to(q.dtype)), q)
        return dk_acc, dv_acc

    @triton.autotune(configs=_DKV_CONFIGS, key=_AUTOTUNE_KEY)
    @triton.jit
    def _bwd_dkv_kernel(
        Q,
        QS,
        K,
        V,
        GradOut,
        Lse,
        Delta,
        DK,
        DV,
        COL_OFFSET,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_dol,
        stride_dod,
        stride_dkb,
        stride_dkh,
        stride_dkl,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvl,
        stride_dvd,
        H,
        L,
        TOKENS_PER_HALF,
        NFB_TOK,
        ROLLING_TOK,
        S_TOK,
        sm_scale,
        IS_TF: tl.constexpr,
        NO_WINDOW: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        q_base = Q + b * stride_qb + h * stride_qh
        qs_base = QS + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        do_base = GradOut + b * stride_dob + h * stride_doh

        tile_lo = COL_OFFSET + pid_n * BN
        c = tile_lo + tl.arange(0, BN)
        cmask = c < L
        d_offs = tl.arange(0, D)

        kt = tl.load(k_base + c[:, None] * stride_kl + d_offs[None, :] * stride_kd, mask=cmask[:, None], other=0.0)
        vt = tl.load(v_base + c[:, None] * stride_vl + d_offs[None, :] * stride_vd, mask=cmask[:, None], other=0.0)

        dk_acc = tl.zeros((BN, D), dtype=tl.float32)
        dv_acc = tl.zeros((BN, D), dtype=tl.float32)

        c_clamped = tl.minimum(c, L - 1)
        if IS_TF:
            c_half = c_clamped // TOKENS_PER_HALF
            c_pos = c_clamped % TOKENS_PER_HALF
        else:
            c_half = tl.zeros_like(c_clamped)
            c_pos = c_clamped
        c_block = c_pos // NFB_TOK

        # Query ranges that can reach this key tile. The first candidate block
        # is the key tile's earliest frame-block (rows before it end too
        # early); the last (windowed geometries) is the one whose window still
        # reaches back to the newest key column. Sink columns stay visible to
        # every later block, so tiles touching the sink scan to the end of the
        # half; per-element visibility keeps every range exact.
        cb_min = tl.min(tl.where(cmask, c_block, L))
        lo1 = (cb_min * NFB_TOK // BM) * BM
        if IS_TF and tile_lo >= TOKENS_PER_HALF:
            # Noisy-half key tile: only same-block noisy rows see it.
            cb_max = tl.max(tl.where(cmask, c_block, 0))
            lo = TOKENS_PER_HALF + lo1
            hi = TOKENS_PER_HALF + (cb_max + 1) * NFB_TOK
            dk_acc, dv_acc = _dkv_accumulate(dk_acc,
                                             dv_acc,
                                             lo,
                                             hi,
                                             c,
                                             cmask,
                                             c_half,
                                             c_block,
                                             kt,
                                             vt,
                                             q_base,
                                             qs_base,
                                             do_base,
                                             Lse,
                                             Delta,
                                             stride_ql,
                                             stride_qd,
                                             stride_dol,
                                             stride_dod,
                                             pid_bh,
                                             L,
                                             TOKENS_PER_HALF,
                                             NFB_TOK,
                                             ROLLING_TOK,
                                             S_TOK,
                                             sm_scale,
                                             IS_TF=IS_TF,
                                             NO_WINDOW=NO_WINDOW,
                                             HAS_DELTA=HAS_DELTA,
                                             BM=BM,
                                             BN=BN,
                                             D=D)
        else:
            if NO_WINDOW:
                hi1 = TOKENS_PER_HALF
            else:
                max_c = tl.max(tl.where(cmask, c, 0))
                hi1 = tl.minimum(TOKENS_PER_HALF, ((max_c + ROLLING_TOK) // NFB_TOK) * NFB_TOK)
                if tile_lo < S_TOK:
                    hi1 = TOKENS_PER_HALF
            dk_acc, dv_acc = _dkv_accumulate(dk_acc,
                                             dv_acc,
                                             lo1,
                                             hi1,
                                             c,
                                             cmask,
                                             c_half,
                                             c_block,
                                             kt,
                                             vt,
                                             q_base,
                                             qs_base,
                                             do_base,
                                             Lse,
                                             Delta,
                                             stride_ql,
                                             stride_qd,
                                             stride_dol,
                                             stride_dod,
                                             pid_bh,
                                             L,
                                             TOKENS_PER_HALF,
                                             NFB_TOK,
                                             ROLLING_TOK,
                                             S_TOK,
                                             sm_scale,
                                             IS_TF=IS_TF,
                                             NO_WINDOW=NO_WINDOW,
                                             HAS_DELTA=HAS_DELTA,
                                             BM=BM,
                                             BN=BN,
                                             D=D)
            if IS_TF:
                # Noisy rows read the same clean context; avoid re-visiting a
                # tile the first range already covered.
                lo_n = (TOKENS_PER_HALF + lo1) // BM * BM
                lo_n = tl.maximum(lo_n, (hi1 + BM - 1) // BM * BM)
                hi_n = TOKENS_PER_HALF + hi1
                dk_acc, dv_acc = _dkv_accumulate(dk_acc,
                                                 dv_acc,
                                                 lo_n,
                                                 hi_n,
                                                 c,
                                                 cmask,
                                                 c_half,
                                                 c_block,
                                                 kt,
                                                 vt,
                                                 q_base,
                                                 qs_base,
                                                 do_base,
                                                 Lse,
                                                 Delta,
                                                 stride_ql,
                                                 stride_qd,
                                                 stride_dol,
                                                 stride_dod,
                                                 pid_bh,
                                                 L,
                                                 TOKENS_PER_HALF,
                                                 NFB_TOK,
                                                 ROLLING_TOK,
                                                 S_TOK,
                                                 sm_scale,
                                                 IS_TF=IS_TF,
                                                 NO_WINDOW=NO_WINDOW,
                                                 HAS_DELTA=HAS_DELTA,
                                                 BM=BM,
                                                 BN=BN,
                                                 D=D)

        tl.store(DK + b * stride_dkb + h * stride_dkh + c[:, None] * stride_dkl + d_offs[None, :] * stride_dkd,
                 dk_acc.to(DK.dtype.element_ty),
                 mask=cmask[:, None])
        tl.store(DV + b * stride_dvb + h * stride_dvh + c[:, None] * stride_dvl + d_offs[None, :] * stride_dvd,
                 dv_acc.to(DV.dtype.element_ty),
                 mask=cmask[:, None])

    @triton.jit
    def _bwd_dkv_sink_kernel(
        Q,
        QS,
        K,
        V,
        GradOut,
        Lse,
        Delta,
        DK32,
        DV32,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_dol,
        stride_dod,
        H,
        L,
        SINK_BOUND,
        QCHUNK,
        TOKENS_PER_HALF,
        NFB_TOK,
        ROLLING_TOK,
        S_TOK,
        sm_scale,
        IS_TF: tl.constexpr,
        NO_WINDOW: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
    ):
        """Split-K dK/dV for the leading [0, SINK_BOUND) key columns.

        Sink columns are visible to every later query block, so their key
        tiles would otherwise be the few longest-running programs of the
        dK/dV grid (a serial scan over the whole query range). Here the query
        range is split across ``program_id(2)``; the partial [BN, D] results
        are accumulated into float32 buffers with atomics. ``QCHUNK`` is
        128-aligned so split boundaries land on tile edges for every BM.
        """
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_s = tl.program_id(2)
        b = pid_bh // H
        h = pid_bh % H

        q_base = Q + b * stride_qb + h * stride_qh
        qs_base = QS + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        do_base = GradOut + b * stride_dob + h * stride_doh

        c = pid_n * BN + tl.arange(0, BN)
        cmask = c < L
        d_offs = tl.arange(0, D)

        kt = tl.load(k_base + c[:, None] * stride_kl + d_offs[None, :] * stride_kd, mask=cmask[:, None], other=0.0)
        vt = tl.load(v_base + c[:, None] * stride_vl + d_offs[None, :] * stride_vd, mask=cmask[:, None], other=0.0)

        c_clamped = tl.minimum(c, L - 1)
        if IS_TF:
            c_half = c_clamped // TOKENS_PER_HALF
            c_pos = c_clamped % TOKENS_PER_HALF
        else:
            c_half = tl.zeros_like(c_clamped)
            c_pos = c_clamped
        c_block = c_pos // NFB_TOK

        q_lo = pid_s * QCHUNK
        q_hi = tl.minimum(q_lo + QCHUNK, L)
        dk_acc = tl.zeros((BN, D), dtype=tl.float32)
        dv_acc = tl.zeros((BN, D), dtype=tl.float32)
        dk_acc, dv_acc = _dkv_accumulate(dk_acc,
                                         dv_acc,
                                         q_lo,
                                         q_hi,
                                         c,
                                         cmask,
                                         c_half,
                                         c_block,
                                         kt,
                                         vt,
                                         q_base,
                                         qs_base,
                                         do_base,
                                         Lse,
                                         Delta,
                                         stride_ql,
                                         stride_qd,
                                         stride_dol,
                                         stride_dod,
                                         pid_bh,
                                         L,
                                         TOKENS_PER_HALF,
                                         NFB_TOK,
                                         ROLLING_TOK,
                                         S_TOK,
                                         sm_scale,
                                         IS_TF=IS_TF,
                                         NO_WINDOW=NO_WINDOW,
                                         HAS_DELTA=HAS_DELTA,
                                         BM=BM,
                                         BN=BN,
                                         D=D)

        offs = ((b * H + h) * SINK_BOUND + c[:, None]) * D + d_offs[None, :]
        tl.atomic_add(DK32 + offs, dk_acc, mask=cmask[:, None])
        tl.atomic_add(DV32 + offs, dv_acc, mask=cmask[:, None])


def _plan_scalars(plan) -> dict:
    tokens_per_half = plan.num_frames * plan.frame_seqlen
    no_window = plan.local_attn_size == -1
    rolling_tok = (0 if no_window else max(0, plan.local_attn_size - plan.sink_size) * plan.frame_seqlen)
    s_tok = 0 if no_window else plan.sink_size * plan.frame_seqlen
    return {
        "tokens_per_half": tokens_per_half,
        "nfb_tok": plan.num_frame_per_block * plan.frame_seqlen,
        "rolling_tok": rolling_tok,
        "s_tok": s_tok,
        "is_tf": plan.kind == "teacher_forcing",
        "no_window": no_window,
    }


def _rotated_sink_queries(q, dc, ds, sc) -> torch.Tensor:
    """R(-delta) on the roped queries via one fused elementwise kernel.

    Returned tensor shares q's (possibly transposed) strides - the attention
    kernels index it with q's stride tuple.
    """
    B, H, L, D = q.shape
    q_sink = torch.empty_like(q)
    grid = (triton.cdiv(L, 128), B * H)
    _rotate_queries_kernel[grid](
        q,
        q_sink,
        dc,
        ds,
        *q.stride(),
        dc.stride(0),
        dc.stride(1),
        H,
        L,
        sc["tokens_per_half"],
        sc["nfb_tok"],
        BM=128,
        D=D,
        num_warps=4,
    )
    return q_sink


class _BlockCausalSinkAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, delta_cos, delta_sin, plan):
        B, H, L, D = q.shape
        sc = _plan_scalars(plan)
        if plan.seq_len != L:
            raise ValueError(f"Plan expects seq_len={plan.seq_len}, got {L}")
        has_delta = delta_cos is not None and sc["s_tok"] > 0

        out = torch.empty_like(q)
        lse = torch.empty((B * H, L), device=q.device, dtype=torch.float32)
        dc = delta_cos if has_delta else q.new_empty((1, D), dtype=torch.float32)
        ds = delta_sin if has_delta else q.new_empty((1, D), dtype=torch.float32)
        q_sink = _rotated_sink_queries(q, dc, ds, sc) if has_delta else q

        grid = lambda META: (triton.cdiv(L, META["BM"]), B * H)  # noqa: E731
        _fwd_kernel[grid](
            q,
            q_sink,
            k,
            v,
            out,
            lse,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            H,
            L,
            sc["tokens_per_half"],
            sc["nfb_tok"],
            sc["rolling_tok"],
            sc["s_tok"],
            plan.sm_scale,
            IS_TF=sc["is_tf"],
            NO_WINDOW=sc["no_window"],
            HAS_DELTA=has_delta,
            D=D,
        )

        ctx.save_for_backward(q, q_sink, k, v, out, lse, dc, ds)
        ctx.plan = plan
        ctx.has_delta = has_delta
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, q_sink, k, v, out, lse, dc, ds = ctx.saved_tensors
        plan = ctx.plan
        sc = _plan_scalars(plan)
        B, H, L, D = q.shape
        grad_out = grad_out.contiguous()

        delta = torch.empty_like(lse)
        BM_P = 128
        _bwd_preprocess[(triton.cdiv(L, BM_P), B * H)](
            out,
            grad_out,
            delta,
            *out.stride(),
            *grad_out.stride(),
            H,
            L,
            BM=BM_P,
            D=D,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid_dq = lambda META: (triton.cdiv(L, META["BM"]), B * H)  # noqa: E731
        _bwd_dq_kernel[grid_dq](
            q,
            q_sink,
            k,
            v,
            dc,
            ds,
            grad_out,
            lse,
            delta,
            dq,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *grad_out.stride(),
            *dq.stride(),
            dc.stride(0),
            dc.stride(1),
            H,
            L,
            sc["tokens_per_half"],
            sc["nfb_tok"],
            sc["rolling_tok"],
            sc["s_tok"],
            plan.sm_scale,
            IS_TF=sc["is_tf"],
            NO_WINDOW=sc["no_window"],
            HAS_DELTA=ctx.has_delta,
            D=D,
        )

        # dK/dV. Sink key columns are visible from every later query block, so
        # their tiles would be the few longest-running programs of a flat kv
        # grid; carve the (128-aligned) sink region out into a split-K kernel
        # with float32 atomic accumulation, and let the main kernel - which
        # then never sees sink columns - run with the delta logic compiled out.
        s_tok = sc["s_tok"]
        sink_bound = ((s_tok + 127) // 128) * 128 if s_tok > 0 else 0
        if 0 < sink_bound < L:
            dk32 = torch.zeros((B, H, sink_bound, D), device=q.device, dtype=torch.float32)
            dv32 = torch.zeros_like(dk32)
            split = 8
            qchunk = (((L + split - 1) // split + 127) // 128) * 128
            _bwd_dkv_sink_kernel[(sink_bound // 64, B * H, split)](
                q,
                q_sink,
                k,
                v,
                grad_out,
                lse,
                delta,
                dk32,
                dv32,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *grad_out.stride(),
                H,
                L,
                sink_bound,
                qchunk,
                sc["tokens_per_half"],
                sc["nfb_tok"],
                sc["rolling_tok"],
                s_tok,
                plan.sm_scale,
                IS_TF=sc["is_tf"],
                NO_WINDOW=sc["no_window"],
                HAS_DELTA=ctx.has_delta,
                BM=64,
                BN=64,
                D=D,
                num_warps=4,
                num_stages=2,
            )
            col_offset = sink_bound
            has_delta_main = False
            grid_dkv = lambda META: (triton.cdiv(L - sink_bound, META["BN"]), B * H)  # noqa: E731
        else:
            col_offset = 0
            has_delta_main = ctx.has_delta
            grid_dkv = lambda META: (triton.cdiv(L, META["BN"]), B * H)  # noqa: E731

        _bwd_dkv_kernel[grid_dkv](
            q,
            q_sink,
            k,
            v,
            grad_out,
            lse,
            delta,
            dk,
            dv,
            col_offset,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *grad_out.stride(),
            *dk.stride(),
            *dv.stride(),
            H,
            L,
            sc["tokens_per_half"],
            sc["nfb_tok"],
            sc["rolling_tok"],
            sc["s_tok"],
            plan.sm_scale,
            IS_TF=sc["is_tf"],
            NO_WINDOW=sc["no_window"],
            HAS_DELTA=has_delta_main,
            D=D,
        )
        if 0 < sink_bound < L:
            dk[:, :, :sink_bound] = dk32.to(dk.dtype)
            dv[:, :, :sink_bound] = dv32.to(dv.dtype)
        return dq, dk, dv, None, None, None


def block_causal_sink_attention(q, k, v, plan):
    """Fused training attention for a ``CausalTrainAttentionPlan``.

    Args:
        q / k / v: ``[B, H, L, D]`` on CUDA; ``D`` must be a power of two
            (pairs interleaved GPT-J style when the sink correction is used).
        plan: ``CausalTrainAttentionPlan`` with ``impl == "triton"``.
    Returns:
        ``[B, H, L, D]`` in the input dtype.
    """
    if not HAS_TRITON:
        raise RuntimeError("causal_train_attention='triton' requires triton; install it or "
                           "use 'flex' / 'reference'")
    if not q.is_cuda:
        raise RuntimeError("causal_train_attention='triton' requires CUDA tensors; use "
                           "'flex' / 'reference' elsewhere")
    head_dim = q.shape[-1]
    if head_dim & (head_dim - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {head_dim}")
    delta_cos = delta_sin = None
    if plan.delta_cos is not None:
        delta_cos = plan.delta_cos.to(device=q.device, dtype=torch.float32).contiguous()
        delta_sin = plan.delta_sin.to(device=q.device, dtype=torch.float32).contiguous()
    return _BlockCausalSinkAttention.apply(q, k, v, delta_cos, delta_sin, plan)
