# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results there from are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input


def _resolve_flash_attn_varlen_func() -> tuple[Any, str]:
    try:
        from fastvideo.attention.utils.flash_attn_cute import (
            flash_attn_varlen_func as flash_attn_varlen_func_cute, )

        return flash_attn_varlen_func_cute, "4"
    except ImportError:
        try:
            from flash_attn_interface import (
                flash_attn_varlen_func as flash_attn_varlen_func_interface, )

            return flash_attn_varlen_func_interface, "3"
        except ImportError:
            from flash_attn import (
                flash_attn_varlen_func as flash_attn_varlen_func_flash, )

            return flash_attn_varlen_func_flash, "2"


flash_attn_varlen_func_impl, _FA_VARLEN_VERSION = _resolve_flash_attn_varlen_func()

# FA2-only: the private varlen backward we register against the custom ops
# below. FA3 / FA4 have different private signatures and validation paths
# (Hopper / Blackwell boxes) — those legs keep the autograd carve-out
# pattern from PR #1373 until their setup PRs land.
if _FA_VARLEN_VERSION == "2":
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward as _fa2_varlen_backward, )


def flash_attn_no_pad(
    qkv: torch.Tensor,
    key_padding_mask: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv: torch.Tensor,
    key_padding_mask: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    from flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3, )

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(rearrange(query, "b s h d -> b s (h d)"),
                                                                      key_padding_mask)
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
    value_unpad, _, _, _, _ = unpad_input(rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_varlen_qk_no_pad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_padding_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    batch_size, q_seqlen, nheads, _ = query.shape

    query_unpad, q_indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(rearrange(query, "b s h d -> b s (h d)"),
                                                                        query_padding_mask)
    key_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
    value_unpad, _, _, _, _ = unpad_input(rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_impl(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            q_indices,
            batch_size,
            q_seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


# ---------------------------------------------------------------------------
# torch.compile traceability + register_autograd parity for the masked /
# varlen attention paths.
#
# Wraps the two entry points `FlashAttentionImpl.forward` calls
# (`flash_attn_no_pad`, `flash_attn_varlen_qk_no_pad`) as
# `torch.library.custom_op`s so dynamo sees one traceable node — the
# internal unpad / pad bookkeeping (data-dependent `nnz` shapes) runs
# eager inside the op, and the op's outputs are the statically-shaped
# padded tensors. This mirrors the FA2 default-path wrapper in
# `fastvideo/attention/backends/flash_attn.py`.
#
# Autograd: on FA2 we register a real backward (`register_autograd`)
# that calls FA2's `_flash_attn_varlen_backward` on the unpadded form
# — re-unpadding the saved padded tensors using the saved mask. The
# `softmax_lse` from the varlen forward is naturally unpadded
# (`[nheads, total_q]`); we pad it to `[batch, nheads, seqlen]` on
# the way out (statically shaped) and re-unpad in backward. So
# training backprops *through* the op (no graph break on the training
# path either).
#
# FA3 / FA4 keep the autograd carve-out pattern from PR #1373: the
# custom op has forward + fake only, and `*_compilable` falls back to
# the original autograd.Function for grad-enabled calls. Those legs
# are gated on Hopper-class / Blackwell-class boxes for backward
# validation and ship as separate follow-ups.

if _FA_VARLEN_VERSION == "2":
    # ---------- masked self-attention: flash_attn_no_pad (FA2) ----------

    @torch.library.custom_op(
        "fastvideo::_flash_attn_no_pad_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def _flash_attn_no_pad_forward(
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, _three, h, d = qkv.shape
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=h)
        out_unpad, lse_unpad, _ = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_seqlens, max_s, dropout_p,
            softmax_scale=softmax_scale, causal=causal,
            deterministic=deterministic, return_attn_probs=True)
        # Pad out: [nnz, h, d] -> [b, s, h, d]
        out_padded = rearrange(
            pad_input(rearrange(out_unpad, "nnz h d -> nnz (h d)"), indices, b, s),
            "b s (h d) -> b s h d", h=h)
        # Pad lse: FA2 varlen returns [nheads, total_q]. Transpose to [total_q,
        # nheads], pad to [b, s, nheads], permute to [b, nheads, s] — statically
        # shaped so register_fake matches.
        lse_padded = pad_input(lse_unpad.t().contiguous(), indices, b, s).permute(0, 2, 1).contiguous()
        return out_padded, lse_padded

    @torch.library.register_fake("fastvideo::_flash_attn_no_pad_forward")
    def _flash_attn_no_pad_forward_fake(
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key_padding_mask, causal, dropout_p, softmax_scale, deterministic
        b, s, _three, h, d = qkv.shape
        out = qkv.new_empty(b, s, h, d)
        lse = qkv.new_empty(b, h, s, dtype=torch.float32)
        return out, lse

    def _flash_attn_no_pad_setup_context(ctx, inputs, output):
        qkv, key_padding_mask, causal, dropout_p, softmax_scale, deterministic = inputs
        out, lse = output
        ctx.save_for_backward(qkv, out, lse, key_padding_mask)
        # FA2's varlen backward requires a concrete float for softmax_scale.
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** -0.5  # head_dim from qkv's last dim
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.deterministic = deterministic

    def _flash_attn_no_pad_backward(ctx, grad_out, grad_lse):
        # lse is saved-for-backward, not differentiated.
        del grad_lse
        qkv, out_padded, lse_padded, key_padding_mask = ctx.saved_tensors
        b, s, _three, h, d = qkv.shape

        # Re-unpad qkv -> q, k, v unpadded ([nnz, h, d] each).
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=h)
        q_unpad, k_unpad, v_unpad = (t.contiguous() for t in x_unpad.unbind(dim=1))

        # Re-unpad out and dout using the same mask.
        out_unpad = rearrange(
            unpad_input(rearrange(out_padded, "b s h d -> b s (h d)"), key_padding_mask)[0],
            "nnz (h d) -> nnz h d", h=h).contiguous()
        dout_unpad = rearrange(
            unpad_input(rearrange(grad_out, "b s h d -> b s (h d)"), key_padding_mask)[0],
            "nnz (h d) -> nnz h d", h=h).contiguous()

        # Re-unpad lse: [b, h, s] -> [b, s, h] -> [nnz, h] -> [h, nnz].
        lse_unpad = unpad_input(lse_padded.permute(0, 2, 1).contiguous(),
                                key_padding_mask)[0].t().contiguous()

        dq_unpad = torch.empty_like(q_unpad)
        dk_unpad = torch.empty_like(k_unpad)
        dv_unpad = torch.empty_like(v_unpad)
        _fa2_varlen_backward(
            dout_unpad, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad,
            dq_unpad, dk_unpad, dv_unpad,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_s, max_seqlen_k=max_s,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=-1, window_size_right=-1,
            softcap=0.0, alibi_slopes=None,
            deterministic=ctx.deterministic,
            rng_state=None,
        )

        # Re-pad each grad and stack into dqkv.
        def _repad(dt_unpad):
            padded = pad_input(rearrange(dt_unpad, "nnz h d -> nnz (h d)"), indices, b, s)
            return rearrange(padded, "b s (h d) -> b s h d", h=h)

        dqkv = torch.stack([_repad(dq_unpad), _repad(dk_unpad), _repad(dv_unpad)], dim=2)
        # 6 inputs total: qkv, key_padding_mask, causal, dropout_p, softmax_scale, deterministic.
        return dqkv, None, None, None, None, None

    torch.library.register_autograd(
        "fastvideo::_flash_attn_no_pad_forward",
        _flash_attn_no_pad_backward,
        setup_context=_flash_attn_no_pad_setup_context,
    )

    # ---------- cross-attention: flash_attn_varlen_qk_no_pad (FA2) ----------

    @torch.library.custom_op(
        "fastvideo::_flash_attn_varlen_qk_no_pad_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def _flash_attn_varlen_qk_no_pad_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, sq, h, d = query.shape
        sk = key.shape[1]
        q_unpad, q_indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
            rearrange(query, "b s h d -> b s (h d)"), query_padding_mask)
        k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
            rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(
            rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)
        q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=h)
        k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=h)
        v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=h)
        out_unpad, lse_unpad, _ = flash_attn_varlen_func_impl(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, deterministic=deterministic,
            return_attn_probs=True)
        # Pad out: [nnz_q, h, d] -> [b, sq, h, d]
        out_padded = rearrange(
            pad_input(rearrange(out_unpad, "nnz h d -> nnz (h d)"), q_indices, b, sq),
            "b s (h d) -> b s h d", h=h)
        # Pad lse: [h, nnz_q] -> [b, h, sq]
        lse_padded = pad_input(lse_unpad.t().contiguous(), q_indices, b, sq).permute(0, 2, 1).contiguous()
        return out_padded, lse_padded

    @torch.library.register_fake("fastvideo::_flash_attn_varlen_qk_no_pad_forward")
    def _flash_attn_varlen_qk_no_pad_forward_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key, value, query_padding_mask, key_padding_mask
        del causal, dropout_p, softmax_scale, deterministic
        b, sq, h, _ = query.shape
        out = query.new_empty(query.shape)
        lse = query.new_empty(b, h, sq, dtype=torch.float32)
        return out, lse

    def _flash_attn_varlen_qk_no_pad_setup_context(ctx, inputs, output):
        (query, key, value, query_padding_mask, key_padding_mask,
         causal, dropout_p, softmax_scale, deterministic) = inputs
        out, lse = output
        ctx.save_for_backward(query, key, value, out, lse,
                              query_padding_mask, key_padding_mask)
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.deterministic = deterministic

    def _flash_attn_varlen_qk_no_pad_backward(ctx, grad_out, grad_lse):
        del grad_lse
        (query, key, value, out_padded, lse_padded,
         query_padding_mask, key_padding_mask) = ctx.saved_tensors
        b, sq, h, d = query.shape
        sk = key.shape[1]

        # Re-unpad q with q_mask; k, v with k_mask.
        q_unpad, q_indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
            rearrange(query, "b s h d -> b s (h d)"), query_padding_mask)
        k_unpad, k_indices, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
            rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(
            rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)
        q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=h).contiguous()
        k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=h).contiguous()
        v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=h).contiguous()

        # out and lse follow q's shape, so re-unpad with q_mask.
        out_unpad = rearrange(
            unpad_input(rearrange(out_padded, "b s h d -> b s (h d)"), query_padding_mask)[0],
            "nnz (h d) -> nnz h d", h=h).contiguous()
        dout_unpad = rearrange(
            unpad_input(rearrange(grad_out, "b s h d -> b s (h d)"), query_padding_mask)[0],
            "nnz (h d) -> nnz h d", h=h).contiguous()
        lse_unpad = unpad_input(lse_padded.permute(0, 2, 1).contiguous(),
                                query_padding_mask)[0].t().contiguous()

        dq_unpad = torch.empty_like(q_unpad)
        dk_unpad = torch.empty_like(k_unpad)
        dv_unpad = torch.empty_like(v_unpad)
        _fa2_varlen_backward(
            dout_unpad, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad,
            dq_unpad, dk_unpad, dv_unpad,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=-1, window_size_right=-1,
            softcap=0.0, alibi_slopes=None,
            deterministic=ctx.deterministic,
            rng_state=None,
        )

        def _repad(dt_unpad, indices, batch, seqlen):
            padded = pad_input(rearrange(dt_unpad, "nnz h d -> nnz (h d)"), indices, batch, seqlen)
            return rearrange(padded, "b s (h d) -> b s h d", h=h)

        dq_padded = _repad(dq_unpad, q_indices, b, sq)
        dk_padded = _repad(dk_unpad, k_indices, b, sk)
        dv_padded = _repad(dv_unpad, k_indices, b, sk)
        # 9 inputs total: query, key, value, q_mask, k_mask, causal, dropout_p,
        # softmax_scale, deterministic.
        return dq_padded, dk_padded, dv_padded, None, None, None, None, None, None

    torch.library.register_autograd(
        "fastvideo::_flash_attn_varlen_qk_no_pad_forward",
        _flash_attn_varlen_qk_no_pad_backward,
        setup_context=_flash_attn_varlen_qk_no_pad_setup_context,
    )

    # ---------- public dispatchers (FA2: autograd flows through the op) -----

    def flash_attn_no_pad_compilable(qkv, key_padding_mask, causal=False,
                                     dropout_p=0.0, softmax_scale=None,
                                     deterministic=False):
        """dynamo-traceable wrapper around ``flash_attn_no_pad`` (registered op,
        full register_autograd on FA2 — both inference and training go through
        the op, no graph break on either)."""
        out, _ = torch.ops.fastvideo._flash_attn_no_pad_forward(
            qkv, key_padding_mask, causal, dropout_p, softmax_scale, deterministic)
        return out

    def flash_attn_varlen_qk_no_pad_compilable(query, key, value, query_padding_mask,
                                               key_padding_mask, causal=False,
                                               dropout_p=0.0, softmax_scale=None,
                                               deterministic=False):
        """dynamo-traceable wrapper around ``flash_attn_varlen_qk_no_pad`` (registered
        op, full register_autograd on FA2)."""
        out, _ = torch.ops.fastvideo._flash_attn_varlen_qk_no_pad_forward(
            query, key, value, query_padding_mask, key_padding_mask,
            causal, dropout_p, softmax_scale, deterministic)
        return out

else:
    # ---------- FA3 / FA4: carve-out (forward+fake only, no real backward) ---
    # Same pattern as the parked varlen-extension and the FA3 default leg in
    # `fastvideo/attention/backends/flash_attn.py`. Real backward for these
    # versions is a follow-up gated on Hopper / Blackwell box validation.

    @torch.library.custom_op(
        "fastvideo::_flash_attn_no_pad_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def _flash_attn_no_pad_forward(
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> torch.Tensor:
        return flash_attn_no_pad(  # type: ignore[no-untyped-call]
            qkv, key_padding_mask, causal=causal, dropout_p=dropout_p,
            softmax_scale=softmax_scale, deterministic=deterministic)

    @torch.library.register_fake("fastvideo::_flash_attn_no_pad_forward")
    def _flash_attn_no_pad_forward_fake(
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> torch.Tensor:
        del key_padding_mask, causal, dropout_p, softmax_scale, deterministic
        b, s, _three, h, d = qkv.shape
        return qkv.new_empty(b, s, h, d)

    @torch.library.custom_op(
        "fastvideo::_flash_attn_varlen_qk_no_pad_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def _flash_attn_varlen_qk_no_pad_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> torch.Tensor:
        return flash_attn_varlen_qk_no_pad(  # type: ignore[no-untyped-call]
            query, key, value,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            causal=causal, dropout_p=dropout_p,
            softmax_scale=softmax_scale, deterministic=deterministic)

    @torch.library.register_fake("fastvideo::_flash_attn_varlen_qk_no_pad_forward")
    def _flash_attn_varlen_qk_no_pad_forward_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool,
        dropout_p: float,
        softmax_scale: float | None,
        deterministic: bool,
    ) -> torch.Tensor:
        del key, value, query_padding_mask, key_padding_mask
        del causal, dropout_p, softmax_scale, deterministic
        return query.new_empty(query.shape)

    def flash_attn_no_pad_compilable(qkv, key_padding_mask, causal=False,
                                     dropout_p=0.0, softmax_scale=None,
                                     deterministic=False):
        if torch.is_grad_enabled() and qkv.requires_grad:
            return flash_attn_no_pad(qkv, key_padding_mask, causal=causal,
                                     dropout_p=dropout_p, softmax_scale=softmax_scale,
                                     deterministic=deterministic)
        return torch.ops.fastvideo._flash_attn_no_pad_forward(
            qkv, key_padding_mask, causal, dropout_p, softmax_scale, deterministic)

    def flash_attn_varlen_qk_no_pad_compilable(query, key, value, query_padding_mask,
                                               key_padding_mask, causal=False,
                                               dropout_p=0.0, softmax_scale=None,
                                               deterministic=False):
        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad
                                        or value.requires_grad):
            return flash_attn_varlen_qk_no_pad(
                query, key, value,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                causal=causal, dropout_p=dropout_p,
                softmax_scale=softmax_scale, deterministic=deterministic)
        return torch.ops.fastvideo._flash_attn_varlen_qk_no_pad_forward(
            query, key, value, query_padding_mask, key_padding_mask,
            causal, dropout_p, softmax_scale, deterministic)
