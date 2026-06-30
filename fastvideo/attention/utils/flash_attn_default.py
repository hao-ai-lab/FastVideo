# SPDX-License-Identifier: Apache-2.0
"""torch.compile-traceable wrapper for the FA2/FA3/FA4 default attention path.

The FA4/cute path (`fa_version == "4"`) is already a registered
`torch.library.custom_op` in `fastvideo.attention.utils.flash_attn_cute`, so
dynamo treats it as a graph node. The external FA2/FA3 ``flash_attn_func`` is
NOT — dynamo breaks the graph at the call site (observed: wanvideo.py
self-attn, once per layer every step), which fragments the compiled region
and blocks CUDA-graph capture. Wrap the FA2/FA3 default call in a custom op
(mirrors the FP4 `flash_attn_cute` template) so it becomes an
opaque-but-traceable node. The kernel still runs eager inside the op
(correct — flash-attn must run eager); only dynamo's treatment of the
boundary changes, so numerics are unchanged (SSIM-gated).

Autograd: FA2 has full ``register_autograd`` parity — the custom op's
backward calls flash_attn's ``_flash_attn_backward`` directly, so training
backprops *through* the op (no graph break on the training path either).
FA3 currently keeps the no-backward + carve-out pattern from PR #1373
because FA3's private backward signature wants validation on a real Hopper
box (gated on Kuan-Hao's Modal FA3 setup PR). Once that lands the FA3 path
can mirror FA2.

Lives in `attention/utils/` (sibling of `flash_attn_cute.py` and
`flash_attn_no_pad.py`) so it can be imported by any backend that wants the
traceable FA default call without pulling in backend dispatch logic. The
backend (`attention/backends/flash_attn.py`) just imports
`flash_attn_func_compilable` and `fa_version` from here.
"""

import torch

# Pick the same backend the rest of FastVideo picked for `flash_attn_func`
# (FA4/cute → FA3 → FA2). Mirror the precedence used in
# `attention/utils/flash_attn_no_pad.py` so the two probes always agree.
try:
    from fastvideo.attention.utils.flash_attn_cute import flash_attn_func

    fa_version = "4"
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_3_func

        # flash_attn 3 no longer has a different API, see following commit:
        # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
        flash_attn_func = flash_attn_3_func
        fa_version = "3"
    except ImportError:
        from flash_attn import flash_attn_func as flash_attn_2_func
        flash_attn_func = flash_attn_2_func
        fa_version = "2"

if fa_version == "2":
    # Scope: this op covers exactly the q/k/v + softmax_scale + causal call
    # shape used by FlashAttentionImpl.forward's default branch (see
    # `flash_attn_func_compilable(...)` call site in
    # `attention/backends/flash_attn.py`). The masked/no-pad and varlen /
    # cross-attn paths use different entry points
    # (`flash_attn_no_pad`, `flash_attn_varlen_*`) which live in
    # `attention/utils/flash_attn_no_pad.py`. The wrapper's signature is the
    # contract: any extra kwarg (dropout_p, window_size, alibi_slopes,
    # deterministic, return_attn_probs, ...) raises TypeError at the call
    # site, so silent loss of kwargs is not a failure mode.
    from flash_attn.flash_attn_interface import _flash_attn_backward as _fa2_backward
    _fa_default = flash_attn_func

    @torch.library.custom_op(
        "fastvideo::_flash_attn_default_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def _flash_attn_default_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `return_attn_probs=True` asks FA2 to also return softmax_lse +
        # S_dmask. We need softmax_lse to feed the backward; S_dmask is the
        # dropout mask (always None here since dropout_p is fixed at 0).
        out, softmax_lse, _ = _fa_default(q, k, v, softmax_scale=softmax_scale, causal=causal, return_attn_probs=True)
        return out, softmax_lse

    @torch.library.register_fake("fastvideo::_flash_attn_default_forward")
    def _flash_attn_default_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del softmax_scale, causal
        # FA2 default path: out = [batch, seqlen_q, nheads, head_dim_v],
        # softmax_lse = [batch, nheads, seqlen_q], fp32 regardless of q dtype.
        b, sq, hq = q.shape[0], q.shape[1], q.shape[2]
        out = q.new_empty(b, sq, hq, v.shape[-1])
        lse = q.new_empty(b, hq, sq, dtype=torch.float32)
        return out, lse

    def _flash_attn_default_setup_context(ctx, inputs, output):
        q, k, v, softmax_scale, causal = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        # `lse` is an auxiliary output we save to feed FA2's backward; nobody
        # should differentiate through it. Mark it non-differentiable so
        # autograd errors loudly if a caller wires it into a loss, rather
        # than silently producing zero/None grads through the `del grad_lse`
        # in our backward.
        ctx.mark_non_differentiable(lse)
        # FA2's *forward* substitutes `1 / sqrt(head_dim)` for `softmax_scale=None`
        # internally; FA2's *backward* (`_flash_attn_backward`) demands a concrete
        # float in its C++ schema and rejects None at the binding boundary. Resolve
        # the default here so the value saved on ctx (and passed to backward) is
        # always a real float — matches what FA2's own autograd.Function does.
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**-0.5
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal

    def _flash_attn_default_backward(ctx, grad_out, grad_lse):
        # We only differentiate `out`; softmax_lse is saved-for-backward, not
        # a real differentiable output. (Mirrors the FP4 cute template.)
        del grad_lse
        q, k, v, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # FA2's `_flash_attn_backward` writes into dq/dk/dv in place. The
        # extra kwargs (window_size_*, softcap, alibi_slopes, deterministic,
        # rng_state) are pinned to the same defaults the forward wrapper
        # uses — flash-attn==2.8.1 (the version FastVideo pins) requires
        # all of them explicitly. `rng_state=None` is correct for our
        # `dropout_p=0` configuration.
        _fa2_backward(
            grad_out,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            dropout_p=0.0,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            rng_state=None,
        )
        return dq, dk, dv, None, None

    torch.library.register_autograd(
        "fastvideo::_flash_attn_default_forward",
        _flash_attn_default_backward,
        setup_context=_flash_attn_default_setup_context,
    )

    def flash_attn_func_compilable(q, k, v, softmax_scale=None, causal=False):
        # Backward is registered: autograd flows through the op (training
        # path is also traceable; no carve-out needed). Public API matches
        # `flash_attn_func` — returns just `out`; we drop the saved-for-
        # backward `lse` here so callers see the original single-tensor
        # contract.
        out, _ = torch.ops.fastvideo._flash_attn_default_forward(q, k, v, softmax_scale, causal)
        return out
elif fa_version == "3":
    # FA3 path: same forward+fake custom op as the original PR #1373, with
    # the autograd carve-out kept. The full backward (mirroring the FA2 leg
    # above) wants a Hopper box for grad-check validation, which we don't
    # have until Kuan-Hao's Modal FA3 setup PR lands. Until then this keeps
    # inference traceable + training correct (via the original
    # autograd.Function path + a pre-PR-style graph break on training).
    _fa_default = flash_attn_func

    @torch.library.custom_op(
        "fastvideo::_flash_attn_default_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def _flash_attn_default_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        return _fa_default(q, k, v, softmax_scale=softmax_scale, causal=causal)

    @torch.library.register_fake("fastvideo::_flash_attn_default_forward")
    def _flash_attn_default_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        del softmax_scale, causal
        return q.new_empty(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])

    def flash_attn_func_compilable(q, k, v, softmax_scale=None, causal=False):
        # Autograd carve-out. The custom op above registers a forward + fake
        # kernel but NO backward (register_autograd), so it is opaque to
        # autograd. Inference runs under no_grad / inference_mode and routes
        # through the traceable custom op — that is the torch.compile win, and
        # the only path this PR claims. Training backprops through attention,
        # so route grad-enabled calls to the original FA2/FA3 `flash_attn_func`
        # (itself an autograd.Function, so backward is correct) at the cost of a
        # dynamo graph break on the training path — i.e. pre-PR behavior, no
        # regression. Full autograd parity for the custom op (mirroring the FP4
        # cute template) is a tracked follow-up.
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            return _fa_default(q, k, v, softmax_scale=softmax_scale, causal=causal)
        return torch.ops.fastvideo._flash_attn_default_forward(q, k, v, softmax_scale, causal)
elif fa_version == "4":
    # FA4 path: `flash_attn_func` is already a torch.library custom op
    # (registered in `fastvideo.attention.utils.flash_attn_cute`), so a
    # passthrough is enough — no extra registration needed.
    def flash_attn_func_compilable(q, k, v, softmax_scale=None, causal=False):
        return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
else:
    # Defensive: the probe above only ever sets fa_version to "2", "3",
    # or "4"; an unexpected value means an import/probe regression and
    # we want a loud error at import, not a silent NameError later.
    raise RuntimeError(f"Unsupported FlashAttention version: {fa_version!r} — expected "
                       f"'2', '3', or '4' from the import probe above.")
