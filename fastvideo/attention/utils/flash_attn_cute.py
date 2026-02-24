from __future__ import annotations

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


def _check_dropout(dropout_p: float) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError(
            f"flash_attn.cute does not support dropout (got dropout_p={dropout_p})"
        )


@torch.library.custom_op(
    "fastvideo::_flash_attn_cute_forward",
    mutates_args=(),
    device_types="cuda",
)
def _flash_attn_cute_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=None,
        window_size_right=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
    )
    return out, lse


@torch.library.register_fake("fastvideo::_flash_attn_cute_forward")
def _flash_attn_cute_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, softmax_scale, causal, deterministic
    batch, seqlen_q, nheads = q.shape[:3]
    out = q.new_empty(batch, seqlen_q, nheads, v.shape[-1])
    lse = q.new_empty(batch, nheads, seqlen_q, dtype=torch.float32)
    return out, lse


def _flash_attn_cute_setup_context(ctx: torch.autograd.function.FunctionCtx,
                                   inputs, output) -> None:
    q, k, v, softmax_scale, causal, deterministic = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.deterministic = deterministic


def _flash_attn_cute_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    grad_lse: torch.Tensor | None,
):
    del grad_lse
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        softmax_scale=ctx.softmax_scale,
        causal=ctx.causal,
        softcap=0.0,
        window_size_left=None,
        window_size_right=None,
        deterministic=ctx.deterministic,
    )
    return dq, dk, dv, None, None, None


torch.library.register_autograd(
    "fastvideo::_flash_attn_cute_forward",
    _flash_attn_cute_backward,
    setup_context=_flash_attn_cute_setup_context,
)


@torch.library.custom_op(
    "fastvideo::_flash_attn_cute_varlen_forward",
    mutates_args=(),
    device_types="cuda",
)
def _flash_attn_cute_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=None,
        window_size_right=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
    )
    return out, lse


@torch.library.register_fake("fastvideo::_flash_attn_cute_varlen_forward")
def _flash_attn_cute_varlen_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale
    del causal
    del deterministic
    total_q, nheads = q.shape[:2]
    out = q.new_empty(total_q, nheads, v.shape[-1])
    lse = q.new_empty(nheads, total_q, dtype=torch.float32)
    return out, lse


def _flash_attn_cute_varlen_setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs, output) -> None:
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        deterministic,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.deterministic = deterministic


def _flash_attn_cute_varlen_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    grad_lse: torch.Tensor | None,
):
    del grad_lse
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        softmax_scale=ctx.softmax_scale,
        causal=ctx.causal,
        softcap=0.0,
        window_size_left=None,
        window_size_right=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=ctx.max_seqlen_q,
        max_seqlen_k=ctx.max_seqlen_k,
        deterministic=ctx.deterministic,
    )
    return dq, dk, dv, None, None, None, None, None, None, None


torch.library.register_autograd(
    "fastvideo::_flash_attn_cute_varlen_forward",
    _flash_attn_cute_varlen_backward,
    setup_context=_flash_attn_cute_varlen_setup_context,
)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
) -> torch.Tensor:
    """Only returns the output, not the lse."""
    _check_dropout(dropout_p)
    out, _ = torch.ops.fastvideo._flash_attn_cute_forward(
        q, k, v, softmax_scale, causal, deterministic)
    return out


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
) -> torch.Tensor:
    """Only returns the output, not the lse."""
    _check_dropout(dropout_p)
    out, _ = torch.ops.fastvideo._flash_attn_cute_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        deterministic,
    )
    return out
