from __future__ import annotations

from collections.abc import Callable

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

if torch.cuda.is_available():
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except ImportError:
        # flash_attn.cute (FA4) is simply not installed -- expected on builds
        # without it; callers handle the ImportError (the FASTVIDEO_FA4 gate in
        # flash_attn.py raises, the FP4 probe treats FA4 as unavailable).
        raise
    except Exception as e:
        # flash_attn.cute IS installed but failed to import -- almost always an
        # nvidia-cutlass-dsl (CuTe DSL) version skew, e.g. "module
        # 'cutlass.cute.core' has no attribute 'ThrMma'" (an AttributeError, not
        # ImportError). This is fixable by pinning a compatible
        # nvidia-cutlass-dsl, so warn loudly, then re-raise as ImportError so
        # callers can handle it uniformly.
        logger.warning(
            "flash_attn.cute (FA4) is installed but failed to import (%r). "
            "This is usually an nvidia-cutlass-dsl version mismatch -- pin a "
            "compatible nvidia-cutlass-dsl to restore FA4.", e)
        raise ImportError(f"flash_attn.cute (FA4) import failed: {e!r}") from e
else:
    # This error will be caught in flash_attn.py or flash_attn_no_pad.py
    raise ImportError("flash_attn.cute is only available on CUDA devices; this error must be handled internally")

try:
    # FA2 serves grad-enabled calls (FA4 cute's backward asserts sm90+ and is
    # unvalidated for training in this repo). Optional so FA4-only installs
    # can still import this module for inference.
    from flash_attn import flash_attn_func as _flash_attn_2_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_2_varlen_func
except ImportError:
    _flash_attn_2_func = None
    _flash_attn_2_varlen_func = None


def _check_dropout(dropout_p: float) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError(f"flash_attn.cute does not support dropout (got dropout_p={dropout_p})")


def _use_fa2(*tensors: torch.Tensor) -> bool:
    # Grad-enabled calls are served by FA2 on every device: the FA4 custom ops
    # below are inference-only (no registered backward).
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _fa2_or_raise(fa2_func: Callable | None) -> Callable:
    if fa2_func is None:
        raise RuntimeError("grad-enabled attention on the FA4 cute path requires FlashAttention-2, "
                           "which is not installed; the FA4 custom ops are inference-only.")
    return fa2_func


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
    # _flash_attn_fwd returns (out, lse) on its empty-sequence early path but
    # (out, lse, p, row_max) on the main path at the pinned FA4 cute ref; take
    # the first two so both arities work.
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
    )[:2]
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
    )[:2]
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
    if _use_fa2(q, k, v):
        return _fa2_or_raise(_flash_attn_2_func)(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
    _check_dropout(dropout_p)
    out, _ = torch.ops.fastvideo._flash_attn_cute_forward(q, k, v, softmax_scale, causal, deterministic)
    return out


# ---------------------------------------------------------------------------
# FP4 (NVFP4 block-scaled) variant
# ---------------------------------------------------------------------------
# The FP4 path needs the mSFQ/mSFK scale-factor tensors that the regular
# wrapper does not expose. We register a separate custom op so that
# torch.compile can treat the kernel as an opaque boundary (the underlying
# CuTeDSL kernel uses cuda.CUstream which dynamo cannot trace).


@torch.library.custom_op(
    "fastvideo::_flash_attn_cute_fp4_forward",
    mutates_args=(),
    device_types="cuda",
)
def _flash_attn_cute_fp4_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor,
    sfk: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    out = _flash_attn_fwd(
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
        mSFQ=sfq,
        mSFK=sfk,
    )[0]
    return out


@torch.library.register_fake("fastvideo::_flash_attn_cute_fp4_forward")
def _flash_attn_cute_fp4_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor,
    sfk: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    del k, sfq, sfk, softmax_scale, causal
    # q is FP4 packed: shape (batch, seqlen, nheads, headdim/2). Output is in
    # V's dtype with full headdim.
    batch, seqlen_q, nheads = q.shape[:3]
    return v.new_empty(batch, seqlen_q, nheads, v.shape[-1])


def flash_attn_fp4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor,
    sfk: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """FP4 (NVFP4 block-scaled) flash attention. q/k are FP4-packed; v is BF16."""
    return torch.ops.fastvideo._flash_attn_cute_fp4_forward(q, k, v, sfq, sfk, softmax_scale, causal)


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
    if _use_fa2(q, k, v):
        return _fa2_or_raise(_flash_attn_2_varlen_func)(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
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
