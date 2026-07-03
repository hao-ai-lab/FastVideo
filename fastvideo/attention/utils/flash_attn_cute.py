from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from flash_attn import flash_attn_func as _flash_attn_2_func
from flash_attn import flash_attn_varlen_func as _flash_attn_2_varlen_func

from fastvideo.logger import init_logger

logger = init_logger(__name__)

if torch.cuda.is_available():
    try:
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
    except ImportError:
        # flash_attn.cute (FA4) is simply not installed -- expected on builds
        # without it; callers fall back to FA3/FA2 quietly.
        raise
    except Exception as e:
        # flash_attn.cute IS installed but failed to import -- almost always an
        # nvidia-cutlass-dsl (CuTe DSL) version skew, e.g. "module
        # 'cutlass.cute.core' has no attribute 'ThrMma'" (an AttributeError, not
        # ImportError). This is fixable by pinning a compatible
        # nvidia-cutlass-dsl, so warn loudly, then re-raise as ImportError so
        # callers fall back to FA3/FA2 instead of crashing worker init.
        logger.warning(
            "flash_attn.cute (FA4) is installed but failed to import (%r); "
            "falling back to FA3/FA2. This is usually an nvidia-cutlass-dsl "
            "version mismatch -- pin a compatible nvidia-cutlass-dsl to "
            "restore FA4.", e)
        raise ImportError(f"flash_attn.cute (FA4) import failed: {e!r}") from e
else:
    # This error will be caught in flash_attn.py or flash_attn_no_pad.py
    raise ImportError("flash_attn.cute is only available on CUDA devices; this error must be handled internally")


def _check_dropout(dropout_p: float) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError(f"flash_attn.cute does not support dropout (got dropout_p={dropout_p})")


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


def _flash_attn_cute_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output) -> None:
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


def _flash_attn_cute_varlen_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output) -> None:
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


# FA4's CuTeDSL kernels JIT-compile per shape family, and some configurations
# fail MLIR op creation at runtime even though the import succeeded (observed:
# GQA models on sm_89 dying in pack_gqa with "ValueError: Operation creation
# failed"). Degrade to FA2 once, process-wide, instead of crashing inference.
class _FA4Policy:
    """Per-call gate for the FA4 cute fast path, with FA2 as the fallback.

    FA4 is skipped when:
      * a previous call failed at runtime -- CuTeDSL JIT compilation is
        shape-dependent, so the first failure disables FA4 for the rest of
        the process instead of retrying a broken JIT on every call; or
      * the call needs autograd -- FA4's backward asserts sm90+ (L40S/sm_89
        dies on its arch check) and is unvalidated for training in this repo
        (its lse is not even allocated through our inference-shaped custom
        op), so training keeps the pre-FA4 behavior: FA2 on every device.
    """

    def __init__(self) -> None:
        self.broken = False

    def use_fa4(self, *tensors: torch.Tensor) -> bool:
        if self.broken:
            return False
        return not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))

    def mark_broken(self, error: Exception) -> None:
        if not self.broken:
            self.broken = True
            logger.warning(
                "flash_attn.cute (FA4) failed at runtime (%r); falling back "
                "to FA2 for the rest of this process.", error)


_FA4 = _FA4Policy()


def _with_fa2_fallback(fa2_func: Callable) -> Callable:
    """Pair an FA4 cute wrapper with its FA2 twin of the same signature.

    The decorated body runs only when ``_FA4`` allows it; otherwise (or after
    the first FA4 runtime failure) the call is served by ``fa2_func``.
    ``NotImplementedError`` is a contract error (e.g. dropout), not a JIT
    failure, so it propagates without disabling FA4.
    """

    def decorator(fa4_func: Callable) -> Callable:

        @functools.wraps(fa4_func)
        def wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if _FA4.use_fa4(q, k, v):
                try:
                    return fa4_func(q, k, v, *args, **kwargs)
                except NotImplementedError:
                    raise
                except Exception as e:  # CuTeDSL compile errors surface as ValueError
                    _FA4.mark_broken(e)
            return fa2_func(q, k, v, *args, **kwargs)

        return wrapper

    return decorator


@_with_fa2_fallback(_flash_attn_2_func)
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


@_with_fa2_fallback(_flash_attn_2_varlen_func)
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
