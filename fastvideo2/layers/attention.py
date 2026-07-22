# Attention backends — the fastvideo-style registry over the official Wan
# attention semantics.
#
# The flash path is vendored from the official Wan2.1 implementation
# (wan/modules/attention.py @ 9737cba, Apache-2.0); the SDPA path is a local,
# mask- and dtype-honoring equivalent. Backend choice is a DEPLOYMENT numerics
# choice inside a measured band (flash vs SDPA = 5.6e-3 rel L2 at the bf16 DiT
# probe; the official goldens are flash-path, so AUTO on a flash-capable box
# is anchor-exact). Selection:
#
#   FASTVIDEO2_ATTENTION_BACKEND = FLASH_ATTN | SDPA        explicit — a
#       requested-but-unavailable backend is a hard error, never a silent
#       fallback (the fastvideo #1494 lesson);
#   unset (AUTO)                                            flash when
#       available on CUDA and not in true-fp32 mode; SDPA for true fp32
#       (the exact-math anchor path — flash cannot run fp32) and off-CUDA.
#
# New backends (Sage, VSA, ...) register here, but serve production only with
# anchor evidence: quantized/sparse attention is a bounded/quality-changing
# substitution, not a drop-in.
"""Attention dispatch. Call :func:`flash_attention` (name kept for the
vendored model's import); q/k/v are ``[B, L, H, C]``, ``k_lens`` masks keys.
torch imports are lazy so the selection logic is testable anywhere.
"""
from __future__ import annotations

import functools
import os
import warnings

__all__ = ["flash_attention", "attention", "select_backend", "backend_policy",
           "available_backends", "BACKENDS"]

_ENV = "FASTVIDEO2_ATTENTION_BACKEND"
BACKENDS = ("FLASH_ATTN", "SDPA")


# --------------------------------------------------------------------------- #
# Selection (pure, torch-free — T0-tested)                                     #
# --------------------------------------------------------------------------- #
def select_backend(requested: str | None, *, flash_available: bool, on_cuda: bool,
                   true_fp32: bool) -> str:
    """Resolve the backend for one call.

    ``requested`` comes from the env override (None = AUTO). Explicit requests
    fail loudly when unsatisfiable. ``true_fp32`` means fp32 inputs with
    autocast off — the exact-math path, which only SDPA can serve; forcing
    FLASH_ATTN there keeps official's own behavior (cast to bf16 + flash).
    """
    if requested is not None:
        req = requested.upper()
        if req not in BACKENDS:
            raise ValueError(f"{_ENV}={requested!r}: unknown backend; known: {list(BACKENDS)}")
        if req == "FLASH_ATTN" and not (flash_available and on_cuda):
            raise RuntimeError(f"{_ENV}=FLASH_ATTN but flash attention is "
                               f"{'unavailable' if not flash_available else 'CUDA-only'} here "
                               f"— refusing to fall back silently")
        return req
    if flash_available and on_cuda and not true_fp32:
        return "FLASH_ATTN"
    return "SDPA"


def backend_policy() -> str:
    """The configured policy, for env fingerprints and traces."""
    return os.environ.get(_ENV, "AUTO").upper()


def available_backends() -> list[str]:
    return [b for b in BACKENDS if b == "SDPA" or _flash_availability()[0]]


@functools.lru_cache(maxsize=1)
def _flash_availability() -> tuple[bool, bool]:
    """(flash2_or_3_available, flash3_available) — probed once, lazily."""
    try:
        import flash_attn_interface  # noqa: F401
        fa3 = True
    except ModuleNotFoundError:
        fa3 = False
    try:
        import flash_attn  # noqa: F401
        fa2 = True
    except ModuleNotFoundError:
        fa2 = False
    return (fa2 or fa3, fa3)


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #
def _sdpa_attention(q, k, v, q_lens=None, k_lens=None, softmax_scale=None, causal=False):
    """Mask- and dtype-honoring SDPA (local backend)."""
    import torch
    qh, kh, vh = (u.transpose(1, 2) for u in (q, k, v))          # [B, H, L, C]
    mask = None
    if k_lens is not None:
        idx = torch.arange(kh.shape[2], device=kh.device)[None]
        mask = (idx < k_lens.to(kh.device)[:, None])[:, None, None]
    out = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=mask, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)


def _flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                     q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                     dtype=None, version=None):
    """The official varlen flash path — vendored verbatim (casts non-half
    inputs to ``dtype`` exactly like upstream)."""
    import torch
    dtype = dtype or torch.bfloat16
    _, fa3 = _flash_availability()
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not fa3:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and fa3:
        import flash_attn_interface
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        import flash_attn
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


# --------------------------------------------------------------------------- #
# Dispatch (the model-facing entrypoint; name kept for the vendored model)     #
# --------------------------------------------------------------------------- #
def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                    dtype=None, version=None):
    import torch
    true_fp32 = q.dtype == torch.float32 and not torch.is_autocast_enabled('cuda')
    backend = select_backend(os.environ.get(_ENV), flash_available=_flash_availability()[0],
                             on_cuda=q.device.type == 'cuda', true_fp32=true_fp32)
    if backend == "SDPA":
        return _sdpa_attention(q, k, v, q_lens=q_lens, k_lens=k_lens,
                               softmax_scale=softmax_scale, causal=causal)
    return _flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                            softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
                            window_size=window_size, deterministic=deterministic,
                            dtype=dtype, version=version)


def attention(q, k, v, **kwargs):
    """Official upstream alias."""
    fa_version = kwargs.pop("fa_version", None)
    return flash_attention(q, k, v, version=fa_version, **kwargs)
