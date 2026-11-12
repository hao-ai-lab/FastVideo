# SPDX-License-Identifier: Apache-2.0
"""Sol-Attn block-sparse attention hook for the Open-H3 refine (GB10 / sm_121).

This wires the reference `sol_attn` backend (the training-free sparse attention
NVlabs used for a 2.8x refine speedup) into MiniMax H3 self-attention.  It is
opt-in and off by default:

    FASTVIDEO_H3_SOL_ATTN=sol_engine       refine step 0 dense, steps 1-2 tau 1.0
    FASTVIDEO_H3_SOL_ATTN=sol_engine_fast  every refine step tau 1.75

``thresh_type="exact"``, and the leading ``[text | condition | audio]`` rows are
kept exact as attention sinks (NVlabs' ``sink_tokens``).  The dense steps and
every other pipeline (IC, t2v) are untouched, so this is an approximate-tier
opt-in only.

The module never imports `sol_attn` at import time so that the exact stack does
not depend on the optional package; :func:`sol_attention` imports lazily.
"""

from __future__ import annotations

import os

import torch

# Presets mirror NVlabs' open_h3/sol_engine_fwd.py SPARSE table: `dense_steps`
# counts the leading refine steps that stay dense (0-based), `tau` is the
# threshold multiplier for the sparse steps.
_PRESETS: dict[str, dict[str, float | int]] = {
    "sol_engine": {
        "dense_steps": 1,
        "tau": 1.0,
    },
    "sol_engine_fast": {
        "dense_steps": 0,
        "tau": 1.75,
    },
}

_sink_tokens = 0
_logged = False


def mode() -> str | None:
    """Return the active preset name, or ``None`` when Sol-Attn is disabled."""
    value = os.environ.get("FASTVIDEO_H3_SOL_ATTN", "").strip().lower()
    if value in ("", "0", "none", "off", "false"):
        return None
    if value not in _PRESETS:
        raise ValueError(
            f"FASTVIDEO_H3_SOL_ATTN must be one of {sorted(_PRESETS)} or empty, got {value!r}")
    return value


def available() -> bool:
    """Whether the optional ``sol_attn`` package can be imported."""
    try:
        import sol_attn  # noqa: F401
    except Exception:
        return False
    return True


def set_sink_tokens(tokens: int) -> None:
    """Set how many leading tokens are always attended exactly (sinks)."""
    global _sink_tokens
    _sink_tokens = max(0, int(tokens))


def get_sink_tokens() -> int:
    return _sink_tokens


def sparse_step(step: int) -> bool:
    """Whether refine step ``step`` (0-based) should use sparse attention."""
    name = mode()
    if name is None:
        return False
    return int(step) >= int(_PRESETS[name]["dense_steps"])


def tau() -> float:
    name = mode()
    if name is None:
        return 0.0
    return float(_PRESETS[name]["tau"])


def sol_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Run Sol-Attn on contiguous ``[B, S, H, 128]`` bf16 Q/K/V.

    The callers pass the post-QK-norm/RoPE tensors; they are already contiguous
    for the H3 prologue and the eager path.  Output is ``[B, S, H, 128]`` bf16.
    """
    from sol_attn import sol_attn

    global _logged
    if not _logged:
        _logged = True
        print(f"[sol-attn] first sparse call: B={query.shape[0]} S={query.shape[1]} H={query.shape[2]} "
              f"sink={get_sink_tokens()} tau={tau()} preset={mode()}", flush=True)
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
    return sol_attn(
        query,
        key,
        value,
        thresh_type="exact",
        tau=tau(),
        sink_tokens=get_sink_tokens(),
        sink_start=0,
    )


__all__ = [
    "available",
    "get_sink_tokens",
    "mode",
    "set_sink_tokens",
    "sol_attention",
    "sparse_step",
    "tau",
]
