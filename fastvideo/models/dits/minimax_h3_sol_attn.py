# SPDX-License-Identifier: Apache-2.0
"""Sol-Attn block-sparse attention hook for the Open-H3 refine (GB10 / sm_121).

This wires the reference `sol_attn` backend (the training-free sparse attention
NVlabs used for a 2.8x refine speedup) into MiniMax H3 self-attention.  It is
opt-in and off by default:

    FASTVIDEO_H3_SOL_ATTN=sol_engine            step 0 dense, steps 1-2 tau 1.0
    FASTVIDEO_H3_SOL_ATTN=sol_engine_fast       every step tau 1.75
    FASTVIDEO_H3_SOL_ATTN=sol_engine_tau0       step 0 tau 0.0, steps 1-2 tau 1.0
    FASTVIDEO_H3_SOL_ATTN=sol_engine_fast_tau0  step 0 tau 0.0, steps 1-2 tau 1.75

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

# Per-step tau schedule for each preset.  ``None`` marks a dense step; a float is
# the Sol-Attn threshold multiplier ``mean + tau*std`` on the proxy scores used on
# that refine step.  The schedule is general in the sequence length and in the
# number of refine steps (a missing entry falls back to the last defined value),
# so the IC two-row-group path is not special-cased.
_PRESETS: dict[str, tuple[float | None, ...]] = {
    "sol_engine": (None, 1.0, 1.0),
    "sol_engine_fast": (1.75, 1.75, 1.75),
    # Mild variant: the first refine step is sparse at tau=0.0.  Measured on real
    # q/k/v (S=133652, leading 452 sink rows) tau=0.0 is 1.6x cuDNN and keeps every
    # block within 0.013-0.043 rel L2 of dense (vs 0.041-0.127 at tau=1.0).
    "sol_engine_tau0": (0.0, 1.0, 1.0),
    "sol_engine_fast_tau0": (0.0, 1.75, 1.75),
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


def _tau_for_step(step: int | None) -> float | None:
    """Scheduled tau for refine ``step`` (``None`` = dense), or ``None`` if disabled."""
    name = mode()
    if name is None:
        return None
    schedule = _PRESETS[name]
    if step is None:
        step = 0
    idx = max(0, int(step))
    if idx >= len(schedule):
        idx = len(schedule) - 1
    return schedule[idx]


def sparse_step(step: int) -> bool:
    """Whether refine step ``step`` (0-based) should use sparse attention."""
    return _tau_for_step(step) is not None


def tau(step: int | None = None) -> float:
    """Sol-Attn tau for refine ``step``; only meaningful on a sparse step."""
    value = _tau_for_step(step)
    return 0.0 if value is None else float(value)


def sol_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  step: int | None = None) -> torch.Tensor:
    """Run Sol-Attn on contiguous ``[B, S, H, 128]`` bf16 Q/K/V.

    The callers pass the post-QK-norm/RoPE tensors; they are already contiguous
    for the H3 prologue and the eager path.  Output is ``[B, S, H, 128]`` bf16.
    """
    from sol_attn import sol_attn

    tau_val = tau(step)
    global _logged
    if not _logged:
        _logged = True
        print(f"[sol-attn] first sparse call: B={query.shape[0]} S={query.shape[1]} H={query.shape[2]} "
              f"sink={get_sink_tokens()} step={step} tau={tau_val} preset={mode()}", flush=True)
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
    return sol_attn(
        query,
        key,
        value,
        thresh_type="exact",
        tau=tau_val,
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
