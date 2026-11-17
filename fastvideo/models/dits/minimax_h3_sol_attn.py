# SPDX-License-Identifier: Apache-2.0
"""Sol-Attn block-sparse attention for the Open-H3-Regenerate-2K refine.

Wires NVlabs' training-free Sol-Attn (``NVlabs/Sana`` ``techniques/sparse_backends``, Apache-2.0) into MiniMax-H3
self-attention. It is opt-in through ``FASTVIDEO_H3_SOL_ATTN`` and off by default:

    sol_engine            step 0 dense, steps 1-2 tau 1.0 (NVlabs' default refine preset)
    sol_engine_fast       every step tau 1.75 (NVlabs' fastest refine preset)
    sol_engine_tau0       step 0 tau 0.0, steps 1-2 tau 1.0
    sol_engine_fast_tau0  step 0 tau 0.0, steps 1-2 tau 1.75

``thresh_type="exact"``, and the leading ``[text | condition | audio]`` rows are kept exact as attention sinks
(NVlabs' ``sink_tokens``), set per request by the refine denoising stage. ``sol_attn`` is imported lazily, so the
dense path never depends on it.
"""

from __future__ import annotations

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Per-step tau for each preset. ``None`` marks a dense step; a float is the Sol-Attn threshold multiplier
# ``mean + tau * std`` on its proxy block scores. A step past the end reuses the last entry, so the schedule does
# not depend on the sequence length or the number of refine steps.
_PRESETS: dict[str, tuple[float | None, ...]] = {
    "sol_engine": (None, 1.0, 1.0),
    "sol_engine_fast": (1.75, 1.75, 1.75),
    # Step 0 sparse at tau 0.0: on real 2K q/k/v this is 1.6x cuDNN with every block within 0.013-0.043 rel L2 of
    # dense (0.041-0.127 at tau 1.0).
    "sol_engine_tau0": (0.0, 1.0, 1.0),
    "sol_engine_fast_tau0": (0.0, 1.75, 1.75),
}

# DGX Spark (GB10) reports compute capability 12.1. Sol-Attn maps only 12.0 to its SM120 CuTe kernel and sends
# everything else to its Triton kernel; GB10 has the same warp-level mma.sync family, and the SM120 kernel matches
# the Triton reference there to three digits while running 1.6x faster at the 2K refine shape.
_GB10_CAPABILITY = (12, 1)

_sink_tokens = 0
_announced = False


def mode() -> str | None:
    """The active preset name, or ``None`` when Sol-Attn is off."""
    value = envs.FASTVIDEO_H3_SOL_ATTN.get()
    return None if value == "off" else value


def set_sink_tokens(tokens: int) -> None:
    """Set how many leading rows every query attends exactly (the refine's text, condition and audio rows)."""
    global _sink_tokens
    _sink_tokens = max(0, int(tokens))


def get_sink_tokens() -> int:
    return _sink_tokens


def _tau_for_step(step: int) -> float | None:
    name = mode()
    if name is None:
        return None
    schedule = _PRESETS[name]
    return schedule[min(max(0, step), len(schedule) - 1)]


def sparse_step(step: int) -> bool:
    """Whether refine step ``step`` (0-based) runs sparse attention."""
    return _tau_for_step(step) is not None


def tau(step: int) -> float:
    """Sol-Attn tau for refine ``step``; only meaningful on a sparse step."""
    value = _tau_for_step(step)
    return 0.0 if value is None else float(value)


def _register_gb10_backend() -> None:
    from sol_attn import interface
    backends = getattr(interface, "_CUTE_BACKENDS", None)
    if isinstance(backends, dict) and (12, 0) in backends:
        backends.setdefault(_GB10_CAPABILITY, backends[(12, 0)])


def sol_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, step: int) -> torch.Tensor:
    """Sol-Attn over ``[B, S, H, 128]`` bf16 Q/K/V (after QK-norm and RoPE); returns ``[B, S, H, 128]``."""
    try:
        from sol_attn import sol_attn
    except ImportError as exc:
        raise ImportError("FASTVIDEO_H3_SOL_ATTN needs the sol_attn package: pip install --no-build-isolation "
                          "'git+https://github.com/NVlabs/Sana.git#subdirectory=techniques/sparse_backends'") from exc
    global _announced
    if not _announced:
        _announced = True
        _register_gb10_backend()
        logger.info("Sol-Attn refine attention: preset=%s, B=%d S=%d H=%d, sink rows=%d", mode(), query.shape[0],
                    query.shape[1], query.shape[2], get_sink_tokens())
    return sol_attn(query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    thresh_type="exact",
                    tau=tau(step),
                    sink_tokens=get_sink_tokens(),
                    sink_start=0)


__all__ = ["get_sink_tokens", "mode", "set_sink_tokens", "sol_attention", "sparse_step", "tau"]
