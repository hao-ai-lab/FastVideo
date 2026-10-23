# SPDX-License-Identifier: Apache-2.0
"""Options for MiniMax-H3 training-free token reduction, parsed without importing model code.

``FastVideoArgs`` validates ``h3_token_reduction`` here, so building args never
imports a model module. The row maps and the pooling arithmetic live in
``fastvideo/models/dits/minimax_h3_token_reduction.py``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TOKEN_REDUCTION_OPTION_KEYS = frozenset({"enabled", "blocks", "early", "update_scale"})


@dataclass(frozen=True)
class MiniMaxH3TokenReductionSchedule:
    """Which DiT blocks run on paired tokens at each denoising step.

    Blocks ``begin <= block < end`` run reduced. During the first
    ``early_steps`` steps the interval extends to ``early_end``.

    h3.c reduces blocks 4..29, and 4..39 during its first 10 of about 19
    forwards, where coarse structure forms. The default keeps that share of an
    eight-forward PDD schedule: blocks 4..39 for the first four forwards.
    """

    begin: int = 4
    end: int = 30
    early_steps: int = 4
    early_end: int = 40
    update_scale: float = 1.0

    @classmethod
    def from_options(cls,
                     options: Mapping[str, Any],
                     num_blocks: int | None = None) -> MiniMaxH3TokenReductionSchedule | None:
        """Parse ``h3_token_reduction`` options; None unless ``enabled`` is true.

        ``num_blocks``, when known, bounds both block intervals by the DiT depth.
        """
        if not isinstance(options, Mapping) or options.keys() - TOKEN_REDUCTION_OPTION_KEYS:
            raise ValueError(
                f"h3_token_reduction must be a mapping with keys from {sorted(TOKEN_REDUCTION_OPTION_KEYS)}")
        enabled = options.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("h3_token_reduction.enabled must be a boolean")
        if not enabled:
            return None
        begin, end = _int_pair(options.get("blocks", (cls.begin, cls.end)), "blocks")
        if not 0 <= begin < end:
            raise ValueError(f"h3_token_reduction.blocks must satisfy 0 <= BEGIN < END, got {begin}:{end}")
        # Same rule as h3.c: the deeper early interval applies only when the
        # base interval restores before block 40.
        default_early = (cls.early_steps, cls.early_end) if end < cls.early_end else (0, end)
        early_steps, early_end = _int_pair(options.get("early", default_early), "early")
        if early_steps < 0:
            raise ValueError("h3_token_reduction.early STEPS must be non-negative")
        if early_steps == 0:
            early_end = end
        elif early_end <= end:
            raise ValueError(f"h3_token_reduction.early END must exceed blocks END {end}, got {early_end}")
        if num_blocks is not None and max(end, early_end) > num_blocks:
            raise ValueError(f"h3_token_reduction block intervals end at {max(end, early_end)}, past the "
                             f"{num_blocks} DiT blocks")
        scale = options.get("update_scale", cls.update_scale)
        if isinstance(scale, bool) or not isinstance(scale, int | float) or not 0.0 <= float(scale) <= 2.0:
            raise ValueError("h3_token_reduction.update_scale must be a number in [0, 2]")
        return cls(begin=begin, end=end, early_steps=early_steps, early_end=early_end, update_scale=float(scale))

    def end_for_step(self, step: int) -> int:
        return self.early_end if step < self.early_steps else self.end


def _int_pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        value = value.split(":")
    if (not isinstance(value, list | tuple) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int | str) for item in value)):
        raise ValueError(f"h3_token_reduction.{name} must be two integers, e.g. [4, 30] or '4:30'")
    try:
        return int(value[0]), int(value[1])
    except ValueError as error:
        raise ValueError(f"h3_token_reduction.{name} must be two integers, got {value!r}") from error


__all__ = ["MiniMaxH3TokenReductionSchedule", "TOKEN_REDUCTION_OPTION_KEYS"]
