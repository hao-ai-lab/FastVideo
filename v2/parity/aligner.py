"""ParityAligner — parity is measured, never assumed.

A read-only observer: *record mode* dumps named taps per step/block from a reference
(the official framework, or a pre-change build); *compare mode* replays with fixed seeds
and reports the first divergence beyond per-tap tolerance. This is the engine behind
the "old loop vs new loop, bit-identical" gate and the standing instrument for every
port, precision change, and kernel swap.
"""
from __future__ import annotations

from typing import Any

from v2.parity.ladder import ConsistencyLevel, Divergence, array_diff


class ParityAligner:
    """Observer that records named taps and compares two runs."""

    def __init__(self, name: str = "parity", default_atol: float = 0.0, default_rtol: float = 0.0):
        self.name = name
        self.default_atol = default_atol
        self.default_rtol = default_rtol
        self.taps: dict[tuple[int, str], Any] = {}  # (step, tap_name) -> value
        self.tolerances: dict[str, tuple[float, float]] = {}  # tap_name -> (atol, rtol)

    def set_tolerance(self, tap_name: str, atol: float = 0.0, rtol: float = 0.0) -> None:
        self.tolerances[tap_name] = (atol, rtol)

    # observer hook (the loop/engine calls this) ------------------------------ #
    def record_tap(self, step: int, tap_name: str, value: Any) -> None:
        self.taps[(step, tap_name)] = value

    def observe(self, event: str, **kw) -> None:
        if event == "tap":
            self.record_tap(kw.get("step", 0), kw["name"], kw["value"])

    def first_divergence(self,
                         reference: ParityAligner,
                         level: ConsistencyLevel = ConsistencyLevel.C1) -> Divergence | None:
        """Compare this (current) run against a reference, reporting the FIRST divergence
        in step order beyond the per-tap tolerance."""
        for (step, tap_name) in sorted(reference.taps, key=lambda k: (k[0], k[1])):
            ref_val = reference.taps[(step, tap_name)]
            if (step, tap_name) not in self.taps:
                return Divergence(f"{tap_name}@{step}", level, float("inf"), float("inf"), "tap missing in current run")
            cur_val = self.taps[(step, tap_name)]
            atol, rtol = self.tolerances.get(tap_name, (self.default_atol, self.default_rtol))
            abs_d, rel_d = array_diff(ref_val, cur_val)
            if not (abs_d <= atol or rel_d <= rtol):  # diverged: outside BOTH tolerances
                return Divergence(f"{tap_name}@{step}", level, abs_d, rel_d,
                                  f"abs={abs_d:.3e} rel={rel_d:.3e} > (atol={atol},rtol={rtol})")
        return None
