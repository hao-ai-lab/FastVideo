# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.callbacks.grad_clip`.

Exercises ``GradNormClipCallback.on_before_optimizer_step`` against
synthetic ``nn.Module`` targets with manually populated gradients.
"""
from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.callbacks.grad_clip import GradNormClipCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingTracker:
    """Tracker stub that records every ``log`` call."""

    def __init__(self) -> None:
        self.entries: list[tuple[dict[str, Any], int]] = []

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.entries.append((payload, step))


class _Method:
    """Minimal stand-in for ``TrainingMethod``."""

    def __init__(
        self,
        targets: dict[str, torch.nn.Module],
        tracker: Any | None = None,
    ) -> None:
        self._targets = targets
        self.tracker = tracker
        self.iter_seen: int | None = None

    def get_grad_clip_targets(
        self, iteration: int
    ) -> dict[str, torch.nn.Module]:
        self.iter_seen = iteration
        return self._targets


def _make_module(*, grad_value: float, n: int = 4) -> torch.nn.Module:
    """Return an ``nn.Linear`` whose grads are filled with ``grad_value``."""
    m = torch.nn.Linear(n, n, bias=False)
    m.weight.grad = torch.full_like(m.weight, fill_value=grad_value)
    return m


def _grad_norm(module: torch.nn.Module) -> float:
    flat = torch.cat([p.grad.flatten() for p in module.parameters()])
    return float(flat.norm(2).item())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGradNormClipCallback:

    def test_disabled_when_max_norm_non_positive(self) -> None:
        m = _make_module(grad_value=10.0)
        before = _grad_norm(m)

        cb = GradNormClipCallback(max_grad_norm=0.0)
        method = _Method(targets={"m": m})
        cb.on_before_optimizer_step(method=method, iteration=0)

        # No clipping applied; ``get_grad_clip_targets`` not consulted.
        assert _grad_norm(m) == before
        assert method.iter_seen is None

    def test_large_grads_get_clipped(self) -> None:
        m = _make_module(grad_value=10.0)
        assert _grad_norm(m) > 1.0

        cb = GradNormClipCallback(max_grad_norm=1.0)
        cb.on_before_optimizer_step(
            method=_Method(targets={"m": m}), iteration=0
        )

        # After clipping the L2 norm should not exceed max_grad_norm
        # (allow a tiny epsilon for the +1e-6 in the clip helper).
        assert _grad_norm(m) <= 1.0 + 1e-4

    def test_small_grads_unchanged(self) -> None:
        m = _make_module(grad_value=0.01)
        before = _grad_norm(m)
        assert before < 1.0

        cb = GradNormClipCallback(max_grad_norm=1.0)
        cb.on_before_optimizer_step(
            method=_Method(targets={"m": m}), iteration=0
        )

        # Clip coef >1 is clamped to 1, so values are preserved
        # (modulo the *1.0 multiply, which is exact for floats).
        assert abs(_grad_norm(m) - before) < 1e-6

    def test_iteration_forwarded_to_targets(self) -> None:
        m = _make_module(grad_value=0.5)
        cb = GradNormClipCallback(max_grad_norm=1.0)
        method = _Method(targets={"m": m})
        cb.on_before_optimizer_step(method=method, iteration=42)
        assert method.iter_seen == 42

    def test_tracker_logged_when_enabled(self) -> None:
        m = _make_module(grad_value=5.0)
        tracker = _RecordingTracker()
        cb = GradNormClipCallback(
            max_grad_norm=1.0, log_grad_norms=True
        )
        cb.on_before_optimizer_step(
            method=_Method(targets={"layer": m}, tracker=tracker),
            iteration=7,
        )
        assert len(tracker.entries) == 1
        payload, step = tracker.entries[0]
        assert step == 7
        assert "grad_norm/layer" in payload
        assert payload["grad_norm/layer"] > 0.0

    def test_tracker_not_logged_when_disabled(self) -> None:
        m = _make_module(grad_value=5.0)
        tracker = _RecordingTracker()
        cb = GradNormClipCallback(
            max_grad_norm=1.0, log_grad_norms=False
        )
        cb.on_before_optimizer_step(
            method=_Method(targets={"m": m}, tracker=tracker),
            iteration=0,
        )
        assert tracker.entries == []

    def test_no_tracker_does_not_raise(self) -> None:
        m = _make_module(grad_value=5.0)
        cb = GradNormClipCallback(max_grad_norm=1.0, log_grad_norms=True)
        # Method without a tracker attribute at all.

        class _BareMethod:

            def get_grad_clip_targets(
                self, iteration: int
            ) -> dict[str, torch.nn.Module]:
                return {"m": m}

        cb.on_before_optimizer_step(method=_BareMethod(), iteration=0)
        # No assertion — must simply not raise.

    def test_multiple_targets_each_logged(self) -> None:
        targets = {
            "head": _make_module(grad_value=4.0),
            "tail": _make_module(grad_value=8.0),
        }
        tracker = _RecordingTracker()
        cb = GradNormClipCallback(max_grad_norm=1.0)
        cb.on_before_optimizer_step(
            method=_Method(targets=targets, tracker=tracker),
            iteration=1,
        )
        keys = {next(iter(p)) for p, _ in tracker.entries}
        assert keys == {"grad_norm/head", "grad_norm/tail"}
