# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.callbacks.ema`.

Exercises the EMA lifecycle (lazy init, ``start_iter`` gating, decay
math, ``ema_context`` swap, state-dict round-trip) on a tiny CPU
``nn.Linear``. ``EMA_FSDP`` works without ``dist.init_process_group``
because ``dist.is_initialized()`` returns False and ``_to_local_tensor``
falls through to raw tensors for non-DTensor inputs.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from fastvideo.train.callbacks.ema import EMACallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Student:

    def __init__(self, transformer: torch.nn.Module | None) -> None:
        self.transformer = transformer


class _RecordingTracker:

    def __init__(self) -> None:
        self.entries: list[tuple[dict[str, Any], int]] = []

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.entries.append((payload, step))


class _Method:

    def __init__(
        self,
        transformer: torch.nn.Module | None,
        tracker: Any | None = None,
    ) -> None:
        self.student = _Student(transformer)
        self.tracker = tracker


def _tiny_transformer(*, fill: float = 0.0) -> torch.nn.Module:
    m = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(fill)
    return m


# ---------------------------------------------------------------------------
# A. on_train_start
# ---------------------------------------------------------------------------


class TestOnTrainStart:

    def test_initializes_ema_from_student(self) -> None:
        transformer = _tiny_transformer(fill=0.5)
        cb = EMACallback(decay=0.9, start_iter=0)
        cb.on_train_start(_Method(transformer), iteration=0)

        assert cb.student_ema is not None
        # Shadow shape matches transformer parameter.
        shadow = cb.student_ema.shadow["weight"]
        assert shadow.shape == transformer.weight.shape
        assert torch.allclose(shadow, transformer.weight.detach().cpu())

    def test_missing_transformer_raises(self) -> None:
        cb = EMACallback()
        with pytest.raises(ValueError, match="No student transformer"):
            cb.on_train_start(_Method(transformer=None), iteration=0)


# ---------------------------------------------------------------------------
# B. on_training_step_end (decay math + start_iter gating)
# ---------------------------------------------------------------------------


class TestOnTrainingStepEnd:

    def test_no_op_before_train_start(self) -> None:
        cb = EMACallback()
        # student_ema is None until on_train_start.
        cb.on_training_step_end(
            _Method(transformer=None), loss_dict={}, iteration=0
        )
        assert not cb._ema_started

    def test_skipped_until_start_iter(self) -> None:
        transformer = _tiny_transformer(fill=1.0)
        cb = EMACallback(decay=0.5, start_iter=10)
        cb.on_train_start(_Method(transformer), iteration=0)

        # Mutate transformer to drift it away from initial shadow.
        with torch.no_grad():
            transformer.weight.fill_(7.0)

        cb.on_training_step_end(
            _Method(transformer), loss_dict={}, iteration=5
        )
        # Below start_iter: shadow is untouched, _ema_started False.
        assert not cb._ema_started
        assert torch.allclose(
            cb.student_ema.shadow["weight"],
            torch.full((2, 4), 1.0),
        )

    def test_first_active_step_reinits_then_updates(self) -> None:
        transformer = _tiny_transformer(fill=1.0)
        cb = EMACallback(decay=0.9, start_iter=10)
        cb.on_train_start(_Method(transformer), iteration=0)

        # Drift transformer so that re-init has a visible effect.
        with torch.no_grad():
            transformer.weight.fill_(5.0)

        cb.on_training_step_end(
            _Method(transformer), loss_dict={}, iteration=10
        )
        # First active step: shadow is re-initialized from the
        # current transformer (5.0) and *then* update() applies decay
        # against the same value, so shadow stays at 5.0.
        assert cb._ema_started
        assert torch.allclose(
            cb.student_ema.shadow["weight"],
            torch.full((2, 4), 5.0),
        )

    def test_subsequent_step_applies_decay(self) -> None:
        transformer = _tiny_transformer(fill=2.0)
        cb = EMACallback(decay=0.9, start_iter=0)
        cb.on_train_start(_Method(transformer), iteration=0)

        # Step 0: re-init at 2.0, then update against 2.0 → still 2.0.
        cb.on_training_step_end(
            _Method(transformer), loss_dict={}, iteration=0
        )
        # Step 1: drift transformer to 12.0, expect
        # shadow = 0.9 * 2.0 + 0.1 * 12.0 = 3.0.
        with torch.no_grad():
            transformer.weight.fill_(12.0)
        cb.on_training_step_end(
            _Method(transformer), loss_dict={}, iteration=1
        )
        assert torch.allclose(
            cb.student_ema.shadow["weight"],
            torch.full((2, 4), 3.0),
            atol=1e-6,
        )

    def test_tracker_logs_decay(self) -> None:
        transformer = _tiny_transformer()
        tracker = _RecordingTracker()
        cb = EMACallback(decay=0.99, start_iter=0)
        method = _Method(transformer, tracker=tracker)
        cb.on_train_start(method, iteration=0)
        cb.on_training_step_end(method, loss_dict={}, iteration=0)

        assert any(
            payload.get("ema/decay") == 0.99 and step == 0
            for payload, step in tracker.entries
        )


# ---------------------------------------------------------------------------
# C. ema_context
# ---------------------------------------------------------------------------


class TestEmaContext:

    def test_passthrough_when_inactive(self) -> None:
        transformer = _tiny_transformer(fill=3.0)
        cb = EMACallback()
        # No on_train_start → student_ema is None.
        with cb.ema_context(transformer) as t:
            assert t is transformer
            assert torch.allclose(
                t.weight, torch.full((2, 4), 3.0)
            )

    def test_swaps_weights_then_restores(self) -> None:
        transformer = _tiny_transformer(fill=1.0)
        cb = EMACallback(decay=0.0, start_iter=0)
        method = _Method(transformer)
        cb.on_train_start(method, iteration=0)

        # decay=0 → after one step the shadow == current weights == 1.0.
        cb.on_training_step_end(method, loss_dict={}, iteration=0)
        # Drift transformer; ema_context should swap shadow (1.0) in
        # for the duration and restore the post-drift value (9.0).
        with torch.no_grad():
            transformer.weight.fill_(9.0)

        with cb.ema_context(transformer) as t:
            assert torch.allclose(t.weight, torch.full((2, 4), 1.0))

        assert torch.allclose(
            transformer.weight, torch.full((2, 4), 9.0)
        )


# ---------------------------------------------------------------------------
# D. State dict round-trip
# ---------------------------------------------------------------------------


class TestStateDict:

    def test_state_dict_empty_before_train_start(self) -> None:
        cb = EMACallback()
        assert cb.state_dict() == {}

    def test_round_trip_preserves_shadow_and_started_flag(self) -> None:
        transformer = _tiny_transformer(fill=4.0)
        cb = EMACallback(decay=0.5, start_iter=0)
        method = _Method(transformer)
        cb.on_train_start(method, iteration=0)
        cb.on_training_step_end(method, loss_dict={}, iteration=0)

        state = cb.state_dict()
        assert "student_ema" in state
        assert state["ema_started"] is True

        # Build a fresh callback and load.
        fresh = EMACallback(decay=0.5, start_iter=0)
        fresh.on_train_start(_Method(_tiny_transformer(fill=0.0)),
                             iteration=0)
        # Sanity: fresh shadow != saved shadow before load.
        assert not torch.allclose(
            fresh.student_ema.shadow["weight"],
            cb.student_ema.shadow["weight"],
        )
        fresh.load_state_dict(state)
        assert fresh._ema_started is True
        assert torch.allclose(
            fresh.student_ema.shadow["weight"],
            cb.student_ema.shadow["weight"],
        )

    def test_load_without_student_ema_only_sets_flag(self) -> None:
        cb = EMACallback()
        # student_ema is None — load must not attempt to assign shadow.
        cb.load_state_dict({"ema_started": True})
        assert cb._ema_started is True
        assert cb.student_ema is None
