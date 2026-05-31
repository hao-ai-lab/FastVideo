# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.callbacks.callback`.

Covers the ``Callback`` base class no-op contract and the
``CallbackDict`` instantiation / dispatch / state-dict logic.

The concrete callback subclasses (``GradNormClipCallback``,
``EMACallback``, ``ValidationCallback``) have their own test files.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from fastvideo.train.callbacks.callback import (
    Callback,
    CallbackDict,
    _BUILTIN_CALLBACKS,
)


# ``fastvideo.logger.init_logger`` sets ``propagate=False`` on its
# loggers, so the standard ``caplog`` fixture cannot observe them.
# This helper attaches a temporary handler directly to the target
# logger and yields the captured records.
@contextmanager
def _capture_logger(
    name: str, level: int = logging.WARNING
) -> Iterator[list[logging.LogRecord]]:
    logger = logging.getLogger(name)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):

        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=level)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCallback(Callback):
    """Callback that records every hook call into a shared list."""

    def __init__(self, *, tag: str, sink: list[str]) -> None:
        self._tag = tag
        self._sink = sink

    def on_train_start(self, method, iteration: int = 0) -> None:
        self._sink.append(f"{self._tag}:on_train_start:{iteration}")

    def on_training_step_end(
        self, method, loss_dict, iteration: int = 0
    ) -> None:
        self._sink.append(f"{self._tag}:on_training_step_end:{iteration}")

    def on_validation_begin(self, method, iteration: int = 0) -> None:
        self._sink.append(f"{self._tag}:on_validation_begin:{iteration}")

    def state_dict(self) -> dict[str, Any]:
        return {"tag": self._tag, "marker": 7}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._sink.append(f"{self._tag}:load:{sd.get('marker')}")


class _NotACallback:
    """Plain class used to exercise the non-Callback type guard."""

    def __init__(self, **_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# A. Callback base class
# ---------------------------------------------------------------------------


class TestCallbackBase:

    def test_default_hooks_return_none(self) -> None:
        cb = Callback()
        assert cb.on_train_start(method=None) is None
        assert (
            cb.on_training_step_end(method=None, loss_dict={}) is None
        )
        assert cb.on_before_optimizer_step(method=None) is None
        assert cb.on_validation_begin(method=None) is None
        assert cb.on_validation_end(method=None) is None
        assert cb.on_train_end(method=None) is None

    def test_default_state_dict_round_trip(self) -> None:
        cb = Callback()
        assert cb.state_dict() == {}
        # Default load_state_dict accepts arbitrary state without raising.
        assert cb.load_state_dict({"unrelated": 1}) is None


# ---------------------------------------------------------------------------
# B. CallbackDict construction
# ---------------------------------------------------------------------------


class TestCallbackDictInit:

    def test_empty_config(self) -> None:
        cb_dict = CallbackDict({}, training_config=object())
        assert cb_dict._callbacks == {}

    def test_builtin_name_resolves_without_target(self) -> None:
        # ``grad_clip`` is a registered builtin.
        cfg = {"grad_clip": {"max_grad_norm": 0.5}}
        tc = object()
        cb_dict = CallbackDict(cfg, training_config=tc)

        assert "grad_clip" in cb_dict._callbacks
        from fastvideo.train.callbacks.grad_clip import (
            GradNormClipCallback,
        )
        cb = cb_dict._callbacks["grad_clip"]
        assert isinstance(cb, GradNormClipCallback)
        # CallbackDict wires up training_config + back-pointer.
        assert cb.training_config is tc
        assert cb._callback_dict is cb_dict

    def test_explicit_target_overrides_name_lookup(self) -> None:
        cfg = {
            "anything_goes": {
                "_target_": (
                    "fastvideo.train.callbacks.grad_clip."
                    "GradNormClipCallback"
                ),
                "max_grad_norm": 1.0,
            }
        }
        cb_dict = CallbackDict(cfg, training_config=object())
        assert "anything_goes" in cb_dict._callbacks

    def test_unknown_name_without_target_is_skipped(self) -> None:
        cfg = {"mystery": {"some_arg": 1}}
        with _capture_logger(
            "fastvideo.train.callbacks.callback"
        ) as records:
            cb_dict = CallbackDict(cfg, training_config=object())
        assert cb_dict._callbacks == {}
        assert any(
            "missing" in r.getMessage() and "mystery" in r.getMessage()
            for r in records
        )

    def test_non_callback_target_raises(self) -> None:
        cfg = {
            "bad": {
                "_target_": (
                    "fastvideo.tests.train.callbacks.test_callback."
                    "_NotACallback"
                )
            }
        }
        with pytest.raises(TypeError, match="expected a Callback"):
            CallbackDict(cfg, training_config=object())

    def test_builtin_registry_has_expected_entries(self) -> None:
        # Sanity: protect the builtin registry from silent shrinkage.
        assert set(_BUILTIN_CALLBACKS) >= {
            "grad_clip",
            "validation",
            "ema",
        }


# ---------------------------------------------------------------------------
# C. Dispatch via __getattr__
# ---------------------------------------------------------------------------


class TestCallbackDictDispatch:

    def _build(
        self, sink: list[str]
    ) -> CallbackDict:
        cb_dict = CallbackDict({}, training_config=object())
        cb_dict._callbacks["first"] = _RecordingCallback(
            tag="first", sink=sink
        )
        cb_dict._callbacks["second"] = _RecordingCallback(
            tag="second", sink=sink
        )
        return cb_dict

    def test_dispatch_calls_all_in_insertion_order(self) -> None:
        sink: list[str] = []
        cb_dict = self._build(sink)

        cb_dict.on_train_start(method=None, iteration=3)
        assert sink == [
            "first:on_train_start:3",
            "second:on_train_start:3",
        ]

    def test_dispatch_to_hook_some_callbacks_skip(self) -> None:
        sink: list[str] = []
        cb_dict = self._build(sink)
        # The base Callback subclass below only implements one hook;
        # dispatch should still fan out without raising.

        class _OnlyValidation(Callback):

            def on_validation_end(
                self, method, iteration: int = 0
            ) -> None:
                sink.append(f"vend:{iteration}")

        cb_dict._callbacks["only_v"] = _OnlyValidation()
        cb_dict.on_validation_end(method=None, iteration=11)
        assert "vend:11" in sink

    def test_dispatch_unknown_hook_is_noop(self) -> None:
        # Methods that don't exist on any callback should not raise.
        cb_dict = self._build([])
        cb_dict.totally_made_up_hook(method=None, iteration=0)

    def test_underscore_attribute_raises(self) -> None:
        cb_dict = self._build([])
        with pytest.raises(AttributeError):
            getattr(cb_dict, "_does_not_exist")


# ---------------------------------------------------------------------------
# D. state_dict / load_state_dict
# ---------------------------------------------------------------------------


class TestCallbackDictStateDict:

    def _build(self) -> tuple[CallbackDict, list[str]]:
        sink: list[str] = []
        cb_dict = CallbackDict({}, training_config=object())
        cb_dict._callbacks["first"] = _RecordingCallback(
            tag="first", sink=sink
        )
        cb_dict._callbacks["second"] = _RecordingCallback(
            tag="second", sink=sink
        )
        return cb_dict, sink

    def test_state_dict_returns_per_callback_dict(self) -> None:
        cb_dict, _ = self._build()
        state = cb_dict.state_dict()
        assert set(state) == {"first", "second"}
        assert state["first"] == {"tag": "first", "marker": 7}
        assert state["second"] == {"tag": "second", "marker": 7}

    def test_load_state_dict_dispatches_to_each(self) -> None:
        cb_dict, sink = self._build()
        cb_dict.load_state_dict(
            {
                "first": {"marker": 1},
                "second": {"marker": 2},
            }
        )
        assert sink == ["first:load:1", "second:load:2"]

    def test_load_state_dict_missing_key_warns_no_raise(self) -> None:
        cb_dict, sink = self._build()
        with _capture_logger(
            "fastvideo.train.callbacks.callback"
        ) as records:
            cb_dict.load_state_dict({"first": {"marker": 99}})
        assert sink == ["first:load:99"]
        assert any(
            "second" in r.getMessage() and "not found" in r.getMessage()
            for r in records
        )
