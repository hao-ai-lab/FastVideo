# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.callbacks.validation`.

Covers the parts of ``ValidationCallback`` that don't need a real
pipeline or distributed init:

* constructor type coercions and defaults,
* ``on_validation_begin`` gating logic (every_steps + modulo),
* ``_find_ema_callback`` lookup via ``_callback_dict``,
* ``state_dict`` / ``load_state_dict`` rng round-trip.

The heavy ``_run_validation`` path needs a real diffusion pipeline plus
distributed init and is exercised by Phase 2/3 tests.
"""
from __future__ import annotations

import torch

from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.callbacks.ema import EMACallback
from fastvideo.train.callbacks.validation import ValidationCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PIPE_TARGET = "fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline"


def _make_callback(
    *,
    every_steps: int = 100,
    sampling_steps: list[int] | None = None,
    guidance_scale: float | None = None,
    num_frames: int | None = None,
    sampling_timesteps: list[int] | None = None,
    output_dir: str | None = None,
) -> ValidationCallback:
    return ValidationCallback(
        pipeline_target=_PIPE_TARGET,
        dataset_file="/tmp/does_not_exist.json",
        every_steps=every_steps,
        sampling_steps=sampling_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        sampling_timesteps=sampling_timesteps,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# A. Constructor coercions / defaults
# ---------------------------------------------------------------------------


class TestConstructor:

    def test_defaults(self) -> None:
        cb = _make_callback()
        assert cb.pipeline_target == _PIPE_TARGET
        assert cb.dataset_file == "/tmp/does_not_exist.json"
        assert cb.every_steps == 100
        assert cb.sampling_steps == [40]
        assert cb.guidance_scale is None
        assert cb.num_frames is None
        assert cb.sampling_timesteps is None
        assert cb.output_dir is None
        # Lazy fields not yet populated.
        assert cb._pipeline is None
        assert cb._sampling_param is None
        assert cb.validation_random_generator is None

    def test_string_inputs_are_coerced(self) -> None:
        # YAML often produces strings for numeric fields; the
        # constructor must coerce them.
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            every_steps="50",  # type: ignore[arg-type]
            sampling_steps=["20", "40"],  # type: ignore[arg-type]
            guidance_scale="4.5",  # type: ignore[arg-type]
            num_frames="77",  # type: ignore[arg-type]
            sampling_timesteps=["1000", "500"],
        )
        assert cb.every_steps == 50
        assert cb.sampling_steps == [20, 40]
        assert cb.guidance_scale == 4.5
        assert cb.num_frames == 77
        assert cb.sampling_timesteps == [1000, 500]

    def test_pipeline_kwargs_collected(self) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            extra_arg=123,
            another="value",
        )
        # Unknown kwargs are stashed for the pipeline factory.
        assert cb.pipeline_kwargs == {
            "extra_arg": 123,
            "another": "value",
        }


# ---------------------------------------------------------------------------
# B. on_validation_begin gating
# ---------------------------------------------------------------------------


class _NoRunValidation(ValidationCallback):
    """Subclass that records ``_run_validation`` calls instead of
    actually running them — lets us assert the gating logic without a
    real pipeline."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[int] = []

    def _run_validation(self, method, step: int) -> None:  # type: ignore[override]
        self.run_calls.append(step)


def _make_recording(**kwargs) -> _NoRunValidation:
    return _NoRunValidation(
        pipeline_target=_PIPE_TARGET,
        dataset_file="x.json",
        **kwargs,
    )


class TestOnValidationBegin:

    def test_skipped_when_every_steps_zero(self) -> None:
        cb = _make_recording(every_steps=0)
        cb.on_validation_begin(method=None, iteration=0)
        cb.on_validation_begin(method=None, iteration=1000)
        assert cb.run_calls == []

    def test_skipped_on_off_iter(self) -> None:
        cb = _make_recording(every_steps=50)
        cb.on_validation_begin(method=None, iteration=49)
        cb.on_validation_begin(method=None, iteration=51)
        assert cb.run_calls == []

    def test_runs_on_match(self) -> None:
        cb = _make_recording(every_steps=50)
        cb.on_validation_begin(method=None, iteration=50)
        cb.on_validation_begin(method=None, iteration=100)
        assert cb.run_calls == [50, 100]

    def test_iter_zero_runs(self) -> None:
        # 0 % anything == 0 → step 0 fires (matches existing
        # validation behavior used by ValidationCallback consumers).
        cb = _make_recording(every_steps=50)
        cb.on_validation_begin(method=None, iteration=0)
        assert cb.run_calls == [0]


# ---------------------------------------------------------------------------
# C. _find_ema_callback
# ---------------------------------------------------------------------------


class TestFindEmaCallback:

    def test_returns_none_without_callback_dict(self) -> None:
        cb = _make_callback()
        # _callback_dict is not set on bare instances.
        assert cb._find_ema_callback() is None

    def test_returns_none_when_no_ema_registered(self) -> None:
        cb = _make_callback()
        cb_dict = CallbackDict({}, training_config=object())
        cb._callback_dict = cb_dict
        assert cb._find_ema_callback() is None

    def test_finds_ema_callback(self) -> None:
        cb = _make_callback()
        cb_dict = CallbackDict({}, training_config=object())
        ema = EMACallback(decay=0.99)
        cb_dict._callbacks["ema"] = ema
        cb_dict._callbacks["validation"] = cb
        cb._callback_dict = cb_dict

        found = cb._find_ema_callback()
        assert found is ema


# ---------------------------------------------------------------------------
# D. state_dict / load_state_dict (rng round-trip)
# ---------------------------------------------------------------------------


class TestStateDict:

    def test_state_dict_empty_without_generator(self) -> None:
        cb = _make_callback()
        # validation_random_generator is None until on_train_start.
        assert cb.state_dict() == {}

    def test_round_trip_preserves_rng_state(self) -> None:
        cb = _make_callback()
        gen = torch.Generator(device="cpu").manual_seed(123)
        # Advance RNG so a default-init generator on the receiving
        # side is observably different.
        for _ in range(5):
            torch.randn(4, generator=gen)
        cb.validation_random_generator = gen

        state = cb.state_dict()
        assert "validation_rng" in state

        # Receiver: fresh generator with a different seed.
        fresh = _make_callback()
        fresh.validation_random_generator = (
            torch.Generator(device="cpu").manual_seed(999)
        )
        fresh.load_state_dict(state)

        # After load, both generators draw the same next sample.
        a = torch.randn(8, generator=cb.validation_random_generator)
        b = torch.randn(8, generator=fresh.validation_random_generator)
        assert torch.equal(a, b)

    def test_load_without_generator_is_noop(self) -> None:
        cb = _make_callback()
        # Generator is None: load must not raise even when state has
        # an rng entry.
        cb.load_state_dict(
            {"validation_rng": torch.tensor([1, 2, 3], dtype=torch.uint8)}
        )
        assert cb.validation_random_generator is None
