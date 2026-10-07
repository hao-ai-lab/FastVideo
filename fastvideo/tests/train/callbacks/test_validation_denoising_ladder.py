# SPDX-License-Identifier: Apache-2.0
"""Validation samples on the ladder the method trains against.

``sampling_steps`` only sets ``num_inference_steps``; the scheduler expands N
of those into an N-point sigma grid, i.e. N-1 forwards on its own spacing. A
few-step recipe therefore validates off its operating point unless
``sampling_timesteps`` repeats the trained ladder exactly.
"""

from __future__ import annotations

import types

import pytest

from fastvideo.train.callbacks.validation import ValidationCallback

LADDER = [1000, 750, 500, 250]


def _callback(**kwargs):
    return ValidationCallback(
        pipeline_target="fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline",
        dataset_file="unused.json",
        **kwargs,
    )


def _method(ladder=LADDER):
    cfg = {} if ladder is None else {"dmd_denoising_steps": list(ladder)}
    return types.SimpleNamespace(method_config=cfg)


def test_ladder_is_inherited_when_unset() -> None:
    cb = _callback(sampling_steps=[4])
    assert cb.sampling_timesteps is None
    cb._adopt_training_denoising_ladder(_method())
    assert cb.sampling_timesteps == LADDER


def test_matching_ladder_is_accepted() -> None:
    cb = _callback(sampling_steps=[4], sampling_timesteps=LADDER)
    cb._adopt_training_denoising_ladder(_method())
    assert cb.sampling_timesteps == LADDER


def test_diverging_ladder_raises() -> None:
    cb = _callback(sampling_steps=[4], sampling_timesteps=[999, 749, 500])
    with pytest.raises(ValueError, match="disagrees with the trained ladder"):
        cb._adopt_training_denoising_ladder(_method())


@pytest.mark.parametrize("ladder", [None, []])
def test_methods_without_a_ladder_are_left_alone(ladder) -> None:
    cb = _callback(sampling_steps=[40])
    cb._adopt_training_denoising_ladder(_method(ladder))
    assert cb.sampling_timesteps is None


def test_method_without_config_is_tolerated() -> None:
    cb = _callback(sampling_steps=[40])
    cb._adopt_training_denoising_ladder(types.SimpleNamespace())
    assert cb.sampling_timesteps is None
