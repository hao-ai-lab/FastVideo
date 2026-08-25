# SPDX-License-Identifier: Apache-2.0
"""Signature contracts for Wan subclasses that reuse ``predict_noise``."""

import inspect

import pytest

from fastvideo.train.models.hunyuan.hunyuan import HunyuanModel
from fastvideo.train.models.longcat.longcat import LongCatModel
from fastvideo.train.models.matrixgame2.matrixgame2 import MatrixGame2Model


@pytest.mark.parametrize("model_cls", [HunyuanModel, LongCatModel, MatrixGame2Model])
def test_distill_kwargs_accept_wan_teacher_forcing_keywords(model_cls: type) -> None:
    parameters = inspect.signature(model_cls._build_distill_input_kwargs).parameters

    assert "clean_x" in parameters
    assert "aug_t" in parameters


@pytest.mark.parametrize("model_cls", [LongCatModel, MatrixGame2Model])
def test_predict_noise_override_preserves_wan_teacher_forcing_keywords(model_cls: type) -> None:
    parameters = inspect.signature(model_cls.predict_noise).parameters

    assert "clean_x" in parameters
    assert "aug_t" in parameters
