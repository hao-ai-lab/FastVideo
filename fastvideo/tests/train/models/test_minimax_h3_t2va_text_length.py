# SPDX-License-Identifier: Apache-2.0
"""Existing T2VA configs keep the fixed text-length crop unless they opt into dynamic collation."""

from types import SimpleNamespace
from typing import Any

import pytest

from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model


def _init_preprocessors(monkeypatch: pytest.MonkeyPatch, **data: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("fastvideo.train.models.minimax_h3.minimax_h3.get_sp_group", lambda: None)
    monkeypatch.setattr("fastvideo.train.utils.dataloader.build_parquet_t2v_train_dataloader",
                        lambda data_config, **kwargs: captured.update(kwargs))
    data_config = SimpleNamespace(preprocessed_data_type="t2va", cross_sample_sequence_packing=None, **data)
    text_encoder = SimpleNamespace(arch_config=SimpleNamespace(text_len=1024))
    training_config = SimpleNamespace(data=data_config,
                                      pipeline_config=SimpleNamespace(text_encoder_configs=(text_encoder, )))
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model.init_preprocessors(training_config)
    return captured


def test_unset_presentation_bound_keeps_the_text_encoder_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    # A prompt longer than text_len was cropped before dynamic collation existed;
    # rejecting it mid-run would kill a run whose config never changed.
    kwargs = _init_preprocessors(monkeypatch, native_shape_bucketing=False, max_presentation_tokens=None)
    assert kwargs["text_len"] == 1024
    assert kwargs["max_presentation_tokens"] is None


@pytest.mark.parametrize("native_shape_bucketing", [False, True])
def test_explicit_presentation_bound_selects_dynamic_collation(monkeypatch: pytest.MonkeyPatch,
                                                               native_shape_bucketing: bool) -> None:
    kwargs = _init_preprocessors(monkeypatch,
                                 native_shape_bucketing=native_shape_bucketing,
                                 max_presentation_tokens=2048)
    assert kwargs["text_len"] is None
    assert kwargs["max_presentation_tokens"] == 2048


def test_native_shapes_require_an_explicit_presentation_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="max_presentation_tokens"):
        _init_preprocessors(monkeypatch, native_shape_bucketing=True, max_presentation_tokens=None)
