# SPDX-License-Identifier: Apache-2.0
"""LTX-2 audio-conditioning kwargs must reach ForwardBatch.extra and
SamplingParam.update() must reject unknown kwargs instead of silently
dropping them.

Background: prior to PR-1288 a chain of LTX-2 audio kwargs
(``audio_num_frames``, ``ltx2_audio_clean_latent`` …) silently flowed
into ``SamplingParam.update()`` which ``logger.error``'d and dropped
them. That made every continuation segment generate audio for the
default ``num_frames`` duration, which in turn fed an A/V duration
mismatch into ``av_streaming.stream_fmp4`` whose ``-shortest`` ffmpeg
invocation closed stdin before the writer thread had pushed every
frame, surfacing as ``BrokenPipeError`` in the streaming server.

These tests pin two contracts:
  1. ``_BATCH_EXTRA_PASSTHROUGH_KEYS`` lists the exact set of kwargs
     pulled out of ``generate_video(**kwargs)`` for ``batch.extra``.
  2. ``SamplingParam.update()`` raises ``ValueError`` on unknown keys.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.entrypoints.video_generator import (
    _BATCH_EXTRA_PASSTHROUGH_KEYS,
)
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import shallow_asdict


def test_passthrough_keys_cover_ltx2_audio_conditioning() -> None:
    expected = {
        "ltx2_audio_latents",
        "ltx2_audio_clean_latent",
        "ltx2_audio_denoise_mask",
        "audio_num_frames",
        "video_position_offset_sec",
    }
    assert set(_BATCH_EXTRA_PASSTHROUGH_KEYS) == expected


def test_passthrough_keys_are_not_sampling_param_fields() -> None:
    """If any of these become SamplingParam fields, the routing block
    in video_generator.py needs to be re-evaluated — they would no
    longer need to be popped before ``sampling_param.update()``."""
    import dataclasses
    sp_fields = {f.name for f in dataclasses.fields(SamplingParam())}
    leaked = sp_fields & set(_BATCH_EXTRA_PASSTHROUGH_KEYS)
    assert not leaked, (
        f"Passthrough keys collide with SamplingParam fields: {leaked}. "
        "Remove from _BATCH_EXTRA_PASSTHROUGH_KEYS or rename the field.")


def test_sampling_param_update_rejects_unknown_keys() -> None:
    sp = SamplingParam()
    with pytest.raises(ValueError, match="unknown field"):
        sp.update({"bogus_field": 42})


def test_sampling_param_update_rejects_partially_unknown_keys() -> None:
    """Even when most keys are valid, a single unknown key must raise.
    Partial-success was the silent-failure mode this regression fixes."""
    sp = SamplingParam()
    with pytest.raises(ValueError, match=r"\['bogus_field'\]"):
        sp.update({"prompt": "hello", "bogus_field": 42})


def test_sampling_param_update_rejects_audio_passthrough_keys() -> None:
    """LTX-2 audio kwargs must NOT slip through ``update()`` — they
    belong to ``ForwardBatch.extra`` and the routing block in
    ``video_generator.py`` is responsible for popping them first."""
    sp = SamplingParam()
    for key in _BATCH_EXTRA_PASSTHROUGH_KEYS:
        with pytest.raises(ValueError, match="unknown field"):
            sp.update({key: object()})


def test_sampling_param_update_accepts_known_fields() -> None:
    sp = SamplingParam()
    sp.update({"prompt": "hello world", "seed": 42, "num_frames": 121})
    assert sp.prompt == "hello world"
    assert sp.seed == 42
    assert sp.num_frames == 121


def test_forward_batch_accepts_ltx2_sampling_param_fields() -> None:
    stage1 = torch.zeros(1)
    stage2 = torch.ones(1)
    sp = SamplingParam(
        ltx2_images=[("image.png", 0, 1.0)],
        ltx2_image_crf=0.0,
        ltx2_conditioning_latent_stage1=stage1,
        ltx2_conditioning_latent_stage2=stage2,
        ltx2_video_conditions=[(["frame0.png", "frame1.png"], 2, 0.5)],
    )

    batch = ForwardBatch(**shallow_asdict(sp))

    assert batch.ltx2_images == [("image.png", 0, 1.0)]
    assert batch.ltx2_image_crf == 0.0
    assert batch.ltx2_conditioning_latent_stage1 is stage1
    assert batch.ltx2_conditioning_latent_stage2 is stage2
    assert batch.ltx2_video_conditions == [(["frame0.png", "frame1.png"], 2, 0.5)]


def test_sampling_param_update_error_mentions_passthrough_route() -> None:
    """The error message should point future contributors at the right
    routing mechanism so they don't re-introduce silent dropping."""
    sp = SamplingParam()
    with pytest.raises(ValueError, match="_BATCH_EXTRA_PASSTHROUGH_KEYS"):
        sp.update({"audio_num_frames": 161})
