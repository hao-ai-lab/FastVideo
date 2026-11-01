# SPDX-License-Identifier: Apache-2.0
"""FASTVIDEO_H3_OVERLAP_AUDIO_DECODE: audio decode beside the video decode."""

import time

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding import (
    MiniMaxH3AudioDecodingStage,
    MiniMaxH3VideoDecodingStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class _FakeStream:

    device = torch.device("cpu")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def synchronize(self) -> None:
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    """The overlap path only needs a stream-like context manager and a device index."""
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: _FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda *args, **kwargs: _FakeStream())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def _pipeline(video: MiniMaxH3VideoDecodingStage, audio: MiniMaxH3AudioDecodingStage) -> MiniMaxH3Pipeline:
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._stage_name_mapping = {"video_decoding_stage": video, "audio_decoding_stage": audio}
    return pipeline


def _fake_video(order: list[str], hold: float):
    video = MiniMaxH3VideoDecodingStage(None)

    def forward(batch, fastvideo_args):
        order.append("video_start")
        batch.extra["video"] = torch.ones(1)
        time.sleep(hold)
        order.append("video_end")
        return batch

    video.forward = forward
    return video


def _fake_audio(order: list[str], hold: float, *, fail: bool = False):
    audio = MiniMaxH3AudioDecodingStage(None)

    def forward(batch, fastvideo_args):
        order.append("audio_start")
        time.sleep(hold)
        if fail:
            raise RuntimeError("audio decode failed")
        batch.extra["audio"] = torch.full((2, 3), 0.5)
        batch.extra["audio_sample_rate"] = 16000
        # The real stage clears the runtime tensors it consumed.
        batch.latents = None
        batch.extra.pop("minimax_h3_layout", None)
        order.append("audio_end")
        return batch

    audio.forward = forward
    return audio


def test_audio_decode_overlaps_the_video_decode(monkeypatch, fake_cuda) -> None:
    monkeypatch.setenv("FASTVIDEO_H3_OVERLAP_AUDIO_DECODE", "1")
    order: list[str] = []
    video = _fake_video(order, hold=0.20)
    audio = _fake_audio(order, hold=0.10)
    batch = ForwardBatch(data_type="video", latents=torch.zeros(1), extra={"minimax_h3_layout": object()})

    result = _pipeline(video, audio)._run_stage_sequence(("video_decoding_stage", "audio_decoding_stage"), batch,
                                                         None)

    # The audio decode started before the video decode finished: they overlap.
    assert order.index("audio_start") < order.index("video_end")
    assert order.index("audio_end") < order.index("video_end")
    # The results merge back onto the caller's batch; the audio thread's copy did
    # not clear the caller's runtime tensors.
    assert torch.equal(result.extra["audio"], torch.full((2, 3), 0.5))
    assert result.extra["audio_sample_rate"] == 16000
    assert result.extra["video"] is not None
    assert result.latents is not None
    assert "minimax_h3_layout" in result.extra


def test_audio_decode_stays_serial_by_default(monkeypatch, fake_cuda) -> None:
    monkeypatch.delenv("FASTVIDEO_H3_OVERLAP_AUDIO_DECODE", raising=False)
    order: list[str] = []
    video = _fake_video(order, hold=0.05)
    audio = _fake_audio(order, hold=0.05)
    batch = ForwardBatch(data_type="video", latents=torch.zeros(1), extra={})

    _pipeline(video, audio)._run_stage_sequence(("video_decoding_stage", "audio_decoding_stage"), batch, None)

    assert order == ["video_start", "video_end", "audio_start", "audio_end"]


def test_audio_decode_error_is_raised_on_the_main_thread(monkeypatch, fake_cuda) -> None:
    monkeypatch.setenv("FASTVIDEO_H3_OVERLAP_AUDIO_DECODE", "1")
    order: list[str] = []
    video = _fake_video(order, hold=0.05)
    audio = _fake_audio(order, hold=0.01, fail=True)
    batch = ForwardBatch(data_type="video", latents=torch.zeros(1), extra={})

    with pytest.raises(RuntimeError, match="audio decode failed"):
        _pipeline(video, audio)._run_stage_sequence(("video_decoding_stage", "audio_decoding_stage"), batch, None)


def test_audio_decode_runs_after_video_when_listed_first(monkeypatch, fake_cuda) -> None:
    """An audio stage that precedes the video stage in the order cannot overlap it."""
    monkeypatch.setenv("FASTVIDEO_H3_OVERLAP_AUDIO_DECODE", "1")
    order: list[str] = []
    video = _fake_video(order, hold=0.05)
    audio = _fake_audio(order, hold=0.05)
    batch = ForwardBatch(data_type="video", latents=torch.zeros(1), extra={})

    _pipeline(video, audio)._run_stage_sequence(("audio_decoding_stage", "video_decoding_stage"), batch, None)

    assert order == ["audio_start", "audio_end", "video_start", "video_end"]
