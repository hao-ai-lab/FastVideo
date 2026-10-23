# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR video / audio input and delivery resizing (CPU)."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import k6_sr_tiny
import pytest
import torch

from fastvideo.pipelines.basic.kandinsky6_sr import sr_io

av = pytest.importorskip("av")


def _resample_then_clip(video: torch.Tensor, src_fps: float) -> tuple[torch.Tensor, int]:
    """The SR input contract written the straightforward way: decode everything, resample to 24 fps, keep the first
    121 frames aligned to 1 + 8k."""
    if abs(src_fps - 24) < 1.5:
        out_fps = round(src_fps)
    elif src_fps > 24:
        step = src_fps / 24
        indices = [i for i in (round(i * step) for i in range(int(video.shape[0] / step))) if i < video.shape[0]]
        video, out_fps = video[indices], 24
    else:
        out_fps = round(src_fps)
    video = video[:121]
    if video.shape[0] == 0:
        raise ValueError("no readable frames")
    return video[:1 + 8 * ((video.shape[0] - 1) // 8)], out_fps


@pytest.mark.parametrize("src_fps", [10, 12, 23.976, 24, 25, 25.5, 29.97, 30, 48, 60, 120, 239.76])
@pytest.mark.parametrize("total", [1, 2, 9, 17, 25, 100, 121, 122, 151, 241, 300, 1000])
def test_streaming_selection_equals_decoding_everything(src_fps, total):
    ids = torch.arange(total, dtype=torch.int32).view(-1, 1, 1, 1)
    plan = sr_io.plan_frame_selection(src_fps)
    kept = sr_io._select_frames(iter(list(ids)), plan)
    try:
        expected, expected_fps = _resample_then_clip(ids, src_fps)
    except ValueError:
        assert kept == []
        return
    got = torch.stack(kept)[:121]
    got = got[:1 + 8 * ((got.shape[0] - 1) // 8)]
    assert got.flatten().tolist() == expected.flatten().tolist()
    assert plan.out_fps == expected_fps


def test_decoding_stops_after_the_last_needed_frame():
    consumed = []

    def frames():
        for i in range(10**7):
            consumed.append(i)
            yield torch.zeros(1, 1, 1)

    plan = sr_io.plan_frame_selection(60.0)
    assert len(sr_io._select_frames(frames(), plan)) == 121
    assert len(consumed) == plan.lookahead == 303  # ceil(121 * 2.5)


def _decode_all(path: Path) -> tuple[torch.Tensor, float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1) for f in container.decode(stream)]
        return torch.stack(frames), float(stream.average_rate)


@pytest.mark.parametrize("fps,num_frames", [(30, 50), (24, 20), (Fraction(24000, 1001), 30), (12, 18), (60, 70)])
def test_read_video_follows_the_sr_input_contract(tmp_path, fps, num_frames):
    path = tmp_path / "clip.mp4"
    k6_sr_tiny.write_mp4(path, num_frames, fps)
    video, out_fps = sr_io.read_video(path)
    expected, expected_fps = _resample_then_clip(*_decode_all(path))
    assert video.dtype == torch.uint8 and torch.equal(video, expected) and out_fps == expected_fps


def test_missing_input_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        sr_io.read_video(tmp_path / "nope.mp4")


def test_audio_is_mono_trimmed_to_the_processed_frames_and_optional(tmp_path):
    with_audio, without = tmp_path / "a.mp4", tmp_path / "b.mp4"
    k6_sr_tiny.write_mp4(with_audio, 24, 24, audio_seconds=3.0)
    k6_sr_tiny.write_mp4(without, 24, 24)
    audio = sr_io.read_audio(with_audio, 17, 24)
    assert audio.dtype == torch.float32 and audio.ndim == 1
    assert audio.shape[0] == round(17 / 24 * sr_io.AUDIO_SAMPLE_RATE)
    assert 0.05 < audio.abs().max() <= 1.0
    assert sr_io.read_audio(without, 17, 24) is None


@pytest.mark.parametrize("spec, mode, expected", [
    (None, "fit", None),
    ("hd", "fit", (694, 1280)),  # isotropic fit, even sizes
    ("hd", "exact", (720, 1280)),
    ("640x360", "fit", (346, 640)),
    ("2k", "fit", None),  # already inside the bucket: never upscales
])
def test_delivery_size(spec, mode, expected):
    assert sr_io.resolve_target_hw(spec, (832, 1536), mode) == expected


@pytest.mark.parametrize("spec, mode", [("8k", "fit"), ("12x", "fit"), ("hd", "stretch")])
def test_bad_delivery_specs_are_rejected(spec, mode):
    with pytest.raises(ValueError):
        sr_io.validate_target_spec(spec, mode)


def test_resize_never_upscales():
    video = torch.zeros(3, 2, 64, 96, dtype=torch.uint8)
    assert sr_io.resize_video(video, (32, 48)).shape == (3, 2, 32, 48)
    with pytest.raises(ValueError, match="exceeds"):
        sr_io.resize_video(video, (128, 192))
