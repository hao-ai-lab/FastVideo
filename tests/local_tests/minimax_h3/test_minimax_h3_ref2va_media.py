# SPDX-License-Identifier: Apache-2.0
"""Implementation-subcomponent parity for MiniMax H3 Ref2VA media preparation."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference, assert_reference_source

PARITY_SCOPE = "implementation_subcomponent"

assert_pinned_reference(
    "src/diffusers/modular_pipelines/minimax_h3/packing.py",
    "969d667d1c0316ed931d1675d474220045390e9a566cc885e3bd9cc6147b3e5b",
)
assert_pinned_reference(
    "src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py",
    "1f025b68af3f5c4316e2b89b026d4976c5b85286dafcc8b5c1d8ceb186c9bc01",
)
sys.path.insert(0, str(REFERENCE_SRC))

from diffusers.modular_pipelines.minimax_h3 import packing as reference_packing  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import packing_ref2va as reference  # noqa: E402

assert_reference_source(reference_packing, "src/diffusers/modular_pipelines/minimax_h3/packing.py")
assert_reference_source(reference, "src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py")

from fastvideo.pipelines.basic.minimax_h3 import packing as base_packing  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3 import reference as actual  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.reference import (  # noqa: E402
    MiniMaxH3Reference, decode_reference_audio, decode_reference_video, prepare_reference,
)


@pytest.mark.parametrize("size", [(80, 48), (48, 80), (64, 64), (128, 32)])
def test_reference_image_size_matches_pinned_diffusers(size: tuple[int, int]) -> None:
    assert actual.resolve_reference_image_size(*size) == reference.resolve_reference_image_size(*size)


def test_reference_media_to_uint8_matches_pinned_diffusers() -> None:
    float_pixels = np.array(
        [
            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]],
            [[1.0, 0.5, 0.0], [0.1, 0.2, 0.3]],
        ],
        dtype=np.float32,
    )
    cases = [
        float_pixels,
        torch.from_numpy(float_pixels).movedim(-1, 0),
        Image.fromarray((float_pixels * 255).round().astype(np.uint8)),
        [
            Image.fromarray((float_pixels * 255).round().astype(np.uint8)),
            Image.fromarray(np.flip((float_pixels * 255).round().astype(np.uint8), axis=1)),
        ],
    ]

    for media in cases:
        np.testing.assert_array_equal(actual.reference_media_to_uint8(media), reference.reference_media_to_uint8(media))


def test_reference_video_24_and_30_fps_resampling_matches_pinned_diffusers() -> None:
    frames = np.arange(30, dtype=np.uint8).reshape(-1, 1, 1, 1) * np.ones((1, 2, 2, 3), dtype=np.uint8)

    assert actual.resample_reference_frames(frames, 24.0) is frames
    assert reference.resample_reference_frames(frames, 24.0) is frames

    result = actual.resample_reference_frames(frames, 30.0)
    expected = reference.resample_reference_frames(frames, 30.0)
    np.testing.assert_array_equal(result, expected)
    assert [int(frame[0, 0, 0])
            for frame in result] == [index for index in range(30) if index not in (2, 7, 12, 17, 22, 27)]


def test_reference_video_resize_and_truncate_matches_pinned_diffusers(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (base_packing, reference_packing):
        monkeypatch.setattr(module, "MINIMAX_H3_SHORT_EDGE", 32)
        monkeypatch.setattr(module, "MINIMAX_H3_MAX_PIXELS", 32 * 64)

    frames = np.arange(5 * 16 * 32 * 3, dtype=np.uint32).reshape(5, 16, 32, 3).astype(np.uint8)
    result = actual.prepare_reference_frames(frames, num_frames=3)
    expected = reference.prepare_reference_frames(frames, num_frames=3)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 32, 64, 3)

    canvas_frames = np.zeros((5, 32, 64, 3), dtype=np.uint8)
    result = actual.prepare_reference_frames(canvas_frames, num_frames=3)
    expected = reference.prepare_reference_frames(canvas_frames, num_frames=3)
    np.testing.assert_array_equal(result, expected)
    assert np.shares_memory(result, canvas_frames)
    assert np.shares_memory(expected, canvas_frames)


def test_qwen_two_fps_sampling_and_timestamps_match_pinned_diffusers() -> None:
    frames = np.arange(25, dtype=np.uint8).reshape(-1, 1, 1, 1) * np.ones((1, 2, 2, 3), dtype=np.uint8)

    sampled, timestamps = actual.sample_reference_video_frames(frames)
    expected_sampled, expected_timestamps = reference.sample_reference_video_frames(frames)

    np.testing.assert_array_equal(np.stack(sampled), np.stack(expected_sampled))
    assert [int(frame[0, 0, 0]) for frame in sampled] == [0, 12, 24]
    assert timestamps == expected_timestamps == [0.25, 1.0]


@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
def test_same_rate_reference_waveform_matches_pinned_diffusers(channels: int) -> None:
    waveform = torch.arange(channels * 10, dtype=torch.float64).reshape(channels, 10) / 10
    kwargs = {
        "waveform": waveform,
        "sample_rate": 10,
        "target_sample_rate": 10,
        "max_duration": 0.7,
    }

    result = actual.prepare_reference_waveform(**kwargs)
    expected = reference.prepare_reference_waveform(**kwargs)

    assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == torch.float32
    assert result.shape == (2, 7)
    if channels == 1:
        assert_close(result[0], result[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [(1, 22), (5, 22), (21, 22), (22, 22), (23, 22), (39, 39), (56, 56), (120, 107)],
)
def test_reference_frame_trimming_matches_pinned_diffusers(num_frames: int, expected: int) -> None:
    assert actual.trim_reference_num_frames(num_frames) == expected
    assert actual.trim_reference_num_frames(num_frames) == reference.trim_reference_num_frames(num_frames)


_FIXTURE_FPS = 8
_FIXTURE_NUM_FRAMES = 8
_FIXTURE_SAMPLE_RATE = 8000


def _write_video_with_audio(path: Path) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        video_stream = container.add_stream("libx264", rate=_FIXTURE_FPS)
        video_stream.width = 64
        video_stream.height = 32
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=_FIXTURE_SAMPLE_RATE)
        audio_stream.codec_context.layout = "stereo"

        for index in range(_FIXTURE_NUM_FRAMES):
            pixels = np.full((32, 64, 3), index * 16, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode())

        num_samples = _FIXTURE_SAMPLE_RATE
        samples = np.zeros((1, 2 * num_samples), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
        frame.sample_rate = _FIXTURE_SAMPLE_RATE
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=_FIXTURE_SAMPLE_RATE,
        )
        pts = 0
        for resampled in resampler.resample(frame):
            resampled.pts = pts
            pts += resampled.samples
            container.mux(audio_stream.encode(resampled))
        container.mux(audio_stream.encode())


def _write_audio(path: Path) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=_FIXTURE_SAMPLE_RATE)
        stream.codec_context.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 2 * _FIXTURE_SAMPLE_RATE), dtype=np.int16),
            format="s16",
            layout="stereo",
        )
        frame.sample_rate = _FIXTURE_SAMPLE_RATE
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_deferred_stage_decodes_local_video_soundtrack_and_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("av")
    video_path = tmp_path / "reference.mp4"
    audio_path = tmp_path / "reference.wav"
    _write_video_with_audio(video_path)
    _write_audio(audio_path)

    actual_video = decode_reference_video(video_path)
    reference_video = reference.decode_reference_video(video_path)
    np.testing.assert_array_equal(actual_video[0], reference_video[0])
    assert actual_video[1] == reference_video[1] == float(_FIXTURE_FPS)
    assert actual_video[2] is not None and reference_video[2] is not None
    assert_close(actual_video[2][0], reference_video[2][0], rtol=0, atol=0)
    assert actual_video[2][1] == reference_video[2][1] == _FIXTURE_SAMPLE_RATE

    actual_audio = decode_reference_audio(audio_path)
    reference_audio = reference.decode_reference_audio(audio_path)
    assert_close(actual_audio[0], reference_audio[0], rtol=0, atol=0)
    assert actual_audio[1] == reference_audio[1] == _FIXTURE_SAMPLE_RATE

    video_reference = MiniMaxH3Reference(source=str(video_path), media_type="video")
    audio_reference = MiniMaxH3Reference(source=str(audio_path), media_type="audio")
    assert isinstance(video_reference.source, str)
    assert isinstance(audio_reference.source, str)

    monkeypatch.setattr(base_packing, "MINIMAX_H3_SHORT_EDGE", 32)
    monkeypatch.setattr(base_packing, "MINIMAX_H3_MAX_PIXELS", 32 * 64)
    prepared_video = prepare_reference(video_reference, num_frames=22, target_sample_rate=_FIXTURE_SAMPLE_RATE)
    prepared_audio = prepare_reference(audio_reference, num_frames=22, target_sample_rate=_FIXTURE_SAMPLE_RATE)

    expected_samples = int(22 / 24 * _FIXTURE_SAMPLE_RATE)
    assert prepared_video.frames.shape == (22, 32, 64, 3)
    assert prepared_video.frames.dtype == np.uint8
    assert prepared_video.has_audio
    assert prepared_video.waveform.shape == (2, expected_samples)
    assert prepared_audio.has_audio
    assert prepared_audio.waveform.shape == (2, expected_samples)
