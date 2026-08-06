# SPDX-License-Identifier: Apache-2.0
"""CPU tests for synchronized validation MP4 encoding."""

from pathlib import Path

import av
import numpy as np
import pytest

from fastvideo.train.utils import validation_media
from fastvideo.train.utils.validation_media import write_validation_mp4


def _rgb_frames(frame_count: int) -> list[np.ndarray]:
    """Create distinct even-sized RGB frames for codec tests."""
    return [
        np.full((16, 24, 3), fill_value=frame_index * 10, dtype=np.uint8)
        for frame_index in range(frame_count)
    ]


def _stereo_waveform(sample_count: int, sample_rate: int) -> np.ndarray:
    """Create a bounded stereo tone in sample-major layout."""
    sample_times = np.arange(sample_count, dtype=np.float32) / sample_rate
    left = 0.25 * np.sin(2 * np.pi * 220 * sample_times)
    right = 0.25 * np.sin(2 * np.pi * 330 * sample_times)
    return np.stack((left, right), axis=1)


def test_write_validation_mp4_encodes_h264_with_stereo_aac(tmp_path: Path) -> None:
    """Verify stream geometry, frame count, stereo layout, and sample rate."""
    output_path = tmp_path / "audio-video.mp4"
    waveform = _stereo_waveform(32_000, 32_000).T

    write_validation_mp4(
        str(output_path),
        _rgb_frames(12),
        fps=24,
        audio=waveform,
        audio_sample_rate=32_000,
    )

    with av.open(str(output_path)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.video[0].average_rate == 24
        assert container.streams.video[0].width == 24
        assert container.streams.video[0].height == 16
        assert container.streams.audio[0].codec_context.name == "aac"
        assert container.streams.audio[0].codec_context.sample_rate == 32_000
        assert len(container.streams.audio[0].codec_context.layout.channels) == 2
        assert sum(1 for _ in container.decode(video=0)) == 12


def test_write_validation_mp4_trims_video_to_shorter_audio(tmp_path: Path) -> None:
    """Verify that a quarter-second waveform limits a 24 FPS video to six frames."""
    output_path = tmp_path / "trimmed.mp4"

    write_validation_mp4(
        str(output_path),
        _rgb_frames(12),
        fps=24,
        audio=_stereo_waveform(8_000, 32_000),
        audio_sample_rate=32_000,
    )

    with av.open(str(output_path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 6


def test_write_validation_mp4_preserves_silent_video(tmp_path: Path) -> None:
    """Verify that pipelines without waveforms retain video-only validation."""
    output_path = tmp_path / "silent.mp4"

    write_validation_mp4(str(output_path), _rgb_frames(4), fps=24)

    with av.open(str(output_path)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 0
        assert sum(1 for _ in container.decode(video=0)) == 4


def test_write_validation_mp4_rejects_audio_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a silent encode cannot replace an audio-bearing destination."""
    output_path = tmp_path / "must-keep-audio.mp4"
    output_path.write_bytes(b"preserved destination")
    real_encoder = validation_media._encode_mp4

    def encode_without_audio(
        temporary_path: str,
        frames: np.ndarray,
        fps: int,
        waveform: np.ndarray | None,
        sample_rate: int | None,
    ) -> None:
        """Simulate an encoder that drops a requested waveform."""
        del waveform, sample_rate
        real_encoder(temporary_path, frames, fps, None, None)

    monkeypatch.setattr(validation_media, "_encode_mp4", encode_without_audio)

    with pytest.raises(RuntimeError, match="exactly one audio stream"):
        write_validation_mp4(
            str(output_path),
            _rgb_frames(4),
            fps=24,
            audio=_stereo_waveform(8_000, 32_000),
            audio_sample_rate=32_000,
        )

    assert output_path.read_bytes() == b"preserved destination"
    assert list(tmp_path.glob(".must-keep-audio.mp4.*.mp4")) == []
