# SPDX-License-Identifier: Apache-2.0
"""The streaming_v2 fMP4 encoder against a real ffmpeg; skipped when ffmpeg is missing."""
from __future__ import annotations

import asyncio
import io
import shutil

import numpy as np
import pytest

from fastvideo import envs
from fastvideo.entrypoints.streaming_v2.video_stream import encode_fmp4, video_mime

pytestmark = pytest.mark.skipif(
    shutil.which(envs.FASTVIDEO_FFMPEG_BIN.get()) is None,
    reason="ffmpeg is not installed",
)

_FPS = 24
_NUM_FRAMES = 24
_SAMPLE_RATE = 16000


def _frames() -> list[np.ndarray]:
    return [np.full((96, 128, 3), 8 * index, dtype=np.uint8) for index in range(_NUM_FRAMES)]


def _encode(**kwargs) -> bytes:

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in encode_fmp4(_frames(), _FPS, **kwargs)])

    return asyncio.run(collect())


def test_encode_fmp4_with_audio_produces_fragmented_h264_and_aac() -> None:
    av = pytest.importorskip("av")
    audio = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi, _SAMPLE_RATE)).astype(np.float32)  # one second, longer than the video
    data = _encode(audio=audio, sample_rate=_SAMPLE_RATE)

    assert data[4:8] == b"ftyp"
    assert b"moof" in data  # media fragments, as MSE needs
    # The avcC box after the init segment's "avcC" tag holds version, profile, constraint flags, and level.
    avcc = data.index(b"avcC") + 4
    assert f"avc1.{data[avcc + 1]:02X}{data[avcc + 2]:02X}{data[avcc + 3]:02X}" == "avc1.42C028"
    with av.open(io.BytesIO(data)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.audio[0].codec_context.name == "aac"
        assert sum(1 for _ in container.decode(video=0)) == _NUM_FRAMES
    assert video_mime(has_audio=True) == 'video/mp4; codecs="avc1.42C028, mp4a.40.2"'


def test_encode_fmp4_without_audio_has_only_video() -> None:
    av = pytest.importorskip("av")
    data = _encode()

    with av.open(io.BytesIO(data)) as container:
        assert len(container.streams.video) == 1 and len(container.streams.audio) == 0
    assert video_mime(has_audio=False) == 'video/mp4; codecs="avc1.42C028"'


def test_encode_fmp4_rejects_non_rgb_frames() -> None:

    async def collect() -> None:
        async for _ in encode_fmp4([np.zeros((8, 8), dtype=np.uint8)], _FPS):
            pass

    with pytest.raises(ValueError, match="HxWx3 uint8 RGB"):
        asyncio.run(collect())
