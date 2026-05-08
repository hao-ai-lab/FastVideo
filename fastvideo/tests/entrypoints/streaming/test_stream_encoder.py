# SPDX-License-Identifier: Apache-2.0
"""Tests for the fMP4 encoder.

These tests require ``ffmpeg`` on PATH. Skip when missing so the suite
stays CPU/CI friendly.
"""
from __future__ import annotations

import asyncio
import shutil

import numpy as np
import pytest

from fastvideo.entrypoints.streaming.stream import (
    FragmentedMP4Chunk,
    FragmentedMP4Encoder,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed",
)


def _frame(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_encoder_emits_init_then_media_chunks():
    async def run():
        enc = FragmentedMP4Encoder(
            width=64, height=64, fps=24, segment_idx=0)
        chunks: list[FragmentedMP4Chunk] = []
        async with enc:
            frames = [_frame(64, 64, v) for v in range(4, 28)]
            async for chunk in enc.encode(frames):
                chunks.append(chunk)
        assert len(chunks) > 0
        assert chunks[0].kind == "init"
        assert all(c.stream_id == enc.stream_id for c in chunks)
        assert all(c.segment_idx == 0 for c in chunks)

    asyncio.run(run())


def test_encoder_init_chunk_is_fmp4():
    """The first chunk must contain the ``ftyp`` box (fMP4 init segment)."""
    async def run():
        enc = FragmentedMP4Encoder(
            width=64, height=64, fps=24, segment_idx=0)
        first_chunk = None
        async with enc:
            async for chunk in enc.encode([_frame(64, 64, 20)] * 24):
                first_chunk = chunk
                break
        assert first_chunk is not None
        assert first_chunk.kind == "init"
        # Box header: 4 bytes length, 4 bytes type. "ftyp" should appear
        # near the start of the init segment.
        assert b"ftyp" in first_chunk.data[:32]

    asyncio.run(run())


def test_encoder_rejects_non_ndarray_frames():
    async def run():
        enc = FragmentedMP4Encoder(
            width=64, height=64, fps=24, segment_idx=0)
        async with enc:
            with pytest.raises(TypeError):
                async for _ in enc.encode(["not-a-frame"]):
                    pass

    asyncio.run(run())


def test_encoder_rejects_wrong_shape():
    async def run():
        enc = FragmentedMP4Encoder(
            width=64, height=64, fps=24, segment_idx=0)
        async with enc:
            with pytest.raises(ValueError):
                async for _ in enc.encode(
                        [np.zeros((64, 64, 4), dtype=np.uint8)]):
                    pass

    asyncio.run(run())


def test_encoder_close_is_idempotent():
    async def run():
        enc = FragmentedMP4Encoder(
            width=64, height=64, fps=24, segment_idx=0)
        await enc.__aenter__()
        await enc.close()
        await enc.close()  # no raise

    asyncio.run(run())
