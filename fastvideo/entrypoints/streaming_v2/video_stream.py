# SPDX-License-Identifier: Apache-2.0
"""Encode decoded frames and audio into one fragmented MP4 through ffmpeg.

The output is an init segment followed by media fragments, which a browser's
Media Source Extensions player accepts. The encoder settings are fixed so that
:func:`video_mime` names the exact codecs: H.264 Constrained Baseline level
4.0 (the profile that x264's ``ultrafast`` preset produces) and AAC-LC.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
import wave
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch

from fastvideo import envs

_VIDEO_CODEC = "avc1.42C028"  # H.264 Constrained Baseline (0x42, constraint flags 0xC0), level 4.0 (0x28)
_AUDIO_CODEC = "mp4a.40.2"  # AAC-LC
_CHUNK_BYTES = 64 * 1024
_GOP_FRAMES = 12  # a keyframe, and so a fragment, every half second at 24 fps


def video_mime(has_audio: bool) -> str:
    """The MIME type with codecs of the stream that :func:`encode_fmp4` produces."""
    codecs = f"{_VIDEO_CODEC}, {_AUDIO_CODEC}" if has_audio else _VIDEO_CODEC
    return f'video/mp4; codecs="{codecs}"'


async def encode_fmp4(
    frames: list[np.ndarray],
    fps: int,
    audio: Any | None = None,
    sample_rate: int | None = None,
) -> AsyncIterator[bytes]:
    """Encode HxWx3 uint8 RGB frames, plus optional audio, and yield the fMP4 bytes in order.

    Audio is padded or trimmed to the video's length. Raises ``RuntimeError``
    when ffmpeg is missing or exits with an error; the ffmpeg process is killed
    if the consumer stops reading early.
    """
    ffmpeg_bin = shutil.which(envs.FASTVIDEO_FFMPEG_BIN.get())
    if ffmpeg_bin is None:
        raise RuntimeError(f"ffmpeg not found: {envs.FASTVIDEO_FFMPEG_BIN.get()!r}")
    if not frames:
        raise ValueError("Cannot encode a video without frames.")
    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])

    with tempfile.TemporaryDirectory(prefix="fastvideo-streamv2-") as tmpdir:
        audio_args: list[str] = []
        audio_output_args: list[str] = []
        if audio is not None:
            if sample_rate is None:
                raise ValueError("Audio requires a sample rate.")
            wav_path = os.path.join(tmpdir, "audio.wav")
            _write_wav(wav_path, audio, int(sample_rate))
            audio_args = ["-i", wav_path]
            audio_output_args = ["-c:a", "aac", "-af", "apad", "-shortest"]
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            *audio_args,
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-profile:v", "baseline", "-level:v", "4.0", "-pix_fmt", "yuv420p",
            "-g", str(_GOP_FRAMES), "-keyint_min", str(_GOP_FRAMES), "-x264-params", "scenecut=0",
            *audio_output_args,
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4", "pipe:1",
        ]  # yapf: disable
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Feed frames and drain stderr concurrently so neither pipe can fill up and stall ffmpeg.
        writer = asyncio.create_task(_write_frames(proc, frames))
        stderr_reader = asyncio.create_task(proc.stderr.read())
        try:
            while chunk := await proc.stdout.read(_CHUNK_BYTES):
                yield chunk
            # A broken pipe means ffmpeg exited early; its exit code and stderr below say why.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await writer
            stderr = await stderr_reader
            if await proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace').strip()}")
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            for task in (writer, stderr_reader):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                        await task


async def _write_frames(proc: asyncio.subprocess.Process, frames: list[np.ndarray]) -> None:
    """Write every frame to ffmpeg's stdin as raw RGB; stdin is closed even on failure so ffmpeg can finish."""
    assert proc.stdin is not None
    try:
        for frame in frames:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Frames must be HxWx3 uint8 RGB, got shape={frame.shape} dtype={frame.dtype}.")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            await proc.stdin.drain()
    finally:
        proc.stdin.close()


def _write_wav(path: str, audio: Any, sample_rate: int) -> None:
    """Write ``[samples]``, ``[samples, channels]``, or ``[channels, samples]`` audio in [-1, 1] as 16-bit PCM WAV."""
    samples = audio.detach().cpu().float().numpy() if torch.is_tensor(audio) else np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim != 2:
        raise ValueError(f"Unexpected audio shape {samples.shape}.")
    elif samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
        samples = samples.T  # channels-first layout
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
