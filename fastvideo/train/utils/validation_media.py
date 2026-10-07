# SPDX-License-Identifier: Apache-2.0
"""Encode and verify validation frames with optional synchronized audio.

Validation artifacts are logged to experiment trackers by filename. Stream
verification prevents the tracker from accepting an MP4 that silently dropped
a requested audio stream.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from fractions import Fraction

import av
import numpy as np
import torch


def _normalize_validation_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Convert RGB frames to the contiguous, even-sized layout required by H.264."""
    if not frames:
        raise ValueError("Validation media requires at least one video frame.")
    frame_array = np.stack(frames)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError("Validation frames must have shape [frames, height, width, 3]; "
                         f"got {frame_array.shape}.")
    if frame_array.dtype != np.uint8:
        raise TypeError("Validation frames must use uint8 RGB values; "
                        f"got {frame_array.dtype}.")
    if frame_array.shape[1] % 2 or frame_array.shape[2] % 2:
        raise ValueError("H.264 yuv420p encoding requires even frame dimensions; "
                         f"got {frame_array.shape[1]} x {frame_array.shape[2]}.")
    return np.ascontiguousarray(frame_array)


def _normalize_audio_waveform(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert supported waveform layouts to sample-major float32.

    Model outputs can be channel-major while PyAV input preparation uses
    sample-major arrays. Ambiguous two-dimensional shapes raise an error so
    validation never exchanges the sample and channel axes silently.
    """
    waveform = (audio.detach().cpu().float().numpy() if torch.is_tensor(audio) else np.asarray(audio, dtype=np.float32))

    if waveform.ndim == 1:
        waveform = waveform[:, None]
    elif waveform.ndim == 2:
        first_axis_is_channels = waveform.shape[0] <= 8
        second_axis_is_channels = waveform.shape[1] <= 8
        if first_axis_is_channels and not second_axis_is_channels:
            waveform = waveform.T
        elif first_axis_is_channels == second_axis_is_channels:
            raise ValueError("A two-dimensional audio waveform must have one channel axis "
                             f"with at most eight entries; got {waveform.shape}.")
    else:
        raise ValueError("Audio must have shape [samples], [samples, channels], or "
                         f"[channels, samples]; got {waveform.shape}.")

    if waveform.shape[0] == 0:
        raise ValueError("Validation audio requires at least one sample.")
    if not np.isfinite(waveform).all():
        raise ValueError("Validation audio contains a nonfinite sample.")
    return np.ascontiguousarray(np.clip(waveform, -1.0, 1.0), dtype=np.float32)


def _trim_to_shared_duration(
    frames: np.ndarray,
    fps: int,
    waveform: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim both streams to a shared duration containing complete video frames."""
    audio_duration = waveform.shape[0] / sample_rate
    frame_count = min(frames.shape[0], math.floor(audio_duration * fps + 1e-9))
    if frame_count < 1:
        raise ValueError("Validation audio must cover at least one complete video frame.")
    audio_sample_count = min(
        waveform.shape[0],
        math.floor(frame_count / fps * sample_rate + 1e-9),
    )
    if audio_sample_count < 1:
        raise ValueError("Validation media has no shared audio-video duration.")
    return frames[:frame_count], waveform[:audio_sample_count]


def _encode_mp4(
    output_path: str,
    frames: np.ndarray,
    fps: int,
    waveform: np.ndarray | None,
    sample_rate: int | None,
) -> None:
    """Encode H.264 and optional Advanced Audio Coding into one MP4 artifact."""
    with av.open(output_path, mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = int(frames.shape[2])
        video_stream.height = int(frames.shape[1])
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"preset": "ultrafast"}
        audio_stream = None
        if waveform is not None and sample_rate is not None:
            channel_count = int(waveform.shape[1])
            if channel_count not in (1, 2):
                raise ValueError("Validation MP4 audio must be mono or stereo; "
                                 f"got {channel_count} channels.")
            layout = "mono" if channel_count == 1 else "stereo"
            audio_stream = container.add_stream("aac", rate=sample_rate, layout=layout)

        for frame_index, frame_array in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            video_frame.pts = frame_index
            video_frame.time_base = Fraction(1, fps)
            for packet in video_stream.encode(video_frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        if waveform is None or sample_rate is None or audio_stream is None:
            return

        layout = "mono" if waveform.shape[1] == 1 else "stereo"
        audio_int16 = (waveform * 32767.0).astype(np.int16)
        # AAC uses 1024-sample frames; flushing after the trailing short chunk lets
        # the codec emit every buffered sample into the container.
        for sample_index in range(0, audio_int16.shape[0], 1024):
            chunk = np.ascontiguousarray(audio_int16[sample_index:sample_index + 1024].T)
            audio_frame = av.AudioFrame.from_ndarray(chunk, format="s16p", layout=layout)
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = sample_index
            audio_frame.time_base = Fraction(1, sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


def _verify_encoded_streams(
    output_path: str,
    *,
    expected_audio_channels: int | None,
    expected_audio_sample_rate: int | None,
) -> None:
    """Verify requested streams before the temporary file replaces its destination."""
    with av.open(output_path) as container:
        if len(container.streams.video) != 1:
            raise RuntimeError("Validation MP4 encoding must produce exactly one video stream.")
        if expected_audio_channels is None:
            if container.streams.audio:
                raise RuntimeError("Silent validation media contains an unexpected audio stream.")
            return
        if len(container.streams.audio) != 1:
            raise RuntimeError("Audio-bearing validation MP4 encoding must produce exactly one audio stream.")
        audio_stream = container.streams.audio[0]
        encoded_channels = len(audio_stream.codec_context.layout.channels)
        encoded_sample_rate = int(audio_stream.codec_context.sample_rate)
        if encoded_channels != expected_audio_channels:
            raise RuntimeError("Validation MP4 audio channel count changed during encoding: "
                               f"expected {expected_audio_channels}, got {encoded_channels}.")
        if encoded_sample_rate != expected_audio_sample_rate:
            raise RuntimeError("Validation MP4 audio sample rate changed during encoding: "
                               f"expected {expected_audio_sample_rate}, got {encoded_sample_rate}.")


def write_validation_mp4(
    output_path: str,
    frames: list[np.ndarray],
    *,
    fps: int,
    audio: torch.Tensor | np.ndarray | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    """Write validation frames and optional synchronized audio atomically.

    The video input must contain RGB uint8 frames. Audio can use
    ``[samples]``, ``[samples, channels]``, or ``[channels, samples]`` layout.
    When both streams are present, the encoder trims them to their shortest
    complete shared duration. The destination changes only after PyAV encodes
    and verifies every requested stream.
    """
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"Validation video FPS must be a positive integer; got {fps!r}.")
    if (audio is None) != (audio_sample_rate is None):
        raise ValueError("Validation audio and its sample rate must be provided together.")
    if audio_sample_rate is not None and audio_sample_rate <= 0:
        raise ValueError("Validation audio sample rate must be positive; "
                         f"got {audio_sample_rate}.")

    frame_array = _normalize_validation_frames(frames)
    waveform = _normalize_audio_waveform(audio) if audio is not None else None
    if waveform is not None and audio_sample_rate is not None:
        frame_array, waveform = _trim_to_shared_duration(
            frame_array,
            fps,
            waveform,
            audio_sample_rate,
        )

    destination = os.path.abspath(output_path)
    destination_dir = os.path.dirname(destination)
    if not os.path.isdir(destination_dir):
        raise FileNotFoundError(f"Validation media destination directory does not exist: {destination_dir}")
    # A temporary file in the destination directory keeps replacement atomic
    # and leaves any previous artifact intact when encoding or verification fails.
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".mp4",
        dir=destination_dir,
    )
    os.close(file_descriptor)
    try:
        _encode_mp4(
            temporary_path,
            frame_array,
            fps,
            waveform,
            audio_sample_rate,
        )
        _verify_encoded_streams(
            temporary_path,
            expected_audio_channels=(int(waveform.shape[1]) if waveform is not None else None),
            expected_audio_sample_rate=audio_sample_rate,
        )
        # Publish only an MP4 whose requested stream contract was verified.
        os.replace(temporary_path, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary_path)


__all__ = ["write_validation_mp4"]
