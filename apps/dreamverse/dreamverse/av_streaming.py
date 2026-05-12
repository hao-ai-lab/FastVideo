# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportConstantRedefinition=false, reportCallIssue=false
# ruff: noqa: UP007, SIM108, SIM105
# mypy: ignore-errors
"""ffmpeg fMP4 muxing with chunk-level event emission.

Self-contained: spawns ffmpeg as a subprocess, pipes raw frames into
its stdin, reads fragmented-MP4 chunks from stdout, and publishes each
chunk as a ``StreamEvent`` via the caller-supplied ``publish``
callback.  Knows nothing about multiprocessing queues, the GPU pool,
or individual users — the caller decides what "publish" means.
"""
import fcntl
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Union
from collections.abc import Callable

import numpy as np
import torch

FFMPEG_BIN = shutil.which(os.getenv("FASTVIDEO_FFMPEG_BIN", "ffmpeg"))
AV_MEDIA_MIME = os.getenv(
    "STREAM_MIME_TYPE",
    'video/mp4; codecs="avc1.42E01E,mp4a.40.2"',
)
AV_CHUNK_SIZE_BYTES = 1048576
TARGET_FPS = 24
AV_FRAGMENT_DURATION_US = int(os.getenv("FASTVIDEO_FRAG_US", "250000"))
X264_GOP_FRAMES = int(os.getenv("FASTVIDEO_X264_GOP", "12"))
X264_PROFILE = os.getenv("FASTVIDEO_X264_PROFILE", "baseline").strip().lower()
if X264_PROFILE not in {"baseline", "main", "high", "main10", "high10"}:
    print(f"[WARN] Unsupported FASTVIDEO_X264_PROFILE={X264_PROFILE}; using baseline")
    X264_PROFILE = "baseline"
USE_SHARED_STREAM_BUFFER = (os.getenv("FASTVIDEO_USE_SHARED_STREAM_BUFFER", "1").strip().lower()
                            not in {"0", "false", "no"})
SHARED_STREAM_BUFFER_BYTES = int(os.getenv("FASTVIDEO_SHARED_STREAM_BUFFER_BYTES", str(256 * 1024 * 1024)))


@dataclass
class StreamInit:
    """First event emitted — tells the consumer the stream is starting."""
    stream_id: str
    mime: str
    uses_shared_buffer: bool


@dataclass
class StreamChunk:
    """One fMP4 chunk.  Either ``chunk`` (raw bytes) or
    ``chunk_offset``+``chunk_length`` (read from the shared buffer)
    will be populated, never both."""
    stream_id: str
    chunk: bytes | None = None
    chunk_offset: int | None = None
    chunk_length: int | None = None
    uses_shared_buffer: bool = False


@dataclass
class StreamComplete:
    """Final event emitted — muxing finished successfully."""
    stream_id: str
    chunks: int


StreamEvent = Union[StreamInit, StreamChunk, StreamComplete]


def generate_stream_id(segment_idx: int) -> str:
    """Convenience: build a stream id of the form ``seg007-abcd1234``."""
    return f"seg{segment_idx:03d}-{uuid.uuid4().hex[:8]}"


def _normalize_audio_tensor(audio: object) -> tuple[np.ndarray, int] | None:
    """Convert audio tensor/array into int16 ndarray [samples, channels]."""
    if audio is None:
        return None

    if torch.is_tensor(audio):
        audio_np = audio.detach().cpu().float().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2:
        if audio_np.shape[0] <= 8 and audio_np.shape[1] > audio_np.shape[0]:
            audio_np = audio_np.T
    else:
        return None

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    num_channels = audio_int16.shape[1]
    return audio_int16, num_channels


def _write_audio_wav(
    audio_int16: np.ndarray,
    num_channels: int,
    sample_rate: int,
) -> str:
    """Write normalized int16 audio to a temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    except Exception:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass
        raise
    return wav_path


def stream_fmp4(
    *,
    frames: list[np.ndarray],
    audio: object,
    audio_sample_rate: int | None,
    stream_id: str,
    timings: dict,
    head_trim_frames: int = 0,
    head_trim_audio_frames: int | None = None,
    shared_buffer=None,
    shared_buffer_bytes: int = 0,
    publish: Callable[[StreamEvent], None],
    log_prefix: str = "",
) -> tuple[bool, str | None]:
    """Encode frames+audio with ffmpeg, publish each fMP4 chunk as an event.

    Args:
        frames: RGB24 video frames as HxWx3 uint8 arrays.
        audio: 1D/2D tensor or ndarray, float values in [-1, 1].
        audio_sample_rate: sample rate of ``audio``.
        stream_id: caller-supplied identifier carried on every event.
        timings: dict mutated in place with ffmpeg/stream timing metrics.
        head_trim_frames: video frames to drop from the start
            (conditioning overlap).
        head_trim_audio_frames: video-frame-equivalent audio to drop.
            Defaults to ``head_trim_frames``.
        shared_buffer: optional ``mp.RawArray``-compatible object; when
            provided, chunks are written into it and emitted by offset
            rather than by bytes (avoids IPC copies).
        shared_buffer_bytes: size of ``shared_buffer`` in bytes.
        publish: callback invoked once per stream event.
        log_prefix: prepended to warning prints (e.g. ``"[GPU 0]"``).

    Returns:
        ``(True, None)`` on success, ``(False, error_message)`` on
        failure.  On mid-stream failure, a ``StreamInit`` may have
        already been published — the caller is responsible for
        handling that.
    """
    if not frames:
        return False, "no frames returned"
    if audio is None:
        return False, "audio is None"
    if audio_sample_rate is None:
        return False, "audio_sample_rate is None"
    if FFMPEG_BIN is None:
        return False, "ffmpeg not found"

    if head_trim_audio_frames is None:
        head_trim_audio_frames = head_trim_frames

    normalized_audio = _normalize_audio_tensor(audio)
    if normalized_audio is None:
        shape_hint = getattr(audio, "shape", None)
        return False, f"unsupported audio shape={shape_hint}"
    audio_int16, num_channels = normalized_audio

    if head_trim_frames < 0:
        return False, (f"head_trim_frames must be >= 0, "
                       f"got {head_trim_frames}")
    if head_trim_frames >= len(frames):
        return False, (f"head_trim_frames={head_trim_frames} removes "
                       f"all {len(frames)} frames in segment")

    out_frames = (frames[head_trim_frames:] if head_trim_frames > 0 else frames)
    sample_rate = int(audio_sample_rate)
    if head_trim_audio_frames > 0:
        trim_start_samples = int(round((head_trim_audio_frames / float(TARGET_FPS)) * sample_rate))
        if trim_start_samples >= audio_int16.shape[0]:
            return False, ("audio too short after overlap trim: "
                           f"trim_start_samples={trim_start_samples}"
                           f", audio_samples={audio_int16.shape[0]}")
        keep_samples = int(round((len(out_frames) / float(TARGET_FPS)) * sample_rate))
        trim_end_samples = min(
            audio_int16.shape[0],
            trim_start_samples + keep_samples,
        )
        if trim_end_samples <= trim_start_samples:
            return False, ("invalid audio trim range: "
                           f"start={trim_start_samples}, "
                           f"end={trim_end_samples}")
        audio_int16 = audio_int16[trim_start_samples:trim_end_samples]

    height = int(out_frames[0].shape[0])
    width = int(out_frames[0].shape[1])
    codec = os.getenv("FASTVIDEO_VIDEO_CODEC", "libx264")
    t_wav_start = time.perf_counter()
    wav_path = _write_audio_wav(audio_int16, num_channels, sample_rate)
    wav_write_ms = (time.perf_counter() - t_wav_start) * 1000

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(TARGET_FPS),
        "-i",
        "pipe:0",
        "-i",
        wav_path,
        "-c:v",
        codec,
    ]
    if codec.endswith("_nvenc"):
        cmd += [
            "-preset",
            os.getenv("FASTVIDEO_NVENC_PRESET", "p1"),
            "-tune",
            os.getenv("FASTVIDEO_NVENC_TUNE", "ull"),
            "-rc",
            os.getenv("FASTVIDEO_NVENC_RC", "constqp"),
            "-qp",
            os.getenv("FASTVIDEO_NVENC_QP", "28"),
            "-bf",
            os.getenv("FASTVIDEO_NVENC_BF", "0"),
        ]
    else:
        cmd += [
            "-preset",
            os.getenv("FASTVIDEO_X264_PRESET", "ultrafast"),
            "-tune",
            "zerolatency",
            "-profile:v",
            X264_PROFILE,
            # Emit frequent keyframes so fragments are independently playable.
            "-g",
            str(X264_GOP_FRAMES),
            "-keyint_min",
            str(X264_GOP_FRAMES),
            "-x264-params",
            "scenecut=0",
        ]
    cmd += [
        "-c:a",
        "aac",
        "-pix_fmt",
        os.getenv("FASTVIDEO_OUTPUT_PIX_FMT", "yuv420p"),
        "-shortest",
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-frag_duration",
        str(AV_FRAGMENT_DURATION_US),
        "-flush_packets",
        "1",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-f",
        "mp4",
        "pipe:1",
    ]

    proc: subprocess.Popen | None = None
    stderr_chunks: list[bytes] = []
    writer_error: list[Exception | None] = [None]
    t_stream_start = time.perf_counter()
    use_shared_buffer = (USE_SHARED_STREAM_BUFFER and shared_buffer is not None and shared_buffer_bytes > 0)
    shared_write_offset = 0
    shared_buffer_fallback = False
    shared_np = (np.frombuffer(
        shared_buffer,
        dtype=np.uint8,
        count=shared_buffer_bytes,
    ) if use_shared_buffer else None)
    chunk_intervals_ms: list[float] = []
    chunk_publish_ms: list[float] = []
    chunk_read_ms: list[float] = []

    try:
        t_proc_spawn_start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        if hasattr(fcntl, "F_SETPIPE_SZ"):
            fcntl.fcntl(proc.stdin.fileno(), fcntl.F_SETPIPE_SZ, 1048576)
            fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETPIPE_SZ, 1048576)
        ffmpeg_spawn_ms = (time.perf_counter() - t_proc_spawn_start) * 1000

        def _write_frames():
            try:
                for frame in out_frames:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                proc.stdin.close()
            except Exception as exc:
                writer_error[0] = exc
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        def _read_stderr():
            try:
                while True:
                    data = proc.stderr.read(4096)
                    if not data:
                        break
                    stderr_chunks.append(data)
            except Exception:
                pass

        writer_thread = threading.Thread(target=_write_frames, daemon=True)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        writer_thread.start()
        stderr_thread.start()

        publish(StreamInit(
            stream_id=stream_id,
            mime=AV_MEDIA_MIME,
            uses_shared_buffer=use_shared_buffer,
        ))

        chunk_count = 0
        total_bytes = 0
        first_chunk_ms: float | None = None
        last_chunk_emit_t = time.perf_counter()
        while True:
            t_read_start = time.perf_counter()
            chunk = proc.stdout.read(AV_CHUNK_SIZE_BYTES)
            t_read_end = time.perf_counter()
            if not chunk:
                break
            chunk_read_ms.append((t_read_end - t_read_start) * 1000)
            chunk_count += 1
            total_bytes += len(chunk)
            if first_chunk_ms is None:
                first_chunk_ms = (t_read_end - t_stream_start) * 1000
            chunk_intervals_ms.append((t_read_end - last_chunk_emit_t) * 1000)
            t_publish_start = time.perf_counter()
            if use_shared_buffer and not shared_buffer_fallback:
                chunk_len = len(chunk)
                write_end = shared_write_offset + chunk_len
                if write_end <= shared_buffer_bytes:
                    shared_np[shared_write_offset:write_end] = np.frombuffer(chunk, dtype=np.uint8)
                    publish(
                        StreamChunk(
                            stream_id=stream_id,
                            chunk_offset=shared_write_offset,
                            chunk_length=chunk_len,
                            uses_shared_buffer=True,
                        ))
                    shared_write_offset = write_end
                    chunk_publish_ms.append((time.perf_counter() - t_publish_start) * 1000)
                    last_chunk_emit_t = time.perf_counter()
                    continue
                shared_buffer_fallback = True
                print(f"{log_prefix} Shared stream buffer exhausted at "
                      f"{shared_write_offset / (1024 * 1024):.1f}MB; "
                      "falling back to queue chunk bytes")
            publish(StreamChunk(
                stream_id=stream_id,
                chunk=chunk,
            ))
            chunk_publish_ms.append((time.perf_counter() - t_publish_start) * 1000)
            last_chunk_emit_t = time.perf_counter()

        writer_thread.join(timeout=5.0)
        rc = proc.wait()
        stderr_thread.join(timeout=1.0)

        if rc != 0:
            stderr_tail = b"".join(stderr_chunks).decode(errors="ignore")[-1200:]
            return False, f"ffmpeg av_fmp4 stream failed (rc={rc}): {stderr_tail}"
        if writer_error[0] is not None:
            return False, f"ffmpeg frame writer failed: {writer_error[0]}"

        timings["av_encode_stream_ms"] = (time.perf_counter() - t_stream_start) * 1000
        timings["av_stream_bytes"] = total_bytes
        timings["av_trim_head_frames"] = head_trim_frames
        timings["av_trim_head_audio_frames"] = head_trim_audio_frames
        timings["av_frames_encoded"] = len(out_frames)
        timings["av_shared_buffer_used"] = (bool(use_shared_buffer and not shared_buffer_fallback))
        timings["av_wav_write_ms"] = wav_write_ms
        timings["av_ffmpeg_spawn_ms"] = ffmpeg_spawn_ms
        timings["av_first_chunk_ms"] = first_chunk_ms or 0.0
        if chunk_intervals_ms:
            timings["av_chunk_interval_ms_min"] = min(chunk_intervals_ms)
            timings["av_chunk_interval_ms_median"] = float(np.median(chunk_intervals_ms))
            timings["av_chunk_interval_ms_p95"] = (float(np.percentile(chunk_intervals_ms, 95)))
            timings["av_chunk_interval_ms_max"] = max(chunk_intervals_ms)
        if chunk_publish_ms:
            timings["av_chunk_publish_ms_median"] = float(np.median(chunk_publish_ms))
            timings["av_chunk_publish_ms_p95"] = float(np.percentile(chunk_publish_ms, 95))
        if chunk_read_ms:
            timings["av_chunk_read_ms_median"] = float(np.median(chunk_read_ms))
            timings["av_chunk_read_ms_p95"] = float(np.percentile(chunk_read_ms, 95))
        publish(StreamComplete(
            stream_id=stream_id,
            chunks=chunk_count,
        ))
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            os.remove(wav_path)
        except OSError:
            pass
