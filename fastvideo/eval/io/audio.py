"""Audio-extraction helper for video files.

Audio metrics (``audio.clap_score``, ``audio.frechet_distance``, …) read
file paths under ``sample["audio"]`` and do their own per-metric
preprocessing (CLAP at 48 kHz, PaSST at 32 kHz, whisper at 16 kHz mono,
…).  When the source video carries an audio track (V2A / T2A-V model
outputs), :func:`samples_from(extract_audio=True)` calls
:func:`extract_audio_track` to pull a ``.wav`` next to it once per video.

Why a separate utility instead of in-pool decode: every audio metric
wants a different sample rate / channel count / format, so the pool
can't usefully pre-load audio into a single canonical tensor the way it
pre-loads video frames.  Paths-in / paths-out is the lingua franca for
audio in this codebase.
"""
from __future__ import annotations

from pathlib import Path


class NoAudioStreamError(ValueError):
    """Raised when a video file has no audio stream to extract."""


def extract_audio_track(
    video_path: str | Path,
    *,
    output_dir: str | Path,
    sample_rate: int | None = None,
    codec: str = "pcm_s16le",
) -> Path:
    """Pull the first audio stream from *video_path* to a ``.wav`` in *output_dir*.

    Idempotent — if ``output_dir / {stem}.wav`` already exists, the
    existing file is returned without re-decoding.  This makes the
    helper safe to call repeatedly from parallel workers on overlapping
    inputs (the cache hit is the fast path).

    Parameters
    ----------
    video_path :
        Source video file (anything PyAV can open: mp4, mkv, webm, …).
    output_dir :
        Directory the ``.wav`` lands in.  Created if it doesn't exist.
    sample_rate :
        If set, resample to this rate.  Default (``None``) preserves the
        source rate — most audio metrics resample again internally to
        their own target rate, so adding a resample step here would just
        be wasted work.
    codec :
        PCM codec for the output.  ``pcm_s16le`` (default) gives 16-bit
        little-endian PCM, the most broadly compatible WAV format.

    Returns
    -------
    Path
        Absolute path to the extracted ``.wav``.

    Raises
    ------
    NoAudioStreamError
        If *video_path* has no audio stream.
    """
    import av

    in_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{in_path.stem}.wav").resolve()
    if out_path.exists():
        return out_path

    container = av.open(str(in_path))
    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise NoAudioStreamError(f"No audio stream in {in_path}")
        in_stream = audio_streams[0]
        target_rate = sample_rate or in_stream.rate

        out = av.open(str(out_path), "w", format="wav")
        try:
            out_stream = out.add_stream(codec, rate=target_rate)
            # Match input layout (mono / stereo / surround) — audio metrics
            # that want mono will downmix themselves.
            if in_stream.layout is not None:
                out_stream.layout = in_stream.layout
            resampler = None
            if sample_rate is not None and sample_rate != in_stream.rate:
                resampler = av.AudioResampler(format=out_stream.format, layout=out_stream.layout, rate=target_rate)
            for frame in container.decode(in_stream):
                frames_out = resampler.resample(frame) if resampler is not None else [frame]
                for f in frames_out:
                    # PyAV needs pts cleared so the encoder reassigns.
                    f.pts = None
                    for packet in out_stream.encode(f):
                        out.mux(packet)
            for packet in out_stream.encode():  # flush
                out.mux(packet)
        finally:
            out.close()
    finally:
        container.close()
    return out_path
