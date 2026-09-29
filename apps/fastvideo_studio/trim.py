# SPDX-License-Identifier: Apache-2.0
"""Cut a finished clip to a chosen in/out range and adjust its color, non-destructively.

Kept apart from ``frames.py`` (which saves the *end* of a clip for another job
to continue from) because this edits a clip's own timeline, not a reference
for someone else. Needs only PyAV and numpy, so it can be exercised without a GPU.

Trimming and grading share one pipeline rather than being two independent
edits, on purpose: both always re-render from the untouched original in a
single pass with the *current* full set of parameters (range and color
together). Two separate "always start from the original" operations would
each undo the other's work the moment they ran -- grading, then trimming,
would silently discard the grade, because the trim would re-cut the original
as if the grade had never happened. Rendering everything from one function
each time is what lets adjusting either one leave the other in place.

Color grading is per razor-cut section, not the whole clip at once: the kept
range is split at zero or more cut points into contiguous sections (no
reordering, no gaps), each with its own brightness/contrast/saturation. A
plain range trim with no cuts is just the one-section case. Audio is one
continuous track regardless -- cuts only change which grade a video frame
gets, they don't touch the audio.
"""

from __future__ import annotations

import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from fastvideo_studio.frames import VIDEO_EXTENSIONS, FrameError

ORIGINAL_FILENAME = "original.mp4"
EDITED_FILENAME = "edited.mp4"

# Neutral (no-op) values for each color control, and the range each is clamped to.
BRIGHTNESS_RANGE = (-100.0, 100.0)  # added directly to 0-255 pixel values
CONTRAST_RANGE = (0.0, 3.0)  # 1.0 = unchanged
SATURATION_RANGE = (0.0, 3.0)  # 1.0 = unchanged, 0 = grayscale

# Perceptual luma weights, matching the ones used elsewhere in Studio for brightness checks.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def clip_duration_seconds(video_path: str) -> float:
    """Decode ``video_path`` and return its length in seconds."""
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 24)
        n = sum(1 for _ in container.decode(stream))
    if n == 0:
        raise ValueError("The video has no frames.")
    return n / fps


def _clamp_range(name: str, value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if not (low <= value <= high):
        raise ValueError(f"{name} must be between {low:g} and {high:g}, got {value:g}.")
    return value


def _apply_grade(array: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Adjust one RGB uint8 frame. Neutral values (0, 1, 1) are a no-op, skipped by the caller."""
    pixels = array.astype(np.float32)
    if saturation != 1.0:
        luma = (pixels * _LUMA_WEIGHTS).sum(axis=-1, keepdims=True)
        pixels = luma + (pixels - luma) * saturation
    if contrast != 1.0:
        pixels = (pixels - 128.0) * contrast + 128.0
    if brightness != 0.0:
        pixels = pixels + brightness
    return np.clip(pixels, 0, 255).astype(np.uint8)


#: A neutral single section covering the whole kept range -- what a never-edited
#: clip (or a call that doesn't care about sections) gets by default.
DEFAULT_SEGMENTS: list[dict[str, Any]] = [{"end_seconds": None, "brightness": 0.0, "contrast": 1.0, "saturation": 1.0}]


class _Segment:
    """One validated, frame-indexed section: [start_frame, end_frame) plus its grade."""

    __slots__ = ("start_frame", "end_frame", "brightness", "contrast", "saturation")

    def __init__(self, start_frame: int, end_frame: int, brightness: float, contrast: float, saturation: float):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    @property
    def graded(self) -> bool:
        return self.brightness != 0.0 or self.contrast != 1.0 or self.saturation != 1.0


def _normalize_segments(segments: list[dict[str, Any]] | None, kept_frames: int, fps: float) -> list[_Segment]:
    """Validate a razor-cut section list and convert each boundary to a frame index.

    Sections are contiguous and cover the whole kept range in order -- there's
    no reordering or gaps, only where to cut and what to grade each side. Cut
    points are seconds into the *kept* (already start/end-trimmed) range, i.e.
    the same timeline the preview video itself plays, not the original file's.
    """
    segments = segments or DEFAULT_SEGMENTS

    out: list[_Segment] = []
    start_frame = 0
    prev_end_seconds = 0.0
    for i, seg in enumerate(segments):
        is_last = i == len(segments) - 1
        end_seconds = seg.get("end_seconds")
        brightness = _clamp_range("brightness", float(seg.get("brightness", 0.0)), BRIGHTNESS_RANGE)
        contrast = _clamp_range("contrast", float(seg.get("contrast", 1.0)), CONTRAST_RANGE)
        saturation = _clamp_range("saturation", float(seg.get("saturation", 1.0)), SATURATION_RANGE)
        if end_seconds is None:
            if not is_last:
                raise ValueError("Only the last section can be open-ended.")
            end_frame = kept_frames
        else:
            if end_seconds <= prev_end_seconds:
                raise ValueError("Sections must be in increasing order.")
            end_frame = min(kept_frames, round(end_seconds * fps))
            prev_end_seconds = end_seconds
        if end_frame <= start_frame:
            raise ValueError("A section is too short to keep any frames.")
        out.append(_Segment(start_frame, end_frame, brightness, contrast, saturation))
        start_frame = end_frame
    if start_frame < kept_frames:
        raise ValueError("The sections must cover the whole clip; the last one can't end early.")
    return out


def _render_edit(
    video_path: str,
    out_path: str,
    start_seconds: float,
    end_seconds: float | None,
    segments: list[dict[str, Any]] | None,
) -> int:
    """Re-encode [``start_seconds``, ``end_seconds``) of ``video_path`` to ``out_path``, graded per section.

    Re-encoded rather than stream-copied, for a frame-accurate cut regardless
    of where the source's keyframes happen to fall. Returns the frame count kept.
    """
    import av

    if start_seconds < 0:
        raise ValueError("The start of the range can't be negative.")
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("The end of the range must come after the start.")

    with av.open(video_path) as src:
        video_stream = src.streams.video[0]
        audio_stream = src.streams.audio[0] if src.streams.audio else None
        fps = float(video_stream.average_rate or video_stream.guessed_rate or 24)
        frames: list[Any] = []
        audio_frames: list[Any] = []
        # One decode pass covers both streams -- the container can't be
        # decoded a second time once its packets are exhausted.
        decode_streams = (video_stream, audio_stream) if audio_stream else (video_stream,)
        for frame in src.decode(*decode_streams):
            (audio_frames if isinstance(frame, av.AudioFrame) else frames).append(frame)
    if not frames:
        raise ValueError("The video has no frames.")

    start_i = max(0, round(start_seconds * fps))
    end_i = len(frames) if end_seconds is None else min(len(frames), round(end_seconds * fps))
    kept = frames[start_i:end_i]
    if not kept:
        raise ValueError("That range keeps no frames.")
    # Audio is cut to the same frame-granular window as the video, not the raw
    # requested seconds, so the two tracks stay aligned. Razor cuts only affect
    # which grade a frame gets, not the audio, which stays one continuous track.
    kept_audio = [f for f in audio_frames if f.time is not None and start_i / fps <= f.time < end_i / fps]
    section_list = _normalize_segments(segments, len(kept), fps)

    tmp_path = f"{out_path}.tmp.mp4"
    # movflags=faststart puts the moov atom (duration, seek table) at the front of
    # the file instead of the end -- without it, a browser has to reach nearly the
    # end of the file just to read the video's metadata, which is painfully slow
    # over a network filesystem.
    with av.open(tmp_path, mode="w", options={"movflags": "faststart"}) as out:
        out_stream = out.add_stream("h264", rate=round(fps))
        out_stream.width, out_stream.height = kept[0].width, kept[0].height
        out_stream.pix_fmt = "yuv420p"
        # Every output stream has to be added before the first packet is muxed
        # (that's what writes the container header), so the audio stream and
        # resampler are set up here even though they're only used after the
        # video loop below.
        out_audio_stream = None
        resampler = None
        if kept_audio:
            out_audio_stream = out.add_stream("aac", rate=audio_stream.codec_context.sample_rate)
            out_audio_stream.layout = audio_stream.codec_context.layout.name
            resampler = av.AudioResampler(
                format=out_audio_stream.format,
                layout=out_audio_stream.layout,
                rate=out_audio_stream.rate,
            )

        # Each frame needs its own increasing pts at a known time base, or the
        # muxed container's frame-rate metadata comes out wrong (see frames.py's
        # save_last_clip, which hit exactly this).
        frame_time_base = Fraction(1, round(fps))
        section_i = 0
        for i, frame in enumerate(kept):
            while i >= section_list[section_i].end_frame:
                section_i += 1
            section = section_list[section_i]
            if section.graded:
                array = _apply_grade(frame.to_ndarray(format="rgb24"), section.brightness, section.contrast,
                                      section.saturation)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = i
            frame.time_base = frame_time_base
            for packet in out_stream.encode(frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)

        if out_audio_stream is not None:
            for frame in kept_audio:
                frame.pts = None  # let the resampler retime relative to the trimmed start
                for rframe in resampler.resample(frame):
                    for packet in out_audio_stream.encode(rframe):
                        out.mux(packet)
            for packet in out_audio_stream.encode(None):
                out.mux(packet)
    os.replace(tmp_path, out_path)  # atomic, so a reader never sees a half-written file
    return len(kept)


def _require_completed_video(job: Any) -> None:
    status = getattr(job.status, "value", job.status)
    if status != "completed" or not job.output_path:
        raise FrameError(404, "No output available for this job")
    if not os.path.isfile(job.output_path):
        raise FrameError(404, "Output file not found on disk")
    if Path(job.output_path).suffix.lower() not in VIDEO_EXTENSIONS:
        raise FrameError(400, "This job's output is not a video")


def trim_video(
    job: Any,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    segments: list[dict[str, Any]] | None = None,
) -> tuple[str, int]:
    """Cut ``job``'s video to [``start_seconds``, ``end_seconds``) and grade it section by section.

    ``segments`` is an ordered, contiguous list of razor-cut sections covering
    the whole kept range -- each a ``{"end_seconds", "brightness", "contrast",
    "saturation"}`` dict, ``end_seconds`` being seconds into the *kept* range
    (only the last section's may be ``None``, meaning to the end). Omitted or
    empty defaults to one neutral section covering everything, i.e. what a
    plain range trim with no grading looks like.

    Always renders from the untouched original with this *full* set of
    parameters, so trimming and grading compose -- adjusting one doesn't
    discard the other, and neither ever re-encodes an already-edited file.
    Returns ``(new_output_path, frames_kept)``. Raises :class:`FrameError` with
    the status the API should report.
    """
    _require_completed_video(job)
    job_dir = os.path.dirname(job.output_path)
    original_path = os.path.join(job_dir, ORIGINAL_FILENAME)
    if not os.path.isfile(original_path):
        # First edit: keep an untouched copy of whatever is there now before
        # changing anything, so every later edit (and a restore) is relative
        # to this, never to a file this function already produced.
        shutil.copy2(job.output_path, original_path)

    edited_path = os.path.join(job_dir, EDITED_FILENAME)
    try:
        kept = _render_edit(original_path, edited_path, start_seconds, end_seconds, segments)
    except FrameError:
        raise
    except Exception as e:
        raise FrameError(400, str(e)) from e
    return edited_path, kept


def restore_original(job: Any) -> tuple[str, int]:
    """Undo every edit: the path of ``job``'s untouched original video, and its frame count.

    Undoes trimming and grading together, since they share one edit state.
    Raises :class:`FrameError` if the job was never edited.
    """
    if not job.output_path:
        raise FrameError(404, "No output available for this job")
    original_path = os.path.join(os.path.dirname(job.output_path), ORIGINAL_FILENAME)
    if not os.path.isfile(original_path):
        raise FrameError(404, "This job has not been edited.")
    try:
        import av

        with av.open(original_path) as container:
            stream = container.streams.video[0]
            n = sum(1 for _ in container.decode(stream))
    except Exception as e:
        raise FrameError(500, f"Could not read the original video: {e}") from e
    return original_path, n
