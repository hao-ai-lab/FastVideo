# SPDX-License-Identifier: Apache-2.0
"""A finished clip's trailing seconds are saved as a short video the next clip can continue from."""
from __future__ import annotations

import os
import time
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from fastvideo_studio import frames
from fastvideo_studio.frames import FrameError, last_clip_for_job, save_last_clip

WIDTH, HEIGHT, FPS, FRAMES = 128, 96, 24, 48  # 2 seconds


class Status(Enum):  # mirrors JobStatus without importing job_runner (which needs fastvideo)
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


def _write_video(path: str, n: int = FRAMES, fps: int = FPS) -> None:
    """A clip whose frames are numbered by brightness (frame i is roughly i*5 grey), so the tail is identifiable."""
    import av

    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(n):
            level = min(i * 5, 255)
            array = np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _read_levels(path: str) -> list[int]:
    import av

    with av.open(path) as container:
        return [int(f.to_ndarray(format="rgb24")[0, 0, 0]) for f in container.decode(container.streams.video[0])]


def _job(job_id="job-1", status=Status.COMPLETED, output_path=None):
    return SimpleNamespace(id=job_id, status=status, output_path=output_path)


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "clip.mp4")
    _write_video(path)
    return path


@pytest.fixture
def uploads(tmp_path):
    return str(tmp_path / "uploads")


class TestSaveLastClip:
    def test_keeps_roughly_the_requested_seconds(self, video, tmp_path):
        out = str(tmp_path / "tail.mp4")
        save_last_clip(video, out, seconds=1.0)
        assert os.path.isfile(out)
        levels = _read_levels(out)
        assert FPS - 2 <= len(levels) <= FPS + 2  # ~1s at 24fps

    def test_the_muxed_frame_rate_is_correct(self, video, tmp_path):
        # Regression: every frame stamped with the same pts (e.g. all left at 0,
        # or all set to `None` and left uncorrected) still decodes back the right
        # *number* of frames, so a frame-count-only check doesn't catch it -- but
        # it corrupts the container's average-frame-rate metadata, which is what
        # H3's own reference pipeline reads to resample the clip. A wrong rate
        # there silently collapses a short reference clip to as little as one
        # usable frame.
        out = str(tmp_path / "tail.mp4")
        save_last_clip(video, out, seconds=1.0)
        import av

        with av.open(out) as container:
            rate = float(container.streams.video[0].average_rate)
        assert FPS - 1 <= rate <= FPS + 1

    def test_frame_timestamps_are_distinct_and_increasing(self, video, tmp_path):
        out = str(tmp_path / "tail.mp4")
        save_last_clip(video, out, seconds=1.0)
        import av

        with av.open(out) as container:
            times = [f.time for f in container.decode(container.streams.video[0])]
        assert len(set(times)) == len(times)
        assert times == sorted(times)

    def test_keeps_the_end_of_the_source_not_the_start(self, video, tmp_path):
        out = str(tmp_path / "tail.mp4")
        save_last_clip(video, out, seconds=1.0)
        levels = _read_levels(out)
        full = _read_levels(video)
        assert levels[-1] == full[-1]  # the very last frame is preserved
        assert levels[0] > full[0]  # it does not start from the source's beginning

    def test_asking_for_the_whole_clip_or_more_keeps_every_frame(self, video, tmp_path):
        out = str(tmp_path / "tail.mp4")
        save_last_clip(video, out, seconds=10.0)
        assert len(_read_levels(out)) == FRAMES

    def test_a_clip_with_no_frames(self, video, tmp_path, monkeypatch):
        # A container that opens fine but yields zero frames (e.g. a truncated encode).
        import av

        class EmptyContainer:
            class _Streams:
                video = [SimpleNamespace(average_rate=24, guessed_rate=24)]

            streams = _Streams()

            def decode(self, *_a):
                return iter(())

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        monkeypatch.setattr(av, "open", lambda *_a, **_k: EmptyContainer())
        with pytest.raises(ValueError, match="no frames"):
            save_last_clip(video, str(tmp_path / "out.mp4"))

    def test_a_nonpositive_duration_is_refused(self, video, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            save_last_clip(video, str(tmp_path / "out.mp4"), seconds=0)

    def test_writes_no_temp_file_leftovers(self, video, tmp_path):
        out = str(tmp_path / "tail.mp4")
        save_last_clip(video, out, seconds=1.0)
        assert os.listdir(tmp_path) == ["clip.mp4", "tail.mp4"]


class TestLastClipForJob:
    def test_extracts_and_returns_a_path(self, video, uploads):
        job = _job(output_path=video)
        path = last_clip_for_job(job, uploads)
        assert os.path.isfile(path)
        assert path.endswith("last_clip_job-1.mp4")
        assert "last_clips" in path

    def test_reuses_an_existing_extraction(self, video, uploads, monkeypatch):
        job = _job(output_path=video)
        first = last_clip_for_job(job, uploads)
        mtime = os.path.getmtime(first)
        calls = []
        monkeypatch.setattr(frames, "save_last_clip", lambda *a, **k: calls.append(a))
        second = last_clip_for_job(job, uploads)
        assert second == first
        assert os.path.getmtime(second) == mtime
        assert calls == []

    def test_re_extracts_when_the_video_changed_since(self, video, uploads, monkeypatch):
        job = _job(output_path=video)
        last_clip_for_job(job, uploads)
        _write_video(video, n=FRAMES + 24)  # a longer re-render
        # Force the source to read as newer, regardless of filesystem mtime resolution.
        monkeypatch.setattr(os.path, "getmtime", lambda p: (time.time() + 1) if p == video else os.stat(p).st_mtime)
        calls = []
        real_save = frames.save_last_clip
        monkeypatch.setattr(frames, "save_last_clip", lambda *a, **k: (calls.append(a), real_save(*a, **k))[1])
        last_clip_for_job(job, uploads)
        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("job", "match"),
        [
            (_job(status=Status.PENDING), "No output"),
            (_job(status=Status.COMPLETED, output_path=None), "No output"),
            (_job(status=Status.COMPLETED, output_path="/nope.mp4"), "not found"),
        ],
    )
    def test_refusals(self, job, match, uploads):
        with pytest.raises(FrameError, match=match):
            last_clip_for_job(job, uploads)

    def test_refuses_a_non_video_output(self, uploads, tmp_path):
        img = tmp_path / "out.png"
        img.write_bytes(b"x")
        job = _job(output_path=str(img))
        with pytest.raises(FrameError, match="not a video"):
            last_clip_for_job(job, uploads)

    def test_refuses_without_an_upload_dir(self, video):
        job = _job(output_path=video)
        with pytest.raises(FrameError, match="Upload directory"):
            last_clip_for_job(job, "")

    def test_a_decode_failure_is_reported_as_a_500(self, uploads, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        job = _job(output_path=str(bad))
        with pytest.raises(FrameError, match="Could not read the end of the clip") as e:
            last_clip_for_job(job, uploads)
        assert e.value.status_code == 500
