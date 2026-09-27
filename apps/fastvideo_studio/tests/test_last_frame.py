# SPDX-License-Identifier: Apache-2.0
"""A finished clip's final frame is saved as an image the next clip can use as a reference."""
from __future__ import annotations

import os
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fastvideo_studio import frames
from fastvideo_studio.frames import FrameError, last_frame_for_job

WIDTH, HEIGHT, FRAMES = 128, 96, 12


class Status(Enum):  # mirrors JobStatus without importing job_runner (which needs fastvideo)
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


def _write_video(path: str, *, last_rgb=(0, 0, 255), other_rgb=(255, 0, 0)) -> None:
    """A clip whose frames are all `other_rgb` except the last, which is `last_rgb`."""
    import av

    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("h264", rate=24)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(FRAMES):
            rgb = last_rgb if i == FRAMES - 1 else other_rgb
            array = np.full((HEIGHT, WIDTH, 3), rgb, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


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


def test_saves_the_final_frame_not_the_first(video, uploads, tmp_path):
    path = last_frame_for_job(_job(output_path=video), uploads)

    assert path == str(tmp_path / "uploads" / "last_frames" / "last_frame_job-1.png")
    assert os.path.isabs(path)
    image = Image.open(path).convert("RGB")
    assert image.size == (WIDTH, HEIGHT)
    r, _g, b = image.getpixel((WIDTH // 2, HEIGHT // 2))
    # the last frame is blue and every other frame is red -- a wrong frame would be red
    assert b > 150 and r < 100, (r, b)


def test_accepts_a_plain_string_status(video, uploads):
    assert last_frame_for_job(_job(status="completed", output_path=video), uploads).endswith(".png")


def test_extraction_is_reused_until_the_video_changes(video, uploads, monkeypatch):
    calls = []
    real = frames.save_last_frame
    monkeypatch.setattr(frames, "save_last_frame", lambda v, o: (calls.append(o), real(v, o))[1])

    first = last_frame_for_job(_job(output_path=video), uploads)
    second = last_frame_for_job(_job(output_path=video), uploads)
    assert first == second
    assert len(calls) == 1

    # the clip is regenerated -> the saved frame is stale and gets rebuilt
    later = os.path.getmtime(first) + 10
    os.utime(video, (later, later))
    last_frame_for_job(_job(output_path=video), uploads)
    assert len(calls) == 2


def test_each_job_gets_its_own_file(video, uploads):
    a = last_frame_for_job(_job("a", output_path=video), uploads)
    b = last_frame_for_job(_job("b", output_path=video), uploads)
    assert a != b and os.path.isfile(a) and os.path.isfile(b)


def test_leaves_no_temp_file_behind(video, uploads):
    last_frame_for_job(_job(output_path=video), uploads)
    assert os.listdir(os.path.join(uploads, "last_frames")) == ["last_frame_job-1.png"]


@pytest.mark.parametrize("status", [Status.PENDING, Status.RUNNING, Status.FAILED, Status.STOPPED])
def test_a_job_that_has_not_finished_is_404(video, uploads, status):
    with pytest.raises(FrameError) as e:
        last_frame_for_job(_job(status=status, output_path=video), uploads)
    assert e.value.status_code == 404
    assert "No output" in e.value.detail


def test_a_completed_job_with_no_output_path_is_404(uploads):
    with pytest.raises(FrameError) as e:
        last_frame_for_job(_job(output_path=None), uploads)
    assert e.value.status_code == 404


def test_missing_output_file_is_404(uploads, tmp_path):
    with pytest.raises(FrameError) as e:
        last_frame_for_job(_job(output_path=str(tmp_path / "gone.mp4")), uploads)
    assert (e.value.status_code, "not found on disk" in e.value.detail) == (404, True)


def test_an_image_output_is_rejected(uploads, tmp_path):
    png = tmp_path / "out.png"
    Image.new("RGB", (8, 8)).save(png)
    with pytest.raises(FrameError) as e:
        last_frame_for_job(_job(output_path=str(png)), uploads)
    assert e.value.status_code == 400


def test_an_unreadable_video_is_a_500_with_a_reason(uploads, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(FrameError) as e:
        last_frame_for_job(_job(output_path=str(bad)), uploads)
    assert e.value.status_code == 500
    assert "Could not read the last frame" in e.value.detail


def test_a_failed_extraction_does_not_leave_a_partial_file(uploads, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(FrameError):
        last_frame_for_job(_job(output_path=str(bad)), uploads)
    assert os.listdir(os.path.join(uploads, "last_frames")) == []


def test_needs_an_upload_directory(video):
    with pytest.raises(FrameError) as e:
        last_frame_for_job(_job(output_path=video), "")
    assert e.value.status_code == 503
