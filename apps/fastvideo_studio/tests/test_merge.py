# SPDX-License-Identifier: Apache-2.0
"""A scene's clips are joined, in order, into one video without re-encoding."""
from __future__ import annotations

import os
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from fastvideo_studio import merge
from fastvideo_studio.merge import MergeError, find_ffmpeg, merge_clips, merge_jobs, merged_path, probe_clip

pytestmark = pytest.mark.skipif(find_ffmpeg() is None, reason="needs an ffmpeg binary")

FPS = 24
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}


class Status(Enum):  # mirrors JobStatus without importing job_runner (which needs fastvideo)
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


def write_clip(path, *, frames=24, color="red", size=(128, 96), fps=FPS, audio=True, audio_extra=0.042):
    """A clip of one flat colour. Like real H3 output, its audio runs a little past the video."""
    import av

    container = av.open(path, mode="w")
    try:
        video = container.add_stream("h264", rate=fps)
        video.width, video.height, video.pix_fmt = size[0], size[1], "yuv420p"
        sound = None
        if audio:
            sound = container.add_stream("aac", rate=32000)
            sound.layout = "stereo"
        for _ in range(frames):
            array = np.full((size[1], size[0], 3), COLORS[color], dtype=np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if sound is not None:
            total = int((frames / fps + audio_extra) * 32000)
            tone = (np.sin(np.arange(total) * 2 * np.pi * 440 / 32000) * 0.2).astype(np.float32)
            pts = 0
            for start in range(0, total, 1024):
                chunk = tone[start:start + 1024]
                chunk = np.pad(chunk, (0, 1024 - len(chunk)))
                frame = av.AudioFrame.from_ndarray(np.stack([chunk, chunk]), format="fltp", layout="stereo")
                frame.sample_rate, frame.pts = 32000, pts
                pts += 1024
                for packet in sound.encode(frame):
                    container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)
    finally:
        container.close()


def read_back(path):
    """(frame colours in order, audio seconds, video seconds) of a finished video."""
    import av

    with av.open(path) as c:
        v = c.streams.video[0]
        colours = []
        for frame in c.decode(v):
            r, g, b = frame.to_ndarray(format="rgb24")[48, 64]
            colours.append("red" if r > 150 else "green" if g > 150 else "blue")
        audio = next((s for s in c.streams if s.type == "audio"), None)
        a_secs = float(audio.duration * audio.time_base) if audio and audio.duration else None
        return colours, a_secs, float(v.duration * v.time_base)


def job(path, name="", status=Status.COMPLETED, job_id="j"):
    return SimpleNamespace(id=job_id, name=name, status=status, output_path=path)


@pytest.fixture
def clips(tmp_path):
    paths = []
    for i, (color, frames) in enumerate([("red", 24), ("green", 36), ("blue", 12)]):
        p = str(tmp_path / f"clip{i}.mp4")
        write_clip(p, frames=frames, color=color)
        paths.append(p)
    return paths


class TestMergeClips:
    def test_joins_clips_in_the_order_given(self, clips, tmp_path):
        out = str(tmp_path / "out.mp4")
        merge_clips(clips, out)
        colours, _, _ = read_back(out)
        assert colours == ["red"] * 24 + ["green"] * 36 + ["blue"] * 12

    def test_the_order_is_the_callers_not_the_filenames(self, clips, tmp_path):
        out = str(tmp_path / "out.mp4")
        merge_clips([clips[2], clips[0], clips[1]], out)
        colours, _, _ = read_back(out)
        assert colours == ["blue"] * 12 + ["red"] * 24 + ["green"] * 36

    def test_keeps_every_frame_and_the_full_length(self, clips, tmp_path):
        out = str(tmp_path / "out.mp4")
        seconds = merge_clips(clips, out)
        colours, _, video_secs = read_back(out)
        assert len(colours) == 72
        assert seconds == pytest.approx(72 / FPS, abs=0.02)
        assert video_secs == pytest.approx(72 / FPS, abs=0.05)

    def test_audio_stays_in_step_with_the_picture(self, clips, tmp_path):
        # Each clip's audio runs ~40 ms past its video; joined naively that adds up to a gap or drift.
        out = str(tmp_path / "out.mp4")
        merge_clips(clips, out)
        _, audio_secs, video_secs = read_back(out)
        assert audio_secs is not None
        assert abs(audio_secs - video_secs) < 0.08  # not 3 x 0.042 and growing

    def test_a_single_clip_is_fine(self, clips, tmp_path):
        out = str(tmp_path / "out.mp4")
        merge_clips(clips[:1], out)
        assert len(read_back(out)[0]) == 24

    def test_clips_without_audio_join_too(self, tmp_path):
        paths = []
        for i, color in enumerate(("red", "green")):
            p = str(tmp_path / f"s{i}.mp4")
            write_clip(p, frames=12, color=color, audio=False)
            paths.append(p)
        out = str(tmp_path / "out.mp4")
        merge_clips(paths, out)
        colours, audio_secs, _ = read_back(out)
        assert colours == ["red"] * 12 + ["green"] * 12 and audio_secs is None

    def test_creates_the_output_folder_and_leaves_no_temp_files(self, clips, tmp_path):
        out = str(tmp_path / "new" / "dir" / "out.mp4")
        merge_clips(clips, out)
        assert os.listdir(os.path.dirname(out)) == ["out.mp4"]

    def test_paths_with_quotes_and_spaces(self, tmp_path):
        paths = []
        for i, color in enumerate(("red", "blue")):
            p = str(tmp_path / f"it's clip {i}.mp4")
            write_clip(p, frames=6, color=color)
            paths.append(p)
        out = str(tmp_path / "out.mp4")
        merge_clips(paths, out)
        assert read_back(out)[0] == ["red"] * 6 + ["blue"] * 6


class TestRefusals:
    def test_clips_of_different_sizes(self, clips, tmp_path):
        odd = str(tmp_path / "odd.mp4")
        write_clip(odd, size=(160, 96))
        with pytest.raises(MergeError, match=r"'clip 3' \(160x96.*doesn't match 'clip 1' \(128x96") as e:
            merge_clips([clips[0], clips[1], odd], str(tmp_path / "o.mp4"), labels=["'clip 1'", "'clip 2'", "'clip 3'"])
        assert e.value.status_code == 409
        assert not (tmp_path / "o.mp4").exists()

    def test_clips_of_different_frame_rates(self, clips, tmp_path):
        odd = str(tmp_path / "odd.mp4")
        write_clip(odd, fps=30)
        with pytest.raises(MergeError, match="doesn't match"):
            merge_clips([clips[0], odd], str(tmp_path / "o.mp4"))

    def test_one_clip_with_audio_and_one_without(self, clips, tmp_path):
        silent = str(tmp_path / "silent.mp4")
        write_clip(silent, audio=False)
        with pytest.raises(MergeError, match="no audio"):
            merge_clips([clips[0], silent], str(tmp_path / "o.mp4"))

    def test_nothing_to_merge(self, tmp_path):
        with pytest.raises(MergeError) as e:
            merge_clips([], str(tmp_path / "o.mp4"))
        assert e.value.status_code == 400

    def test_a_file_that_is_not_a_video(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(MergeError, match="Could not read bad.mp4"):
            merge_clips([str(bad)], str(tmp_path / "o.mp4"))

    def test_no_ffmpeg(self, clips, tmp_path, monkeypatch):
        monkeypatch.setattr(merge, "find_ffmpeg", lambda: None)
        with pytest.raises(MergeError) as e:
            merge_clips(clips, str(tmp_path / "o.mp4"))
        assert e.value.status_code == 503 and "ffmpeg" in e.value.detail


class TestMergeJobs:
    def test_merges_the_jobs_videos_into_a_merged_folder(self, clips, tmp_path):
        jobs = [job(p, name=f"clip {i + 1}", job_id=f"j{i}") for i, p in enumerate(clips)]
        result = merge_jobs(jobs, str(tmp_path / "out"), name="Wolf Lunch")
        assert result.clips == 3 and result.seconds == pytest.approx(3.0, abs=0.02)
        assert result.filename.startswith("Wolf-Lunch-") and result.filename.endswith(".mp4")
        assert result.path == str(tmp_path / "out" / "merged" / result.filename)
        assert len(read_back(result.path)[0]) == 72

    def test_a_name_cannot_escape_the_folder(self, clips, tmp_path):
        result = merge_jobs([job(clips[0])], str(tmp_path / "out"), name="../../etc/passwd")
        assert os.path.dirname(result.path) == str(tmp_path / "out" / "merged")
        assert "/" not in result.filename

    def test_an_empty_name_still_gets_a_filename(self, clips, tmp_path):
        assert merge_jobs([job(clips[0])], str(tmp_path / "out"), name="  ").filename.startswith("scene-")

    def test_refuses_a_clip_that_has_not_finished_and_names_it(self, clips, tmp_path):
        jobs = [job(clips[0], name="clip 1"), job(None, name="clip 2", status=Status.PENDING)]
        with pytest.raises(MergeError, match=r"'clip 2' hasn't finished \(pending\)") as e:
            merge_jobs(jobs, str(tmp_path / "out"))
        assert e.value.status_code == 409
        assert not (tmp_path / "out").exists()  # nothing partial

    def test_refuses_a_video_missing_from_disk(self, clips, tmp_path):
        jobs = [job(clips[0], name="a"), job(str(tmp_path / "gone.mp4"), name="b")]
        with pytest.raises(MergeError, match="'b'.*missing from disk"):
            merge_jobs(jobs, str(tmp_path / "out"))

    def test_refuses_an_output_that_is_not_a_video(self, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        with pytest.raises(MergeError, match="isn't a video"):
            merge_jobs([job(str(img), name="a")], str(tmp_path / "out"))

    def test_refuses_no_jobs(self, tmp_path):
        with pytest.raises(MergeError) as e:
            merge_jobs([], str(tmp_path))
        assert e.value.status_code == 400


class TestMergedPath:
    def test_finds_a_merged_video(self, clips, tmp_path):
        result = merge_jobs([job(clips[0])], str(tmp_path))
        assert merged_path(str(tmp_path), result.filename) == result.path

    @pytest.mark.parametrize("name", ["../secret.mp4", "a/b.mp4", "..", "x.txt", "", "nope.mp4", "a b.mp4"])
    def test_refuses_anything_else(self, tmp_path, name):
        (tmp_path / "merged").mkdir()
        (tmp_path / "secret.mp4").write_bytes(b"x")
        with pytest.raises(MergeError) as e:
            merged_path(str(tmp_path), name)
        assert e.value.status_code == 404


def test_probe_reads_real_clip_settings(clips):
    info = probe_clip(clips[1])
    assert (info.width, info.height, info.video_codec, info.pix_fmt) == (128, 96, "h264", "yuv420p")
    assert info.fps == pytest.approx(24) and info.video_seconds == pytest.approx(1.5, abs=0.03)
    assert info.audio == ("aac", 32000, 2)
