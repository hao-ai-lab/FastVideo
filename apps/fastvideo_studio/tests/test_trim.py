# SPDX-License-Identifier: Apache-2.0
"""Cutting a finished clip to a chosen in/out range and adjusting its color, non-destructively."""
from __future__ import annotations

import os
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from fastvideo_studio.frames import FrameError
from fastvideo_studio.trim import (
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    EDITED_FILENAME,
    ORIGINAL_FILENAME,
    SATURATION_RANGE,
    clip_duration_seconds,
    restore_original,
    trim_video,
)

WIDTH, HEIGHT, FPS, FRAMES = 128, 96, 24, 48  # 2 seconds, one frame per level 0..47


class Status(Enum):  # mirrors JobStatus without importing job_runner (which needs fastvideo)
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


def _write_video(path: str, n: int = FRAMES, fps: int = FPS) -> None:
    """A clip whose frames are numbered by brightness (frame i is roughly i*5 grey), so a range is identifiable."""
    import av

    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(n):
            level = min(i * 5, 255)
            frame = av.VideoFrame.from_ndarray(np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


AUDIO_SAMPLE_RATE = 32000
AUDIO_SAMPLES_PER_FRAME = 1024


def _write_video_with_audio(path: str, n: int = FRAMES, fps: int = FPS) -> None:
    """Same as ``_write_video``, plus a silent stereo AAC track, to test that edits keep audio."""
    import av
    from fractions import Fraction

    container = av.open(path, mode="w")
    try:
        vstream = container.add_stream("h264", rate=fps)
        vstream.width, vstream.height, vstream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        astream = container.add_stream("aac", rate=AUDIO_SAMPLE_RATE)
        astream.layout = "stereo"

        for i in range(n):
            level = min(i * 5, 255)
            frame = av.VideoFrame.from_ndarray(np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8), format="rgb24")
            for packet in vstream.encode(frame):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)

        audio_time_base = Fraction(1, AUDIO_SAMPLE_RATE)
        n_audio_frames = int(n / fps * AUDIO_SAMPLE_RATE / AUDIO_SAMPLES_PER_FRAME) + 2
        pts = 0
        for _ in range(n_audio_frames):
            arr = np.zeros((2, AUDIO_SAMPLES_PER_FRAME), dtype=np.float32)
            aframe = av.AudioFrame.from_ndarray(arr, format="fltp", layout="stereo")
            aframe.sample_rate = AUDIO_SAMPLE_RATE
            aframe.pts = pts
            aframe.time_base = audio_time_base
            pts += AUDIO_SAMPLES_PER_FRAME
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode(None):
            container.mux(packet)
    finally:
        container.close()


def _write_colored_video(path: str, rgb: tuple[int, int, int], n: int = 12, fps: int = FPS) -> None:
    """A clip of one solid, non-gray color, so saturation changes are visible (unlike on a gray clip)."""
    import av

    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for _ in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((HEIGHT, WIDTH, 3), rgb, dtype=np.uint8), format="rgb24")
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


def _read_mean_rgb(path: str, index: int = 0) -> np.ndarray:
    import av

    with av.open(path) as container:
        frames = list(container.decode(container.streams.video[0]))
    return frames[index].to_ndarray(format="rgb24").astype(float).mean(axis=(0, 1))


def _job(job_id="job-1", status=Status.COMPLETED, output_path=None):
    return SimpleNamespace(id=job_id, status=status, output_path=output_path)


def _top_level_atom_offset(path: str, atom_type: bytes) -> int:
    """Byte offset of a top-level MP4 box (e.g. b"moov", b"mdat"), or -1 if absent."""
    import struct

    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        pos = 0
        while pos < size:
            f.seek(pos)
            header = f.read(8)
            if len(header) < 8:
                break
            atom_size, found_type = struct.unpack(">I4s", header)
            if found_type == atom_type:
                return pos
            if atom_size in (0, 1):
                break
            pos += atom_size
    return -1


def _moov_offset(path: str) -> int:
    return _top_level_atom_offset(path, b"moov")


def _mdat_offset(path: str) -> int:
    return _top_level_atom_offset(path, b"mdat")


def seg(brightness=0.0, contrast=1.0, saturation=1.0, end_seconds=None):
    """One razor-cut section dict, as the API/trim_video expects."""
    return {"end_seconds": end_seconds, "brightness": brightness, "contrast": contrast, "saturation": saturation}


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "job-1"
    d.mkdir()
    return d


@pytest.fixture
def video(job_dir):
    path = str(job_dir / "wolf-lunch-clip-06.mp4")
    _write_video(path)
    return path


@pytest.fixture
def colored_video(job_dir):
    path = str(job_dir / "wolf-lunch-clip-07.mp4")
    _write_colored_video(path, (200, 100, 50))  # a warm orange, clearly non-gray
    return path


@pytest.fixture
def video_with_audio(job_dir):
    path = str(job_dir / "wolf-lunch-clip-08.mp4")
    _write_video_with_audio(path)
    return path


class TestTrimVideo:
    def test_keeps_only_the_requested_range(self, video, job_dir):
        path, kept = trim_video(_job(output_path=video), start_seconds=0.5, end_seconds=1.5)
        assert path == str(job_dir / EDITED_FILENAME)
        levels = _read_levels(path)
        assert kept == len(levels) == FPS  # 1 second kept
        assert levels[0] == _read_levels(video)[12]  # starts at 0.5s = frame 12
        assert levels[-1] == _read_levels(video)[35]  # ends just before 1.5s = frame 36

    def test_an_open_end_keeps_to_the_end_of_the_source(self, video):
        path, kept = trim_video(_job(output_path=video), start_seconds=1.0, end_seconds=None)
        assert kept == FRAMES - FPS
        assert _read_levels(path)[-1] == _read_levels(video)[-1]

    def test_start_at_zero_keeps_from_the_beginning(self, video):
        path, _ = trim_video(_job(output_path=video), start_seconds=0.0, end_seconds=1.0)
        assert _read_levels(path)[0] == _read_levels(video)[0]

    def test_the_original_is_preserved_byte_for_byte(self, video, job_dir):
        before = open(video, "rb").read()
        trim_video(_job(output_path=video), start_seconds=0.5, end_seconds=1.5)
        original = job_dir / ORIGINAL_FILENAME
        assert original.is_file()
        assert original.read_bytes() == before

    def test_trimming_again_re_cuts_from_the_original_not_the_edited_file(self, video, job_dir):
        # If it re-cut the already-edited 0.5-1.5s file, asking for 0-2s again
        # could not recover the parts outside that first window.
        trim_video(_job(output_path=video), start_seconds=0.5, end_seconds=1.5)
        path, kept = trim_video(_job(output_path=video), start_seconds=0.0, end_seconds=2.0)
        assert kept == FRAMES
        assert _read_levels(path) == _read_levels(video)

    def test_narrowing_the_range_further_still_re_cuts_from_the_original(self, video):
        trim_video(_job(output_path=video), start_seconds=0.0, end_seconds=2.0)
        path, kept = trim_video(_job(output_path=video), start_seconds=1.0, end_seconds=1.5)
        assert kept == FPS // 2
        assert _read_levels(path)[0] == _read_levels(video)[24]

    def test_muxed_frame_rate_stays_correct(self, video):
        path, _ = trim_video(_job(output_path=video), start_seconds=0.0, end_seconds=1.0)
        import av

        with av.open(path) as c:
            assert FPS - 1 <= float(c.streams.video[0].average_rate) <= FPS + 1

    def test_output_is_faststart_moov_before_mdat(self, video):
        # Otherwise a browser has to reach nearly the end of the file just to read
        # its duration/seek table -- painfully slow over a network filesystem.
        path, _ = trim_video(_job(output_path=video), start_seconds=0.0, end_seconds=1.0)
        assert _moov_offset(path) < _mdat_offset(path)

    @pytest.mark.parametrize(
        ("start", "end", "match"),
        [
            (-1.0, None, "negative"),
            (1.0, 1.0, "after the start"),
            (1.0, 0.5, "after the start"),
            (100.0, None, "no frames"),
        ],
    )
    def test_refuses_an_invalid_range(self, video, start, end, match):
        with pytest.raises(FrameError, match=match) as e:
            trim_video(_job(output_path=video), start_seconds=start, end_seconds=end)
        assert e.value.status_code == 400

    @pytest.mark.parametrize(
        ("job", "match"),
        [
            (_job(status=Status.PENDING), "No output"),
            (_job(status=Status.COMPLETED, output_path=None), "No output"),
            (_job(status=Status.COMPLETED, output_path="/nope.mp4"), "not found"),
        ],
    )
    def test_refusals(self, job, match):
        with pytest.raises(FrameError, match=match):
            trim_video(job, 0.0, 1.0)

    def test_refuses_a_non_video_output(self, tmp_path):
        img = tmp_path / "out.png"
        img.write_bytes(b"x")
        with pytest.raises(FrameError, match="not a video"):
            trim_video(_job(output_path=str(img)), 0.0, 1.0)


class TestGrading:
    def test_neutral_values_do_not_alter_the_pixels(self, colored_video):
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg()])
        got = _read_mean_rgb(path)
        want = _read_mean_rgb(colored_video)
        assert np.allclose(got, want, atol=2)  # h264 is lossy even unchanged

    def test_neutral_values_skip_the_per_frame_numpy_round_trip(self, colored_video, monkeypatch):
        import fastvideo_studio.trim as trim_module

        calls = []
        monkeypatch.setattr(trim_module, "_apply_grade", lambda *a: calls.append(a) or a[0])
        trim_video(_job(output_path=colored_video), 0.0, None)  # no segments -> the default single neutral one
        assert calls == []

    def test_positive_brightness_lightens_every_channel(self, colored_video):
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(brightness=50.0)])
        got = _read_mean_rgb(path)
        want = _read_mean_rgb(colored_video)
        assert (got > want + 30).all()

    def test_negative_brightness_darkens(self, colored_video):
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(brightness=-50.0)])
        got = _read_mean_rgb(path)
        want = _read_mean_rgb(colored_video)
        assert (got < want - 30).all()

    def test_zero_saturation_makes_every_channel_equal(self, colored_video):
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(saturation=0.0)])
        r, g, b = _read_mean_rgb(path)
        assert abs(r - g) < 2 and abs(g - b) < 2  # gray: channels converge on the frame's luma

    def test_saturation_above_one_widens_the_channel_spread(self, colored_video):
        base_r, base_g, base_b = _read_mean_rgb(colored_video)
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(saturation=2.0)])
        r, g, b = _read_mean_rgb(path)
        assert (r - b) > (base_r - base_b)  # the warm/cool spread grows, not shrinks

    def test_high_contrast_pushes_values_away_from_mid_gray(self, colored_video):
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(contrast=2.0)])
        got = _read_mean_rgb(path)
        want = _read_mean_rgb(colored_video)
        # 128 = mid-gray. This clip's R (200) sits above it and should be pushed further up;
        # its G and B (100, 50) sit below and should be pushed further down -- the spread
        # around 128 widens either way.
        assert got[0] > want[0]  # R: 200 -> further above 128
        assert got[1] < want[1]  # G: 100 -> further below 128
        assert got[2] < want[2]  # B: 50 -> further below 128

    def test_grading_and_trimming_the_same_call_compose(self, video, colored_video):
        # A range trim and a color grade requested together both take effect --
        # neither is silently dropped by the other.
        path, kept = trim_video(
            _job(output_path=colored_video), start_seconds=0.2, end_seconds=0.3, segments=[seg(brightness=40.0)])
        assert kept == pytest.approx(FPS * 0.1, abs=1)
        base = _read_mean_rgb(colored_video)
        assert (_read_mean_rgb(path) > base + 20).all()

    @pytest.mark.parametrize(
        ("kwargs", "bounds"),
        [
            ({"brightness": BRIGHTNESS_RANGE[1] + 1}, BRIGHTNESS_RANGE),
            ({"brightness": BRIGHTNESS_RANGE[0] - 1}, BRIGHTNESS_RANGE),
            ({"contrast": CONTRAST_RANGE[1] + 1}, CONTRAST_RANGE),
            ({"contrast": CONTRAST_RANGE[0] - 1}, CONTRAST_RANGE),
            ({"saturation": SATURATION_RANGE[1] + 1}, SATURATION_RANGE),
            ({"saturation": SATURATION_RANGE[0] - 1}, SATURATION_RANGE),
        ],
    )
    def test_refuses_an_out_of_range_value(self, colored_video, kwargs, bounds):
        with pytest.raises(FrameError, match="between") as e:
            trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(**kwargs)])
        assert e.value.status_code == 400

    @pytest.mark.parametrize("kwargs", [{"brightness": BRIGHTNESS_RANGE[1]}, {"contrast": CONTRAST_RANGE[0]},
                                        {"saturation": SATURATION_RANGE[1]}])
    def test_the_edges_of_each_range_are_allowed(self, colored_video, kwargs):
        trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(**kwargs)])  # does not raise


class TestSections:
    """Razor-cut sections: an ordered, contiguous list of grades within the kept range."""

    def test_two_sections_grade_independently(self, colored_video):
        path, kept = trim_video(
            _job(output_path=colored_video), 0.0, None,
            segments=[seg(brightness=80.0, end_seconds=0.25), seg()],
        )
        base = _read_mean_rgb(colored_video)
        assert (_read_mean_rgb(path, index=0) > base + 40).all()  # first section, graded
        assert np.allclose(_read_mean_rgb(path, index=kept - 1), base, atol=2)  # second section, untouched

    def test_three_sections_each_get_their_own_grade(self, colored_video):
        path, kept = trim_video(
            _job(output_path=colored_video), 0.0, None,
            segments=[
                seg(saturation=0.0, end_seconds=4 / FPS),
                seg(brightness=80.0, end_seconds=8 / FPS),
                seg(),
            ],
        )
        base = _read_mean_rgb(colored_video)
        r0, g0, b0 = _read_mean_rgb(path, index=0)
        assert abs(r0 - g0) < 2 and abs(g0 - b0) < 2  # first section desaturated
        assert (_read_mean_rgb(path, index=6) > base + 40).all()  # second section brightened
        assert np.allclose(_read_mean_rgb(path, index=kept - 1), base, atol=2)  # third section untouched

    def test_sections_compose_with_the_range_trim(self, colored_video):
        # The cut point (0.1) is relative to the KEPT range, not the original file's timeline.
        path, kept = trim_video(
            _job(output_path=colored_video), start_seconds=0.1, end_seconds=0.4,
            segments=[seg(brightness=80.0, end_seconds=0.1), seg()],
        )
        base = _read_mean_rgb(colored_video)
        assert (_read_mean_rgb(path, index=0) > base + 40).all()
        assert np.allclose(_read_mean_rgb(path, index=kept - 1), base, atol=2)

    def test_an_empty_section_list_defaults_to_one_neutral_section(self, colored_video):
        path, _ = trim_video(_job(output_path=colored_video), 0.0, None, segments=[])
        assert np.allclose(_read_mean_rgb(path), _read_mean_rgb(colored_video), atol=2)

    def test_a_single_explicit_end_that_does_not_cover_the_clip_is_refused(self, colored_video):
        with pytest.raises(FrameError, match="cover the whole clip"):
            trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(end_seconds=0.1)])

    def test_only_the_last_section_can_be_open_ended(self, colored_video):
        with pytest.raises(FrameError, match="open-ended"):
            trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(), seg(end_seconds=0.25)])

    def test_sections_must_be_in_increasing_order(self, colored_video):
        with pytest.raises(FrameError, match="increasing order"):
            trim_video(
                _job(output_path=colored_video), 0.0, None,
                segments=[seg(end_seconds=0.3), seg(end_seconds=0.2), seg()],
            )

    def test_a_section_too_short_to_keep_a_frame_is_refused(self, colored_video):
        # 0.001s at 24fps rounds to 0 frames -- passes the ordering check (> 0) but keeps nothing.
        with pytest.raises(FrameError, match="too short"):
            trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(end_seconds=0.001), seg()])

    def test_a_section_with_an_out_of_range_grade_is_refused(self, colored_video):
        with pytest.raises(FrameError, match="between"):
            trim_video(
                _job(output_path=colored_video), 0.0, None,
                segments=[seg(end_seconds=0.25), seg(brightness=500.0)],
            )

    def test_only_frames_in_a_graded_section_go_through_the_numpy_round_trip(self, colored_video, monkeypatch):
        import fastvideo_studio.trim as trim_module

        calls = []
        original = trim_module._apply_grade
        monkeypatch.setattr(trim_module, "_apply_grade", lambda *a: calls.append(a) or original(*a))
        trim_video(
            _job(output_path=colored_video), 0.0, None,
            segments=[seg(end_seconds=0.25), seg(brightness=50.0)],  # first neutral, second graded
        )
        assert 0 < len(calls) < 12  # some frames graded, not all -- the neutral section was skipped


class TestAudio:
    def test_a_source_with_no_audio_stays_video_only(self, video):
        path, _ = trim_video(_job(output_path=video), start_seconds=0.5, end_seconds=1.5)
        import av

        with av.open(path) as c:
            assert [s.type for s in c.streams] == ["video"]

    def test_a_range_trim_keeps_the_audio_track(self, video_with_audio):
        path, _ = trim_video(_job(output_path=video_with_audio), start_seconds=0.5, end_seconds=1.5)
        import av

        with av.open(path) as c:
            assert "audio" in [s.type for s in c.streams]
            a = c.streams.audio[0]
            assert a.codec_context.sample_rate == AUDIO_SAMPLE_RATE
            assert sum(1 for _ in c.decode(a)) > 0

    def test_grading_alongside_a_trim_also_keeps_the_audio_track(self, video_with_audio):
        path, _ = trim_video(
            _job(output_path=video_with_audio), start_seconds=0.0, end_seconds=1.0, segments=[seg(brightness=40.0)])
        import av

        with av.open(path) as c:
            assert "audio" in [s.type for s in c.streams]

    def test_restore_original_keeps_the_audio_track(self, video_with_audio, job_dir):
        trim_video(_job(output_path=video_with_audio), start_seconds=0.5, end_seconds=1.5)
        path, _ = restore_original(_job(output_path=str(job_dir / EDITED_FILENAME)))
        import av

        with av.open(path) as c:
            assert "audio" in [s.type for s in c.streams]


class TestRestoreOriginal:
    def test_restores_the_original_video_and_its_frame_count(self, video, job_dir):
        trim_video(_job(output_path=video), start_seconds=0.5, end_seconds=1.5)
        path, n = restore_original(_job(output_path=str(job_dir / EDITED_FILENAME)))
        assert path == str(job_dir / ORIGINAL_FILENAME)
        assert n == FRAMES
        assert _read_levels(path) == _read_levels(video)

    def test_restores_past_a_grade_too(self, colored_video, job_dir):
        trim_video(_job(output_path=colored_video), 0.0, None, segments=[seg(brightness=50.0, saturation=0.0)])
        path, _ = restore_original(_job(output_path=str(job_dir / EDITED_FILENAME)))
        assert np.allclose(_read_mean_rgb(path), _read_mean_rgb(colored_video), atol=2)

    def test_refuses_a_job_that_was_never_edited(self, video):
        with pytest.raises(FrameError, match="not been edited"):
            restore_original(_job(output_path=video))

    def test_refuses_a_job_with_no_output(self):
        with pytest.raises(FrameError, match="No output"):
            restore_original(_job(output_path=None))


def test_clip_duration_seconds(video):
    assert clip_duration_seconds(video) == pytest.approx(FRAMES / FPS, abs=0.05)
