"""Exercise real FFmpeg/PyAV packets on CPU; no model or API keys are needed."""

import json
import shutil
import subprocess
from fractions import Fraction

import av
import pytest

from infinite_livestream.metadata import clip_view, encode_id3
from infinite_livestream.muxer import MetadataMuxer


def descriptor(name):
    return clip_view({"clip_id": name, "prompt": name})


def read_record(raw):
    # Parse the TXXX UTF-8 value independently of the production serializer.
    assert raw[:3] == b"ID3" and raw[10:14] == b"TXXX" and raw[20] == 3
    description, value = raw[21:].split(b"\x00", 1)
    assert description == b"infinite-livestream"
    return json.loads(value)


@pytest.fixture()
def encoded_source(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg with libx264/AAC is required for the CPU media integration test")
    source = tmp_path / "source.ts"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=red:s=160x96:r=24:d=1.125[a];color=green:s=160x96:r=24:d=1.5[b];"
        "color=blue:s=160x96:r=24:d=3.375[c];[a][b][c]concat=n=3:v=1:a=0",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "6",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-g", "48", "-sc_threshold", "0", "-bf", "0", "-c:a", "aac",
        "-f", "mpegts", str(source),
    ], check=True, capture_output=True, timeout=20)
    return source


def run_mux(source, playlist, prefix=""):
    with source.open("rb") as stream:
        muxer = MetadataMuxer(stream, playlist, 24, 120)
        for frame in range(144):
            name = "A" if frame < 27 else "B" if frame < 63 else "C"
            muxer.frames.put_nowait(encode_id3(descriptor(prefix + name)))
        muxer._mux()
        assert muxer.frames.empty()
        return muxer


def segment_paths(playlist):
    return [playlist.parent / line for line in playlist.read_text().splitlines()
            if line and not line.startswith("#")]


def media_packets(paths):
    result = {"video": [], "audio": []}
    for path in paths:
        with av.open(str(path)) as media:
            for packet in media.demux():
                if packet.dts is not None and packet.stream.type in result:
                    result[packet.stream.type].append((packet.pts * packet.time_base,
                                                       packet.dts * packet.time_base, bytes(packet)))
    return result


def test_metadata_matches_decoded_frames_and_every_segment_start(encoded_source, tmp_path):
    playlist = tmp_path / "stream.m3u8"
    run_mux(encoded_source, playlist)
    segments = segment_paths(playlist)
    assert len(segments) == 3
    # No second encode, no retiming, no lost video/audio packets.
    assert media_packets(segments) == media_packets([encoded_source])
    transitions = []
    for path in segments:
        cues = []
        with av.open(str(path)) as media:
            for packet in media.demux():
                if packet.stream.type == "data" and packet.pts is not None:
                    cues.append((packet.pts * packet.time_base, read_record(bytes(packet))["clip"]["clip_id"]))
        with av.open(str(path)) as media:
            frames = list(media.decode(video=0))
        assert cues[0][0] == frames[0].pts * frames[0].time_base
        for frame in frames:
            at = frame.pts * frame.time_base
            title = [name for pts, name in cues if pts <= at][-1]
            # The generated color is an independent oracle for clip identity.
            rgb = frame.to_ndarray(format="rgb24").mean(axis=(0, 1))
            assert title == "ABC"[int(rgb.argmax())]
        transitions.extend(cues)
    first = transitions[0][0]
    assert (first + Fraction(27, 24), "B") in transitions
    assert (first + Fraction(63, 24), "C") in transitions


def test_restart_keeps_old_segments_and_marks_the_new_timeline(encoded_source, tmp_path):
    playlist = tmp_path / "stream.m3u8"
    first = run_mux(encoded_source, playlist)
    old_segments = segment_paths(playlist)
    old_bytes = [p.read_bytes() for p in old_segments]
    second = run_mux(encoded_source, playlist, prefix="restart-")
    segments = segment_paths(playlist)
    assert len(segments) == 6
    assert first.epoch != second.epoch
    assert segments[:3] == old_segments
    assert [p.read_bytes() for p in old_segments] == old_bytes
    text = playlist.read_text()
    assert "#EXT-X-DISCONTINUITY\n" + "#EXTINF" in text
    assert len({p.name for p in segments}) == 6
    for path in segments[3:]:
        with av.open(str(path)) as media:
            packet = next(p for p in media.demux() if p.stream.type == "data" and p.pts is not None)
            assert read_record(bytes(packet))["clip"]["clip_id"].startswith("restart-")


def test_cleanup_preserves_listed_history_and_removes_expired_orphans(tmp_path):
    import os
    import time
    playlist = tmp_path / "stream.m3u8"
    listed = tmp_path / "seg_previous_01.ts"
    orphan = tmp_path / "seg_previous_02.ts"
    recent = tmp_path / "seg_previous_03.ts"
    for p in (listed, orphan, recent):
        p.write_bytes(b"media")
    playlist.write_text("#EXTM3U\n#EXTINF:2,\n" + listed.name + "\n")
    for p in (listed, orphan):
        os.utime(p, (time.time() - 300, time.time() - 300))
    with listed.open("rb") as source:
        muxer = MetadataMuxer(source, playlist, 24, 120)
        muxer._cleanup()
    assert listed.exists() and recent.exists()
    assert not orphan.exists()
