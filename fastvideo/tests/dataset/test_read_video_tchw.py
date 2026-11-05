# SPDX-License-Identifier: Apache-2.0

import wave

import av
import numpy as np
import pytest
import torch

from fastvideo.dataset.utils import read_video_tchw


def _write_video(path, frames: np.ndarray, fps: int = 8) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)  # lossless RGB H.264
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "rgb24"
        stream.options = {"qp": "0"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def pyav_only(monkeypatch):
    """Simulate torchvision >= 0.26, where torchvision.io.read_video no longer exists."""
    import torchvision.io

    monkeypatch.delattr(torchvision.io, "read_video", raising=False)


def test_decodes_every_frame_as_uint8_tchw_exactly(tmp_path, pyav_only) -> None:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(9, 32, 48, 3), dtype=np.uint8)
    path = tmp_path / "clip.mkv"
    _write_video(path, frames)

    video = read_video_tchw(str(path))

    assert video.dtype == torch.uint8
    assert tuple(video.shape) == (9, 3, 32, 48)
    torch.testing.assert_close(video, torch.from_numpy(frames).permute(0, 3, 1, 2))


def test_frame_index_selection_matches_the_old_call_pattern(tmp_path, pyav_only) -> None:
    frames = np.stack([np.full((16, 16, 3), i * 20, dtype=np.uint8) for i in range(6)])
    path = tmp_path / "ramp.mkv"
    _write_video(path, frames)

    picked = read_video_tchw(str(path))[[0, 2, 5]]

    assert picked[:, 0, 0, 0].tolist() == [0, 40, 100]


def test_missing_video_stream_raises(tmp_path, pyav_only) -> None:
    path = tmp_path / "audio_only.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 800)

    with pytest.raises(RuntimeError, match="No video stream"):
        read_video_tchw(str(path))


def test_prefers_torchvision_when_available(tmp_path, monkeypatch) -> None:
    import torchvision.io

    calls = []

    def fake_read_video(path, output_format):
        calls.append((path, output_format))
        return torch.zeros(2, 3, 4, 4, dtype=torch.uint8), torch.empty(0), {}

    monkeypatch.setattr(torchvision.io, "read_video", fake_read_video, raising=False)

    video = read_video_tchw("clip.mp4")

    assert calls == [("clip.mp4", "TCHW")]
    assert tuple(video.shape) == (2, 3, 4, 4)

