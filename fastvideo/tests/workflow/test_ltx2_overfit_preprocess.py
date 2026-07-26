# SPDX-License-Identifier: Apache-2.0

import wave

import pytest

from fastvideo.pipelines.preprocess.preprocess_ltx2_overfit import (
    load_video, )


def test_load_video_rejects_audio_only_input(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8)

    with pytest.raises(RuntimeError, match="No video stream found"):
        load_video(str(path), num_frames=1, target_fps=24, height=8, width=8)
