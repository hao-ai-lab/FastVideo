# SPDX-License-Identifier: Apache-2.0
"""Ref2VA SFT preprocessing keeps each target clip's own canvas and native frame count."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_TEXT_TAG,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.ref2va_manifest import MiniMaxH3Ref2VARawSample
from fastvideo.pipelines.preprocess import preprocess_minimax_h3_overfit as overfit
from fastvideo.pipelines.preprocess import preprocess_minimax_h3_ref2va as ref2va

_SAMPLES_PER_AUDIO_LATENT = 800  # 32 kHz / 40 latents per second


@pytest.mark.parametrize(
    ("available", "expected"),
    [(124, 124), (140, 124), (141, 141), (157, 141), (209, 209), (362, 362), (1000, 362)],
)
def test_native_num_frames_rounds_down_to_17n_plus_5(available: int, expected: int) -> None:
    assert overfit.native_num_frames(available) == expected


def test_native_num_frames_rejects_clips_under_five_seconds() -> None:
    with pytest.raises(ValueError, match="at least 124"):
        overfit.native_num_frames(123)


def _stub_media(monkeypatch: pytest.MonkeyPatch, num_source_frames: int, seconds: float) -> None:
    """Decode to tiny 30-fps frames and a 48 kHz stereo soundtrack; skip the canvas resize."""
    frames = np.zeros((num_source_frames, 8, 4, 3), dtype=np.uint8)
    waveform = torch.zeros(2, int(seconds * 48000))
    monkeypatch.setattr(overfit, "decode_reference_video", lambda _path: (frames, 30.0, (waveform, 48000)))
    monkeypatch.setattr(overfit, "prepare_reference_frames", lambda decoded, num_frames: decoded[:num_frames])


def test_native_media_keeps_native_length_and_whole_audio_latents(monkeypatch: pytest.MonkeyPatch) -> None:
    # 210 frames at 30 fps resample to 168 at 24 fps; the longest native count is 158.
    _stub_media(monkeypatch, num_source_frames=210, seconds=7.0)

    frames, waveform = overfit.load_training_media(Path("clip.mp4"), num_frames=None)

    assert frames.shape[0] == 158
    # 158 frames last 210666 samples, past the 263rd whole latent; the VAE would pad a 264th.
    assert audio_latent_num_frames(158) == 263
    assert waveform.shape == (2, 263 * _SAMPLES_PER_AUDIO_LATENT)


def test_fixed_overfit_media_audio_length_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_media(monkeypatch, num_source_frames=200, seconds=7.0)
    monkeypatch.setattr(
        overfit,
        "prepare_reference_frames",
        lambda decoded, num_frames: np.zeros((num_frames, overfit.VIDEO_HEIGHT, overfit.VIDEO_WIDTH, 3), np.uint8),
    )

    frames, waveform = overfit.load_training_media(Path("clip.mp4"))

    # 124 frames end inside the 207th audio latent, so the whole-latent cap cuts nothing.
    uncapped = overfit.prepare_reference_waveform(torch.zeros(2, 7 * 48000),
                                                  48000,
                                                  overfit.AUDIO_SAMPLE_RATE,
                                                  max_duration=overfit.NUM_FRAMES / 24)
    assert frames.shape[0] == overfit.NUM_FRAMES
    assert uncapped.shape[-1] < audio_latent_num_frames(overfit.NUM_FRAMES) * _SAMPLES_PER_AUDIO_LATENT
    assert waveform.shape == uncapped.shape


def test_build_record_uses_the_clip_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A portrait 158-frame clip yields a record the training collator accepts."""
    num_frames = 158
    height, width = resolve_canvas_size(9, 16)
    # np.zeros is not touched below, so the full-size canvas costs no memory.
    target = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    waveform = torch.zeros(2, audio_latent_num_frames(num_frames) * _SAMPLES_PER_AUDIO_LATENT)
    monkeypatch.setattr(ref2va, "load_training_media", lambda _path, num_frames: (target, waveform))
    monkeypatch.setattr(ref2va, "_prepare_references", lambda _refs, _num_frames: [])
    monkeypatch.setattr(
        ref2va,
        "encode_video_latents",
        lambda frames, *_args: torch.zeros(24, video_latent_num_frames(frames.shape[0]), frames.shape[1] // 16,
                                           frames.shape[2] // 16),
    )
    monkeypatch.setattr(
        ref2va,
        "encode_audio_latents",
        lambda audio, *_args: torch.zeros(2, 32, audio.shape[-1] // _SAMPLES_PER_AUDIO_LATENT),
    )
    monkeypatch.setattr(ref2va, "encode_ref_visual_anchor", lambda *_args: torch.empty(0, 96))
    monkeypatch.setattr(ref2va, "encode_ref_audio_anchor", lambda *_args: torch.empty(0, 32))
    monkeypatch.setattr(
        ref2va,
        "encode_ref2va_conditioning",
        lambda *_args: (torch.zeros(4, 5120), torch.full((4, ), MINIMAX_H3_TEXT_TAG)),
    )
    sample = MiniMaxH3Ref2VARawSample(
        sample_id="portrait",
        target_file="portrait.mp4",
        target_video_path=Path("portrait.mp4"),
        caption="A person talks to the camera.",
        references=(),
    )

    record = ref2va._build_record(
        sample,
        model_path=Path("model"),
        model_index={},
        fastvideo_args=None,
        patch_size=(1, 2, 2),
    )

    assert (record["num_frames"], record["height"], record["width"]) == (num_frames, height, width)
    assert record["vae_latent_shape"] == [24, video_latent_num_frames(num_frames), height // 16, width // 16]
    assert record["audio_latent_shape"] == [2, 32, audio_latent_num_frames(num_frames)]
