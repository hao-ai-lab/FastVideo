# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 Ref2VA continuation clips: validation, decoding and their place in the packed sequence."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MiniMaxH3ContinuationLayout,
    _reference_temporal_span,
    audio_latent_num_frames,
    build_ref2va_packed_sequence,
    video_latent_num_frames,
)
from fastvideo.models import pinned_offload
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MINIMAX_H3_PREPARED_CONTINUATION_KEY,
    prepare_continuation,
    validate_references,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import _h3_vsa_ref2va_segments
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MiniMaxH3InputPreparationStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MiniMaxH3LatentPreparationStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

TEXT_TOKENS = 137
LATENT_HEIGHT, LATENT_WIDTH = 48, 84
ROWS_PER_FRAME = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)


def _image_reference():
    return SimpleNamespace(media_type="image",
                           num_latent_frames=1,
                           latent_height=64,
                           latent_width=40,
                           has_audio=False,
                           num_audio_latents=0)


def _layout(clip_audio_latents: int, num_frames: int = 141, clip_frames: int = 22):
    clip = MiniMaxH3ContinuationLayout(video_latent_num_frames(clip_frames), clip_audio_latents)
    packed = build_ref2va_packed_sequence(
        torch.ones(TEXT_TOKENS, dtype=torch.long),
        [_image_reference()],
        video_latent_num_frames(num_frames),
        LATENT_HEIGHT,
        LATENT_WIDTH,
        audio_latent_num_frames(num_frames),
        (1, 2, 2),
        continuation=clip,
    )
    return packed, clip


@pytest.mark.parametrize("clip_audio_latents", [37, 0])
def test_clip_rows_take_the_coordinates_of_the_target_start(clip_audio_latents):
    packed, clip = _layout(clip_audio_latents)
    positions = packed.position_ids
    clip_rows = clip.num_latent_frames * ROWS_PER_FRAME
    conditions = packed.num_condition_video_rows
    clip_video = packed.video_indices[conditions - clip_rows:conditions]
    target_start = packed.video_indices[conditions:conditions + clip_rows]
    assert torch.equal(positions[clip_video], positions[target_start])

    image_rows = (64 // 2) * (40 // 2)
    assert conditions == image_rows + clip_rows
    origin = TEXT_TOKENS + 1.0 + max(float(clip_audio_latents), _reference_temporal_span(clip.num_latent_frames))
    assert positions[target_start[0], 0].item() == pytest.approx(origin)

    audio_conditions = packed.num_condition_audio_rows
    assert audio_conditions == clip_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    clip_audio = packed.audio_indices[:audio_conditions]
    target_audio = packed.audio_indices[audio_conditions:]
    for channel in range(MINIMAX_H3_AUDIO_CHANNELS if clip_audio_latents else 0):
        clip_channel = clip_audio[channel * clip_audio_latents:(channel + 1) * clip_audio_latents]
        start = channel * packed.num_audio_latents
        assert torch.equal(positions[clip_channel], positions[target_audio[start:start + clip_audio_latents]])


@pytest.mark.parametrize("clip_audio_latents", [37, 0])
def test_vsa_keeps_the_clip_dense(clip_audio_latents):
    packed, clip = _layout(clip_audio_latents)
    segments = _h3_vsa_ref2va_segments(packed, (1, 2, 2))
    assert [segment for segment in segments if isinstance(segment, tuple)] == [
        (packed.num_video_latent_frames, LATENT_HEIGHT, LATENT_WIDTH)
    ]
    assert clip.num_latent_frames * ROWS_PER_FRAME in segments


def test_a_clip_as_long_as_the_target_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        _layout(37, num_frames=22, clip_frames=22)


def test_a_continuation_is_not_a_reference():
    image = MiniMaxH3Reference(source="picture.png")
    clip = MiniMaxH3Reference(source="clip.mkv", media_type="continuation")
    assert len(validate_references([image] * 9 + [MiniMaxH3Reference("a.wav", media_type="audio")] * 3 + [clip])) == 13
    with pytest.raises(ValueError, match="at most one continuation"):
        validate_references([image, clip, clip])
    with pytest.raises(ValueError, match="besides its clip"):
        validate_references([clip])


def test_continuation_clips_are_checked_against_the_contract():
    height, width = 768, 1344
    frames = np.zeros((22, height, width, 3), dtype=np.uint8)
    prepared = prepare_continuation(MiniMaxH3Reference(frames, media_type="continuation"), height, width, 32000)
    assert prepared.frames.shape == (22, height, width, 3) and prepared.waveform is None
    with pytest.raises(ValueError, match="5, 22, 39 frames"):
        prepare_continuation(MiniMaxH3Reference(frames[:20], media_type="continuation"), height, width, 32000)
    with pytest.raises(ValueError, match="target canvas"):
        prepare_continuation(MiniMaxH3Reference(frames, media_type="continuation"), width, height, 32000)


class _Posterior:

    def __init__(self, latents: torch.Tensor) -> None:
        self.latents = latents

    def sample(self, generator=None) -> torch.Tensor:
        return self.latents

    def mode(self) -> torch.Tensor:
        return self.latents


class _VideoVAE:
    """Encodes each clip to latent frames of 2.0 and each still image to 1.0, on the H3 latent geometry."""

    latent_channels = 4
    spatial_compression_ratio = 16

    def normalize_pixels(self, pixels):
        return pixels

    def normalize_latents(self, latents):
        return latents

    def _latents(self, pixels: torch.Tensor, frames: int, value: float):
        height, width = pixels.shape[-2] // 16, pixels.shape[-1] // 16
        return SimpleNamespace(latent_dist=_Posterior(torch.full((1, 4, frames, height, width), value)))

    def encode_keyframe(self, pixels):
        return self._latents(pixels, 1, 1.0)

    def encode_pixels(self, pixels):
        return self._latents(pixels, video_latent_num_frames(pixels.shape[2]), 2.0)


class _AudioVAE:
    """Encodes each stereo channel to one latent per 800 samples whose values number the channel and step."""

    latent_channels = 4
    sampling_rate = 32000

    def normalize_latents(self, latents):
        return latents

    def encode(self, waveform):
        channels, steps = waveform.shape[0], waveform.shape[-1] // 800
        index = torch.arange(channels * steps, dtype=torch.float32).reshape(channels, 1, steps)
        return SimpleNamespace(latent_dist=_Posterior(index.repeat(1, self.latent_channels, 1)))


def _args():
    return SimpleNamespace(
        pin_cpu_memory=False,
        vae_cpu_offload=False,
        vae_parallel_encode=False,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 2, 2)),
            vae_config=SimpleNamespace(arch_config=SimpleNamespace(latent_channels=4)),
        ),
    )


def test_a_clip_with_sound_reaches_the_packed_sequence():
    height = width = 64
    clip_frames = np.zeros((22, height, width, 3), dtype=np.uint8)
    clip_samples = round(22 / 24 * 32000)
    clip = MiniMaxH3Reference(clip_frames,
                              media_type="continuation",
                              soundtrack=torch.linspace(-0.5, 0.5, clip_samples).repeat(2, 1),
                              sample_rate=32000)
    image = MiniMaxH3Reference(Image.new("RGB", (64, 64), "red"))
    batch = ForwardBatch(data_type="video",
                         prompt="x",
                         height=height,
                         width=width,
                         num_frames=141,
                         references=[image, clip],
                         generator=torch.Generator("cpu").manual_seed(0))
    video, audio = _VideoVAE(), _AudioVAE()
    MiniMaxH3InputPreparationStage(video, audio, ref2va=True)._prepare_ref2va(batch)
    assert [reference.media_type for reference in batch.references] == ["image"]
    prepared = batch.extra[MINIMAX_H3_PREPARED_CONTINUATION_KEY]
    assert prepared.frames.shape == (22, height, width, 3) and prepared.waveform.shape == (2, clip_samples)

    stage = MiniMaxH3LatentPreparationStage(video, audio, MiniMaxH3Scheduler(), ref2va=True)
    with patch.object(pinned_offload, "load"), patch.object(pinned_offload, "unload"):
        video_rows, audio_rows = stage._encode_ref2va_conditions(batch, _args(), torch.device("cpu"))
    image_rows = (batch.references[0].latent_height // 2) * (batch.references[0].latent_width // 2)
    clip_rows = video_latent_num_frames(22) * (height // 16 // 2) * (width // 16 // 2)
    assert video_rows.shape[0] == image_rows + clip_rows
    clip_audio_latents = clip_samples // 800
    expected_audio = torch.arange(2 * clip_audio_latents, dtype=torch.float32)[:, None].repeat(1, 4)
    assert torch.equal(audio_rows, expected_audio)

    batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = torch.ones(5, dtype=torch.long)
    layout = stage._build_layout(batch, _args())
    assert layout.num_condition_video_rows == image_rows + clip_rows
    assert layout.num_condition_audio_rows == 2 * clip_audio_latents
    assert layout.reference_segments[-2:] == (
        ("audio", 2 * clip_audio_latents, (0, 0, 0)),
        ("continuation", clip_rows, (video_latent_num_frames(22), height // 16, width // 16)),
    )
