# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 Ref2VA continuation clips: validation, decoding and their place in the packed sequence."""

from types import SimpleNamespace

import numpy as np
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
from fastvideo.pipelines.basic.minimax_h3.reference import prepare_continuation, validate_references
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import _h3_vsa_ref2va_segments

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
    # The clip advances the clock as a video reference with its soundtrack would.
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
    # Only the target video is a sparse region; the clip is a dense row count.
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
    # Twelve references plus a clip stay within the reference limit.
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
