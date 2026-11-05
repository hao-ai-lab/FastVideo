# SPDX-License-Identifier: Apache-2.0
"""Device-boundary contracts for MiniMax H3 packed-layout metadata."""

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MiniMaxH3PackedBatchLayout,
    MiniMaxH3PackedLayout,
    build_packed_batch_sequence,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference


def _assert_cpu_layout_matches(actual: MiniMaxH3PackedLayout, expected: MiniMaxH3PackedLayout) -> None:
    for field in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        actual_tensor = getattr(actual, field)
        assert actual_tensor.device.type == "cpu"
        torch.testing.assert_close(actual_tensor, getattr(expected, field), rtol=0, atol=0)

    actual_unique, actual_inverse = build_row_timesteps(actual, 0.5, 0.25, 0.999, 1.0)
    expected_unique, expected_inverse = build_row_timesteps(expected, 0.5, 0.25, 0.999, 1.0)
    torch.testing.assert_close(actual_unique, expected_unique, rtol=0, atol=0)
    torch.testing.assert_close(actual_inverse, expected_inverse, rtol=0, atol=0)


def test_cross_sample_layout_offsets_indices_and_restarts_positions() -> None:
    first = build_packed_sequence(
        torch.ones(3, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )
    second = build_packed_sequence(
        torch.ones(5, dtype=torch.long),
        num_latent_frames=3,
        latent_height=4,
        latent_width=6,
        num_audio_latents=4,
        patch_size=(1, 2, 2),
    )
    packed = build_packed_batch_sequence((first, second))

    assert isinstance(packed, MiniMaxH3PackedBatchLayout)
    assert packed.document_lengths == (first.sequence_length, second.sequence_length)
    assert packed.document_offsets == (0, first.sequence_length, first.sequence_length + second.sequence_length)
    assert packed.text_lengths == (3, 5)
    torch.testing.assert_close(packed.position_ids[:first.sequence_length], first.position_ids)
    torch.testing.assert_close(packed.position_ids[first.sequence_length:], second.position_ids)
    torch.testing.assert_close(
        packed.video_indices,
        torch.cat((first.video_indices, second.video_indices + first.sequence_length)),
    )

    unique, inverse = build_row_timesteps(
        packed,
        video_timestep=(0.25, 0.75),
        audio_timestep=(0.1, 0.9),
        condition_video_timestep=(0.999, 0.999),
        condition_audio_timestep=(1.0, 1.0),
    )
    reconstructed = unique[inverse]
    assert reconstructed.shape == (packed.sequence_length, )
    first_unique, first_inverse = build_row_timesteps(first, 0.25, 0.1, 0.999, 1.0)
    second_unique, second_inverse = build_row_timesteps(second, 0.75, 0.9, 0.999, 1.0)
    torch.testing.assert_close(reconstructed[:first.sequence_length], first_unique[first_inverse])
    torch.testing.assert_close(reconstructed[first.sequence_length:], second_unique[second_inverse])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise moved training batches")
def test_packed_layouts_accept_cuda_text_tags_and_keep_metadata_on_cpu() -> None:
    cpu_tags = torch.tensor([1, 0, 1], dtype=torch.long)
    cuda_tags = cpu_tags.to("cuda")
    standard_kwargs = {
        "num_latent_frames": 2,
        "latent_height": 4,
        "latent_width": 4,
        "num_audio_latents": 2,
        "patch_size": (1, 2, 2),
    }
    references = [
        MiniMaxH3PreparedReference(
            media_type="video",
            has_audio=True,
            num_latent_frames=2,
            latent_height=4,
            latent_width=4,
            num_audio_latents=2,
        )
    ]

    _assert_cpu_layout_matches(
        build_packed_sequence(cuda_tags, **standard_kwargs),
        build_packed_sequence(cpu_tags, **standard_kwargs),
    )
    _assert_cpu_layout_matches(
        build_ref2va_packed_sequence(cuda_tags, references=references, **standard_kwargs),
        build_ref2va_packed_sequence(cpu_tags, references=references, **standard_kwargs),
    )
