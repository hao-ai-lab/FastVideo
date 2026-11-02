# SPDX-License-Identifier: Apache-2.0
"""Ref2VA packing parity: the MLX layout vs the upstream PyTorch builder.

A reference-conditioned H3 request packs ``[text | ordered references |
target audio | target video]`` with a rotary clock that advances over every
reference. The MLX runtime builds that layout in NumPy and the CUDA runtime
builds it in torch; if the two disagree on a single row the two runtimes
generate different clips from the same request, so the builders are gated
against each other here, exactly as the T2VA/FL2VA layout is in
``test_mlx_minimax_h3_parity.py``.

Runs anywhere MLX and torch are installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastvideo.mlx_runtime.minimax_h3 import (
    MINIMAX_H3_TEXT_TAG,
    build_ref2va_packed_layout,
)
from fastvideo.mlx_runtime.minimax_h3_reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.packing import build_ref2va_packed_sequence
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference as TorchPreparedReference,
)

from fastvideo.tests.mlx.tiny_h3 import (  # noqa: E402  (after the mlx import guard)
    LATENT_HEIGHT,
    LATENT_WIDTH,
    NUM_AUDIO_LATENTS,
    NUM_LATENT_FRAMES,
    NUM_TEXT_TOKENS,
    TINY_ARCH,
)

PATCH_SIZE = tuple(TINY_ARCH["patch_size"])


def _reference(
    media_type: str,
    *,
    has_audio: bool = False,
    frames: int | None = None,
    height: int = 4,
    width: int = 4,
    audio_latents: int = 2,
):
    """The same prepared reference in both runtimes' dataclasses.

    An image is a single-latent-frame reference: its rows are one frame grid,
    which is what both builders assign to its span.
    """
    if frames is None:
        frames = 1 if media_type == "image" else 2
    has_audio = has_audio or media_type == "audio"
    torch_reference = TorchPreparedReference(media_type=media_type, has_audio=has_audio)
    mlx_reference = MiniMaxH3PreparedReference(media_type=media_type, has_audio=has_audio)
    for reference in (torch_reference, mlx_reference):
        reference.num_latent_frames = frames if media_type != "audio" else 1
        reference.latent_height = height if media_type != "audio" else 0
        reference.latent_width = width if media_type != "audio" else 0
        reference.num_audio_latents = audio_latents if has_audio else 0
    return torch_reference, mlx_reference


def _assert_layouts_match(torch_references, mlx_references) -> None:
    tags = np.full(NUM_TEXT_TOKENS, MINIMAX_H3_TEXT_TAG, dtype=np.int64)
    reference = build_ref2va_packed_sequence(
        torch.from_numpy(tags),
        torch_references,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        PATCH_SIZE,
    )
    layout = build_ref2va_packed_layout(
        tags,
        mlx_references,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        PATCH_SIZE,
    )
    assert layout.sequence_length == reference.sequence_length
    np.testing.assert_array_equal(layout.position_ids, reference.position_ids.numpy())
    np.testing.assert_array_equal(layout.token_tags, reference.token_tags.numpy())
    np.testing.assert_array_equal(layout.video_indices, reference.video_indices.numpy())
    np.testing.assert_array_equal(layout.audio_indices, reference.audio_indices.numpy())
    np.testing.assert_array_equal(layout.text_indices, reference.text_indices.numpy())
    assert layout.num_condition_video_rows == reference.num_condition_video_rows
    assert layout.num_condition_audio_rows == reference.num_condition_audio_rows
    assert layout.reference_segments == reference.reference_segments


@pytest.mark.parametrize(
    "kinds",
    [
        ("image",),
        ("video",),
        ("image", "video"),
        ("audio", "image"),
        ("image", "video", "audio"),
    ],
)
def test_ref2va_layout_matches_upstream_builder(kinds: tuple[str, ...]) -> None:
    """Every supported ordered reference mix packs identically on both runtimes."""
    torch_references, mlx_references = [], []
    for index, kind in enumerate(kinds):
        # A video reference carries a soundtrack on every other case, so both
        # the interleaved-audio and the silent branch are covered.
        torch_reference, mlx_reference = _reference(kind, has_audio=kind == "video" and index % 2 == 0)
        torch_references.append(torch_reference)
        mlx_references.append(mlx_reference)
    _assert_layouts_match(torch_references, mlx_references)


def test_ref2va_layout_target_rows_follow_every_reference() -> None:
    """Target rows start after the references and keep their own region."""
    torch_reference, mlx_reference = _reference("image")
    second_torch, second_mlx = _reference("video", has_audio=True)
    tags = np.full(NUM_TEXT_TOKENS, MINIMAX_H3_TEXT_TAG, dtype=np.int64)
    layout = build_ref2va_packed_layout(
        tags, [mlx_reference, second_mlx], NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS,
        PATCH_SIZE)
    rows_per_frame = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    expected_condition_rows = rows_per_frame + NUM_LATENT_FRAMES * rows_per_frame
    assert layout.num_condition_video_rows == expected_condition_rows
    assert layout.video_indices[:layout.num_condition_video_rows].max() == NUM_TEXT_TOKENS + expected_condition_rows + \
        layout.num_condition_audio_rows - 1
    # The target rows are contiguous and land at the end of the sequence.
    np.testing.assert_array_equal(
        layout.video_indices[layout.num_condition_video_rows:],
        np.arange(layout.sequence_length - NUM_LATENT_FRAMES * rows_per_frame, layout.sequence_length),
    )
    assert [segment[0] for segment in layout.reference_segments] == ["image", "audio", "video"]


def test_ref2va_layout_rejects_non_ref2va_patch() -> None:
    _, reference = _reference("image")
    with pytest.raises(ValueError, match="patch_size"):
        build_ref2va_packed_layout(
            np.full(NUM_TEXT_TOKENS, MINIMAX_H3_TEXT_TAG, dtype=np.int64), [reference], NUM_LATENT_FRAMES,
            LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS, (1, 1, 1))
    with pytest.raises(ValueError, match="at least one prepared reference"):
        build_ref2va_packed_layout(
            np.full(NUM_TEXT_TOKENS, MINIMAX_H3_TEXT_TAG, dtype=np.int64), [], NUM_LATENT_FRAMES, LATENT_HEIGHT,
            LATENT_WIDTH, NUM_AUDIO_LATENTS, PATCH_SIZE)
