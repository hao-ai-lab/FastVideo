# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from fastvideo.models.dits.longcat import build_longcat_block_causal_mask


def _frame_visibility(mask: torch.Tensor, tokens_per_frame: int) -> list[list[bool]]:
    token_mask = mask[0, 0]
    frames = []
    for row in token_mask[::tokens_per_frame]:
        frames.append([
            bool(row[frame * tokens_per_frame])
            for frame in range(token_mask.shape[1] // tokens_per_frame)
        ])
    return frames


def test_longcat_block_causal_mask_blocks_future_chunks():
    mask = build_longcat_block_causal_mask(
        query_frames=4,
        key_frames=4,
        tokens_per_frame=2,
        causal_block_size=2,
        device="cpu",
    )

    assert mask.shape == (1, 1, 8, 8)
    assert _frame_visibility(mask, tokens_per_frame=2) == [
        [True, True, False, False],
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]
    assert mask[0, 0, 1].tolist() == mask[0, 0, 0].tolist()


def test_longcat_block_causal_mask_handles_cached_prefix():
    mask = build_longcat_block_causal_mask(
        query_frames=2,
        key_frames=5,
        tokens_per_frame=1,
        causal_block_size=2,
        device="cpu",
    )

    assert _frame_visibility(mask, tokens_per_frame=1) == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_longcat_block_causal_mask_allows_first_single_block_only():
    mask = build_longcat_block_causal_mask(
        query_frames=1,
        key_frames=1,
        tokens_per_frame=2,
        causal_block_size=2,
        device="cpu",
    )

    assert _frame_visibility(mask, tokens_per_frame=2) == [
        [True],
    ]
    assert mask[0, 0, 1].tolist() == mask[0, 0, 0].tolist()


def test_longcat_block_causal_mask_rejects_invalid_shape():
    with pytest.raises(ValueError, match="query_frames must be <= key_frames"):
        build_longcat_block_causal_mask(
            query_frames=3,
            key_frames=2,
            tokens_per_frame=1,
            causal_block_size=1,
            device="cpu",
        )
