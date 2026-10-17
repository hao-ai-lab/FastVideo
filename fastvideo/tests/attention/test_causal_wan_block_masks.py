# SPDX-License-Identifier: Apache-2.0
"""Causal Wan training masks when the last block has fewer frames.

DF-SFT and TF-SFT split the latent frames into ceil(num_frames / num_frames_per_block)
chunks, so the last chunk can be partial. The block-wise causal mask and the
teacher-forcing mask must then end that block at the last real token: clean tokens
never see noisy tokens, and real tokens never see the zero padding that rounds the
sequence up to a multiple of 128. Each mask is compared with a reference built
from the intended rule, frame by frame.
"""

import pytest
import torch

from fastvideo.models.wan.causal_transformer import CausalWanTransformer3DModel

SHAPES = [
    pytest.param(6, 8, 3, id="6frames-block3"),
    pytest.param(7, 8, 3, id="7frames-block3"),
    pytest.param(5, 10, 2, id="5frames-block2"),
    pytest.param(4, 8, 1, id="4frames-block1"),
]


def _materialize(block_mask) -> torch.Tensor:
    n = block_mask.seq_lengths[0]
    q_idx = torch.arange(n)[:, None]
    kv_idx = torch.arange(n)[None, :]
    return block_mask.mask_mod(0, 0, q_idx, kv_idx)


def _block_causal_reference(num_frames: int, frame_seqlen: int, frames_per_block: int, n: int) -> torch.Tensor:
    total = num_frames * frame_seqlen
    block = frames_per_block * frame_seqlen
    mask = torch.eye(n, dtype=torch.bool)
    for q in range(total):
        end = min((q // block + 1) * block, total)
        mask[q, :end] = True
    return mask


def _teacher_forcing_reference(num_frames: int, frame_seqlen: int, frames_per_block: int, n: int) -> torch.Tensor:
    clean = num_frames * frame_seqlen
    block = frames_per_block * frame_seqlen
    mask = torch.eye(n, dtype=torch.bool)
    for q in range(clean):
        mask[q, :min((q // block + 1) * block, clean)] = True
    for q in range(clean, 2 * clean):
        index = (q - clean) // block
        mask[q, :index * block] = True
        mask[q, clean + index * block:clean + min((index + 1) * block, clean)] = True
    return mask


@pytest.mark.parametrize("num_frames, frame_seqlen, frames_per_block", SHAPES)
def test_block_causal_mask_matches_reference(num_frames, frame_seqlen, frames_per_block):
    block_mask = CausalWanTransformer3DModel._prepare_blockwise_causal_attn_mask(
        device="cpu", num_frames=num_frames, frame_seqlen=frame_seqlen, num_frame_per_block=frames_per_block)
    got = _materialize(block_mask)
    expected = _block_causal_reference(num_frames, frame_seqlen, frames_per_block, got.shape[0])
    assert torch.equal(got, expected)


@pytest.mark.parametrize("num_frames, frame_seqlen, frames_per_block", SHAPES)
def test_teacher_forcing_mask_matches_reference(num_frames, frame_seqlen, frames_per_block):
    block_mask = CausalWanTransformer3DModel._prepare_teacher_forcing_mask(
        device="cpu", num_frames=num_frames, frame_seqlen=frame_seqlen, num_frame_per_block=frames_per_block)
    got = _materialize(block_mask)
    expected = _teacher_forcing_reference(num_frames, frame_seqlen, frames_per_block, got.shape[0])
    assert torch.equal(got, expected)
