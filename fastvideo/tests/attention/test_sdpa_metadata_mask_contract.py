# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression test for the SDPA metadata mask contract.

SDPAMetadata is cross-backend: HYWorld and HunyuanVideo15 build it while the
layer's attention selector may pick FLASH_ATTN, whose impl requires the
tokenizer-style 2D [batch, key_len] padding mask (flash_attn.py pads
``attn_mask.shape[1]`` assuming 2D). The builder must therefore store 2D
masks unchanged; only the torch-sdpa impl lifts them to 4D internally.

Regression guard for the CI break where the builder lifted 2D -> 4D and the
FLASH_ATTN consumer produced a garbage-length pad (CUDA device-side assert).
"""

from __future__ import annotations

import torch

from fastvideo.attention.backends.sdpa import (SDPAMetadataBuilder, _normalize_attn_mask_for_sdpa)


def test_builder_keeps_2d_padding_mask_2d() -> None:
    """A tokenizer-style 2D int mask must be stored as-is (cross-backend contract)."""
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)
    md = SDPAMetadataBuilder().build(current_timestep=0, attn_mask=mask)
    assert md.attn_mask is not None
    assert md.attn_mask.dim() == 2
    assert md.attn_mask.shape == (2, 3)
    assert torch.equal(md.attn_mask, mask)


def test_normalize_lifts_2d_int_mask_to_4d_bool_for_torch_sdpa() -> None:
    """The torch-sdpa consumer coerces int -> bool and lifts 2D -> [B,1,1,K]."""
    batch, q_len, k_len, heads, head_dim = 2, 4, 3, 1, 8
    query = torch.zeros(batch, heads, q_len, head_dim)
    key = torch.zeros(batch, heads, k_len, head_dim)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)

    out = _normalize_attn_mask_for_sdpa(mask, query, key)
    assert out is not None
    assert out.dtype == torch.bool
    assert out.shape == (batch, 1, 1, k_len)
    assert torch.equal(out[:, 0, 0, :], mask != 0)
