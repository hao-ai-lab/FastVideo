# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the CUDA TAEH3 preview decoder."""
from __future__ import annotations

import torch

from fastvideo.models.vaes.minimax_h3_taeh3 import TorchTAEH3Decoder, ensure_taeh3_checkpoint, taeh3_decoded_pixel_shape


def test_taeh3_pixel_shape_matches_h3_temporal_contract() -> None:
    assert taeh3_decoded_pixel_shape((1, 24, 37, 30, 52)) == (1, 3, 124, 480, 832)
    assert taeh3_decoded_pixel_shape((1, 24, 2, 4, 4)) == (1, 3, 5, 64, 64)


def test_taeh3_chunk_sizes_agree_on_cpu() -> None:
    checkpoint = ensure_taeh3_checkpoint()
    decoder = TorchTAEH3Decoder(checkpoint, dtype=torch.float32)
    latents = torch.randn(1, 7, 24, 4, 4)
    full = decoder.decode_ntchw(latents, chunk_size=7)
    chunked = decoder.decode_ntchw(latents, chunk_size=3)
    torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5)
    assert full.shape == (1, 22, 3, 64, 64)
