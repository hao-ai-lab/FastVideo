# SPDX-License-Identifier: Apache-2.0
"""Bitwise parity for the fused VSA64 tile-to-BHSD Triton kernel.

Production only routes SM100 here, but the kernel is plain Triton, so parity is
checked on any CUDA device.
"""

import pytest
import torch

pytest.importorskip("triton")

from fastvideo_kernel.triton_kernels.vsa_tile_layout import tile_to_bhsd  # noqa: E402


@pytest.mark.parametrize("dit_shape", [(5, 5, 6), (16, 28, 52)])
def test_tile_to_bhsd_matches_scatter_and_four_transposes(dit_shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    from fastvideo.attention.backends.video_sparse_attn import (VideoSparseAttentionImpl,
                                                                VideoSparseAttentionMetadataBuilder)

    impl = object.__new__(VideoSparseAttentionImpl)
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=dit_shape,
        patch_size=(1, 1, 1),
        VSA_sparsity=0.8,
        device=torch.device("cuda"),
        cache_tile_buf=True,
    )
    sequence = metadata.total_seq_length
    padded_sequence = metadata.variable_block_sizes.numel() * 64
    source = torch.randn((4, sequence + 7, 2, 128), device="cuda", dtype=torch.bfloat16)
    qkvg = source[:, :sequence]
    expected = tuple(piece.transpose(1, 2).contiguous() for piece in impl.tile(qkvg, metadata).chunk(4))

    source_index = torch.full((padded_sequence, ), -1, device="cuda", dtype=torch.int32)
    source_index[metadata.non_pad_index] = metadata.tile_partition_indices.to(torch.int32)
    out = torch.empty((4, 2, padded_sequence, 128), device="cuda", dtype=torch.bfloat16)
    actual = tile_to_bhsd(qkvg, source_index, out, sequence, padded_sequence, 2, 128).chunk(4)
    for name, old, new in zip(("q", "k", "v", "gate"), expected, actual):
        assert old.shape == new.shape, name
        assert torch.equal(old, new), name
