"""Tests for vsa_utils — standalone VSA metadata utilities.

These tests are CPU-only (no GPU/kernel required) since vsa_utils
only does index computation with pure PyTorch.
"""

import math
import pytest
import torch

from fastvideo_kernel.vsa_utils import (
    VSA_TILE_SIZE,
    get_tile_partition_indices,
    get_reverse_tile_partition_indices,
    construct_variable_block_sizes,
    get_non_pad_index,
    build_vsa_metadata,
)


class TestGetTilePartitionIndices:

    @pytest.mark.parametrize("dit_seq_shape,tile_size", [
        ((8, 16, 16), (4, 4, 4)),
        ((4, 8, 8), (4, 4, 4)),
        ((9, 10, 7), (4, 4, 4)),
    ])
    def test_is_valid_permutation(self, dit_seq_shape, tile_size):
        """Output must be a permutation of 0..N-1."""
        device = torch.device("cpu")
        idx = get_tile_partition_indices(dit_seq_shape, tile_size, device)
        n = math.prod(dit_seq_shape)
        assert idx.shape == (n,)
        assert idx.dtype == torch.long
        assert set(idx.tolist()) == set(range(n))

    def test_exact_values_small(self):
        """Manually verify a small case: (2,2,2) with tile (2,2,2) = 1 tile."""
        device = torch.device("cpu")
        idx = get_tile_partition_indices((2, 2, 2), (2, 2, 2), device)
        assert idx.tolist() == list(range(8))

    def test_non_divisible_shape(self):
        """When shape doesn't divide evenly by tile_size, all tokens still covered."""
        device = torch.device("cpu")
        shape = (5, 7, 3)
        idx = get_tile_partition_indices(shape, (4, 4, 4), device)
        assert idx.shape == (5 * 7 * 3,)
        assert set(idx.tolist()) == set(range(5 * 7 * 3))


class TestGetReverseTilePartitionIndices:

    @pytest.mark.parametrize("dit_seq_shape", [
        (8, 16, 16),
        (9, 10, 7),
    ])
    def test_inverse_of_forward(self, dit_seq_shape):
        """reverse[forward[i]] == i for all i."""
        device = torch.device("cpu")
        tile_size = (4, 4, 4)
        fwd = get_tile_partition_indices(dit_seq_shape, tile_size, device)
        rev = get_reverse_tile_partition_indices(dit_seq_shape, tile_size, device)
        n = math.prod(dit_seq_shape)
        identity = torch.arange(n, device=device)
        assert torch.equal(rev[fwd], identity)
        assert torch.equal(fwd[rev], identity)


class TestConstructVariableBlockSizes:

    def test_sum_equals_total_tokens(self):
        """Sum of block sizes must equal T*H*W."""
        device = torch.device("cpu")
        shape = (8, 16, 16)
        tile_size = (4, 4, 4)
        num_tiles = tuple(math.ceil(s / t) for s, t in zip(shape, tile_size))
        vbs = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        assert vbs.sum().item() == math.prod(shape)

    def test_max_block_size(self):
        """No block can exceed tile volume."""
        device = torch.device("cpu")
        shape = (9, 10, 7)
        tile_size = (4, 4, 4)
        num_tiles = tuple(math.ceil(s / t) for s, t in zip(shape, tile_size))
        vbs = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        assert vbs.max().item() <= math.prod(tile_size)

    def test_num_blocks(self):
        """Number of blocks = product of num_tiles."""
        device = torch.device("cpu")
        shape = (8, 16, 16)
        tile_size = (4, 4, 4)
        num_tiles = tuple(math.ceil(s / t) for s, t in zip(shape, tile_size))
        vbs = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        assert vbs.shape[0] == math.prod(num_tiles)

    def test_exact_divisible(self):
        """When perfectly divisible, all blocks have the same size."""
        device = torch.device("cpu")
        shape = (8, 8, 8)
        tile_size = (4, 4, 4)
        num_tiles = (2, 2, 2)
        vbs = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        assert (vbs == 64).all()

    def test_non_divisible_last_tile_smaller(self):
        """When not divisible, at least one block is smaller than max."""
        device = torch.device("cpu")
        shape = (9, 8, 8)
        tile_size = (4, 4, 4)
        num_tiles = (3, 2, 2)
        vbs = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        assert vbs.min().item() < math.prod(tile_size)

    def test_custom_tile_size(self):
        """tile_size parameter overrides default VSA_TILE_SIZE."""
        device = torch.device("cpu")
        shape = (6, 6, 6)
        tile_size = (3, 3, 3)
        num_tiles = (2, 2, 2)
        vbs = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        assert vbs.sum().item() == 6 * 6 * 6
        assert (vbs == 27).all()


class TestGetNonPadIndex:

    def test_length_equals_sum_block_sizes(self):
        """Output length must equal sum of variable_block_sizes."""
        vbs = torch.tensor([32, 48, 64], dtype=torch.long)
        idx = get_non_pad_index(vbs, 64)
        assert idx.shape[0] == 32 + 48 + 64

    def test_indices_in_valid_range(self):
        """All indices must be in [0, num_blocks * max_block_size)."""
        vbs = torch.tensor([32, 48], dtype=torch.long)
        idx = get_non_pad_index(vbs, 64)
        assert idx.min().item() >= 0
        assert idx.max().item() < 2 * 64

    def test_block_boundary_alignment(self):
        """First token of block i starts at i * max_block_size."""
        vbs = torch.tensor([20, 40], dtype=torch.long)
        idx = get_non_pad_index(vbs, 64)
        assert idx[0].item() == 0
        assert idx[20].item() == 64

    def test_full_blocks(self):
        """When all blocks are full, output is just 0..N-1."""
        vbs = torch.tensor([64, 64], dtype=torch.long)
        idx = get_non_pad_index(vbs, 64)
        assert torch.equal(idx, torch.arange(128))


class TestBuildVsaMetadata:

    def test_all_keys_present(self):
        """build_vsa_metadata returns all expected keys."""
        meta = build_vsa_metadata((8, 16, 16), device="cpu")
        expected_keys = {
            "tile_indices", "reverse_tile_indices",
            "variable_block_sizes", "non_pad_index",
            "num_tiles", "max_block_size",
        }
        assert set(meta.keys()) == expected_keys

    def test_types(self):
        """Return types are correct."""
        meta = build_vsa_metadata((8, 16, 16), device="cpu")
        assert isinstance(meta["tile_indices"], torch.Tensor)
        assert isinstance(meta["reverse_tile_indices"], torch.Tensor)
        assert isinstance(meta["variable_block_sizes"], torch.Tensor)
        assert isinstance(meta["non_pad_index"], torch.Tensor)
        assert isinstance(meta["num_tiles"], tuple)
        assert isinstance(meta["max_block_size"], int)

    def test_num_tiles_correct(self):
        meta = build_vsa_metadata((9, 10, 7), tile_size=(4, 4, 4), device="cpu")
        assert meta["num_tiles"] == (3, 3, 2)
        assert meta["max_block_size"] == 64

    def test_custom_tile_size(self):
        meta = build_vsa_metadata((6, 6, 6), tile_size=(3, 3, 3), device="cpu")
        assert meta["num_tiles"] == (2, 2, 2)
        assert meta["max_block_size"] == 27

    def test_consistency(self):
        """All components are internally consistent."""
        shape = (8, 16, 16)
        meta = build_vsa_metadata(shape, device="cpu")
        n = math.prod(shape)
        assert meta["tile_indices"].shape == (n,)
        assert meta["reverse_tile_indices"].shape == (n,)
        assert meta["variable_block_sizes"].sum().item() == n
        assert meta["non_pad_index"].shape[0] == n


class TestConsistencyWithFramework:
    """Verify vsa_utils matches the framework-level functions exactly.

    Only runs if fastvideo is importable (skip otherwise).
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_framework(self):
        try:
            from fastvideo.attention.backends.video_sparse_attn import (
                get_tile_partition_indices as fw_get_tile,
            )
        except ImportError:
            pytest.skip("fastvideo framework not installed")

    @pytest.mark.parametrize("shape", [(8, 16, 16), (9, 10, 7)])
    def test_tile_indices_match(self, shape):
        from fastvideo.attention.backends.video_sparse_attn import (
            get_tile_partition_indices as fw_get_tile,
        )
        device = torch.device("cpu")
        tile_size = (4, 4, 4)
        ours = get_tile_partition_indices(shape, tile_size, device)
        theirs = fw_get_tile(shape, tile_size, device)
        assert torch.equal(ours, theirs)

    @pytest.mark.parametrize("shape", [(8, 16, 16), (9, 10, 7)])
    def test_variable_block_sizes_match(self, shape):
        from fastvideo.attention.backends.video_sparse_attn import (
            construct_variable_block_sizes as fw_construct_vbs,
        )
        device = torch.device("cpu")
        tile_size = (4, 4, 4)
        num_tiles = tuple(math.ceil(s / t) for s, t in zip(shape, tile_size))
        ours = construct_variable_block_sizes(shape, num_tiles, device, tile_size)
        theirs = fw_construct_vbs(shape, num_tiles, device)
        assert torch.equal(ours, theirs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
