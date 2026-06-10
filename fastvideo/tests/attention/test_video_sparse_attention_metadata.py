import torch

from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
)


def _build_metadata(cache_tile_buf: bool):
    return VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(4, 4, 4),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
        cache_tile_buf=cache_tile_buf,
    )


def test_vsa_tile_does_not_cache_training_scratch_when_disabled():
    metadata = _build_metadata(cache_tile_buf=False)
    impl = object.__new__(VideoSparseAttentionImpl)
    x = torch.ones(1, 64, 2, 2)

    tiled = impl.tile(x, metadata)

    assert tiled.shape == x.shape
    assert metadata.tile_buf is None


def test_vsa_tile_caches_scratch_by_default():
    metadata = _build_metadata(cache_tile_buf=True)
    impl = object.__new__(VideoSparseAttentionImpl)
    x = torch.ones(1, 64, 2, 2)

    tiled = impl.tile(x, metadata)

    assert metadata.tile_buf is tiled
