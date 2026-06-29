import torch

from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
)


def _build_metadata(cache_tile_buf: bool, raw_latent_shape=(4, 4, 4)):
    return VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=raw_latent_shape,
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


def test_vsa_tile_cached_and_uncached_produce_identical_values():
    # The cache flag must be a pure performance knob: enabling it (inference /
    # memory-rich training) and disabling it (#1423 OOM-safe training) must
    # yield bit-identical tilings. Use a padded shape (dit T=5 -> padded T=8)
    # so the zero-pad-position scatter is actually exercised.
    impl = object.__new__(VideoSparseAttentionImpl)
    md_cached = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 4, 4))
    md_uncached = _build_metadata(cache_tile_buf=False, raw_latent_shape=(5, 4, 4))

    total_seq_length = md_cached.total_seq_length
    x = torch.randn(2, total_seq_length, 3, 4)

    cached = impl.tile(x, md_cached).clone()
    uncached = impl.tile(x, md_uncached)

    assert cached.shape == uncached.shape
    assert torch.equal(cached, uncached)
    # The cached path stashes the buffer; the uncached path does not.
    assert md_cached.tile_buf is not None
    assert md_uncached.tile_buf is None
