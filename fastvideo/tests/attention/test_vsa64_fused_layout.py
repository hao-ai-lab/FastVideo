"""VSA64 fused layout routing, fallback, and forward parity.

The fused route is gated on SM100, but the Triton layout kernel itself is
architecture-agnostic. Routing and layout tests therefore run on any CUDA GPU
(the unit CI runs on L40S) with the capability check pinned to SM100; only the
end-to-end ``video_sparse_attn`` parity test needs a real SM100 device.
"""

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.attention.backends import video_sparse_attn as vsa_mod
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionImpl, VideoSparseAttentionMetadataBuilder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 CUDA device required",
)


@pytest.fixture
def as_sm100(monkeypatch):
    """Pin the capability gate to SM100 so fallbacks are caused by the condition under test."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))


def _fused_layout(enabled: bool):
    return envs.FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT.override(not enabled)


@pytest.fixture
def tile_kernel():
    """Require the in-tree layout kernel instead of skipping without it.

    CI builds fastvideo-kernel from source, so a missing module means the build
    is broken or stale (e.g. the pinned PyPI wheel won), and a skip would let
    the lane pass without exercising the fused route.
    """
    pytest.importorskip("triton")
    try:
        import fastvideo_kernel.triton_kernels.vsa_tile_layout  # noqa: F401
    except ImportError as exc:
        pytest.fail("fastvideo_kernel.triton_kernels.vsa_tile_layout is missing; build fastvideo-kernel from "
                    f"source (cd fastvideo-kernel && ./build.sh): {exc}")


def _build(dit_shape: tuple[int, int, int]):
    impl = object.__new__(VideoSparseAttentionImpl)
    builder = VideoSparseAttentionMetadataBuilder()
    kwargs = dict(
        current_timestep=0,
        raw_latent_shape=dit_shape,
        patch_size=(1, 1, 1),
        VSA_sparsity=0.8,
        device=torch.device("cuda"),
        cache_tile_buf=True,
    )
    return impl, builder, kwargs


@pytest.mark.parametrize("dit_shape", [(5, 5, 6), (16, 28, 52)])
def test_vsa64_fused_layout_matches_legacy_tiles(as_sm100, tile_kernel, dit_shape):
    impl, builder, kwargs = _build(dit_shape)
    fused_metadata = builder.build(**kwargs)
    legacy_metadata = builder.build(**kwargs)
    sequence = fused_metadata.total_seq_length
    # Non-contiguous input, as produced by the SP-padding trim in the attention layer.
    qkvg = torch.randn((4, sequence + 7, 2, 128), device="cuda", dtype=torch.bfloat16)[:, :sequence]

    with _fused_layout(True), torch.no_grad():
        fused = impl.preprocess_qkv(qkvg, fused_metadata)
    assert fused_metadata.fused_layout_active

    with _fused_layout(False), torch.no_grad():
        legacy = impl.preprocess_qkv(qkvg, legacy_metadata).transpose(1, 2).contiguous()
    assert not legacy_metadata.fused_layout_active

    assert fused.shape == legacy.shape
    assert torch.equal(fused, legacy)


@pytest.mark.parametrize("dit_shape", [(5, 5, 6), (16, 28, 52)])
def test_vsa64_fused_layout_can_be_disabled(as_sm100, dit_shape):
    impl, builder, kwargs = _build(dit_shape)
    metadata = builder.build(**kwargs)
    sequence = metadata.total_seq_length
    qkvg = torch.randn((4, sequence, 2, 128), device="cuda", dtype=torch.bfloat16)
    with _fused_layout(False), torch.no_grad():
        disabled = impl.preprocess_qkv(qkvg, metadata)
    assert not metadata.fused_layout_active
    assert disabled.ndim == 4 and disabled.shape[1] == metadata.variable_block_sizes.numel() * 64


def test_vsa64_fused_layout_training_falls_back(as_sm100):
    impl, builder, kwargs = _build((5, 5, 6))
    metadata = builder.build(**kwargs)
    qkvg = torch.randn((4, 150, 2, 128), device="cuda", dtype=torch.bfloat16)
    with _fused_layout(True):
        training = impl.preprocess_qkv(qkvg.requires_grad_(), metadata)
    assert not metadata.fused_layout_active
    assert training.shape == (4, 512, 2, 128)
    assert training.requires_grad


def test_vsa64_mismatched_metadata_uses_original_path(monkeypatch, as_sm100):
    impl, builder, kwargs = _build((5, 5, 6))
    metadata = builder.build(**kwargs)
    qkvg = torch.empty((4, metadata.total_seq_length - 1, 2, 128), device="cuda", dtype=torch.bfloat16)
    fallback = object()
    calls = []

    def tile(x, meta):
        calls.append((x, meta))
        return fallback

    monkeypatch.setattr(impl, "tile", tile)
    with _fused_layout(True), torch.no_grad():
        result = impl.preprocess_qkv(qkvg, metadata)
    assert result is fallback
    assert len(calls) == 1
    assert calls[0][0] is qkvg and calls[0][1] is metadata
    assert not metadata.fused_layout_active


def test_vsa64_fused_layout_falls_back_when_kernel_missing(monkeypatch, as_sm100):
    impl, builder, kwargs = _build((5, 5, 6))
    metadata = builder.build(**kwargs)
    qkvg = torch.randn((4, metadata.total_seq_length, 2, 128), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(vsa_mod, "_get_tile_to_bhsd", lambda: None)
    with _fused_layout(True), torch.no_grad():
        result = impl.preprocess_qkv(qkvg, metadata)
    assert not metadata.fused_layout_active
    assert result.shape[1] == metadata.variable_block_sizes.numel() * 64


@requires_sm100
@pytest.mark.parametrize("dit_shape", [(5, 5, 6), (16, 28, 52)])
def test_vsa64_fused_forward_matches_legacy(dit_shape):
    try:
        from fastvideo_kernel import video_sparse_attn  # noqa: F401
    except ImportError:
        pytest.skip("video_sparse_attn is not installed")

    impl, builder, kwargs = _build(dit_shape)
    fused_metadata = builder.build(**kwargs)
    legacy_metadata = builder.build(**kwargs)
    sequence = fused_metadata.total_seq_length
    qkvg = torch.randn((4, sequence, 2, 128), device="cuda", dtype=torch.bfloat16)

    with _fused_layout(True), torch.no_grad():
        fused_tiled = impl.preprocess_qkv(qkvg, fused_metadata)
        assert fused_metadata.fused_layout_active
        q, k, v, gate = fused_tiled.chunk(4)
        fused_out = impl.forward(q, k, v, gate, fused_metadata)

    with _fused_layout(False), torch.no_grad():
        legacy_tiled = impl.preprocess_qkv(qkvg, legacy_metadata)
        lq, lk, lv, lg = legacy_tiled.chunk(4)
        legacy_out = impl.forward(lq, lk, lv, lg, legacy_metadata)

    assert torch.equal(fused_out, legacy_out)
