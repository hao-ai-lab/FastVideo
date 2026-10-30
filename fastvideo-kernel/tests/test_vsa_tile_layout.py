# SPDX-License-Identifier: Apache-2.0
"""Bitwise parity for the fused VSA64 tile-to-BHSD Triton kernel.

Production only routes SM100 here, but the kernel is plain Triton, so parity is
checked on any CUDA device.
"""

import importlib.util
from pathlib import Path

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
    for name, old, new in zip(("q", "k", "v", "gate"), expected, actual, strict=True):
        assert old.shape == new.shape, name
        assert torch.equal(old, new), name


@pytest.mark.parametrize("fused", [False, True])
def test_layout_benchmark_matches_forward_preprocessing(monkeypatch, fused):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    from fastvideo.attention.backends import video_sparse_attn as vsa

    path = Path(__file__).resolve().parents[1] / "benchmarks/bench_vsa_tile_layout.py"
    spec = importlib.util.spec_from_file_location("bench_vsa_tile_layout", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    original_gate = vsa._can_fuse_vsa64_layout

    def allow_test_device(x, metadata):
        with monkeypatch.context() as gate_patch:
            gate_patch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 0))
            return original_gate(x, metadata)

    monkeypatch.setattr(vsa, "_can_fuse_vsa64_layout", allow_test_device)
    impl = object.__new__(vsa.VideoSparseAttentionImpl)
    builder = vsa.VideoSparseAttentionMetadataBuilder()
    kwargs = dict(current_timestep=0,
                  raw_latent_shape=(5, 5, 6),
                  patch_size=(1, 1, 1),
                  VSA_sparsity=0.8,
                  device=torch.device("cuda"),
                  cache_tile_buf=True)
    qkvg = torch.randn((4, 157, 2, 128), device="cuda", dtype=torch.bfloat16)[:, :150]
    captured = []

    def capture_attention_inputs(q, k, v, *args, **kwargs):
        captured[:] = [q, k, v, kwargs["compress_attn_weight"]]
        return q

    monkeypatch.setattr(vsa, "video_sparse_attn", capture_attention_inputs)
    with torch.no_grad():
        layout_step = benchmark.make_step(impl, qkvg, builder, kwargs, fused=fused, include_attn=False)
        attention_step = benchmark.make_step(impl, qkvg, builder, kwargs, fused=fused, include_attn=True)

        def reject_cat(*args, **kwargs):
            pytest.fail("The timed layout step must not concatenate its outputs")

        with monkeypatch.context() as timed_patch:
            timed_patch.setattr(torch, "cat", reject_cat)
            layout = layout_step()
            attention = attention_step()

    assert isinstance(layout, tuple) and len(layout) == 4
    for actual, expected in zip(layout, captured, strict=True):
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()
    assert all(t.is_contiguous() for t in layout[:3])
    assert layout[3].is_contiguous() is fused
    assert isinstance(attention, tuple) and len(attention) == 1
    assert torch.equal(attention[0], captured[0].transpose(1, 2))
