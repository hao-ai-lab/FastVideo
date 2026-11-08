# SPDX-License-Identifier: Apache-2.0
"""H3 routing and compression parity for the opt-in Gluon forward on SM120."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Requires an SM120 CUDA GPU",
)


def _inputs():
    torch.manual_seed(123)
    q = torch.randn(1, 3, 192, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    sizes = torch.tensor([1, 63, 64], device="cuda", dtype=torch.int32)
    valid = torch.arange(192, device="cuda") % 64 < sizes.repeat_interleave(64)
    k[:, :, ~valid] = 0
    v[:, :, ~valid] = 0
    query, key, value = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
    return query, key, value, sizes


@pytest.mark.parametrize("gated", [False, True])
@torch.no_grad()
def test_h3_dispatch_preserves_mask_and_compression(gated, monkeypatch):
    import fastvideo.envs as envs
    import fastvideo.attention.backends.video_sparse_attn_h3 as backend

    query, key, value, sizes = _inputs()
    metadata = SimpleNamespace(
        tile_elems=64, variable_block_sizes=sizes, dense_layers=(), VSA_sparsity=0.5,
        num_prefix_tiles=1, exempt=True, video_tile_spans=((1, 3),), span_sparsities=(0.5,),
    )
    gate = torch.randn_like(query) * 0.1 if gated else None
    monkeypatch.setattr(backend, "probe_enabled", lambda: None)
    with envs.FASTVIDEO_H3_VSA_GLUON.override(False), envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(False):
        native = backend.MiniMaxH3VSAImpl(3, 128, False, 128**-0.5)
        expected = native.forward(query, key, value, gate, metadata)
        with envs.FASTVIDEO_H3_VSA_GLUON.override(True):
            candidate = backend.MiniMaxH3VSAImpl(3, 128, False, 128**-0.5)
            actual = candidate.forward(query, key, value, gate, metadata)
            with pytest.raises(RuntimeError, match="eager execution"):
                candidate.prepare_for_compile(query.device)
            metadata.tile_elems = 128
            with pytest.raises(RuntimeError, match="tile-64"):
                candidate.forward(query, key, value, gate, metadata)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_h3_conflicting_routes():
    import fastvideo.envs as envs
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl

    with envs.FASTVIDEO_H3_VSA_GLUON.override(True):
        with pytest.raises(ValueError, match="head dimension 128"):
            MiniMaxH3VSAImpl(3, 64, False, 64**-0.5)
        with envs.FASTVIDEO_VSA_PTX_BLACKWELL.override(True):
            with pytest.raises(ValueError, match="another fine-attention override"):
                MiniMaxH3VSAImpl(3, 128, False, 128**-0.5)
