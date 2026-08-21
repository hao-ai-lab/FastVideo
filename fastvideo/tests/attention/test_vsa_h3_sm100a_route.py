# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the VSA-H3 tile-64 sm_100a route selection.

The opt-in third kernel route (``FASTVIDEO_VSA_SM100A=1``) must (a) stay off by
default, (b) engage only when the extension is present, the device qualifies,
and the forward carries no grad, and (c) fall back to the Triton-64 entry with
one warning when the env is set but a precondition fails. All device/extension
probes are monkeypatched; no GPU or kernel install needed.
"""

import pytest
import torch

import fastvideo.attention.backends.video_sparse_attn_h3 as vsa_h3
from fastvideo.attention.backends.video_sparse_attn_h3 import (VSA_SM100A_ENV, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder, _sm100a_unavailable_reason)

# Small tile-64 geometry: 2 prefix segments + a (4,4,8)-token video grid.
_SPEC = dict(raw_latent_shape=(4, 8, 16), patch_size=(1, 2, 2), prefix_segments=(70, 30))
_HEADS, _DIM = 2, 128


def _build_meta():
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=_SPEC["raw_latent_shape"],
        patch_size=_SPEC["patch_size"],
        VSA_sparsity=0.0,
        prefix_segments=_SPEC["prefix_segments"],
        device=torch.device("cpu"),
        tile_size=64,
    )


def _tiled_qkv(meta, requires_grad=False):
    # bf16 like the real tiled buffers, so forward()'s dtype-cast warning
    # stays out of the warning assertions below.
    s_pad = meta.variable_block_sizes.numel() * 64
    return tuple(
        torch.randn(1, s_pad, _HEADS, _DIM, dtype=torch.bfloat16, requires_grad=requires_grad) for _ in range(3))


class _FakeSm100a:
    """Stands in for fastvideo_kernel.block_sparse_attn_sm100a."""

    def __init__(self, supported=True):
        self.supported = supported
        self.calls = []

    def is_supported(self, q, variable_block_sizes):
        return self.supported

    def block_sparse_attn_sm100a(self, q, k, v, q2k_idx, q2k_num, variable_block_sizes, need_lse=True):
        self.calls.append(dict(q=q, q2k_idx=q2k_idx, q2k_num=q2k_num, vbs=variable_block_sizes,
                               need_lse=need_lse))
        return q.clone(), None


def _fake_map_to_index(block_map):
    """Pure-torch stand-in for the Triton map_to_index (same contract)."""
    b, h, t, n = block_map.shape
    idx = torch.full((b, h, t, n), -1, dtype=torch.int32)
    num = block_map.sum(dim=-1, dtype=torch.int32)
    for bi in range(b):
        for hi in range(h):
            for ti in range(t):
                cols = torch.nonzero(block_map[bi, hi, ti], as_tuple=False).flatten()
                idx[bi, hi, ti, :cols.numel()] = cols.to(torch.int32)
    return idx, num


class _FakeTriton:
    def __init__(self):
        self.calls = 0

    def __call__(self, q, k, v, mask, variable_block_sizes):
        self.calls += 1
        return q.clone(), None


@pytest.fixture()
def routed(monkeypatch):
    """Backend with both kernel entries faked; returns (fakes, run)."""
    fake_sm = _FakeSm100a()
    fake_triton = _FakeTriton()
    monkeypatch.setattr(vsa_h3, "_sm100a", fake_sm)
    monkeypatch.setattr(vsa_h3, "block_sparse_attn_64_bhsd", fake_triton)
    monkeypatch.setattr(vsa_h3, "map_to_index", _fake_map_to_index)
    meta = _build_meta()
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)

    def run(requires_grad=False):
        q, k, v = _tiled_qkv(meta, requires_grad=requires_grad)
        return impl.forward(q, k, v, None, meta)

    return fake_sm, fake_triton, run, meta


def test_reason_covers_every_precondition():
    q = torch.randn(1, _HEADS, 128, _DIM)
    vbs = torch.full((2, ), 64, dtype=torch.long)
    assert "not installed" in _sm100a_unavailable_reason(None, q, vbs, grad_mode=False)
    ok = _FakeSm100a(supported=True)
    assert "forward-only" in _sm100a_unavailable_reason(ok, q, vbs, grad_mode=True)
    bad = _FakeSm100a(supported=False)
    assert "is_supported" in _sm100a_unavailable_reason(bad, q, vbs, grad_mode=False)
    assert _sm100a_unavailable_reason(ok, q, vbs, grad_mode=False) is None


def test_default_off_routes_triton(routed, monkeypatch):
    fake_sm, fake_triton, run, _ = routed
    monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
    run()
    assert fake_triton.calls == 1
    assert fake_sm.calls == []


def test_env_on_routes_sm100a_with_index_metadata(routed, monkeypatch):
    fake_sm, fake_triton, run, meta = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    out = run()
    assert fake_triton.calls == 0
    assert len(fake_sm.calls) == 1
    call = fake_sm.calls[0]
    n_tiles = meta.variable_block_sizes.numel()
    # sparsity 0 -> all-True mask -> every row's count is n_tiles
    assert call["q2k_num"].dtype == torch.int32 and (call["q2k_num"] == n_tiles).all()
    assert call["q2k_idx"].shape[-1] == n_tiles and call["q2k_idx"].dtype == torch.int32
    assert call["vbs"].dtype == torch.int32
    assert call["need_lse"] is False
    # BHSD kernel result comes back in the backend's BSHD layout
    assert out.shape == (1, n_tiles * 64, _HEADS, _DIM)


def test_env_on_grad_inputs_fall_back_to_triton(routed, monkeypatch):
    fake_sm, fake_triton, run, _ = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    run(requires_grad=True)
    assert fake_triton.calls == 1
    assert fake_sm.calls == []
    # ...but the same process still routes no-grad forwards to sm_100a
    run(requires_grad=False)
    assert len(fake_sm.calls) == 1


def test_env_on_unsupported_warns_once_and_falls_back(routed, monkeypatch):
    fake_sm, fake_triton, run, _ = routed
    fake_sm.supported = False
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    warnings = []
    monkeypatch.setattr(vsa_h3.logger, "warning_once", warnings.append)
    run()
    run()
    assert fake_triton.calls == 2
    assert fake_sm.calls == []
    assert len(warnings) == 2  # warning_once dedups by message; both carry the same one line
    assert warnings[0] == warnings[1]
    assert VSA_SM100A_ENV in warnings[0] and "is_supported" in warnings[0]


def test_env_on_missing_module_warns_and_falls_back(routed, monkeypatch):
    fake_sm, fake_triton, run, _ = routed
    monkeypatch.setattr(vsa_h3, "_sm100a", None)
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    warnings = []
    monkeypatch.setattr(vsa_h3.logger, "warning_once", warnings.append)
    run()
    assert fake_triton.calls == 1
    assert warnings and "not installed" in warnings[0]


def test_env_on_no_grad_context_detaches_route_from_leaf_flags(routed, monkeypatch):
    """A requires_grad leaf under torch.no_grad() is still a no-grad forward."""
    fake_sm, fake_triton, run, meta = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    q, k, v = _tiled_qkv(meta, requires_grad=True)
    with torch.no_grad():
        impl.forward(q, k, v, None, meta)
    assert len(fake_sm.calls) == 1
    assert fake_triton.calls == 0
