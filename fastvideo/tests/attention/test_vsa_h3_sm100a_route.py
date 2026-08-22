# SPDX-License-Identifier: Apache-2.0
"""Checks for the VSA-H3 tile-64 sm_100a route and odd-tile transport.

The opt-in third kernel route (``FASTVIDEO_VSA_SM100A=1``) must (a) stay off by
default, (b) engage only when the extension is present, the device qualifies,
and the forward carries no grad, (c) add one zero-valid partner for an odd
logical tile count, and (d) strip that partner before any fallback or returned
output. Most probes are monkeypatched; the final test is a GB200 route receipt.
"""

import logging

import pytest
import torch

import fastvideo.attention.backends.video_sparse_attn_h3 as vsa_h3
import fastvideo.logger as fastvideo_logger
from fastvideo.attention.backends.video_sparse_attn_h3 import (VSA_SM100A_ENV, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder, _sm100a_unavailable_reason)

# Small tile-64 geometry: 2 prefix segments + a (4,4,8)-token video grid.
_SPEC = dict(raw_latent_shape=(4, 8, 16), patch_size=(1, 2, 2), prefix_segments=(70, 30))
_HEADS, _DIM = 2, 128


def _build_meta(
    *,
    device: torch.device = torch.device("cpu"),
    prefix_segments: tuple[int, ...] = _SPEC["prefix_segments"],
    sparsity: float = 0.0,
    builder: MiniMaxH3VSAMetadataBuilder | None = None,
):
    builder = builder or MiniMaxH3VSAMetadataBuilder()
    return builder.build(
        current_timestep=0,
        raw_latent_shape=_SPEC["raw_latent_shape"],
        patch_size=_SPEC["patch_size"],
        VSA_sparsity=sparsity,
        prefix_segments=prefix_segments,
        device=device,
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
    messages = []
    monkeypatch.setattr(vsa_h3.logger, "info_once", messages.append)
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
    assert messages == ["MiniMax-H3 VSA tile-64 forward: using the sm100a CUDA block-sparse kernel"]
    # BHSD kernel result comes back in the backend's BSHD layout
    assert out.shape == (1, n_tiles * 64, _HEADS, _DIM)


def test_sm100a_engagement_receipt_logs_once_without_stacklevel_conflict(routed, monkeypatch):
    fake_sm, fake_triton, run, _ = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    records = []

    def capture_log(level, msg, *args, **kwargs):
        records.append((level, msg, args, kwargs))

    fastvideo_logger._print_info_once.cache_clear()
    monkeypatch.setattr(vsa_h3.logger, "log", capture_log)
    try:
        run()
        run()
    finally:
        fastvideo_logger._print_info_once.cache_clear()

    assert fake_triton.calls == 0
    assert len(fake_sm.calls) == 2
    assert records == [(logging.INFO, "MiniMax-H3 VSA tile-64 forward: using the sm100a CUDA block-sparse kernel", (),
                        {"stacklevel": 2})]


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


@pytest.mark.parametrize(
    ("env_enabled", "requires_grad", "extra_tiles"),
    [
        (False, False, 0),
        (True, False, 1),
        (True, True, 0),
    ],
)
def test_preprocess_adds_partner_only_for_odd_no_grad_sm100a(
    monkeypatch,
    env_enabled,
    requires_grad,
    extra_tiles,
):
    if env_enabled:
        monkeypatch.setenv(VSA_SM100A_ENV, "1")
    else:
        monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
    meta = _build_meta()
    n_tiles = meta.variable_block_sizes.numel()
    assert n_tiles % 2 == 1
    raw = torch.randn(1,
                      meta.total_seq_length,
                      _HEADS,
                      _DIM,
                      dtype=torch.bfloat16,
                      requires_grad=requires_grad)

    tiled = MiniMaxH3VSAImpl(num_heads=_HEADS,
                             head_size=_DIM,
                             causal=False,
                             softmax_scale=_DIM**-0.5).preprocess_qkv(raw, meta)

    assert tiled.shape[1] == (n_tiles + extra_tiles) * 64
    assert torch.equal(tiled[:, meta.untile_combined_index], raw)
    if extra_tiles:
        assert torch.count_nonzero(tiled[:, n_tiles * 64:]) == 0


def test_odd_partner_is_visible_only_to_sm100a_transport(routed, monkeypatch):
    fake_sm, fake_triton, _, meta = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    raw_qkv = torch.randn(3, meta.total_seq_length, _HEADS, _DIM, dtype=torch.bfloat16)
    tiled_qkv = impl.preprocess_qkv(raw_qkv, meta)
    query, key, value = tiled_qkv.chunk(3, dim=0)

    output = impl.forward(query, key, value, None, meta)

    n_tiles = meta.variable_block_sizes.numel()
    logical_seq_len = n_tiles * 64
    assert fake_triton.calls == 0 and len(fake_sm.calls) == 1
    assert tiled_qkv.shape[1] == logical_seq_len + 64
    assert torch.count_nonzero(tiled_qkv[:, logical_seq_len:]) == 0
    call = fake_sm.calls[0]
    assert call["q"].shape[2] == logical_seq_len + 64
    assert call["vbs"].numel() == n_tiles + 1
    assert torch.equal(call["vbs"][:-1], meta.variable_block_sizes.to(torch.int32))
    assert call["vbs"][-1].item() == 0
    assert (call["q2k_num"][..., :n_tiles] == n_tiles).all()
    assert (call["q2k_num"][..., n_tiles] == 0).all()
    assert (call["q2k_idx"][..., :n_tiles, n_tiles] == -1).all()
    assert (call["q2k_idx"][..., n_tiles, :] == -1).all()
    assert output.shape == (1, logical_seq_len, _HEADS, _DIM)


def test_unsupported_odd_route_strips_partner_before_triton(routed, monkeypatch):
    fake_sm, _, _, meta = routed
    fake_sm.supported = False
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    monkeypatch.setattr(vsa_h3.logger, "warning_once", lambda *_args, **_kwargs: None)
    calls = []

    def capture_triton(q, k, v, mask, variable_block_sizes):
        calls.append((q.shape, k.shape, v.shape, mask.shape, variable_block_sizes.clone()))
        return q.clone(), None

    monkeypatch.setattr(vsa_h3, "block_sparse_attn_64_bhsd", capture_triton)
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    raw_qkv = torch.randn(3, meta.total_seq_length, _HEADS, _DIM, dtype=torch.bfloat16)
    tiled_qkv = impl.preprocess_qkv(raw_qkv, meta)
    query, key, value = tiled_qkv.chunk(3, dim=0)
    output = impl.forward(query, key, value, None, meta)

    n_tiles = meta.variable_block_sizes.numel()
    assert tiled_qkv.shape[1] == (n_tiles + 1) * 64
    assert output.shape[1] == n_tiles * 64
    assert len(calls) == 1
    q_shape, k_shape, v_shape, mask_shape, sizes = calls[0]
    assert q_shape[2] == k_shape[2] == v_shape[2] == n_tiles * 64
    assert mask_shape[-2:] == (n_tiles, n_tiles)
    assert torch.equal(sizes, meta.variable_block_sizes)


def test_geometry_change_clears_reused_partner_tile(monkeypatch):
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    builder = MiniMaxH3VSAMetadataBuilder()
    even_meta = _build_meta(prefix_segments=(70, 70), builder=builder)
    odd_meta = _build_meta(prefix_segments=(70, 30), builder=builder)
    assert even_meta.variable_block_sizes.numel() == odd_meta.variable_block_sizes.numel() + 1
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)

    even_raw = torch.ones(1, even_meta.total_seq_length, _HEADS, _DIM, dtype=torch.bfloat16)
    even_tiled = impl.preprocess_qkv(even_raw, even_meta)
    allocation = even_tiled.data_ptr()
    assert torch.count_nonzero(even_tiled[:, -64:]) > 0

    odd_raw = torch.randn(1, odd_meta.total_seq_length, _HEADS, _DIM, dtype=torch.bfloat16)
    odd_tiled = impl.preprocess_qkv(odd_raw, odd_meta)
    assert odd_tiled.data_ptr() == allocation
    assert torch.count_nonzero(odd_tiled[:, -64:]) == 0
    assert torch.equal(odd_tiled[:, odd_meta.untile_combined_index], odd_raw)


def test_forward_rejects_non_contract_transport_shapes(routed):
    del routed
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    meta = _build_meta()
    n_tiles = meta.variable_block_sizes.numel()
    too_many = torch.zeros(1, (n_tiles + 2) * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="tiled query has length"):
        impl.forward(too_many, too_many, too_many, None, meta)

    logical = torch.zeros(1, n_tiles * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    partner = torch.zeros(1, (n_tiles + 1) * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="tiled key length"):
        impl.forward(logical, partner, logical, None, meta)
    with pytest.raises(ValueError, match="tiled gate length"):
        impl.forward(logical, logical, logical, partner, meta)


def test_real_sm100a_odd_route_matches_triton_oracle(monkeypatch):
    """Exercise the padded route through the actual GB200 extension."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires a GB200 (sm_100a) compute node")
    if vsa_h3._sm100a is None or not vsa_h3._sm100a._HAS_VSA_SM100A:
        pytest.skip("requires a fastvideo_kernel build containing sm_100a VSA")

    device = torch.device("cuda")
    meta = _build_meta(device=device, sparsity=0.5)
    assert meta.variable_block_sizes.numel() % 2 == 1
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    torch.manual_seed(11)
    raw_qkv = torch.randn(3, meta.total_seq_length, _HEADS, _DIM, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
        triton_tiled = impl.preprocess_qkv(raw_qkv, meta)
        triton_query, triton_key, triton_value = triton_tiled.chunk(3, dim=0)
        triton_output = impl.postprocess_output(
            impl.forward(triton_query, triton_key, triton_value, None, meta), meta)

        def reject_triton(*args, **kwargs):
            raise AssertionError("odd no-grad VSA-H3 unexpectedly fell back to Triton-64")

        monkeypatch.setattr(vsa_h3, "block_sparse_attn_64_bhsd", reject_triton)
        monkeypatch.setenv(VSA_SM100A_ENV, "1")
        sm100a_tiled = impl.preprocess_qkv(raw_qkv, meta)
        sm100a_query, sm100a_key, sm100a_value = sm100a_tiled.chunk(3, dim=0)
        sm100a_output = impl.postprocess_output(
            impl.forward(sm100a_query, sm100a_key, sm100a_value, None, meta), meta)
        torch.cuda.synchronize()

    assert sm100a_tiled.shape[1] == triton_tiled.shape[1] + 64
    assert sm100a_output.shape == triton_output.shape == (1, meta.total_seq_length, _HEADS, _DIM)
    torch.testing.assert_close(sm100a_output.float(), triton_output.float(), atol=0.04, rtol=0.02)
