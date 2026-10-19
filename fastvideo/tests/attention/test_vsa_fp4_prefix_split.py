# SPDX-License-Identifier: Apache-2.0
"""The FP4 prefix split's row bookkeeping and dense prefix attention.

The split takes the dense prefix query tiles (text, reference, audio rows) off
the FP4 sparse kernel and attends them densely at 8-bit Q/K. These tests cover
the parts that do not need a Blackwell GPU: the FP8 rounding, the valid-key
vector the kernel's half counts expand to, and the dense attention itself -
checked against the training emulator, which is the numeric spec.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.models.dits.minimax_h3_vsa_fp4 import (_fp8_per_token, _half_row_valid, _prefix_dense_attention,
                                                      _resolve_prefix_mode)

HALF = 64


def _emulator_fp8(x: torch.Tensor) -> torch.Tensor:
    """The emulator's per-token E4M3 fake quant; it needs Triton to import."""
    try:
        from fastvideo_kernel.triton_kernels.attn_qat_vsa_train import fp8_per_token_fake_quant
    except (ImportError, KeyError):  # KeyError: a failed fastvideo_kernel import left a None entry
        pytest.skip("the Triton emulator is unavailable")
    return fp8_per_token_fake_quant(x)


def _reference_dense_attention(q, k, v, valid, scale):
    """fp32 softmax attention over ``[B, L, H, D]`` with the invalid keys masked out."""
    scores = torch.einsum("blhd,bkhd->bhlk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~valid[None, None, None, :], float("-inf"))
    return torch.einsum("bhlk,bkhd->blhd", torch.softmax(scores, dim=-1), v.float()).to(v.dtype)


def _case(rows=1, prefix=8, kv_len=16, heads=2, dim=128, tail_invalid=4):
    torch.manual_seed(0)
    q = torch.randn(rows, prefix, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(rows, kv_len, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(rows, kv_len, heads, dim, dtype=torch.bfloat16)
    valid = torch.ones(kv_len, dtype=torch.bool)
    valid[kv_len - tail_invalid:] = False
    return q, k, v, valid, dim


def test_fp8_per_token_rounds_within_one_e4m3_step() -> None:
    """FP8 E4M3 has 3 mantissa bits, so a per-row scale bounds the error at ~6%."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 5, 128, dtype=torch.bfloat16)
    y = _fp8_per_token(x)
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    assert torch.allclose(y, x, rtol=0.1, atol=1e-2)
    # Per (row, head) scale: a row whose magnitude is 1000x the others keeps its own precision.
    x[0, 0, 0] *= 1000.0
    assert ((_fp8_per_token(x)[0, 0, 0] - x[0, 0, 0]).norm() / x[0, 0, 0].norm()).item() < 0.1


def test_fp8_per_token_keeps_zero_rows_zero() -> None:
    """Pad rows stay exactly zero, which the kernel's zero-padded k/v relies on."""
    x = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(_fp8_per_token(x), x)


def test_fp8_per_token_matches_the_emulator() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 16, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(_fp8_per_token(x), _emulator_fp8(x).to(torch.bfloat16))


def test_half_row_valid_expands_counts_to_rows() -> None:
    kv_valid = torch.tensor([HALF, 3, 0, HALF], dtype=torch.int32)
    valid = _half_row_valid(kv_valid, HALF)
    assert valid.shape == (4 * HALF, )
    assert valid.sum().item() == 2 * HALF + 3
    assert valid[:HALF].all() and not valid[HALF + 3:HALF + HALF].any()
    assert not valid[2 * HALF:3 * HALF].any()


@pytest.mark.parametrize(("mode", "steps", "step", "expected"), [
    ("fp8", -1, 0, "fp8"),
    ("fp8", -1, 7, "fp8"),
    ("bf16", 3, 2, "bf16"),
    ("fp8", 3, 3, ""),
    ("fp8", 0, 0, ""),
    ("", 4, 0, ""),
])
def test_prefix_mode_step_cutoff(mode: str, steps: int, step: int, expected: str) -> None:
    """A non-negative step limit applies the split only to the leading steps."""
    assert _resolve_prefix_mode(mode, steps, step) == expected


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_prefix_dense_attention_matches_the_emulator_numerics(precision: str) -> None:
    """Smooth K, then round it - the order the emulator and the student use."""
    q, k, v, valid, dim = _case()

    out = _prefix_dense_attention(q, k, v, q.shape[1], k.shape[1], valid, precision)

    assert out.shape == (1, 2, 8, dim) and out.dtype == torch.bfloat16
    k_ref = k - k.mean(dim=1, keepdim=True)
    q_ref = q
    if precision == "fp8":
        q_ref, k_ref = _emulator_fp8(q), _emulator_fp8(k_ref)
    expected = _reference_dense_attention(q_ref, k_ref, v, valid, dim**-0.5).transpose(1, 2)

    assert (out.float() - expected.float()).norm() / expected.float().norm() < 2e-2


def test_smoothing_happens_before_the_fp8_rounding() -> None:
    """Rounding before smoothing gives different keys, so the order is load-bearing."""
    q, k, v, valid, _ = _case()
    smooth_then_round = _emulator_fp8(k - k.mean(dim=1, keepdim=True))
    round_then_smooth = _emulator_fp8(k) - _emulator_fp8(k).mean(dim=1, keepdim=True)
    assert not torch.equal(smooth_then_round, round_then_smooth)

    out = _prefix_dense_attention(q, k, v, q.shape[1], k.shape[1], valid, "fp8")
    expected = _reference_dense_attention(q, smooth_then_round, v, valid, k.shape[-1]**-0.5).transpose(1, 2)
    wrong = _reference_dense_attention(q, round_then_smooth, v, valid, k.shape[-1]**-0.5).transpose(1, 2)

    error = (out.float() - expected.float()).norm()
    assert error < (out.float() - wrong.float()).norm()


def test_prefix_dense_attention_ignores_invalid_keys() -> None:
    """Masked-out keys must not reach the output, however extreme they are."""
    torch.manual_seed(0)
    rows, prefix, valid_len, heads, dim = 1, 4, 6, 2, 128
    q = torch.randn(rows, prefix, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(rows, valid_len + 2, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(rows, valid_len + 2, heads, dim, dtype=torch.bfloat16)
    valid = torch.ones(valid_len + 2, dtype=torch.bfloat16).bool()
    valid[valid_len:] = False

    masked = _prefix_dense_attention(q, k, v, prefix, valid_len + 2, valid, "bf16")

    # Truncating to the valid rows changes only the smoothing mean, an exact shift.
    k[:, valid_len:] = 0.0
    v[:, valid_len:] = 0.0
    k = k - (k[:, :valid_len].mean(dim=1, keepdim=True) - k.mean(dim=1, keepdim=True))
    truncated = _prefix_dense_attention(q, k[:, :valid_len], v[:, :valid_len], prefix, valid_len,
                                        torch.ones(valid_len, dtype=torch.bool), "bf16")

    assert (masked.float() - truncated.float()).norm() / truncated.float().norm() < 5e-3


@pytest.mark.parametrize("tile,prefix_tiles,n_tiles", [(64, 1, 5), (64, 2, 5), (64, 3, 6), (128, 1, 3)])
@pytest.mark.parametrize("heads", [1, 4])
def test_prefix_split_keeps_original_video_rows_and_quadrants(monkeypatch, tile, prefix_tiles, n_tiles, heads):
    from types import SimpleNamespace
    from fastvideo.models.dits import minimax_h3_vsa_fp4 as impl

    logical = tile * n_tiles
    padded = (logical + 127) // 128 * 128
    prefix_rows = prefix_tiles * tile
    head = prefix_rows // 128
    query = torch.arange(padded).view(1, padded, 1, 1).expand(1, padded, heads, 8).float()
    key, value = torch.randn_like(query), torch.randn_like(query)
    blocks = padded // 128
    indices = torch.arange(heads * blocks * blocks, dtype=torch.int32).view(1, heads, blocks, blocks)
    counts = torch.ones(1, heads, blocks, dtype=torch.int32)
    quadrants = (indices % 15 + 1).to(torch.uint8) if tile == 64 else None
    valid = torch.full((padded // 64,), 64, dtype=torch.int32)
    valid[logical // 64 - 1] = 17
    valid[logical // 64:] = 0
    calls = []

    def native(q, k, v, idx, num, kv, quad, **kwargs):
        calls.append(q.shape[1])
        assert q.shape[1] % 128 == 0
        assert k is key and v is value and kv is valid
        assert torch.equal(q, query[:, head * 128:])
        assert torch.equal(idx, indices[:, :, head:]) and idx.is_contiguous()
        assert torch.equal(num, counts[:, :, head:]) and num.is_contiguous()
        if quadrants is None:
            assert quad is None
        else:
            assert torch.equal(quad, quadrants[:, :, head:]) and quad.is_contiguous()
        return q.transpose(1, 2)

    monkeypatch.setattr(impl, "_sparse_fp4_attention", lambda api, *args, **kwargs: native(*args, **kwargs))
    api = SimpleNamespace(KEY_LENGTH_SIZING=True, HALF_N=64)
    result = impl._fp4_fine_attention(api, query, key, value, indices, counts, valid, quadrants,
                                      logical=logical, prefix_rows=prefix_rows, precision="bf16", options={})
    expected_prefix = impl._prefix_dense_attention(query, key, value, prefix_rows, logical,
                                                    impl._half_row_valid(valid, 64), "bf16")
    torch.testing.assert_close(result[:, :, :prefix_rows], expected_prefix)
    torch.testing.assert_close(result[:, :, prefix_rows:], query[:, prefix_rows:logical].transpose(1, 2))
    assert result.shape == (1, heads, logical, 8)
    assert calls == [padded - head * 128]


@pytest.mark.parametrize("tile", [64, 128])
def test_explicit_prefix_precision_overrides_environment(tile):
    from types import SimpleNamespace
    from fastvideo import envs
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _prefix_settings

    meta = SimpleNamespace(tile_elems=tile, num_prefix_tiles=3, current_timestep=0)
    with envs.FASTVIDEO_H3_VSA_FP4_PREFIX.override(""), envs.FASTVIDEO_H3_VSA_FP4_PREFIX_STEPS.override(-1):
        assert _prefix_settings(SimpleNamespace(_vsa_fp4_options={"prefix_precision": "fp8"}), meta, 5) == (
            "fp8", 3 * tile)
        assert _prefix_settings(SimpleNamespace(), meta, 5) == ("", 0)
    with envs.FASTVIDEO_H3_VSA_FP4_PREFIX.override("fp8"), envs.FASTVIDEO_H3_VSA_FP4_PREFIX_STEPS.override(0):
        assert _prefix_settings(SimpleNamespace(), meta, 5) == ("", 0)


def test_explicit_video_numerics_override_environment():
    from types import SimpleNamespace
    from fastvideo import envs
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _sparse_fp4_attention

    key = torch.randn(1, 128, 2, 8)
    captured = []
    api = SimpleNamespace(sageattn_blackwell_sparse_bshd=lambda *args, **kwargs: captured.append((args, kwargs)))
    with envs.FASTVIDEO_H3_VSA_FP4_K_SMOOTH.override(False), envs.FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P.override(False):
        _sparse_fp4_attention(api, key, key, key, two_level_p=True, smooth_k=True)
    torch.testing.assert_close(captured[0][0][1], key - key.mean(dim=1, keepdim=True))
    assert captured[0][1] == {"validate": False, "single_level_p_quant": False}
    captured.clear()
    with envs.FASTVIDEO_H3_VSA_FP4_K_SMOOTH.override(True), envs.FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P.override(True):
        _sparse_fp4_attention(api, key, key, key, two_level_p=False, smooth_k=False)
    assert captured[0][0][1] is key
    assert captured[0][1] == {"validate": False}


def test_missing_native_extension_reports_required_build(monkeypatch):
    import builtins
    from fastvideo.models.dits import minimax_h3_vsa_fp4 as impl

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "attn_qat_infer.api":
            raise ImportError("test missing extension")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(impl, "_fp4_api", None)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="Build fastvideo-kernel"):
        impl._api()


@pytest.mark.parametrize("tile,prefix_tiles,n_tiles", [(64, 1, 5), (64, 3, 6), (128, 1, 3)])
def test_native_prefix_split_preserves_video_attention(tile, prefix_tiles, n_tiles):
    """Requires consumer Blackwell and the native sparse FP4 extension from this checkout."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires sm_120 or sm_121 native FP4 attention")
    api = pytest.importorskip("attn_qat_infer.api")
    from fastvideo.models.dits import minimax_h3_vsa_fp4 as impl

    torch.manual_seed(1234)
    logical, prefix_rows = n_tiles * tile, prefix_tiles * tile
    padded = (logical + 127) // 128 * 128
    query, key, value = [torch.randn(1, padded, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    valid = torch.full((n_tiles,), tile, device="cuda", dtype=torch.int32)
    valid[-1] = tile // 2
    mask = torch.eye(n_tiles, device="cuda", dtype=torch.bool).expand(1, 4, -1, -1).clone()
    mask[:, :, :prefix_tiles] = True
    mask[:, :, :, :prefix_tiles] = True
    lists = api.vsa_tile_mask_to_fp4_blocks(mask, tile, valid)
    options = {"two_level_p": True, "smooth_k": True}
    baseline = impl._fp4_fine_attention(api, query, key, value, *lists, logical=logical, prefix_rows=0,
                                        precision="", options=options)
    split = impl._fp4_fine_attention(api, query, key, value, *lists, logical=logical, prefix_rows=prefix_rows,
                                     precision="fp8", options=options)
    assert split.isfinite().all()
    torch.testing.assert_close(split[:, :, prefix_rows:], baseline[:, :, prefix_rows:logical], rtol=0, atol=0)
    expected = impl._prefix_dense_attention(query, key, value, prefix_rows, logical,
                                             impl._half_row_valid(lists[2], 64), "fp8")
    torch.testing.assert_close(split[:, :, :prefix_rows], expected, rtol=0, atol=0)
