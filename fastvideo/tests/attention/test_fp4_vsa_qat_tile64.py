# SPDX-License-Identifier: Apache-2.0
"""The FP4 VSA emulator at 64-token tiles: quadrant lists, 128-row padding and the trimmed output.

Needs a CUDA GPU with Triton (the emulator runs the Triton kernel).
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fastvideo.attention.backends.fp4_vsa_qat import (  # noqa: E402
    FP4AttentionNumerics, FP4VSAFineAttention, load_qat_vsa_modules,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU with Triton")


def _problem(n_tiles: int, heads: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    tile_valid = torch.full((n_tiles, ), 64, dtype=torch.int32)
    tile_valid[-1] = 23
    seq_len = int(tile_valid.sum())
    mask = torch.rand(1, heads, n_tiles, n_tiles, generator=gen) < 0.5
    mask[..., 0] = True
    q, k, v = (torch.randn(1, seq_len, heads, 128, generator=gen).to(torch.bfloat16) for _ in range(3))
    metadata = SimpleNamespace(tile_elems=64, variable_block_sizes=tile_valid.cuda(), num_prefix_tiles=0)
    return q.cuda(), k.cuda(), v.cuda(), mask.cuda(), metadata


@pytest.mark.parametrize("quantize", [False, True])
def test_tile64_emulator_matches_the_reference_on_padded_rows(quantize) -> None:
    reference, _ = load_qat_vsa_modules()
    n_tiles, heads = 5, 2
    q, k, v, mask, metadata = _problem(n_tiles, heads, seed=3)
    out = FP4VSAFineAttention(FP4AttentionNumerics(quantize=quantize))(q, k, v, mask, metadata)
    assert out.shape == q.shape and torch.isfinite(out).all()

    q2k_idx, q2k_num, kv_valid, q2k_quad = reference.tile_mask_to_fp4_blocks(mask, 64, metadata.variable_block_sizes)
    padded = q2k_idx.shape[2] * reference.BLOCK
    q_ref, k_ref, v_ref = (torch.nn.functional.pad(t.transpose(1, 2), (0, 0, 0, padded - q.shape[1]))
                           for t in (q, k, v))
    expected = reference.fp4_vsa_attention_reference(q_ref, k_ref, v_ref, q2k_idx, q2k_num, kv_valid, q2k_quad,
                                                     quantize=quantize)
    expected = expected[:, :, :q.shape[1]].transpose(1, 2).float()
    error = (out.float() - expected).norm() / expected.norm()
    # BF16 P and output rounding: the same bounds as fastvideo-kernel/tests/test_attn_qat_vsa_train.py.
    assert error < (2e-3 if quantize else 5e-3)


def test_tile64_rejects_prefix_options_counted_in_128_token_tiles() -> None:
    q, k, v, mask, metadata = _problem(4, 1, seed=4)
    attention = FP4VSAFineAttention(FP4AttentionNumerics(quantize=True, prefix_precision="fp8"))
    with pytest.raises(ValueError, match="128-token prefix tiles"):
        attention(q, k, v, mask, metadata)


def test_stale_installed_kernel_modules_fall_back_to_the_checkout(monkeypatch) -> None:
    import fastvideo.attention.backends.fp4_vsa_qat as fp4_vsa_qat

    def stale_attn(q, k, v, q2k_idx, q2k_num, kv_valid=None, *, quantize=True):
        raise AssertionError("the stale kernel must not be used")

    stale = {
        "attn_qat_vsa_reference": SimpleNamespace(tile128_mask_to_fp4_blocks=None),
        "attn_qat_vsa_train": SimpleNamespace(fp4_vsa_attn_qat=stale_attn),
    }
    monkeypatch.setattr(fp4_vsa_qat, "_load_installed", stale.get)
    reference, train = fp4_vsa_qat.load_qat_vsa_modules()
    assert hasattr(reference, "tile_mask_to_fp4_blocks")
    assert train.fp4_vsa_attn_qat is not stale_attn
