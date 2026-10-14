# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the FA4 VSA-H3 list construction and call grouping, and of the emulator's MXFP8 quantizer."""
import pytest
import torch

from fastvideo.attention.backends.fa4_vsa_h3 import _groups, _pair_lists


def _dense_lists(limit):
    """Reference: per Q256 pair, the key blocks either tile lists, split by 'both listed and 128 valid'."""
    b, h, two_m, n = limit.shape
    full, partial = [], []
    for p in range(two_m // 2):
        a, c = limit[0, 0, 2 * p], limit[0, 0, 2 * p + 1]
        full.append({j for j in range(n) if a[j] == 128 and c[j] == 128})
        partial.append({j for j in range(n) if (a[j] > 0 or c[j] > 0)} - full[-1])
    return full, partial


def test_pair_lists_split_full_and_masked_blocks():
    torch.manual_seed(0)
    n = 9
    vbs = torch.tensor([128, 128, 77, 128, 128, 5, 128, 128, 0], dtype=torch.int32)
    selected = torch.rand(1, 1, 10, n) < 0.6
    limit = (selected & (vbs > 0)) * vbs
    mask_cnt, mask_idx, full_cnt, full_idx = _pair_lists(limit.to(torch.int32))
    full, partial = _dense_lists(limit)
    for p in range(5):
        assert set(full_idx[0, 0, p, :full_cnt[0, 0, p]].tolist()) == full[p]
        assert set(mask_idx[0, 0, p, :mask_cnt[0, 0, p]].tolist()) == partial[p]
    assert mask_idx.dtype == torch.int32 and full_cnt.dtype == torch.int32


@pytest.mark.parametrize("prefix,expected", [
    (0, [("nvfp4", 0, 11, 0, 11)]),
    (11, [("mxfp8", 0, 11, 0, 11)]),
    (4, [("nvfp4", 4, 11, 4, 11), ("mxfp8", 0, 4, 0, 4)]),
    (5, [("nvfp4", 4, 11, 5, 11), ("mxfp8", 0, 6, 0, 5)]),
])
def test_groups_start_on_pairs_and_own_each_tile_once(prefix, expected):
    groups = _groups("mixmx", 11, prefix)
    assert groups == expected
    owned = sorted(t for _, _, _, o0, o1 in groups for t in range(o0, o1))
    assert owned == list(range(11))
    assert all(t0 % 2 == 0 for _, t0, _, _, _ in groups)


def test_groups_single_format_modes():
    assert _groups("qk", 7, 3) == [("nvfp4", 0, 7, 0, 7)]


def test_emulator_mxfp8_power_of_two_scales():
    from fastvideo.attention.backends.fp4_vsa_qat import load_qat_vsa_modules

    try:
        _, train = load_qat_vsa_modules()
    except ImportError:
        pytest.skip("Triton emulator modules unavailable")
    x = torch.randn(4, 64) * torch.tensor([1e-3, 1.0, 37.0, 900.0])[:, None]
    y = train.mxfp8_fake_quant(x)
    groups = x.view(4, 2, 32)
    amax = groups.abs().amax(-1)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    codes = (y.view(4, 2, 32) / scale[..., None])
    assert torch.equal(codes, codes.to(torch.float8_e4m3fn).float())  # every value is E4M3 x its group scale
    assert ((y - x).norm() / x.norm()) < 0.05
