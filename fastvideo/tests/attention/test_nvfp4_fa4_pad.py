# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 quantization needs CUDA")


@pytest.mark.parametrize("seqlen", [77, 300])
def test_fa4_quantize_pads_like_zero_padding(seqlen: int) -> None:
    flash_attn = pytest.importorskip("fastvideo.attention.backends.flash_attn")
    try:
        flash_attn._resolve_fa4_quant_ops()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NVFP4 quantize ops unavailable: {exc}")
    torch.manual_seed(0)
    x = torch.randn(1, seqlen, 4, 128, device="cuda", dtype=torch.bfloat16)
    padded = F.pad(x, (0, 0, 0, 0, 0, (-seqlen) % 128))

    fp4, sf = flash_attn._nvfp4_quantize_for_fa4_impl(x)
    ref_fp4, ref_sf = flash_attn._nvfp4_quantize_for_fa4_impl(padded)

    assert torch.equal(fp4.view(torch.uint8), ref_fp4.view(torch.uint8))
    assert torch.equal(sf, ref_sf)
