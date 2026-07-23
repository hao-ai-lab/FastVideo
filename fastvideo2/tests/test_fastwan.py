"""T0: FastWan-QAD card contract + fp8 formula checks (torch parts skip on
machines without torch — they run on the cluster before GPU work)."""
import pytest

from fastvideo2.registry import resolve
from fastvideo2.wan21 import FASTWAN_QAD_FP8_1_3B, WAN21_T2V_1_3B


def test_qad_card_resolves_and_pins_its_loop():
    card, builder = resolve("fastwan-qad-fp8-1.3b")
    assert card is FASTWAN_QAD_FP8_1_3B
    assert card.provenance.assumes_loop == "wan.dmd.fvmain/v1"
    assert card.provenance.parents == ("wan2.1-t2v-1.3b",)
    assert card.components["transformer"].module.endswith(":WanModelFVFP8")
    assert card.components["text_encoder"].dtype == "fp32"  # main runs fp32 UMT5
    assert card.loops["denoise"].params["timesteps"] == [1000, 757, 522]
    assert card.sampling_defaults.guidance_scale == 1.0  # DMD is CFG-free


def test_base_card_digest_untouched_by_fastwan_addition():
    # the blessed T1 baseline is keyed to this digest; schema or card drift
    # here invalidates human-owned evidence.
    assert WAN21_T2V_1_3B.digest() == "7171d8729ae72ae4"


def test_qad_card_round_trips():
    from fastvideo2.card import ModelCard
    clone = ModelCard.from_dict(FASTWAN_QAD_FP8_1_3B.to_dict())
    assert clone.digest() == FASTWAN_QAD_FP8_1_3B.digest()


def test_fp8_quantize_formula_matches_main():
    torch = pytest.importorskip("torch")
    from fastvideo2.layers.fp8 import FP8_MAX, FP8_MIN_SCALE, quantize_tensorwise
    x = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.bfloat16)
    x_fp8, scale = quantize_tensorwise(x)
    assert x_fp8.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    expected_scale = (x.abs().amax().float() / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    assert torch.equal(scale, expected_scale.view(1))
    # division happens in the INPUT dtype (bf16) before the fp8 cast
    expected = (x / expected_scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    assert torch.equal(x_fp8.view(torch.uint8), expected.view(torch.uint8))


def test_fp8_linear_cpu_dequant_path():
    torch = pytest.importorskip("torch")
    from fastvideo2.layers.fp8 import make_fp8_linear_class, quantize_fp8_
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 4).to(torch.bfloat16)
    q = make_fp8_linear_class()(lin)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    # CPU falls back to main's _apply_dequant: F.linear on dequantized bf16
    w = q._fp8_weight.to(x.dtype) * q._fp8_weight_scale.to(x.dtype).unsqueeze(1)
    expected = torch.nn.functional.linear(x, w, q.bias)
    assert torch.equal(q(x), expected)
    # swap-in-place targets exact module paths, including Sequential indices
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16))
    quantize_fp8_(model, ["0"])
    assert type(model[0]).__name__ == "FP8Linear"
