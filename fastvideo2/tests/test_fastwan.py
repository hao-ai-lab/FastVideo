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
    out = q(x)  # first forward quantizes (device-time, like main's convert)
    assert not hasattr(q, "weight")  # original weight dropped after quantization
    # CPU falls back to main's _apply_dequant: F.linear on dequantized bf16
    w = q._fp8_weight.to(x.dtype) * q._fp8_weight_scale.to(x.dtype).unsqueeze(1)
    expected = torch.nn.functional.linear(x, w, q.bias)
    assert torch.equal(out, expected)
    # swap-in-place targets exact module paths, including Sequential indices
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16))
    quantize_fp8_(model, ["0"])
    assert type(model[0]).__name__ == "FP8Linear"


def test_vsa_card_resolves_and_fails_closed_by_construction():
    card, _ = resolve("fastwan-t2v-1.3b")
    assert card.components["transformer"].module.endswith(":WanModelFVVSA")
    assert card.loops["denoise"].params["vsa_sparsity"] == 0.8
    assert card.provenance.assumes_loop == "wan.dmd.fvmain/v1"
    assert card.digest() != FASTWAN_QAD_FP8_1_3B.digest()


def test_vsa_meta_math_matches_main():
    torch = pytest.importorskip("torch")
    from fastvideo2.layers.vsa import VSA_TILE_SIZE, build_vsa_meta
    assert VSA_TILE_SIZE == (4, 4, 4)
    # grid smaller than one tile: single block of S voxels, identity-ish maps
    m = build_vsa_meta((2, 3, 3), 0.8, "cpu")
    S = 2 * 3 * 3
    assert m.block_sizes.tolist() == [S] and m.topk == 1
    assert sorted(m.tile_index.tolist()) == list(range(S))
    x = torch.arange(S).view(1, S, 1, 1)
    buf = torch.zeros(1, m.padded_len, 1, 1, dtype=x.dtype)
    buf[:, m.non_pad_index] = x[:, m.tile_index]
    assert torch.equal(buf[:, m.untile_index][0, :, 0, 0], torch.arange(S))  # round trip
    # partial tiles along T: (5,4,4) -> T tile sizes [4,1] -> blocks [64,16]
    m2 = build_vsa_meta((5, 4, 4), 0.9, "cpu")
    assert m2.block_sizes.tolist() == [64, 16]
    assert m2.topk == 1  # ceil(0.1*2)=1
    # topk never exceeds block count and never hits 0
    m3 = build_vsa_meta((8, 8, 8), 0.0, "cpu")
    assert m3.topk == m3.block_sizes.numel()


def test_sfwan_card_and_table():
    card, _ = resolve("sfwan-t2v-1.3b")
    assert card.components["transformer"].module.endswith(":WanModelFVCausal")
    assert card.provenance.assumes_loop == "wan.causal_dmd.chunked/v1"
    assert card.loops["denoise"].params["num_frames_per_block"] == 3
    torch = pytest.importorskip("torch")
    from fastvideo2.wan21.loop import self_forcing_table
    ts, sg = self_forcing_table(5.0)
    assert len(ts) == 1000 and float(sg[0]) == 1.0
    # warp rows main feeds the model: table[1000 - t]
    warped = [float(ts[1000 - t]) for t in (1000, 750, 500, 250)]
    assert warped[0] == 1000.0
    assert abs(warped[1] - 937.5) < 1e-3
    assert abs(warped[2] - 833.3333) < 1e-3
    assert abs(warped[3] - 625.0) < 1e-3


def test_causal_loop_plans_denoise_then_context_per_chunk():
    from fastvideo2.wan21.loop import WanCausalDMDLoop
    loop = WanCausalDMDLoop(loop_id="d")
    assert loop._phase(0) == (0, 0)      # chunk0 first denoise step
    assert loop._phase(4) == (0, 4)      # chunk0 context pass (4 timesteps)
    assert loop._phase(5) == (1, 0)      # chunk1 begins
    assert loop._phase(34) == (6, 4)     # last chunk's context pass
