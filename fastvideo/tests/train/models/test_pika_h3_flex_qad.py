# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Pika H3 Flex QAD profile: ladder, re-noise determinism, contract, adapter merge."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import scheduler_step
from fastvideo.train.models.minimax_h3.flex import (CONTRACT_FILE, CONTRACT_SCHEMA, FlexContract,
                                                    MiniMaxH3FlexRef2VAModel, RowRenoiseScheduler, flex_sigmas,
                                                    renoise_seed)

# Pika's published per-modality ladders (knots 1000, 750, 500, 250, 100, 40, 0 shifted by 12 / 3).
PIKA_VIDEO_SIGMAS = (1.0, 0.97297, 0.92308, 0.8, 0.57143, 0.33333, 0.0)
PIKA_AUDIO_SIGMAS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.11111, 0.0)
STEPS = (1000, 750, 500, 250, 100, 40)


def _merge_module():
    path = Path(__file__).resolve().parents[4] / "scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py"
    spec = importlib.util.spec_from_file_location("merge_pika_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CPUFlexModel(MiniMaxH3FlexRef2VAModel):
    device = torch.device("cpu")


def _model_stub() -> MiniMaxH3FlexRef2VAModel:
    model = object.__new__(_CPUFlexModel)
    model.contract = FlexContract("ref2va", "transformer_ref", STEPS, 12.0, 3.0)
    return model


def _row(seed: int) -> SimpleNamespace:
    return SimpleNamespace(spec=SimpleNamespace(seed=seed))


def test_flex_sigmas_match_pika_ladders():
    torch.testing.assert_close(flex_sigmas(STEPS, 12.0), torch.tensor(PIKA_VIDEO_SIGMAS), atol=1e-5, rtol=0)
    torch.testing.assert_close(flex_sigmas(STEPS, 3.0), torch.tensor(PIKA_AUDIO_SIGMAS), atol=1e-5, rtol=0)


def test_schedulers_expose_six_forwards_in_clean_time():
    video, audio = _model_stub().schedulers(_row(7))
    assert video.timesteps.numel() == audio.timesteps.numel() == 6
    torch.testing.assert_close(video.timesteps, 1 - torch.tensor(PIKA_VIDEO_SIGMAS[:-1]), atol=1e-5, rtol=0)
    assert video.stochastic_sampling and audio.stochastic_sampling


def test_schedulers_require_the_row():
    with pytest.raises(ValueError, match="prepared row"):
        _model_stub().schedulers()


def test_renoise_step_is_x0_plus_fixed_fresh_noise():
    video, _ = _model_stub().schedulers(_row(11))
    state, velocity = torch.randn(5, 4), torch.randn(5, 4)
    rung = 2
    out = scheduler_step(video, rung, velocity, state)
    timestep = video.timesteps[rung]
    x0 = state + (1 - timestep) * velocity
    generator = torch.Generator().manual_seed(renoise_seed(11, rung, 0))
    noise = torch.randn(state.shape, generator=generator, dtype=torch.float32)
    sigma_next = video.sigmas[rung + 1]
    torch.testing.assert_close(out, (1 - sigma_next) * x0 + sigma_next * noise)


def test_renoise_noise_depends_only_on_row_forward_and_modality():
    """Teacher-forced pass, rollout and re-evaluation of one row must draw identical noise at each forward."""
    state, velocity = torch.randn(6, 3), torch.randn(6, 3)
    first = _model_stub().schedulers(_row(3))
    second = _model_stub().schedulers(_row(3))
    # Out-of-order use (a rollout after a full teacher-forced pass) still reproduces forward 1.
    for rung in range(5):
        scheduler_step(first[0], rung, velocity, state)
    torch.testing.assert_close(scheduler_step(first[0], 1, velocity, state),
                               scheduler_step(second[0], 1, velocity, state))
    assert not torch.equal(scheduler_step(first[0], 1, velocity, state), scheduler_step(first[0], 2, velocity,
                                                                                          state))
    other_row = _model_stub().schedulers(_row(4))
    assert not torch.equal(scheduler_step(first[0], 1, velocity, state),
                           scheduler_step(other_row[0], 1, velocity, state))
    assert renoise_seed(3, 1, 0) != renoise_seed(3, 1, 1)


def test_last_forward_returns_x0_without_noise():
    video, audio = _model_stub().schedulers(_row(5))
    state, velocity = torch.randn(4, 2), torch.randn(4, 2)
    for scheduler in (video, audio):
        out = scheduler_step(scheduler, 5, velocity, state)
        torch.testing.assert_close(out, state + (1 - scheduler.timesteps[5]) * velocity)


def test_row_renoise_scheduler_refuses_an_implicit_index():
    scheduler = RowRenoiseScheduler(12.0, 0, 0)
    scheduler.set_timesteps(sigmas=flex_sigmas(STEPS, 12.0))
    with pytest.raises(ValueError, match="explicit step index"):
        scheduler.step(torch.zeros(2, 2), scheduler.timesteps[0], torch.zeros(2, 2))


def _write_contract(path: Path, **overrides) -> None:
    data = {
        "schema": CONTRACT_SCHEMA,
        "task": "ref2va",
        "trunk": "transformer_ref",
        "denoising_steps": list(STEPS),
        "video_scheduler_shift": 12.0,
        "audio_scheduler_shift": 3.0,
        "sampler": "renoise",
    }
    data.update(overrides)
    (path / CONTRACT_FILE).write_text(json.dumps(data))


def test_contract_reads_the_released_ladder(tmp_path):
    _write_contract(tmp_path)
    assert FlexContract.read(str(tmp_path)) == FlexContract("ref2va", "transformer_ref", STEPS, 12.0, 3.0)


@pytest.mark.parametrize("overrides", [{
    "denoising_steps": [1000, 500, 0]
}, {
    "video_scheduler_shift": 10.0
}, {
    "trunk": "transformer"
}, {
    "sampler": "euler"
}])
def test_contract_rejects_anything_but_the_release(tmp_path, overrides):
    _write_contract(tmp_path, **overrides)
    with pytest.raises(ValueError):
        FlexContract.read(str(tmp_path))


def test_contract_must_match_the_profile_trunk(tmp_path):
    _write_contract(tmp_path, task="fl2va", trunk="transformer")
    with pytest.raises(ValueError, match="needs ref2va on transformer_ref"):
        _model_stub()._read_contract(str(tmp_path))


def _reference_module():
    from fastvideo.attention.backends.fp4_vsa_qat import _load_checkout
    return _load_checkout("attn_qat_vsa_reference")


class _ExactKernel:
    """Stand-in for ``attn_qat_vsa_train``: exact softmax over the listed (here: all) blocks and valid keys."""

    def __init__(self, reference):
        self.reference = reference
        self.calls = []

    def fp4_vsa_attn_qat(self, q, k, v, q2k_idx, q2k_num, kv_valid, **numerics):
        self.calls.append((q.shape, numerics))
        n_blocks = q.shape[2] // 128
        assert bool((q2k_num == n_blocks).all())
        keys = self.reference.key_valid_mask(kv_valid, n_blocks, q.device).reshape(-1)
        scores = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1]**0.5
        scores = scores.masked_fill(~keys[None, None, None, :], float("-inf"))
        return (scores.softmax(-1) @ v.float()).to(torch.bfloat16)


def test_dense_block_lists_cover_every_key_tile():
    from fastvideo.attention.backends.fp4_dense_qat import dense_block_lists
    reference = _reference_module()
    q2k_idx, q2k_num, kv_valid = dense_block_lists(300, 1, 2, reference, torch.device("cpu"))
    assert q2k_num.shape == (1, 2, 3) and bool((q2k_num == 3).all())
    assert sorted(q2k_idx[0, 0, 0].tolist()) == [0, 1, 2]
    assert kv_valid.tolist() == [64, 64, 64, 64, 44, 0]


def test_fp4_dense_attention_pads_to_tiles_and_matches_dense_attention():
    from fastvideo.attention.backends.fp4_dense_qat import FP4DenseAttention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics
    reference = _reference_module()
    kernel = _ExactKernel(reference)
    attention = FP4DenseAttention(FP4AttentionNumerics(quantize=True, quantize_pv=False), (reference, kernel))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 200, 2, 16, dtype=torch.bfloat16) for _ in range(3))
    out = attention(q, k, v, None)
    expected = torch.nn.functional.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)))
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out.float(), expected.transpose(1, 2), atol=2e-2, rtol=2e-2)
    shape, numerics = kernel.calls[0]
    assert shape == (1, 2, 256, 16) and numerics["quantize_pv"] is False and numerics["quantize"] is True


def test_install_fp4_dense_attention_overrides_and_restores_flash_attention():
    from fastvideo.attention.backends.fp4_dense_qat import install_fp4_dense_attention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics
    try:
        from fastvideo.attention.backends.flash_attn import FlashAttentionImpl
    except (ImportError, RuntimeError) as error:  # FlashAttention is a CUDA-host dependency
        pytest.skip(f"FlashAttention backend not importable here: {error}")

    class Layer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.distributed_attention = SimpleNamespace(attn_impl=object.__new__(FlashAttentionImpl))

    model = torch.nn.Sequential(Layer(), Layer())
    impls = [layer.distributed_attention.attn_impl for layer in model]
    originals = [impl.forward for impl in impls]
    reference = _reference_module()
    kernel = _ExactKernel(reference)
    assert install_fp4_dense_attention(model, FP4AttentionNumerics(quantize_pv=False), (reference, kernel)) == 2
    q = torch.randn(1, 130, 1, 8, dtype=torch.bfloat16)
    impls[0].forward(q, q, q, None)
    assert len(kernel.calls) == 1
    install_fp4_dense_attention(model, None)
    assert all(impl.forward == original for impl, original in zip(impls, originals, strict=True))


def test_install_fp4_dense_attention_covers_sdpa_fallback():
    """Without flash-attn the FLASH_ATTN selection resolves to SDPA; the emulator must still be installed."""
    from fastvideo.attention.backends.fp4_dense_qat import install_fp4_dense_attention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics
    from fastvideo.attention.backends.sdpa import SDPAImpl

    class Layer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.distributed_attention = SimpleNamespace(attn_impl=SDPAImpl(1, 8, causal=False, softmax_scale=8**-0.5))

    model = torch.nn.Sequential(Layer(), Layer())
    impls = [layer.distributed_attention.attn_impl for layer in model]
    originals = [impl.forward for impl in impls]
    reference = _reference_module()
    kernel = _ExactKernel(reference)
    assert install_fp4_dense_attention(model, FP4AttentionNumerics(quantize_pv=False), (reference, kernel)) == 2
    q = torch.randn(1, 130, 1, 8, dtype=torch.bfloat16)
    assert impls[1].forward(q, q, q, None).shape == q.shape
    assert len(kernel.calls) == 1
    with pytest.raises(NotImplementedError, match="packed"):
        impls[0].forward(q, q, q, SimpleNamespace(packed_sequence=object()))
    install_fp4_dense_attention(model, None)
    assert all(impl.forward == original for impl, original in zip(impls, originals, strict=True))


@pytest.mark.parametrize("name", ["smooth_k", "hadamard", "smooth_k+hadamard"])
def test_precondition_qk_leaves_attention_unchanged(name):
    from fastvideo.attention.utils.nvfp4_qk_precondition import parse_qk_transform, precondition_qk
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 37, 3, 16, dtype=torch.float32) for _ in range(3))
    k = k + 4.0  # shared K bias that smoothing removes
    smooth, hadamard = parse_qk_transform(name)
    tq, tk = precondition_qk(q, k, smooth_k=smooth, hadamard=hadamard)

    def attend(a, b):
        return torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (a, b, v))).transpose(1, 2)

    torch.testing.assert_close(attend(tq, tk), attend(q, k), atol=1e-5, rtol=1e-5)
    if smooth:
        assert float(tk.mean(dim=1).abs().max()) < 1e-4


def test_parse_qk_transform_rejects_unknown_names():
    from fastvideo.attention.utils.nvfp4_qk_precondition import parse_qk_transform
    assert parse_qk_transform(None) == (False, False)
    with pytest.raises(ValueError, match="unknown NVFP4 Q/K transform"):
        parse_qk_transform("rotate")


def test_fp4_dense_attention_with_qk_transform_matches_dense_attention():
    from fastvideo.attention.backends.fp4_dense_qat import FP4DenseAttention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics
    reference = _reference_module()
    kernel = _ExactKernel(reference)
    attention = FP4DenseAttention(FP4AttentionNumerics(quantize=True, quantize_pv=False), (reference, kernel),
                                  qk_transform=(True, True))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 200, 2, 16, dtype=torch.bfloat16) for _ in range(3))
    out = attention(q, k, v, None)
    expected = torch.nn.functional.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)))
    torch.testing.assert_close(out.float(), expected.transpose(1, 2), atol=3e-2, rtol=3e-2)


def test_install_fp4_dense_attention_keeps_bf16_blocks_on_original_forward():
    from fastvideo.attention.backends.fp4_dense_qat import block_index, install_fp4_dense_attention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics
    from fastvideo.attention.backends.sdpa import SDPAImpl

    class Block(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.distributed_attention = SimpleNamespace(
                attn_impl=SDPAImpl(1, 8, causal=False, softmax_scale=8**-0.5))

    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([Block(), Block(), Block()])
    impls = [block.attn.distributed_attention.attn_impl for block in model.transformer_blocks]
    originals = [impl.forward for impl in impls]
    reference = _reference_module()
    kernel = _ExactKernel(reference)
    numerics = FP4AttentionNumerics(quantize_pv=False)
    assert block_index("transformer_blocks.12._checkpoint_wrapped_module.attn") == 12
    assert block_index("token_refiner.blocks.0.attn") is None
    assert install_fp4_dense_attention(model, numerics, (reference, kernel), bf16_blocks=(2, )) == 2
    assert impls[2].forward == originals[2] and impls[0].forward != originals[0]
    q = torch.randn(1, 130, 1, 8, dtype=torch.bfloat16)
    impls[2].forward(q, q, q, None)
    assert kernel.calls == []
    impls[1].forward(q, q, q, None)
    assert len(kernel.calls) == 1
    with pytest.raises(ValueError, match="match no transformer_blocks"):
        install_fp4_dense_attention(model, numerics, (reference, kernel), bf16_blocks=(7, ))


def _write_trunk(trunk: Path, weights: dict[str, torch.Tensor]) -> None:
    trunk.mkdir(parents=True)
    names = sorted(weights)
    shards = {"model-00001-of-00002.safetensors": names[:2], "model-00002-of-00002.safetensors": names[2:]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({key: weights[key] for key in keys}, str(trunk / shard))
        weight_map.update({key: shard for key in keys})
    (trunk / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (trunk / "config.json").write_text("{}")


def test_merge_trunk_adds_scaled_lora_in_fp32(tmp_path):
    merge = _merge_module()
    torch.manual_seed(0)
    weights = {
        "blocks.0.attn.to_q.weight": torch.randn(6, 4).to(torch.bfloat16),
        "blocks.0.attn.to_q.bias": torch.randn(6).to(torch.bfloat16),
        "blocks.0.ff.net.2.weight": torch.randn(4, 8).to(torch.bfloat16),
        "proj_out.weight": torch.randn(3, 4),
    }
    factors = {
        "blocks.0.attn.to_q": (torch.randn(2, 4), torch.randn(6, 2)),
        "blocks.0.ff.net.2": (torch.randn(2, 8), torch.randn(4, 2)),
    }
    _write_trunk(tmp_path / "base", weights)
    report = merge.merge_trunk(tmp_path / "base", tmp_path / "out", factors, scale=0.5)
    assert report["modules"] == 2
    from safetensors.torch import load_file
    merged = {}
    for shard in (tmp_path / "out").glob("*.safetensors"):
        merged.update(load_file(str(shard)))
    for module, (lora_a, lora_b) in factors.items():
        key = f"{module}.weight"
        expected = (weights[key].float() + 0.5 * lora_b @ lora_a).to(torch.bfloat16)
        assert merged[key].dtype == torch.bfloat16
        assert torch.equal(merged[key], expected)
    assert torch.equal(merged["blocks.0.attn.to_q.bias"], weights["blocks.0.attn.to_q.bias"])
    assert torch.equal(merged["proj_out.weight"], weights["proj_out.weight"])


def test_merge_trunk_refuses_adapter_modules_without_weights(tmp_path):
    merge = _merge_module()
    _write_trunk(tmp_path / "base", {f"w{i}.weight": torch.zeros(2, 2) for i in range(3)})
    with pytest.raises(ValueError, match="no weight"):
        merge.merge_trunk(tmp_path / "base", tmp_path / "out", {"missing": (torch.zeros(1, 2), torch.zeros(2, 1))},
                          scale=1.0)


def test_adapter_factors_pair_peft_keys(tmp_path):
    merge = _merge_module()
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "a.to_q.lora_A.default.weight": torch.zeros(2, 4),
            "a.to_q.lora_B.default.weight": torch.zeros(6, 2),
        }, str(path))
    factors = merge.adapter_factors(str(path))
    assert list(factors) == ["a.to_q"] and factors["a.to_q"][0].shape == (2, 4)
    save_file({"a.to_q.lora_A.default.weight": torch.zeros(2, 4)}, str(path))
    with pytest.raises(ValueError, match="both LoRA factors"):
        merge.adapter_factors(str(path))


def _calibrate_module():
    path = Path(__file__).resolve().parents[4] / "scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py"
    spec = importlib.util.spec_from_file_location("calibrate_flex_nvfp4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_keys_follow_the_qad_table():
    calibrate = _calibrate_module()
    assert calibrate.amax_key("transformer.transformer_blocks.7.ff.fc_out") == "b7.ff.fc_out"
    assert calibrate.amax_key("transformer_blocks.3._checkpoint_wrapped_module.ff.fc_in") == "b3.ff.fc_in"
    assert calibrate.amax_key("transformer_blocks.3.attn.to_q") is None


def test_calibration_candidates_and_outliers():
    calibrate = _calibrate_module()
    rows = [{"id": f"r{i}", "amax": {"b0.ff.fc_in": [float(i), 0.0, 1.0], "b0.ff.fc_out": [1.0, 2.0, 3.0]}}
            for i in range(1, 101)]
    rows[0]["amax"]["b0.ff.fc_out"] = [50.0, 0.0, 0.0]
    table = calibrate.candidate_table(rows)
    assert table["b0.ff.fc_in"]["max"] == 100.0
    assert table["b0.ff.fc_in"]["0.5x_max"] == 50.0
    assert table["b0.ff.fc_in"]["row_p90"] == pytest.approx(90.1)
    assert set(calibrate.outlier_row_ids(rows, 2)) == {"r1", "r100"}



def test_calibration_candidates_accept_a_single_row(tmp_path):
    import argparse
    calibrate = _calibrate_module()
    row = {"id": "r1", "amax": {"b0.ff.fc_in": [2.0, 1.0, 3.0]}}
    (tmp_path / "collect-shard0-dp0.jsonl").write_text(json.dumps(row) + "\n")
    calibrate.run_candidates(argparse.Namespace(output_dir=str(tmp_path), outlier_rows=8))
    assert json.loads((tmp_path / "amax-max.json").read_text()) == {"b0.ff.fc_in": 3.0}

def test_calibration_choice_all_tokens_or_balanced():
    calibrate = _calibrate_module()
    candidates = {"b0.ff.fc_in": {"max": 10.0, "0.5x_max": 5.0}}
    # "max" wins on all-token error (video dominates) but loses badly on audio; the balanced objective picks the clip.
    mse = {"b0.ff.fc_in": {"ref": [100.0, 1.0, 10.0], "max": [1.0, 0.5, 0.1], "0.5x_max": [2.0, 0.05, 0.1]}}
    table, report = calibrate.choose_table(mse, candidates, "balanced")
    assert table == {"b0.ff.fc_in": 5.0}
    assert report["b0.ff.fc_in"]["best"] == "0.5x_max"
    table, _ = calibrate.choose_table(mse, candidates, "all")
    assert table == {"b0.ff.fc_in": 10.0}


def _builder_module():
    path = Path(__file__).resolve().parents[4] / "scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py"
    spec = importlib.util.spec_from_file_location("build_flex_ref2va_rows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROMPT = "subject_definitions: <Picture 1> is the exact first frame of the target video.\n\nsummary: A dog runs."
_ASSET_A = "asset-" + "a" * 32


def _entry(case: str, clip: str, parquet: str, resolution: str = "480p") -> dict:
    entry_id = f"{case}:{clip}" if case in ("first_frame", "first_last_frame", "storyboard") else clip
    return {"case": case, "id": entry_id, "parquet": parquet, "resolution": resolution}


def test_builder_adds_subject_definitions_in_order():
    builder = _builder_module()
    after = builder._subject(_PROMPT, "<Audio 1> is the soundtrack of the target video.")
    assert after.startswith("subject_definitions: <Picture 1> is the exact first frame of the target video. <Audio 1>")
    assert after.endswith("\n\nsummary: A dog runs.")
    before = builder.apply_prompt_edits(_PROMPT, [("<Picture 1> is the first frame.", True)])
    assert before.startswith("subject_definitions: <Picture 1> is the first frame. <Picture 1> is the exact")
    assert builder.source_row_id("storyboard", "c") == "storyboard:c" and builder.source_row_id("continue_shot", "c") == "c"


def test_builder_new_cases_from_stored_rows():
    import random as stdlib_random
    from collections import defaultdict
    builder = _builder_module()
    counts = {"ff.pq": 1, "sb.pq": 3, "ff2.pq": 1, "sb2.pq": 5, "cs.pq": 1, "cs2.pq": 1}
    first = _entry("first_frame", f"{_ASSET_A}_f1_f124_x", "ff.pq")
    board = _entry("storyboard", f"{_ASSET_A}_f1_f124_x", "sb.pq")
    scene = _entry("continue_scene", f"{_ASSET_A}_omniref_continue_1", "cs.pq")
    shot = _entry("continue_shot", f"{_ASSET_A}_omniref_continue_2", "cs2.pq")
    by_asset = defaultdict(list)
    by_asset[(_ASSET_A, "480p", "image")] = [first, board, _entry("first_frame", f"{_ASSET_A}_f9_x", "ff2.pq"),
                                            _entry("storyboard", f"{_ASSET_A}_f9_x", "sb2.pq")]
    by_asset[(_ASSET_A, "480p", "continue")] = [scene, shot]
    rng = stdlib_random.Random(0)
    audio = builder.new_case_spec("audio_reference", first, rng, by_asset, counts.get)
    assert [r["kind"] for r in audio["references"]] == ["stored", "target_audio"]
    assert audio["id"] == f"audio_reference:{_ASSET_A}_f1_f124_x" and audio["source_case"] == "first_frame"
    mixed = builder.new_case_spec("mixed_image_video", scene, rng, by_asset, counts.get)
    assert [r["kind"] for r in mixed["references"]] == ["target_frame", "stored"] and mixed["prompt_edits"][0][1]
    many = builder.new_case_spec("many_images", board, rng, by_asset, counts.get)
    assert 6 <= len(many["references"]) <= 9
    assert {r["parquet"] for r in many["references"][3:]} <= {"ff.pq", "ff2.pq", "sb2.pq"}
    assert len(many["prompt_edits"]) == len(many["references"]) - 3
    multi = builder.new_case_spec("multi_video", scene, rng, by_asset, counts.get)
    assert [r["parquet"] for r in multi["references"]] == ["cs.pq", "cs2.pq"] and multi["references"][1]["audio"] is False
    lonely = defaultdict(list, {(_ASSET_A, "480p", "continue"): [scene]})
    assert builder.new_case_spec("multi_video", scene, rng, lonely, counts.get) is None


def test_builder_plan_retiers_only_480p_and_keeps_ids():
    builder = _builder_module()
    first = _entry("first_frame", f"{_ASSET_A}_f1_x", "a.pq")
    hi = _entry("first_frame", f"{_ASSET_A}_f1_x", "b.pq", "768p")
    src_rows = {(first["id"], "480p"): first, (hi["id"], "768p"): hi}
    specs = builder.plan_rows(src_rows, {(first["id"], "480p"), (hi["id"], "768p")}, {}, 0, set(), lambda _: 1)
    assert len(specs) == 1 and specs[0]["id"] == first["id"] and specs[0]["case"] == "first_frame"
    assert specs[0]["references"] == [{"kind": "stored", "parquet": "a.pq", "index": 0, "audio": True}]



def test_builder_plan_keeps_held_out_siblings_out_of_references():
    builder = _builder_module()
    counts = {"sb.pq": 3, "ok.pq": 9, "held.pq": 9}
    board = _entry("storyboard", f"{_ASSET_A}_f1_x", "sb.pq")
    allowed = _entry("first_frame", f"{_ASSET_A}_f2_x", "ok.pq")
    held_out = _entry("first_frame", f"{_ASSET_A}_f9_x", "held.pq")
    src_rows = {(entry["id"], "480p"): entry for entry in (board, allowed, held_out)}
    specs = builder.plan_rows(src_rows, set(), {"480p": 1}, 0, {f"{_ASSET_A}_f9_x"}, counts.get)
    assert any(row["case"] == "many_images" for row in specs)
    assert all(ref.get("parquet") != "held.pq" for row in specs for ref in row["references"])

def test_fl2va_prompt_keeps_the_description_and_defines_the_keyframes():
    builder = _builder_module()
    caption = ("subject_definitions: <Picture 1> is the exact first frame.\n\nsummary: A dog runs.\n\n"
               "retention_analysis: <Picture 1> is preserved.\n\noverall_soundscape: barking")
    assert builder.fl2va_prompt(caption, "t2va") == "summary: A dog runs.\n\noverall_soundscape: barking"
    last = builder.fl2va_prompt(caption, "l2va")
    assert last.startswith("subject_definitions: <Picture 1> is the exact last frame of the target video.\n\n")
    assert "retention_analysis" not in last
    both = builder.fl2va_prompt(caption, "fl2va")
    assert "<Picture 2> is the exact last frame" in both.split("\n\n")[0]


def test_fl2va_plan_takes_keyframes_from_the_matching_source_case():
    builder = _builder_module()
    src_rows = {}
    for n in range(10):
        for case in ("first_frame", "first_last_frame"):
            for res in ("480p", "768p"):
                entry = _entry(case, f"{_ASSET_A}_f{n}_x", f"{case}-{n}-{res}.pq", res)
                src_rows[(entry["id"], res)] = entry
    specs = builder.plan_fl2va_rows(src_rows, {"t2va": 5, "i2va": 5, "l2va": 5, "fl2va": 5}, {"480p": 0.8, "768p": 0.2},
                                    0, set())
    by_case = {}
    for row in specs:
        by_case.setdefault(row["case"], []).append(row)
    assert {case: len(rows) for case, rows in by_case.items()} == {"t2va": 5, "i2va": 5, "l2va": 5, "fl2va": 5}
    assert all(not r["keyframes"] for r in by_case["t2va"])
    assert all([k["index"] for k in r["keyframes"]] == [1] and "first_last_frame" in r["base_parquet"]
               for r in by_case["l2va"])
    assert all([k["index"] for k in r["keyframes"]] == [0, 1] for r in by_case["fl2va"])
    assert all(r["id"].startswith(f"{r['case']}:") for r in specs)
    assert sum(r["resolution"] == "768p" for r in by_case["i2va"]) == 1


def test_fl2va_row_packs_keyframe_anchors():
    from types import SimpleNamespace
    import numpy as np
    from fastvideo.train.models.minimax_h3.flex import prepare_fl2va_row
    lat_h, lat_w, frames = 30, 52, 124
    rows_per_frame = (lat_h // 2) * (lat_w // 2)

    def row(n_keyframes):
        def put(name, array):
            return {f"{name}_bytes": array.astype(np.float32).tobytes(), f"{name}_shape": list(array.shape)}
        out = {"num_frames": frames, "height": lat_h * 16, "width": lat_w * 16,
               "reference_media_types": ["image"] * n_keyframes, "reference_has_audio": [False] * n_keyframes,
               "reference_num_latent_frames": [1] * n_keyframes, "reference_latent_heights": [lat_h] * n_keyframes,
               "reference_latent_widths": [lat_w] * n_keyframes, "reference_num_audio_latents": [0] * n_keyframes}
        out.update(put("text_token_tags", np.ones(7)))
        out.update(put("text_embedding", np.zeros((7, 8))))
        out.update(put("reference_video_latent_rows", np.random.rand(n_keyframes * rows_per_frame, 96)))
        out.update(put("reference_audio_latent_rows", np.zeros((0, 32))))
        return out

    spec = SimpleNamespace(id="x", case="fl2va", seed=3)
    prepared = prepare_fl2va_row(spec, row(2), patch_size=(1, 2, 2), latent_channels=24, audio_channels=32,
                                 device=torch.device("cpu"))
    assert prepared.layout.num_condition_video_rows == 2 * rows_per_frame
    assert prepared.video.shape[0] == prepared.layout.video_indices.numel()
    t2va = prepare_fl2va_row(SimpleNamespace(id="y", case="t2va", seed=3), row(0), patch_size=(1, 2, 2),
                             latent_channels=24, audio_channels=32, device=torch.device("cpu"))
    assert t2va.layout.num_condition_video_rows == 0
    with pytest.raises(ValueError, match="needs 1"):
        prepare_fl2va_row(SimpleNamespace(id="z", case="i2va", seed=3), row(2), patch_size=(1, 2, 2),
                          latent_channels=24, audio_channels=32, device=torch.device("cpu"))
