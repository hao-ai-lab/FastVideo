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


def test_fp4_attention_is_refused():
    model = _model_stub()
    model.qad_config = {"fp4_attention": {"enabled": True}, "linears": {"enabled": True}}
    with pytest.raises(ValueError, match="dense attention"):
        model._install_qad()


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
    assert builder.v4_id("storyboard", "c") == "storyboard:c" and builder.v4_id("continue_shot", "c") == "c"


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
    v4 = {(first["id"], "480p"): first, (hi["id"], "768p"): hi}
    specs = builder.plan_rows(v4, {(first["id"], "480p"), (hi["id"], "768p")}, {}, 0, set(), lambda _: 1)
    assert len(specs) == 1 and specs[0]["id"] == first["id"] and specs[0]["case"] == "first_frame"
    assert specs[0]["references"] == [{"kind": "stored", "parquet": "a.pq", "index": 0, "audio": True}]


class _ToyBlock(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return hidden + torch.tanh(self.linear(hidden)) + shift


class _ToyStack(torch.nn.Module):

    def __init__(self, blocks: int, dim: int = 4) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(_ToyBlock(dim) for _ in range(blocks))

    def forward(self, hidden: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden = block(hidden, shift)
        return hidden


def test_theseus_schedule_widens_short_to_long():
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import theseus_schedule, theseus_widths
    assert theseus_widths(50) == (1, 2, 4, 8, 16, 32, 50)
    assert theseus_widths(28) == (1, 2, 4, 8, 16, 28)
    plan = theseus_schedule(40, theseus_widths(50), 2.0)
    assert len(plan) == 40 and plan == sorted(plan) and set(plan) == {1, 2, 4, 8, 16, 32, 50}
    assert plan.count(50) > plan.count(32) > plan.count(4)
    assert theseus_schedule(7, (1, 2, 4, 8, 16, 32, 50), 2.0) == [1, 2, 4, 8, 16, 32, 50]
    with pytest.raises(ValueError):
        theseus_schedule(3, (1, 2, 4, 8), 2.0)


@pytest.mark.parametrize("width", [1, 3, 5])
def test_segment_reconstruction_is_zero_at_identity_and_covers_every_block(width):
    import copy
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import SegmentReconstruction
    torch.manual_seed(0)
    teacher = _ToyStack(5)
    student = copy.deepcopy(teacher)
    hidden, shift = torch.randn(3, 4), torch.randn(4)
    with torch.no_grad(), SegmentReconstruction(teacher, student, width, 1.0) as segments:
        teacher(hidden, shift)
    assert len(segments.rel) == -(-5 // width)
    assert float(segments.loss()) == 0.0
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in student.parameters())


def test_segment_reconstruction_trains_from_teacher_segment_inputs():
    import copy
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import SegmentReconstruction
    torch.manual_seed(0)
    teacher = _ToyStack(4)
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.transformer_blocks[0].linear.weight.add_(0.5)  # only block 0 differs
    hidden, shift = torch.randn(3, 4), torch.randn(4)
    with torch.no_grad(), SegmentReconstruction(teacher, student, 2, 1.0) as segments:
        teacher(hidden, shift)
    # Segment [0, 1] carries block 0's error; segment [2, 3] starts from the teacher's block-2 input, so it is exact.
    assert float(segments.rel[0]) > 0.0 and float(segments.rel[1]) == 0.0
    assert float(student.transformer_blocks[0].linear.weight.grad.abs().max()) > 0.0
    assert float(student.transformer_blocks[1].linear.weight.grad.abs().max()) > 0.0
    assert student.transformer_blocks[2].linear.weight.grad is None or \
        float(student.transformer_blocks[2].linear.weight.grad.abs().max()) == 0.0


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


def test_fl2va_plan_takes_keyframes_from_the_matching_v4_case():
    builder = _builder_module()
    v4 = {}
    for n in range(10):
        for case in ("first_frame", "first_last_frame"):
            for res in ("480p", "768p"):
                entry = _entry(case, f"{_ASSET_A}_f{n}_x", f"{case}-{n}-{res}.pq", res)
                v4[(entry["id"], res)] = entry
    specs = builder.plan_fl2va_rows(v4, {"t2va": 5, "i2va": 5, "l2va": 5, "fl2va": 5}, {"480p": 0.8, "768p": 0.2},
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
