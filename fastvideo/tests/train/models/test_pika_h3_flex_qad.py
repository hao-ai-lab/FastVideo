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
