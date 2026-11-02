# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for PDDMethod and DMPDDMethod on fake packed models.

The fakes implement the ``PDDStudent`` contract on a two-modality packed
latent (video columns then audio columns) with analytic heads
``head_j = scale * (j + 1) * state`` so every rollout, loss, and carry
contract can be checked against closed-form references while gradients still
flow into one trainable parameter. Ports the behavioural tests of FastGen's
``tests/test_pddmodel.py`` and ``tests/test_pddmdmodel.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pytest
import torch

from fastvideo.layers.pdd import PDDModalitySchedule, pdd_fine_grid
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.methods.distribution_matching.dmpdd import DMPDDMethod
from fastvideo.train.methods.trajectory_matching.pdd import PDDMethod, parse_pdd_config
from fastvideo.train.methods.trajectory_matching.pdd_core import PDDModalityBlock, advance_block
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)

_VIDEO_NUMEL = 6
_AUDIO_NUMEL = 2
_NUMEL = _VIDEO_NUMEL + _AUDIO_NUMEL
_VIDEO = slice(0, _VIDEO_NUMEL)
_AUDIO = slice(_VIDEO_NUMEL, _NUMEL)


class _Scale(torch.nn.Module):

    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))


class _FakePackedModel:
    """Packed two-modality fake implementing the PDD and DMD2 model contracts."""

    device = torch.device("cpu")
    pdd_model_dtype = torch.float32
    num_train_timesteps = 1000

    def __init__(self, *, trainable: bool, scale: float, pdd_steps: int | None = None) -> None:
        self._trainable = trainable
        self.transformer = _Scale(scale)
        self.transformer.requires_grad_(trainable)
        self.pdd_steps = pdd_steps
        self.schedules = {
            "video": PDDModalitySchedule(shift=12.0),
            "audio": PDDModalitySchedule(shift=3.0),
        }
        self.head_calls: list[dict[str, Any]] = []
        self.fused_calls: list[dict[str, Any]] = []
        self.noise_calls: list[dict[str, Any]] = []
        self.prepare_calls: list[str] = []
        self.backward_calls: list[Any] = []

    # -- ModelBase pieces the methods touch ---------------------------------------

    def init_preprocessors(self, training_config: Any) -> None:
        del training_config

    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep

    def modality_slices(self) -> tuple[tuple[str, slice], ...]:
        return (("video", _VIDEO), ("audio", _AUDIO))

    def prepare_batch(self, raw_batch: dict[str, Any], *, generator: torch.Generator, latents_source: str) -> Any:
        self.prepare_calls.append(latents_source)
        if latents_source == "data":
            latents = raw_batch["latent"].clone()
        else:
            latents = torch.zeros(1, _NUMEL)
        return SimpleNamespace(
            latents=latents,
            timesteps=torch.tensor([0.0]),
            attn_metadata=None,
            attn_metadata_vsa="vsa-metadata",
            dmd_latent_vis_dict={},
            fake_score_latent_vis_dict={},
            minimax_h3_dmd_layout=None,
            prompt=raw_batch.get("prompt"),
        )

    def _sigmas(self, timestep: torch.Tensor) -> dict[str, torch.Tensor]:
        base = (timestep.reshape(-1)[:1].to(torch.float64) / 1000.0).clamp(0.0, 0.999)
        return {name: schedule.sigma(base) for name, schedule in self.schedules.items()}

    def add_noise_for_batch(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor,
                            batch: Any) -> torch.Tensor:
        del batch
        return self.add_noise(clean, noise, timestep)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigmas = self._sigmas(timestep)
        out = clean.clone().to(torch.float64)
        for name, columns in self.modality_slices():
            sigma = sigmas[name].to(torch.float64)
            out[:, columns] = (1.0 - sigma) * clean[:, columns].to(torch.float64) + sigma * noise[:, columns].to(
                torch.float64)
        return out.to(clean.dtype)

    def backward(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        self.backward_calls.append(ctx)
        (loss / max(1, int(grad_accum_rounds))).backward()

    # -- teacher / critic ---------------------------------------------------------

    def predict_noise(self, state: torch.Tensor, timestep: torch.Tensor, batch: Any, *, conditional: bool,
                      cfg_uncond: dict[str, Any] | None, attn_kind: str) -> torch.Tensor:
        self.noise_calls.append({
            "timestep": timestep.detach().clone(),
            "conditional": conditional,
            "attn_kind": attn_kind,
            "dtype": state.dtype,
        })
        # Like the H3 adapter, a forward rewrites the batch's clean-time field.
        batch.timesteps = -timestep.detach().clone()
        return self.transformer.scale * state + (0.25 if conditional else 0.0)

    def predict_x0(self, state: torch.Tensor, timestep: torch.Tensor, batch: Any, *, conditional: bool,
                   cfg_uncond: dict[str, Any] | None, attn_kind: str) -> torch.Tensor:
        del batch, cfg_uncond, attn_kind, timestep
        return self.transformer.scale * state * (1.0 if conditional else 0.5)

    # -- PDD student contract -------------------------------------------------------

    def pdd_fine_grid(self, device: torch.device) -> torch.Tensor:
        assert self.pdd_steps is not None
        return pdd_fine_grid(self.pdd_steps, device=device)

    def pdd_modality_schedules(self, batch: Any) -> tuple[PDDModalityBlock, ...]:
        del batch
        return tuple(PDDModalityBlock(name, columns, self.schedules[name]) for name, columns in self.modality_slices())

    def pdd_timestep(self, base: torch.Tensor) -> torch.Tensor:
        return base.to(torch.float64).reshape(-1) * 1000.0

    def _heads(self, state: torch.Tensor) -> torch.Tensor:
        assert self.pdd_steps is not None
        factors = torch.arange(1, self.pdd_steps + 1, dtype=state.dtype)
        return self.transformer.scale * factors[None, :, None] * state[:, None, :]

    def predict_pdd_heads(self, state: torch.Tensor, timestep: torch.Tensor, batch: Any, *, conditional: bool,
                          cfg_uncond: dict[str, Any] | None, attn_kind: str,
                          head_window: tuple[int, int] | None = None) -> torch.Tensor:
        del cfg_uncond
        self.head_calls.append({
            "timestep": timestep.detach().clone(),
            "grad_enabled": torch.is_grad_enabled(),
            "attn_kind": attn_kind,
            "conditional": conditional,
            "dtype": state.dtype,
            "head_window": head_window,
        })
        batch.timesteps = timestep.detach().clone()
        heads = self._heads(state)
        if head_window is None:
            return heads
        return heads[:, head_window[0]:head_window[1]]

    def predict_pdd_fused(self, state: torch.Tensor, timestep: torch.Tensor, start: int, end: int,
                          fine_grid: torch.Tensor, batch: Any, *, precision_decoding: torch.dtype, conditional: bool,
                          cfg_uncond: dict[str, Any] | None, attn_kind: str) -> torch.Tensor:
        del timestep, cfg_uncond, conditional, attn_kind
        self.fused_calls.append({"start": start, "end": end, "grad_enabled": torch.is_grad_enabled()})
        heads = self._heads(state).to(precision_decoding)
        out = torch.zeros_like(state, dtype=precision_decoding)
        for block in self.pdd_modality_schedules(batch):
            weights = block.schedule.integration_weights(fine_grid)[start:end]
            alpha = (weights / weights.sum()).to(precision_decoding)
            out[:, block.columns] = torch.einsum("n,bnk->bk", alpha, heads[:, start:end, block.columns])
        return out.to(state.dtype)


def _training_config(*, grad_accum: int = 1, data_type: str = "text_only") -> TrainingConfig:
    return TrainingConfig(
        distributed=DistributedConfig(num_gpus=1, sp_size=1),
        data=DataConfig(preprocessed_data_type=data_type, seed=0, train_batch_size=1),
        optimizer=OptimizerConfig(learning_rate=1e-3, betas=(0.0, 0.999), lr_scheduler="constant"),
        loop=TrainingLoopConfig(max_train_steps=8, gradient_accumulation_steps=grad_accum),
    )


def _pdd_method_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "rollout_mode": "simulate",
        "block_size_min": 2,
        "block_size_max": 4,
        "type_rk": "euler",
        "precision_decoding": "fp32",
        "cfg_uncond": {
            "text": "zero"
        },
    }
    config.update(overrides)
    return config


def _dmd_method_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "rollout_mode": "simulate",
        "generator_update_interval": 5,
        "real_score_guidance_scale": 1.0,
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "fake_score_loss_space": "x0",
        "cfg_uncond": {
            "text": "zero"
        },
        "fake_score_learning_rate": 1e-3,
        "fake_score_betas": [0.0, 0.999],
        "fake_score_lr_scheduler": "constant",
        "block_size_min": 2,
        "block_size_max": 8,
        "type_rk": "euler",
        "precision_decoding": "fp32",
        "pdd_loss_weight": 1.0,
    }
    config.update(overrides)
    return config


def _roles(*, pdd_steps: int = 8, critic: bool = True) -> dict[str, _FakePackedModel]:
    roles = {
        "student": _FakePackedModel(trainable=True, scale=1.0, pdd_steps=pdd_steps),
        "teacher": _FakePackedModel(trainable=False, scale=0.5),
    }
    if critic:
        roles["critic"] = _FakePackedModel(trainable=True, scale=0.25)
    return roles


def _build(method_cls: type, method_config: dict[str, Any], *, grad_accum: int = 1, data_type: str = "text_only",
           pdd_steps: int = 8, critic: bool = True, validation: dict[str, Any] | None = None) -> Any:
    cfg = SimpleNamespace(
        training=_training_config(grad_accum=grad_accum, data_type=data_type),
        method=method_config,
        validation=validation or {},
    )
    method = method_cls(cfg=cfg, role_models=_roles(pdd_steps=pdd_steps, critic=critic))
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(0)
    return method


def _raw_batch(seed: int = 0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {"latent": torch.randn(1, _NUMEL, generator=generator), "prompt": f"prompt-{seed}"}


# ----------------------------------------------------------------------------
# Config parsing
# ----------------------------------------------------------------------------


def test_parse_pdd_config_defaults_and_validation() -> None:
    config = parse_pdd_config({})
    assert (config.block_size_min, config.block_size_max, config.type_rk) == (32, 128, "euler")
    assert config.precision_decoding == torch.float32
    assert config.teacher_guidance_scale == 1.0
    assert config.pdd_loss_weight == 1.0
    parsed = parse_pdd_config({
        "block_size_min": 4,
        "block_size_max": 8,
        "type_rk": "midpoint",
        "precision_decoding": "bf16",
        "student_grid_indices": {
            "3": [0, 2, 5, 8]
        },
    })
    assert parsed.precision_decoding == torch.bfloat16
    assert parsed.student_grid_indices == {3: [0, 2, 5, 8]}
    with pytest.raises(ValueError, match="must not exceed"):
        parse_pdd_config({"block_size_min": 8, "block_size_max": 4})
    with pytest.raises(ValueError, match="type_rk must be one of"):
        parse_pdd_config({"type_rk": "rk9"})
    with pytest.raises(ValueError, match="precision_decoding must name a floating-point torch dtype"):
        parse_pdd_config({"precision_decoding": "int8"})
    assert parse_pdd_config({"precision_decoding": "torch.float64"}).precision_decoding == torch.float64
    with pytest.raises(ValueError, match="pdd_loss_weight must be a non-negative"):
        parse_pdd_config({"pdd_loss_weight": -1.0})


# ----------------------------------------------------------------------------
# Standalone PDD
# ----------------------------------------------------------------------------


def test_pdd_data_latent_step_trains_the_student_from_aligned_block_starts() -> None:
    method = _build(PDDMethod, _pdd_method_config(rollout_mode="data_latent"), data_type="t2va", critic=False)
    student, teacher = method.student, method.teacher

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), iteration=1)

    assert student.prepare_calls == ["data"]
    assert set(loss_map) == {"total_loss", "pdd_loss", "pdd_loss_video", "pdd_loss_audio"}
    assert torch.isfinite(loss_map["total_loss"])
    torch.testing.assert_close(loss_map["total_loss"], loss_map["pdd_loss_video"] + loss_map["pdd_loss_audio"])
    assert metrics["pdd_block_start"] in {0.0, 2.0, 4.0, 6.0}
    # One student forward at the model boundary and one Euler teacher forward.
    assert len(student.head_calls) == 1
    assert student.head_calls[0]["attn_kind"] == "vsa"
    assert [call["attn_kind"] for call in teacher.noise_calls] == ["dense"]
    assert method.latent_vis["generator_pred_video"].shape == (1, _NUMEL)

    method.backward(loss_map, outputs)
    assert student.transformer.scale.grad is not None
    assert torch.isfinite(student.transformer.scale.grad)
    assert teacher.transformer.scale.grad is None
    assert method.get_optimizers(1) == [method._student_optimizer]
    before = student.transformer.scale.detach().clone()
    method.optimizers_schedulers_step(1)
    assert student.transformer.scale.detach() != before


def test_pdd_loss_supervises_one_head_with_the_teacher_target(monkeypatch: pytest.MonkeyPatch) -> None:
    method = _build(PDDMethod, _pdd_method_config(rollout_mode="data_latent"), data_type="t2va", critic=False)
    student = method.student
    fine_grid = student.pdd_fine_grid(torch.device("cpu"))
    batch = student.prepare_batch(_raw_batch(), generator=method.cuda_generator, latents_source="data")
    x_n = batch.latents
    step_idx = torch.tensor([2])
    monkeypatch.setattr(
        "fastvideo.train.methods.trajectory_matching.pdd.sample_supervision_index",
        lambda *args, **kwargs: torch.tensor([4]),
    )
    pred = student.predict_pdd_heads(x_n, student.pdd_timestep(fine_grid[step_idx]), batch, conditional=True,
                                     cfg_uncond=None, attn_kind="vsa")

    head = method._compute_pdd_loss_from_prediction(x_n, step_idx, pred, fine_grid, batch)

    assert head.k.tolist() == [4]
    torch.testing.assert_close(head.p_k, pred[:, 4])
    blocks = student.pdd_modality_schedules(batch)
    expected_x_k = advance_block(x_n, pred.detach(), step_idx, head.k, fine_grid, blocks)
    torch.testing.assert_close(head.x_k, expected_x_k)
    assert not head.x_k.requires_grad
    # Euler target: the teacher's native direction at x_k on the grid time.
    expected_target = method.teacher.transformer.scale * head.x_k + 0.25
    for name, columns in student.modality_slices():
        expected = torch.mean((head.p_k[:, columns] - expected_target[:, columns])**2)
        torch.testing.assert_close(head.per_modality[name], expected)
    torch.testing.assert_close(head.total, head.per_modality["video"] + head.per_modality["audio"])
    # A head window supervises the same head from the sliced prediction.
    windowed = method._compute_pdd_loss_from_prediction(x_n, step_idx, pred[:, 2:6], fine_grid, batch,
                                                        block_end_idx=torch.tensor([6]), head_offset=2)
    torch.testing.assert_close(windowed.total, head.total)


def test_pdd_carried_walk_advances_one_rung_per_call_and_clears_at_the_end() -> None:
    # block_size_min=2 over 8 intervals: a 4-rung carried grid [0, 2, 4, 6, 8].
    method = _build(PDDMethod, _pdd_method_config(), critic=False)
    student = method.student

    rungs, starts, head_counts, fused_counts = [], [], [], []
    for call in range(6):
        heads_before, fused_before = len(student.head_calls), len(student.fused_calls)
        loss_map, outputs, metrics = method.single_train_step(_raw_batch(call), iteration=call + 1)
        assert torch.isfinite(loss_map["total_loss"])
        method.backward(loss_map, outputs)
        rungs.append(metrics["rollout_step"])
        starts.append(metrics["pdd_block_start"])
        head_counts.append(len(student.head_calls) - heads_before)
        fused_counts.append(len(student.fused_calls) - fused_before)

    # Rank 0, one slot: the stream starts at rung 0, walks to rung 3, then
    # restarts fresh from noise (no second pre-walk).
    assert rungs == [0.0, 1.0, 2.0, 3.0, 0.0, 1.0]
    assert starts == [0.0, 2.0, 4.0, 6.0, 0.0, 2.0]
    assert head_counts == [1] * 6
    # A single stagger group at offset 0 has nothing to pre-walk (FastGen
    # skips it too); every call pays exactly one materialized forward.
    assert fused_counts == [0] * 6
    # A trajectory keeps the prompt it set out with.
    assert method._pdd_carry.slots[0]["raw_batch"]["prompt"] == "prompt-4"
    assert student.prepare_calls == ["zeros"] * 6


def test_pdd_second_slot_is_staggered_by_a_no_grad_pre_walk() -> None:
    method = _build(PDDMethod, _pdd_method_config(), grad_accum=2, critic=False)
    student = method.student
    rungs = []
    for call in range(4):
        _, _, metrics = method.single_train_step(_raw_batch(call), iteration=call + 1)
        rungs.append(metrics["rollout_step"])
    # Slot 0 starts at rung 0 without a walk; slot 1 pre-walks the whole grid
    # under no_grad and keeps the snapshot at rung 1.
    assert rungs == [0.0, 1.0, 1.0, 2.0]
    assert len(student.fused_calls) == 3
    assert all(call["grad_enabled"] is False for call in student.fused_calls)


def test_pdd_carry_state_is_the_no_grad_block_advance_of_the_same_heads() -> None:
    method = _build(PDDMethod, _pdd_method_config(), critic=False)
    student = method.student
    fine_grid = student.pdd_fine_grid(torch.device("cpu"))
    method.single_train_step(_raw_batch(), iteration=1)
    carried = method._pdd_carry.slots[0]
    assert carried["rung"] == 1
    state = carried["state"]
    assert not state.requires_grad and state.grad_fn is None
    # Replay the walk: fresh noise at u_0 then heads integrated over [0, 2).
    batch = student.prepare_batch(_raw_batch(), generator=method.cuda_generator, latents_source="zeros")
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(1, _NUMEL, generator=generator)
    fresh = student.add_noise(torch.zeros(1, _NUMEL), noise, student.pdd_timestep(fine_grid[:1]))
    blocks = student.pdd_modality_schedules(batch)
    with torch.no_grad():
        expected = advance_block(fresh, student._heads(fresh), torch.tensor([0]), torch.tensor([2]), fine_grid, blocks)
    torch.testing.assert_close(state, expected)


def test_pdd_method_configuration_guards() -> None:
    with pytest.raises(ValueError, match="rollout_mode must be set"):
        _build(PDDMethod, {"block_size_min": 2, "block_size_max": 4}, critic=False)
    with pytest.raises(ValueError, match="requires method.rollout_mode='simulate'"):
        _build(PDDMethod, _pdd_method_config(rollout_mode="data_latent"), data_type="text_only", critic=False)
    with pytest.raises(ValueError, match="rollout_carry_slots must equal"):
        _build(PDDMethod, _pdd_method_config(rollout_carry_slots=2), grad_accum=1, critic=False)
    with pytest.raises(ValueError, match="exceeds the student's pdd_steps"):
        _build(PDDMethod, _pdd_method_config(block_size_max=16), critic=False)
    # Validation NFEs must not need blocks wider than the trained window.
    _build(PDDMethod, _pdd_method_config(), critic=False, validation={"sampling_steps": [2, 4, 8]})
    with pytest.raises(ValueError, match="wider than the trained method.block_size_max=4"):
        _build(PDDMethod, _pdd_method_config(), critic=False, validation={"sampling_steps": [4, 1]})
    with pytest.raises(ValueError, match="requires method.cfg_uncond"):
        config = _pdd_method_config(teacher_guidance_scale=2.0)
        config.pop("cfg_uncond")
        _build(PDDMethod, config, critic=False)
    with pytest.raises(ValueError, match="pdd_steps must be an int >= 2"):
        cfg = SimpleNamespace(training=_training_config(), method=_pdd_method_config(), validation={})
        PDDMethod(cfg=cfg, role_models={
            "student": _FakePackedModel(trainable=True, scale=1.0),
            "teacher": _FakePackedModel(trainable=False, scale=0.5),
        })
    with pytest.raises(TypeError, match="PDD-widened"):
        narrow = _FakePackedModel(trainable=True, scale=1.0)
        del narrow.pdd_steps
        cfg = SimpleNamespace(training=_training_config(), method=_pdd_method_config(), validation={})
        PDDMethod(cfg=cfg, role_models={
            "student": narrow,
            "teacher": _FakePackedModel(trainable=False, scale=0.5),
        })


def test_pdd_teacher_guidance_runs_both_forwards_and_mixes_in_decoding_precision() -> None:
    method = _build(PDDMethod, _pdd_method_config(teacher_guidance_scale=3.0), critic=False)
    teacher = method.teacher
    batch = method.student.prepare_batch(_raw_batch(), generator=method.cuda_generator, latents_source="zeros")
    predict = method._pdd_teacher_predictor(batch)
    state = torch.randn(1, _NUMEL)
    out = predict(state, torch.tensor([0.5], dtype=torch.float64))
    assert [call["conditional"] for call in teacher.noise_calls] == [True, False]
    conditional = teacher.transformer.scale * state + 0.25
    unconditional = teacher.transformer.scale * state
    torch.testing.assert_close(out, unconditional + (conditional - unconditional) * 3.0)
    assert out.dtype == torch.float32


# ----------------------------------------------------------------------------
# DMPDD
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("rollout_mode", ["simulate", "data_latent"])
def test_dmpdd_matches_fastgen_clip_policy(rollout_mode: str) -> None:
    method = _build(
        DMPDDMethod,
        _dmd_method_config(rollout_mode=rollout_mode, block_size_max=4, generator_update_interval=5),
        data_type="t2va" if rollout_mode == "data_latent" else "text_only",
    )

    assert method.get_grad_clip_targets(5) == {"student": method.student.transformer}
    assert method.get_grad_clip_targets(1) == {}
    assert method.get_grad_sanitize_targets(5) == {"student": method.student.transformer}
    assert method.get_grad_sanitize_targets(1) == {"critic": method.critic.transformer}


def test_dmpdd_rejects_negative_fake_score_weight_decay() -> None:
    with pytest.raises(ValueError, match="fake_score_weight_decay must be non-negative"):
        _build(DMPDDMethod, _dmd_method_config(fake_score_weight_decay=-0.01))


# ----------------------------------------------------------------------------
# DMPDD
# ----------------------------------------------------------------------------


def test_dmpdd_student_step_uses_the_full_tail_and_carries_one_block() -> None:
    method = _build(DMPDDMethod, _dmd_method_config(block_size_max=4))
    student = method.student
    fine_grid = student.pdd_fine_grid(torch.device("cpu"))

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), iteration=5)

    assert metrics["update_student"] == 1.0
    assert metrics["rollout_step"] == 0.0
    assert metrics["pdd_block_start"] == 0.0
    assert {"pdd_loss_video", "pdd_loss_audio", "generator_loss_video", "generator_loss_audio"} <= metrics.keys()
    torch.testing.assert_close(loss_map["generator_loss"], loss_map["dmd_loss"] + loss_map["pdd_loss"])
    # One materialized student forward with gradients; the tail estimate fed DMD.
    assert len(student.head_calls) == 1 and student.head_calls[0]["grad_enabled"]
    generator_pred = method.latent_vis["generator_pred_video"]
    batch = student.prepare_batch(_raw_batch(), generator=method.cuda_generator, latents_source="zeros")
    blocks = student.pdd_modality_schedules(batch)
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(1, _NUMEL, generator=generator)
    fresh = student.add_noise(torch.zeros(1, _NUMEL), noise, student.pdd_timestep(fine_grid[:1]))
    with torch.no_grad():
        heads = student._heads(fresh)
        expected_tail = advance_block(fresh, heads, torch.tensor([0]), torch.tensor([8]), fine_grid, blocks)
        expected_carry = advance_block(fresh, heads, torch.tensor([0]), torch.tensor([2]), fine_grid, blocks)
    torch.testing.assert_close(generator_pred, expected_tail)
    carried = method._pdd_carry.slots[0]
    assert carried["rung"] == 1
    torch.testing.assert_close(carried["state"], expected_carry)

    method.backward(loss_map, outputs)
    assert student.transformer.scale.grad is not None
    assert method.critic.transformer.scale.grad is None


def test_dmpdd_critic_step_skips_pdd_loss_but_still_advances_the_carry() -> None:
    method = _build(DMPDDMethod, _dmd_method_config(block_size_max=4, generator_update_interval=5))
    student = method.student

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), iteration=1)

    assert metrics["update_student"] == 0.0
    assert "pdd_loss" not in loss_map
    assert loss_map["fake_score_loss"].item() > 0.0
    assert loss_map["generator_loss"].item() == 0.0
    assert not student.head_calls[0]["grad_enabled"]
    assert method.teacher.noise_calls == []
    assert method._pdd_carry.slots[0]["rung"] == 1
    method.backward(loss_map, outputs)
    assert method.critic.transformer.scale.grad is not None
    assert student.transformer.scale.grad is None


def test_dmpdd_carry_walks_the_highest_nfe_grid_and_clears_after_the_last_rung() -> None:
    method = _build(DMPDDMethod, _dmd_method_config(block_size_max=4))
    rungs = []
    for call in range(5):
        _, _, metrics = method.single_train_step(_raw_batch(call), iteration=call + 1)
        rungs.append(metrics["rollout_step"])
    assert rungs == [0.0, 1.0, 2.0, 3.0, 0.0]
    assert method._pdd_carry.slots[0]["rung"] == 1


def test_student_export_contract_samples_the_balanced_highest_nfe_partition() -> None:
    method = _build(DMPDDMethod, _dmd_method_config(block_size_max=4))
    requested = []
    method.student.inference_contract = lambda *, pdd_step_indices: requested.append(pdd_step_indices) or {"ok": 1}
    assert method.inference_checkpoint_contract("student") == {"ok": 1}
    # pdd_steps=8 with block_size_min=2: four fused blocks, as validation samples four steps.
    assert requested == [[0, 2, 4, 6, 8]]
    assert method.inference_checkpoint_contract("critic") is None


def test_dmpdd_data_latent_mode_carries_nothing() -> None:
    method = _build(DMPDDMethod, _dmd_method_config(rollout_mode="data_latent", block_size_max=4), data_type="t2va")
    assert method._pdd_carry is None
    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), iteration=5)
    assert method.student.prepare_calls == ["data"]
    assert metrics["pdd_block_start"] in {0.0, 2.0, 4.0, 6.0}
    assert torch.isfinite(loss_map["total_loss"])
    method.backward(loss_map, outputs)
    assert method.student.transformer.scale.grad is not None


def test_dmpdd_configuration_guards() -> None:
    with pytest.raises(ValueError, match="Carried PDD block 6 exceeds"):
        _build(
            DMPDDMethod,
            _dmd_method_config(block_size_min=3, block_size_max=3, student_grid_indices={3: [0, 1, 2, 8]}),
        )
    with pytest.raises(ValueError, match="generator_update_interval must be at least 2"):
        # DMD2 reads 1 as "update both roles"; DMPDD would never train its critic.
        _build(DMPDDMethod, _dmd_method_config(block_size_max=4, generator_update_interval=1))
    with pytest.raises(ValueError, match="Carried DMPDDMethod cannot cover every student rung"):
        # Four carried rungs with a student interval of 4 leave three phases untrained.
        _build(DMPDDMethod, _dmd_method_config(block_size_max=4, generator_update_interval=4))
    with pytest.raises(ValueError, match="rollout_carry_slots must equal"):
        _build(DMPDDMethod, _dmd_method_config(block_size_max=4, rollout_carry_slots=2))
    with pytest.raises(ValueError, match="rollout_carry must be false"):
        _build(DMPDDMethod, _dmd_method_config(block_size_max=4, rollout_carry=True))
    with pytest.raises(ValueError, match="remove method.dmd_denoising_steps"):
        _build(DMPDDMethod, _dmd_method_config(block_size_max=4, dmd_denoising_steps=[999]))


def test_dmpdd_updates_exactly_one_role_and_scheduler_per_step() -> None:
    method = _build(DMPDDMethod, _dmd_method_config(rollout_mode="data_latent", generator_update_interval=5),
                    data_type="t2va")
    for iteration in range(1, 6):
        for optimizer in method._optimizer_dict.values():
            optimizer.zero_grad(set_to_none=True)
        expected = "student" if iteration == 5 else "critic"
        inactive = "critic" if expected == "student" else "student"
        assert method.get_optimizers(iteration) == [method._optimizer_dict[expected]]
        assert method.get_lr_schedulers(iteration) == [method._lr_scheduler_dict[expected]]
        parameters = {role: model.transformer.scale.detach().clone() for role, model in method._role_models.items()}
        loss_map, outputs, _ = method.single_train_step(_raw_batch(iteration), iteration)
        method.backward(loss_map, outputs, grad_accum_rounds=2)
        assert getattr(method, expected).transformer.scale.grad is not None
        assert getattr(method, inactive).transformer.scale.grad is None
        method.optimizers_schedulers_step(iteration)
        assert not torch.equal(getattr(method, expected).transformer.scale, parameters[expected])
        assert torch.equal(getattr(method, inactive).transformer.scale, parameters[inactive])
    assert method._critic_lr_scheduler.last_epoch == 4
    assert method._student_lr_scheduler.last_epoch == 1
    # Teacher forwards mutate batch times after the student. Backward uses
    # the context captured by that student forward, never the teacher's time.
    context = method.student.backward_calls[0]
    assert torch.equal(context[0], method.student.head_calls[-1]["timestep"])
    assert context[1] == "vsa-metadata"


def test_public_dmd2_keeps_critic_update_on_student_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    method = _build(DMD2Method, _dmd_method_config(rollout_mode="data_latent", generator_update_interval=5),
                    data_type="t2va")

    def rollout(batch, *, with_grad):
        with torch.set_grad_enabled(with_grad):
            return method.student.transformer.scale * batch.latents

    monkeypatch.setattr(method, "_student_rollout", rollout)
    loss_map, outputs, _ = method.single_train_step(_raw_batch(), 5)
    assert outputs["_fv_backward"]["student_ctx"] is not None
    assert outputs["_fv_backward"]["critic_ctx"] is not None
    method.backward(loss_map, outputs)
    assert method.student.transformer.scale.grad is not None
    assert method.critic.transformer.scale.grad is not None
    assert method.get_optimizers(5) == [method._critic_optimizer, method._student_optimizer]
    assert method.get_lr_schedulers(5) == [method._critic_lr_scheduler, method._student_lr_scheduler]
    assert set(method.get_grad_clip_targets(5)) == {"student", "critic"}


def test_resume_seeds_both_alternating_roles_and_preserves_existing_state() -> None:
    method = _build(DMPDDMethod, _dmd_method_config())
    # Iteration zero selects only the student for DMPDD. A resumed checkpoint
    # still needs slots for the critic's saved Adam state.
    method.seed_optimizer_state_for_resume()
    for role in ("student", "critic"):
        optimizer = method._optimizer_dict[role]
        parameter = getattr(method, role).transformer.scale
        state = optimizer.state[parameter]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["step"].device.type == "cpu"
        state["exp_avg"].fill_(3.0)
    method.seed_optimizer_state_for_resume()
    for role in ("student", "critic"):
        parameter = getattr(method, role).transformer.scale
        assert method._optimizer_dict[role].state[parameter]["exp_avg"].item() == 3.0


def test_fsdp_boundary_uses_active_unclipped_critic(monkeypatch: pytest.MonkeyPatch) -> None:
    method = _build(DMPDDMethod, _dmd_method_config(generator_update_interval=5))
    calls = []
    for role in ("student", "critic"):
        model = getattr(method, role)
        model.supports_gradient_accumulation_no_sync = True
        monkeypatch.setattr(model.transformer, "set_is_last_backward",
                            lambda flag, role=role: calls.append((role, "last", flag)), raising=False)
        monkeypatch.setattr(model.transformer, "set_requires_gradient_sync",
                            lambda flag, *, recurse, role=role: calls.append((role, "sync", flag, recurse)), raising=False)
    method.set_is_last_backward(1, False)
    method.set_requires_gradient_sync(1, False)
    assert calls == [("critic", "last", False), ("critic", "sync", False, True), ("critic", "last", False)]
    assert method.get_grad_clip_targets(1) == {}
    assert method.supports_gradient_accumulation_no_sync(1)
    method.critic.supports_gradient_accumulation_no_sync = False
    assert not method.supports_gradient_accumulation_no_sync(1)
    assert method.supports_gradient_accumulation_no_sync(5)


def test_public_ref2va_example_constructs_and_trains_toy_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.train.utils.config import load_run_config

    # Resolve the placeholder model path locally; exercise real config parsing
    # without fetching weights or a Hub manifest.
    monkeypatch.setattr("fastvideo.registry.get_pipeline_config_cls_from_name", lambda _: MiniMaxH3PipelineConfig)
    repository = Path(__file__).resolve().parents[4]
    cfg = load_run_config(str(repository / "examples/train/configs/distribution_matching/minimax_h3/dmpdd_ref2va.yaml"))
    method = DMPDDMethod(cfg=cfg, role_models=_roles(pdd_steps=32))
    method.cuda_generator = torch.Generator().manual_seed(43)
    assert method._rollout_mode == "data_latent" and method._pdd_carry is None
    assert method._fake_score_guidance_scale == 2.0
    assert method._fake_score_cfg_dropout_rate == 0.1
    assert method.pdd_config.pdd_loss_weight == 0.3
    assert method.pdd_config.teacher_guidance_scale == 2.0
    assert method.pdd_config.block_size_min == method.pdd_config.block_size_max == 4
    assert method._score_timestep_continuous
    assert method._fake_score_loss_space == {"__default__": "x0"}
    for iteration in (1, 5):
        losses, outputs, metrics = method.single_train_step(_raw_batch(iteration), iteration)
        method.backward(losses, outputs)
        assert torch.isfinite(losses["total_loss"])
        assert metrics["pdd_block_start"] in set(map(float, range(0, 32, 4)))
        assert method.student.prepare_calls[-1] == "data"
        if iteration == 5:
            assert metrics["dmd_real_uncond_forwards"] == 1
            assert metrics["dmd_fake_uncond_forwards"] == 1
            assert metrics["pdd_teacher_uncond_forwards"] == 1
            torch.testing.assert_close(losses["generator_loss"], losses["dmd_loss"] + 0.3 * losses["pdd_loss"])
