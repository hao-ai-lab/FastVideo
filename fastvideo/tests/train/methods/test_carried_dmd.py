# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for single-head carried DMD, using analytic two-modality models."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.distribution_matching.carried_dmd import CarriedDMDMethod
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.methods.trajectory_carry import snapshot_raw_batch
from fastvideo.train.utils.training_config import (
    DataConfig, DistributedConfig, OptimizerConfig, TrainingConfig, TrainingLoopConfig,
)


class _Scale(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))


class _FakePackedModel:
    device = torch.device("cpu")
    num_train_timesteps = 1000
    video_scheduler_shift = 10.0
    audio_scheduler_shift = 3.0

    def __init__(self, *, trainable, scale):
        self._trainable = trainable
        self.transformer = _Scale(scale)
        self.transformer.requires_grad_(trainable)
        self.calls = []
        self.prepare_calls = []
        self.backward_calls = []

    def init_preprocessors(self, config):
        pass

    def shift_and_clamp_timestep(self, timestep):
        return timestep

    def modality_slices(self):
        return (("video", slice(0, 6)), ("audio", slice(6, 8)))

    def modality_slices_for_batch(self, batch):
        return (("video", slice(0, 6)), ("audio", slice(6, batch.latents.shape[1]))) if hasattr(batch, "latents") else self.modality_slices()

    def prepare_batch(self, raw, *, generator, latents_source):
        self.prepare_calls.append((raw.get("prompt"), latents_source))
        return SimpleNamespace(
            latents=torch.zeros(1, raw.get("numel", 8)), prompt=raw.get("prompt"),
            timesteps=None, attn_metadata=None, attn_metadata_vsa=None,
            dmd_latent_vis_dict={}, fake_score_latent_vis_dict={}, minimax_h3_dmd_layout=None,
        )

    def _sigma(self, timestep, shift):
        t = (timestep.to(torch.float64) / 1000).clamp(0, 0.999)
        return t * shift * 0.999 / (t * (shift - 1) + 0.999)

    def add_noise_for_batch(self, clean, noise, timestep, batch):
        result = torch.empty_like(clean, dtype=torch.float64)
        for name, columns in self.modality_slices_for_batch(batch):
            sigma = self._sigma(timestep, getattr(self, name + "_scheduler_shift"))
            result[:, columns] = (1 - sigma) * clean[:, columns].double() + sigma * noise[:, columns].double()
        return result.to(clean.dtype)

    def extract_eps_for_batch(self, state, clean, timestep, batch):
        result = torch.empty_like(state, dtype=torch.float64)
        for name, columns in self.modality_slices_for_batch(batch):
            sigma = self._sigma(timestep, getattr(self, name + "_scheduler_shift"))
            result[:, columns] = (state[:, columns].double() - (1 - sigma) * clean[:, columns].double()) / sigma
        return result.to(state.dtype)

    def predict_x0(self, state, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        self.calls.append({"rung": timestep.clone(), "state": state.detach().clone(),
                           "prompt": batch.prompt, "grad": torch.is_grad_enabled(),
                           "conditional": conditional, "attn_kind": attn_kind})
        batch.timesteps = timestep.detach().clone()
        setattr(batch, "attn_metadata_vsa" if attn_kind == "vsa" else "attn_metadata", (attn_kind, batch.prompt))
        return self.transformer.scale * state * (1 if conditional else 0.5)

    def backward(self, loss, context, *, grad_accum_rounds):
        self.backward_calls.append(context)
        (loss / grad_accum_rounds).backward()


def _training_config(*, grad_accum=1):
    return TrainingConfig(
        distributed=DistributedConfig(num_gpus=1, sp_size=1),
        data=DataConfig(preprocessed_data_type="text_only", seed=0, train_batch_size=1,
                        native_shape_bucketing=True),
        optimizer=OptimizerConfig(learning_rate=1e-3, betas=(0.9, 0.999), lr_scheduler="constant"),
        loop=TrainingLoopConfig(max_train_steps=40, gradient_accumulation_steps=grad_accum),
    )


def _method_config(**overrides):
    result = dict(rollout_mode="simulate", rollout_carry=True, rollout_sample_type="ode",
                  generator_update_interval=5, dmd_denoising_steps=[999, 874, 749, 624, 500, 375, 250, 125],
                  min_timestep_ratio=0.001, max_timestep_ratio=0.999,
                  score_timestep_continuous=True, score_timestep_shift=2.4, score_timestep_warp_max=0.999,
                  fake_score_loss_space="x0", real_score_guidance_scale=1.0,
                  fake_score_learning_rate=1e-3, fake_score_betas=[0.9, 0.999],
                  fake_score_lr_scheduler="constant", cfg_uncond={"text": "zero"})
    result.update(overrides)
    return result


def _roles():
    return {"student": _FakePackedModel(trainable=True, scale=0.8),
            "critic": _FakePackedModel(trainable=True, scale=0.25),
            "teacher": _FakePackedModel(trainable=False, scale=0.5)}


def _build(config=None, *, grad_accum=1):
    cfg = SimpleNamespace(training=_training_config(grad_accum=grad_accum),
                          method=_method_config() if config is None else config)
    method = CarriedDMDMethod(cfg=cfg, role_models=_roles())
    method.cuda_generator = torch.Generator().manual_seed(0)
    return method


def test_carried_dmd_alternates_active_optimizer_scheduler_backward_and_clipping():
    method = _build()
    for iteration in range(1, 6):
        for optimizer in method._optimizer_dict.values():
            optimizer.zero_grad(set_to_none=True)
        role = "student" if iteration == 5 else "critic"
        other = "critic" if role == "student" else "student"
        assert method.get_optimizers(iteration) == [method._optimizer_dict[role]]
        assert method.get_lr_schedulers(iteration) == [method._lr_scheduler_dict[role]]
        assert method.get_grad_clip_targets(iteration) == {role: getattr(method, role).transformer}
        before = {key: model.transformer.scale.detach().clone() for key, model in method._role_models.items()}
        losses, outputs, _ = method.single_train_step({"prompt": str(iteration)}, iteration)
        assert losses["fake_score_loss" if role == "student" else "generator_loss"].item() == 0
        method.backward(losses, outputs, grad_accum_rounds=2)
        assert getattr(method, role).transformer.scale.grad is not None
        assert getattr(method, other).transformer.scale.grad is None
        method.optimizers_schedulers_step(iteration)
        assert not torch.equal(getattr(method, role).transformer.scale, before[role])
        assert torch.equal(getattr(method, other).transformer.scale, before[other])
        assert torch.equal(method.teacher.transformer.scale, before["teacher"])
    assert method._critic_lr_scheduler.last_epoch == 4
    assert method._student_lr_scheduler.last_epoch == 1
    # The student context must precede the dense teacher/critic mutation of the batch.
    assert method.student.backward_calls[0][0].item() == 500
    assert method.student.backward_calls[0][1] == ("vsa", "1")
    assert len(method.teacher.calls) == 1
    assert all(not call["grad"] for call in method.teacher.calls)
    assert len(method.critic.calls) == 5
    assert [call["grad"] for call in method.critic.calls] == [True] * 4 + [False]


def test_existing_dmd2_retains_critic_update_on_student_iterations():
    method = DMD2Method(cfg=SimpleNamespace(training=_training_config(), method=_method_config()), role_models=_roles())
    assert method.get_optimizers(5) == [method._critic_optimizer, method._student_optimizer]
    assert set(method.get_grad_clip_targets(5)) == {"critic", "student"}


def test_carry_staggers_slots_uniformly_then_preserves_prompt_and_geometry_until_clear():
    method = _build(grad_accum=4)
    actual = [[] for _ in range(4)]
    prompts = [[] for _ in range(4)]
    for iteration in range(1, 10):
        for slot in range(4):
            before = len(method.student.calls)
            _, _, metrics = method.single_train_step({"prompt": f"{iteration}-{slot}", "numel": 8 + iteration}, iteration)
            assert len(method.student.calls) - before == (8 if iteration == 1 else 1)
            actual[slot].append(metrics["rollout_step"])
            prompts[slot].append(method.student.calls[-1]["prompt"])
            if method._carry.slots[slot] is not None:
                assert not method._carry.slots[slot]["state"].requires_grad
                assert method._carry.slots[slot]["state"].grad_fn is None
    for slot in range(4):
        assert actual[slot] == [float((slot + i) % 8) for i in range(9)]
        assert prompts[slot][:8-slot] == [f"1-{slot}"] * (8-slot)
        assert prompts[slot][8-slot] == f"{9-slot}-{slot}"
    assert all(call["state"].shape[1] == 9 for call in method.student.calls[:32])


def test_native_shape_stagger_is_rank_independent_and_prewalk_count_uniform(monkeypatch):
    import fastvideo.train.methods.trajectory_carry as carry
    seen = []
    for rank in (0, 7):
        monkeypatch.setattr(carry, "carry_rank_world", lambda config, rank=rank: (rank, 8))
        method = _build(grad_accum=4)
        for slot in range(4):
            _, _, metrics = method.single_train_step({"prompt": str(slot)}, 1)
            seen.append((rank, slot, metrics["rollout_step"], len(method.student.calls)))
    assert [entry[1:] for entry in seen[:4]] == [entry[1:] for entry in seen[4:]]
    assert [entry[-1] for entry in seen[:4]] == [8, 16, 24, 32]


def test_ode_transition_matches_per_modality_affine_solver_without_rng_draw():
    method = _build()
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    state = torch.arange(8).float().reshape(1, -1)
    x0 = 0.8 * state
    current = torch.tensor([749])
    steps = method._get_denoising_step_list(torch.device("cpu"))
    rng = method.cuda_generator.get_state().clone()
    actual = method._renoise(state, x0, current, 3, steps, batch)
    expected = torch.empty_like(state)
    for name, region in method.student.modality_slices():
        shift = 10 if name == "video" else 3
        sigma = lambda t: (t * shift * 0.999) / (t * (shift - 1) + 0.999)
        now, nxt = sigma(0.749), sigma(0.624)
        eps = ((state[:, region].double() - (1 - now) * x0[:, region].double()) / now).float()
        expected[:, region] = ((1 - nxt) * x0[:, region].double() + nxt * eps.double()).float()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(rng, method.cuda_generator.get_state())


def test_raw_snapshot_owns_nested_tensor_storage():
    tensor = torch.ones(2, requires_grad=True)
    source = {"nested": [{"tensor": tensor}], "shape": [1, 2]}
    result = snapshot_raw_batch(source, torch.device("cpu"))
    with torch.no_grad():
        tensor.zero_()
    source["shape"][0] = 99
    assert torch.equal(result["nested"][0]["tensor"], torch.ones(2))
    assert not result["nested"][0]["tensor"].requires_grad
    assert result["shape"] == [1, 2]


def test_model_state_restore_does_not_restore_transient_carry():
    first = _build()
    first.single_train_step({"prompt": "old"}, 1)
    resumed = _build()
    resumed.role_modules.load_state_dict(first.role_modules.state_dict())
    assert resumed._carry.slots == [None]
    _, _, metrics = resumed.single_train_step({"prompt": "new"}, 2)
    assert metrics["rollout_step"] == 0
    assert resumed.student.calls[-1]["prompt"] == "new"


@pytest.mark.parametrize("overrides", [
    {"rollout_carry": False}, {"rollout_sample_type": "rk4"}, {"rollout_data_forcing": True},
    {"warp_denoising_step": True}, {"rollout_carry_slots": 2}, {"generator_update_interval": 0},
    {"generator_update_interval": 4}, {"dmd_denoising_steps": []},
    {"dmd_denoising_steps": [999, 999]}, {"dmd_denoising_steps": [999, 0]},
    {"dmd_denoising_steps": [999, True]}, {"dmd_denoising_steps": [999, 1.5]},
])
def test_invalid_carry_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        _build(_method_config(**overrides))


def test_example_constructs_release_method_and_exports_public_inference_contract(monkeypatch):
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.train.utils.config import load_run_config
    monkeypatch.setattr("fastvideo.registry.get_pipeline_config_cls_from_name", lambda _: MiniMaxH3PipelineConfig)
    path = Path(__file__).resolve().parents[4] / "examples/distill/minimax_h3/t2av_dmd.yaml"
    cfg = load_run_config(str(path))
    method = CarriedDMDMethod(cfg=cfg, role_models=_roles())
    assert method._carry.num_slots == cfg.training.loop.gradient_accumulation_steps == 4
    assert cfg.training.distributed.num_gpus == 64 and cfg.training.distributed.sp_size == 4
    assert method._score_timestep_shift == 2.4 and method._score_timestep_warp_max == 0.999
    assert method._fake_score_loss_space == {"__default__": "x0"}
    assert cfg.training.vsa_sparsity == 0.8 and cfg.training.vsa_tile_size == 64
    assert cfg.training.pipeline_config.video_scheduler_shift == 10.0
    assert cfg.training.pipeline_config.audio_scheduler_shift == 3.0
    contract = method.inference_checkpoint_json_overrides()
    sidecar = contract["fastvideo_inference.json"]
    assert sidecar["num_inference_steps"] == 9
    assert sidecar["transformer_forwards"] == 8
    assert sidecar["dmd_denoising_steps"] == [999, 874, 749, 624, 500, 375, 250, 125]
    assert contract["scheduler/scheduler_config.json"] == {"shift": 10.0}
    assert contract["audio_scheduler/scheduler_config.json"] == {"shift": 3.0}
    for role in ("critic", "teacher", "missing"):
        with pytest.raises(ValueError, match="only for role 'student'"):
            method.inference_checkpoint_json_overrides(role)


def test_validation_adopts_trained_ladder_and_refuses_different_one():
    from fastvideo.train.callbacks.validation import ValidationCallback
    callback = ValidationCallback(pipeline_target="unused.Pipeline", dataset_file="unused.json", sampling_steps=[9])
    method = _build()
    callback._adopt_training_denoising_ladder(method)
    assert callback.sampling_timesteps == method.method_config["dmd_denoising_steps"]
    callback.sampling_timesteps = [999, 500]
    with pytest.raises(ValueError, match="disagrees"):
        callback._adopt_training_denoising_ladder(method)
