# SPDX-License-Identifier: Apache-2.0
"""Fake-model tests for the TDM training method."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.train.methods.distribution_matching import tdm as tdm_module
from fastvideo.train.methods.distribution_matching.tdm import (
    TDMMethod,
    flow_effective_noise,
    flow_transition_to_noisier_sigma,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.training_config import (
    DataConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)


class _FakeFlowScheduler:

    num_train_timesteps = 1000

    def __init__(self) -> None:
        self.timesteps = torch.arange(1000, -1, -1, dtype=torch.float32)
        self.sigmas = self.timesteps / 1000.0

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        t = timestep.float().reshape(-1)
        idx = torch.argmin(
            (self.timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma = self.sigmas[idx].to(device=clean_latent.device, dtype=clean_latent.dtype)
        if sigma.numel() == 1:
            sigma = sigma.reshape((1, ) + (1, ) * (clean_latent.ndim - 1))
        else:
            sigma = sigma.reshape((clean_latent.shape[0], ) + (1, ) * (clean_latent.ndim - 1))
        return (1.0 - sigma) * clean_latent + sigma * noise


class _ShiftedFlowScheduler(_FakeFlowScheduler):

    shift = 8.0

    def __init__(self) -> None:
        raw_sigmas = torch.arange(1000, -1, -1, dtype=torch.float32) / 1000.0
        shifted_sigmas = self.shift * raw_sigmas / (1.0 + (self.shift - 1.0) * raw_sigmas)
        self.timesteps = shifted_sigmas * 1000.0
        self.sigmas = shifted_sigmas
        self.config = SimpleNamespace(
            num_train_timesteps=1000,
            use_dynamic_shifting=False,
            shift_terminal=None,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
        )


class _TinyRoleModel(ModelBase):

    def __init__(
        self,
        *,
        role: str,
        trainable: bool,
    ) -> None:
        super().__init__(trainable=trainable)
        self.role = role
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.transformer.weight.fill_(0.15 if trainable else 0.05)
        self.transformer.requires_grad_(trainable)
        self.noise_scheduler = _FakeFlowScheduler()
        self.backward_calls = 0
        self.predict_calls: list[tuple[bool, str, float, bool]] = []
        self.predict_inputs: list[torch.Tensor] = []
        self.predict_timestep_shapes: list[tuple[int, ...]] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        latents = torch.zeros(
            2,
            2,
            1,
            2,
            2,
        )
        return TrainingBatch(
            latents=latents,
            timesteps=torch.tensor([1000]),
            attn_metadata=None,
            attn_metadata_vsa=None,
        )

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.noise_scheduler.add_noise(clean_latents, noise, timestep)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del cfg_uncond
        assert batch.timesteps is not None
        assert_close(batch.timesteps, timestep)
        timestep_value = float(timestep.reshape(-1)[0].item())
        self.predict_calls.append((
            conditional,
            attn_kind,
            timestep_value,
            torch.is_grad_enabled(),
        ))
        self.predict_inputs.append(noisy_latents.detach().clone())
        self.predict_timestep_shapes.append(tuple(timestep.shape))
        scale = self.transformer.weight.reshape(())
        offset = {
            ("student", True): 0.05,
            ("critic", True): -0.1,
            ("teacher", True): 0.2,
            ("teacher", False): -0.2,
        }.get((self.role, conditional), 0.0)
        return noisy_latents * scale + offset

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del ctx
        self.backward_calls += 1
        (loss / max(1, int(grad_accum_rounds))).backward()


def _build_method(
    *,
    generator_update_interval: int = 1,
    method_overrides: dict[str, Any] | None = None,
) -> tuple[TDMMethod, _TinyRoleModel, _TinyRoleModel]:
    training = TrainingConfig(
        data=DataConfig(preprocessed_data_type="text_only", seed=0),
        optimizer=OptimizerConfig(
            learning_rate=1.0e-3,
            betas=(0.0, 0.999),
            weight_decay=0.0,
            lr_scheduler="constant",
        ),
        loop=TrainingLoopConfig(max_train_steps=4),
    )
    method_cfg = {
        "rollout_mode": "simulate",
        "generator_update_interval": generator_update_interval,
        "real_score_guidance_scale": 4.5,
        "tdm_denoising_steps": [1000, 750, 500, 250],
        "student_sample_type": "sde",
        "noise_interval_mode": "separate",
        "use_randmid": True,
        "fake_score_learning_rate": 1.0e-3,
        "fake_score_betas": [0.0, 0.999],
        "fake_score_lr_scheduler": "constant",
    }
    if method_overrides:
        method_cfg.update(method_overrides)
    cfg = SimpleNamespace(
        training=training,
        method=method_cfg,
        validation={},
    )
    student = _TinyRoleModel(role="student", trainable=True)
    teacher = _TinyRoleModel(role="teacher", trainable=False)
    critic = _TinyRoleModel(role="critic", trainable=True)
    method = TDMMethod(
        cfg=cfg,
        role_models={
            "student": student,
            "teacher": teacher,
            "critic": critic,
        },
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(123)
    return method, student, critic


def test_tdm_single_train_step_reports_losses_and_routes_backward() -> None:
    method, student, critic = _build_method(generator_update_interval=1)

    loss_map, outputs, metrics = method.single_train_step({}, iteration=0)

    assert set(loss_map) == {"total_loss", "generator_loss", "fake_score_loss"}
    assert bool(loss_map["total_loss"].isfinite().item())
    assert bool(loss_map["generator_loss"].isfinite().item())
    assert bool(loss_map["fake_score_loss"].isfinite().item())
    assert metrics["update_student"] == 1.0
    generator_timestep = float(torch.as_tensor(metrics["tdm/generator/source_timestep"]).item())
    generator_sigma = float(torch.as_tensor(metrics["tdm/generator/source_sigma"]).item())
    intermediate_sigma = float(torch.as_tensor(metrics["tdm/generator/intermediate_sigma"]).item())
    target_timestep = float(torch.as_tensor(metrics["tdm/generator/target_timestep"]).item())
    target_sigma = float(torch.as_tensor(metrics["tdm/generator/target_sigma"]).item())
    assert 0.0 < target_sigma < generator_sigma <= 1.0
    assert 0.0 <= intermediate_sigma <= target_sigma
    assert_close(torch.tensor(generator_timestep), torch.tensor(generator_sigma * 1000.0))
    assert_close(torch.tensor(target_timestep), torch.tensor(target_sigma * 1000.0))
    assert "tdm/generator/raw_delta_abs_mean" in metrics
    assert "tdm/generator/target_delta_abs_mean" in metrics
    assert "tdm/generator/normalization_denom" in metrics
    assert 0.0 <= float(metrics["tdm/generator/source_trajectory_index"]) <= 3.0
    assert metrics["tdm/generator/normalize_delta"] == 1.0
    assert "tdm/fake_score/source_sigma" in metrics
    assert "tdm/fake_score/intermediate_sigma" in metrics
    assert "tdm/fake_score/target_sigma" in metrics
    assert "tdm/fake_score/snr_weight" in metrics
    assert "tdm/fake_score/importance_mean" in metrics
    assert "tdm/fake_score/per_sample_loss_mean" in metrics
    assert "tdm/fake_score/source_trajectory_index" in metrics
    assert outputs["_fv_backward"]["update_student"] is True
    assert outputs["_fv_backward"]["student_ctx"][0].shape == (2, )
    assert outputs["_fv_backward"]["critic_ctx"][0].shape == (2, )

    method.backward(loss_map, outputs)

    assert student.backward_calls == 1
    assert critic.backward_calls == 1

    grad_student_calls = [
        call for call in student.predict_calls
        if call[0] is True and call[1] == "vsa" and call[3] is True
    ]
    assert len(grad_student_calls) == 1
    assert 0.0 < grad_student_calls[0][2] <= 1000.0


def test_tdm_generator_reconstructs_intermediate_before_scoring_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, student, critic = _build_method(method_overrides={"use_randmid": False})
    teacher = method.teacher
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)
    context = method._sample_tdm_context(trajectory)
    monkeypatch.setattr(method, "_sample_tdm_context", lambda _: context)
    student.predict_calls.clear()
    student.predict_inputs.clear()
    student.predict_timestep_shapes.clear()
    critic.predict_calls.clear()
    critic.predict_inputs.clear()
    critic.predict_timestep_shapes.clear()
    teacher.predict_calls.clear()
    teacher.predict_inputs.clear()
    teacher.predict_timestep_shapes.clear()

    _, metrics, student_ctx = method._tdm_generator_loss(trajectory, batch)

    assert bool(torch.all(context.sigma_intermediate <= context.sigma_target).item())
    assert bool(torch.all(context.sigma_target < context.sigma_source).item())
    assert student.predict_calls == [(True, "vsa", context.timestep_source[0].item(), True)]
    assert [call[:3] for call in critic.predict_calls] == [
        (True, "dense", context.timestep_target[0].item()),
    ]
    assert [call[:3] for call in teacher.predict_calls] == [
        (True, "dense", context.timestep_target[0].item()),
        (False, "dense", context.timestep_target[0].item()),
    ]

    source_noisy = context.noisy_source
    sigma_source = context.sigma_source.reshape(2, 1, 1, 1, 1)
    generator_flow = source_noisy * student.transformer.weight.reshape(()) + 0.05
    generator_x0 = source_noisy - sigma_source * generator_flow
    eps_source = flow_effective_noise(source_noisy, generator_x0, context.sigma_source)
    sigma_intermediate = context.sigma_intermediate.reshape(2, 1, 1, 1, 1)
    noisy_intermediate = (1.0 - sigma_intermediate) * generator_x0 + sigma_intermediate * eps_source
    expected_target, _, _ = flow_transition_to_noisier_sigma(
        noisy_from=noisy_intermediate,
        clean_latents=generator_x0,
        eps_from=eps_source,
        sigma_from=context.sigma_intermediate,
        sigma_to=context.sigma_target,
        proposal_noise=context.proposal_noise,
    )
    assert_close(critic.predict_inputs[0], teacher.predict_inputs[0])
    assert_close(critic.predict_inputs[0], teacher.predict_inputs[1])
    assert_close(critic.predict_inputs[0], expected_target)
    assert "tdm/generator/intermediate_sigma" in metrics
    assert student_ctx[0].shape == (2, )
    assert student.predict_timestep_shapes == [(2, )]
    assert critic.predict_timestep_shapes == [(2, )]
    assert teacher.predict_timestep_shapes == [(2, ), (2, )]


def test_tdm_sigma_lookup_treats_explicit_steps_as_raw_wan_timesteps() -> None:
    method, student, _ = _build_method()
    student.noise_scheduler = _ShiftedFlowScheduler()

    sigmas = method._timestep_to_sigma(torch.tensor([1000, 750, 500, 250]))

    expected = torch.tensor([
        1.0,
        8.0 * 0.75 / (1.0 + 7.0 * 0.75),
        8.0 * 0.5 / (1.0 + 7.0 * 0.5),
        8.0 * 0.25 / (1.0 + 7.0 * 0.25),
    ])
    assert_close(sigmas, expected)


def test_tdm_real_shifted_scheduler_aligns_noising_model_labels_and_x0_conversion() -> None:
    method, student, critic = _build_method(method_overrides={"student_sample_type": "ode"})
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    student.noise_scheduler = scheduler
    critic.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    method.teacher.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")

    trajectory = method._student_trajectory(batch)

    expected_sigmas = method._timestep_to_sigma(torch.tensor([1000, 750, 500, 250]))
    model_labels = torch.tensor([call[2] for call in student.predict_calls])
    resolved_sigmas = scheduler.sigmas[
        torch.argmin((scheduler.timesteps.unsqueeze(0) - model_labels.unsqueeze(1)).abs(), dim=1)
    ]
    assert_close(trajectory.sigmas, expected_sigmas)
    assert_close(resolved_sigmas, expected_sigmas)
    assert model_labels[1].item() > 900.0

    for noisy, clean, sigma in zip(trajectory.noisy_latents, trajectory.clean_latents, trajectory.sigmas):
        predicted_flow = noisy * student.transformer.weight.reshape(()) + 0.05
        assert_close(clean, noisy - sigma * predicted_flow)

    for index in range(len(trajectory.noisy_latents) - 1):
        noisy = trajectory.noisy_latents[index]
        clean = trajectory.clean_latents[index]
        sigma = trajectory.sigmas[index]
        next_sigma = trajectory.sigmas[index + 1]
        eps = (noisy - (1.0 - sigma) * clean) / sigma
        expected_next = (1.0 - next_sigma) * clean + next_sigma * eps
        assert_close(trajectory.noisy_latents[index + 1], expected_next)


def test_tdm_warped_student_trajectory_uses_scheduler_sigmas_without_double_shift() -> None:
    method, student, _ = _build_method(method_overrides={
        "warp_denoising_step": True,
        "student_sample_type": "ode",
    })
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    student.noise_scheduler = scheduler
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")

    trajectory = method._student_trajectory(batch)

    step_indices = torch.tensor([0, 250, 500, 750])
    expected_steps = scheduler.timesteps[step_indices]
    expected_sigmas = scheduler.sigmas[step_indices]
    model_labels = torch.tensor([call[2] for call in student.predict_calls])
    assert_close(trajectory.timesteps, expected_steps)
    assert_close(trajectory.sigmas, expected_sigmas)
    assert_close(model_labels, expected_steps)

    for index in range(len(trajectory.noisy_latents) - 1):
        noisy = trajectory.noisy_latents[index]
        clean = trajectory.clean_latents[index]
        sigma = trajectory.sigmas[index]
        next_sigma = trajectory.sigmas[index + 1]
        eps = (noisy - (1.0 - sigma) * clean) / sigma
        expected_next = (1.0 - next_sigma) * clean + next_sigma * eps
        assert_close(trajectory.noisy_latents[index + 1], expected_next)


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        ([1000], "at least two"),
        ([750, 500], "terminal sigma"),
        ([1000, 750, 750], "strictly decreasing"),
        ([1000, 500, 750], "strictly decreasing"),
    ],
)
def test_tdm_rejects_invalid_denoising_step_schedules(
    steps: list[int],
    match: str,
) -> None:
    method, _, _ = _build_method(method_overrides={"tdm_denoising_steps": steps})

    with pytest.raises(ValueError, match=match):
        method._get_denoising_step_list(torch.device("cpu"))


def test_tdm_respects_generator_update_interval() -> None:
    method, student, critic = _build_method(generator_update_interval=2)

    loss_map, outputs, metrics = method.single_train_step({}, iteration=1)

    assert metrics["update_student"] == 0.0
    assert "tdm/generator/source_timestep" not in metrics
    assert outputs["_fv_backward"]["update_student"] is False

    method.backward(loss_map, outputs)

    assert student.backward_calls == 0
    assert critic.backward_calls == 1


def test_tdm_separate_noise_interval_samples_each_batch_element() -> None:
    method, _, _ = _build_method(method_overrides={"noise_interval_mode": "separate"})
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)
    method.cuda_generator.manual_seed(0)

    context = method._sample_tdm_context(trajectory)

    assert context.trajectory_indices.tolist() == [0, 3]
    assert (context.sigma_source.shape == context.sigma_intermediate.shape
            == context.sigma_target.shape == (2, ))
    assert bool(torch.all(context.sigma_intermediate <= context.sigma_target).item())
    assert bool(torch.all(context.sigma_target < context.sigma_source).item())
    sigma_mid = context.sigma_intermediate.reshape(2, 1, 1, 1, 1)
    expected_intermediate = ((1.0 - sigma_mid) * context.clean_latents
                             + sigma_mid * context.eps_source)
    assert_close(context.noisy_intermediate, expected_intermediate)


@pytest.mark.parametrize(
    ("sigma", "expected_weight"),
    [
        (0.96, 1.0 / 576.0),
        (8.0 / 9.0, 1.0 / 64.0),
        (8.0 / 11.0, 9.0 / 64.0),
        (0.25, 5.0),
    ],
)
def test_tdm_fake_score_uses_clipped_flow_snr_directly(
    sigma: float,
    expected_weight: float,
) -> None:
    method, _, _ = _build_method()
    context = SimpleNamespace(
        sigma_target=torch.tensor([sigma], dtype=torch.float32),
        mixed_noise=torch.zeros(2, 1, dtype=torch.float32),
        proposal_noise=torch.zeros(2, 1, dtype=torch.float32),
    )

    components = method._tdm_fake_score_weight_components(context)

    assert_close(
        components["snr_weight"],
        torch.tensor([expected_weight], dtype=torch.float32),
    )
    assert_close(
        components["weights"],
        torch.full((2, ), expected_weight, dtype=torch.float32),
    )


def test_tdm_next_step_noise_interval_uses_rank_stratified_adjacent_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, _, _ = _build_method(method_overrides={
        "noise_interval_mode": "next_step",
        "use_randmid": False,
    })
    monkeypatch.setattr(tdm_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tdm_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tdm_module.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(tdm_module.dist, "get_world_size", lambda: 4)
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)

    context = method._sample_tdm_context(trajectory)

    assert context.trajectory_indices.tolist() == [2, 3]
    assert_close(context.sigma_source, torch.tensor([0.5, 0.25]))
    assert_close(context.sigma_intermediate, torch.tensor([0.25, 0.0]))
    assert_close(context.sigma_target, torch.tensor([0.25, 0.001]))
    assert bool(torch.all(context.sigma_target >= context.sigma_intermediate).item())
    assert bool((method._tdm_fake_score_weights(context) > 0).all().item())


def test_tdm_next_step_noise_interval_cycles_undercovered_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, _, _ = _build_method(method_overrides={
        "noise_interval_mode": "next_step",
        "use_randmid": False,
    })
    monkeypatch.setattr(tdm_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tdm_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tdm_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(tdm_module.dist, "get_world_size", lambda: 1)
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)

    first_context = method._sample_tdm_context(trajectory, iteration=0)
    second_context = method._sample_tdm_context(trajectory, iteration=1)

    assert first_context.trajectory_indices.tolist() == [0, 1]
    assert second_context.trajectory_indices.tolist() == [2, 3]
    assert_close(first_context.sigma_source, torch.tensor([1.0, 0.75]))
    assert_close(second_context.sigma_source, torch.tensor([0.5, 0.25]))


def test_tdm_next_step_generator_update_iteration_respects_update_interval() -> None:
    method, _, _ = _build_method(
        generator_update_interval=2,
        method_overrides={
            "noise_interval_mode": "next_step",
            "use_randmid": False,
        },
    )

    assert method._generator_update_iteration(0) == 0
    assert method._generator_update_iteration(2) == 1
    assert method._generator_update_iteration(4) == 2


def test_tdm_next_step_noise_interval_requires_fixed_intermediate() -> None:
    with pytest.raises(ValueError, match="use_randmid"):
        _build_method(method_overrides={
            "noise_interval_mode": "next_step",
            "use_randmid": True,
        })


def test_tdm_to_terminal_noise_interval_stays_below_terminal_sigma() -> None:
    method, _, _ = _build_method(method_overrides={
        "noise_interval_mode": "to_terminal",
        "use_randmid": False,
    })
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)

    context = method._sample_tdm_context(trajectory)

    assert bool(torch.all(context.sigma_target < trajectory.sigmas.max()).item())
    assert bool(torch.all(context.sigma_target >= context.sigma_intermediate).item())
    assert bool(torch.all(context.sigma_target > 0).item())
    assert bool((method._tdm_fake_score_weights(context) > 0).all().item())


def test_tdm_to_terminal_fake_score_loss_trains_critic() -> None:
    method, _, critic = _build_method(
        generator_update_interval=2,
        method_overrides={
            "noise_interval_mode": "to_terminal",
            "use_randmid": False,
        },
    )

    loss_map, outputs, metrics = method.single_train_step({}, iteration=1)

    assert float(loss_map["fake_score_loss"].item()) > 0.0
    assert float(torch.as_tensor(metrics["tdm/fake_score/weight_mean"]).item()) > 0.0

    method.backward(loss_map, outputs)

    grad = critic.transformer.weight.grad
    assert grad is not None
    assert float(grad.abs().item()) > 0.0


def test_tdm_generator_delta_normalization_is_per_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    method, student, critic = _build_method(method_overrides={
        "real_score_guidance_scale": 1.0,
        "use_huber": False,
    })
    teacher = method.teacher
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    pred = torch.zeros((2, 1, 1, 1, 1), requires_grad=True)
    real_x0 = torch.tensor([1.0, 100.0]).reshape(2, 1, 1, 1, 1)
    fake_x0 = torch.zeros_like(real_x0)
    context = SimpleNamespace(
        noisy_source=torch.zeros_like(pred),
        timestep_source=torch.tensor([500.0, 500.0]),
        timestep_target=torch.tensor([400.0, 400.0]),
        sigma_source=torch.tensor([0.5, 0.5]),
        sigma_intermediate=torch.tensor([0.25, 0.25]),
        sigma_target=torch.tensor([0.4, 0.4]),
        proposal_noise=torch.zeros_like(pred),
        trajectory_indices=torch.tensor([2, 2]),
    )

    monkeypatch.setattr(method, "_sample_tdm_context", lambda trajectory: context)
    monkeypatch.setattr(student, "predict_x0", lambda *args, **kwargs: pred)
    monkeypatch.setattr(critic, "predict_x0", lambda *args, **kwargs: fake_x0)
    monkeypatch.setattr(
        teacher,
        "predict_x0",
        lambda *args, conditional, **kwargs: real_x0 if conditional else torch.zeros_like(real_x0),
    )

    loss, metrics, _ = method._tdm_generator_loss(SimpleNamespace(), batch)
    loss.backward()

    assert_close(pred.grad, torch.full_like(pred, -1.0))
    assert_close(
        torch.as_tensor(metrics["tdm/generator/normalization_denom"]),
        torch.tensor(50.5),
    )


def test_tdm_pseudo_huber_applies_per_sample_normalization_after_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    huber_c = 0.25
    method, student, critic = _build_method(method_overrides={
        "real_score_guidance_scale": 1.0,
        "use_huber": True,
        "huber_c": huber_c,
    })
    teacher = method.teacher
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    pred = torch.zeros((2, 1, 1, 1, 1), requires_grad=True)
    real_x0 = torch.tensor([1.0, 100.0]).reshape_as(pred)
    fake_x0 = torch.zeros_like(real_x0)
    context = SimpleNamespace(
        noisy_source=torch.zeros_like(pred),
        timestep_source=torch.tensor([500.0, 500.0]),
        timestep_target=torch.tensor([400.0, 400.0]),
        sigma_source=torch.tensor([0.5, 0.5]),
        sigma_intermediate=torch.tensor([0.25, 0.25]),
        sigma_target=torch.tensor([0.4, 0.4]),
        proposal_noise=torch.zeros_like(pred),
        trajectory_indices=torch.tensor([2, 2]),
    )

    monkeypatch.setattr(method, "_sample_tdm_context", lambda trajectory: context)
    monkeypatch.setattr(student, "predict_x0", lambda *args, **kwargs: pred)
    monkeypatch.setattr(critic, "predict_x0", lambda *args, **kwargs: fake_x0)
    monkeypatch.setattr(
        teacher,
        "predict_x0",
        lambda *args, conditional, **kwargs: real_x0 if conditional else torch.zeros_like(real_x0),
    )

    loss, _, _ = method._tdm_generator_loss(SimpleNamespace(), batch)
    loss.backward()

    reference_pred = torch.zeros_like(pred, requires_grad=True)
    reference_error = reference_pred - real_x0
    reference_denom = torch.abs(reference_pred.detach() - real_x0)
    reference_loss = ((torch.sqrt(reference_error.square() + huber_c**2) - huber_c) / reference_denom).mean()
    reference_loss.backward()

    assert_close(loss, reference_loss.detach())
    assert_close(pred.grad, reference_pred.grad)
    assert not torch.isclose(pred.grad[0], pred.grad[1]).item()


def test_tdm_use_huber_is_generator_only_and_matches_reference_pseudo_huber() -> None:
    mse_method, _, _ = _build_method(method_overrides={"use_huber": False})
    huber_method, _, _ = _build_method(method_overrides={"use_huber": True, "huber_c": 0.25})
    pred = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 0.0])

    expected = torch.sqrt((pred - target).square() + 0.25**2) - 0.25
    assert_close(huber_method._generator_elementwise_loss(pred, target), expected)
    assert_close(mse_method._generator_elementwise_loss(pred, target), (pred - target).square())

    mse_method.cuda_generator.manual_seed(321)
    huber_method.cuda_generator.manual_seed(321)
    mse_batch = mse_method.student.prepare_batch({}, generator=mse_method.cuda_generator, latents_source="zeros")
    huber_batch = huber_method.student.prepare_batch({}, generator=huber_method.cuda_generator, latents_source="zeros")
    mse_method.cuda_generator.manual_seed(654)
    huber_method.cuda_generator.manual_seed(654)
    mse_loss, _, _, _ = mse_method._tdm_fake_score_loss(mse_batch)
    huber_loss, _, _, _ = huber_method._tdm_fake_score_loss(huber_batch)
    assert_close(huber_loss, mse_loss)


def test_tdm_fake_score_weights_each_batch_element_at_its_target_sigma() -> None:
    method, _, _ = _build_method()
    context = SimpleNamespace(
        sigma_target=torch.tensor([0.96, 0.25], dtype=torch.float32),
        mixed_noise=torch.zeros(2, 1, dtype=torch.float32),
        proposal_noise=torch.zeros(2, 1, dtype=torch.float32),
    )

    components = method._tdm_fake_score_weight_components(context)

    assert_close(
        components["weights"],
        torch.tensor([1.0 / 576.0, 5.0], dtype=torch.float32),
    )
