"""Warmup phase and step-ladder tests for the modular TDM method.

The standalone experiments established the winning recipe: a regression
warmup to the CFG-combined teacher x0 at the student's own rollout
states, followed by TDM at progressively smaller step counts. These
tests pin the modular wiring: the Wan-family default, the
student-only optimizer gating, the warmup target/backward composition
(with gradients), and ladder stage resolution.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from fastvideo.train.methods.distribution_matching.tdm import TDMMethod

from tests.local_tests.tdm.test_tdm_method_unit import _build_method


def _build_state(method_overrides: dict[str, Any] | None = None):
    method, student, critic = _build_method(method_overrides=method_overrides)
    batch = student.prepare_batch(
        {},
        generator=torch.Generator(device="cpu").manual_seed(0),
        latents_source="zeros",
    )
    return method, student, critic, batch


def _grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters() if parameter.grad is not None
    }


def _assert_grads_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-6)


def test_warmup_defaults_to_200_for_wan_and_zero_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    method, _, _, _ = _build_state()
    assert method._model_family() == "unknown"
    assert method._warmup_steps == 0

    monkeypatch.setattr(TDMMethod, "_model_family", lambda self: "wan")
    wan_method, _, _, _ = _build_state()
    assert wan_method._warmup_steps == 200
    assert wan_method.method_config["tdm_warmup_steps"] == 200

    override_method, _, _, _ = _build_state({"warmup_steps": 7})
    assert override_method._warmup_steps == 7

    with pytest.raises(ValueError, match="non-negative"):
        _build_state({"warmup_steps": -1})


def test_warmup_step_updates_student_only() -> None:
    warmup_steps = 4
    method, student, critic, _ = _build_state({
        "warmup_steps": warmup_steps,
        "tdm_denoising_steps": [1000, 750, 500, 250],
    })
    data_stream = iter([{} for _ in range(8)])
    loss_map, outputs, metrics = method.managed_train_step(data_stream, iteration=0)

    assert metrics["tdm/warmup"] == 1.0
    assert metrics["tdm/warmup_steps"] == float(warmup_steps)
    assert "tdm/warmup/loss" in metrics
    assert float(loss_map["fake_score_loss"]) == 0.0
    torch.testing.assert_close(loss_map["generator_loss"], loss_map["total_loss"])
    assert outputs == {}
    assert len(method._student_optimizer.state) > 0
    assert len(method._critic_optimizer.state) == 0
    assert critic.transformer.weight.grad is None
    assert student.transformer.weight.grad is None


def test_warmup_phase_gates_critic_out_of_optimizers_and_clip_targets() -> None:
    method, _, _, _ = _build_state({"warmup_steps": 4})
    assert method.get_optimizers(0) == [method._student_optimizer]
    assert method.get_lr_schedulers(0) == [method._student_lr_scheduler]
    assert method.get_grad_clip_targets(0) == {"student": method.student.transformer}
    optimizers = method.get_optimizers(4)
    assert method._critic_optimizer in optimizers
    assert method._student_optimizer in optimizers


def test_warmup_loss_matches_cfg_teacher_target(monkeypatch: pytest.MonkeyPatch) -> None:
    method, student, critic, batch = _build_state({"warmup_steps": 1})
    teacher = method.teacher
    with torch.no_grad():
        trajectory = method._student_trajectory(batch)

    fixed = {
        "noisy_source": trajectory.noisy_latents[0].detach(),
        "sigma_source": trajectory.sigmas[0].reshape(1).expand(2),
        "trajectory_indices": torch.zeros(2, dtype=torch.long),
    }

    def fixed_source(_trajectory: Any):
        timestep_source = method._model_timestep_for_sigma(fixed["sigma_source"], student)
        return (
            fixed["noisy_source"],
            fixed["sigma_source"],
            timestep_source,
            fixed["trajectory_indices"],
        )

    monkeypatch.setattr(method, "_sample_warmup_source", fixed_source)
    guidance_scale = float(method.method_config["real_score_guidance_scale"])
    assert guidance_scale == 4.5

    student.transformer.zero_grad(set_to_none=True)
    loss, metrics, _ = method._tdm_warmup_loss(trajectory, batch)
    loss.backward()
    method_grads = _grads(student.transformer)

    student.transformer.zero_grad(set_to_none=True)
    timestep_source = method._model_timestep_for_sigma(fixed["sigma_source"], student)
    batch.timesteps = timestep_source
    pred_x0 = student.predict_x0(
        fixed["noisy_source"],
        timestep_source,
        batch,
        conditional=True,
        cfg_uncond=method._cfg_uncond,
        attn_kind="vsa",
    )
    with torch.no_grad():
        teacher_timestep = method._model_timestep_for_sigma(fixed["sigma_source"], teacher)
        batch.timesteps = teacher_timestep
        real_cond_x0 = teacher.predict_x0(
            fixed["noisy_source"],
            teacher_timestep,
            batch,
            conditional=True,
            cfg_uncond=method._cfg_uncond,
            attn_kind="dense",
        )
        real_uncond_x0 = teacher.predict_x0(
            fixed["noisy_source"],
            teacher_timestep,
            batch,
            conditional=False,
            cfg_uncond=method._cfg_uncond,
            attn_kind="dense",
        )
        real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale
    reference = (pred_x0.float() - real_cfg_x0.float()).square().mean()
    reference.backward()
    reference_grads = _grads(student.transformer)

    torch.testing.assert_close(loss.detach(), reference.detach(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(metrics["tdm/warmup/guidance"], guidance_scale)
    _assert_grads_equal(method_grads, reference_grads)


LADDER_STAGES = [
    {
        "denoising_steps": list(range(1000, 0, -125)),
        "until_iteration": 5,
    },
    {
        "denoising_steps": [1000, 750, 500, 250],
        "until_iteration": 10,
    },
    {
        "denoising_steps": [1000, 500],
    },
]


def test_step_ladder_selects_stages_and_recomputes_sigmas() -> None:
    method, student, _, batch = _build_state({"tdm_step_ladder": LADDER_STAGES})

    expectations = [(0, 0), (4, 0), (5, 1), (9, 1), (10, 2), (99, 2)]
    for iteration, expected_index in expectations:
        method._select_schedule_stage(iteration)
        assert method._active_schedule_index == expected_index

    for iteration, expected_index, expected_steps in ((0, 0, 8), (6, 1, 4), (20, 2, 2)):
        method._select_schedule_stage(iteration)
        steps = method._get_denoising_step_list(torch.device("cpu"))
        assert steps.shape[0] == expected_steps
        assert torch.equal(steps, torch.tensor(LADDER_STAGES[expected_index]["denoising_steps"],
                                               dtype=torch.float32))
        assert method._denoising_sigma_list is not None
        assert method._denoising_sigma_list.shape[0] == expected_steps
        assert torch.all(method._denoising_sigma_list[:-1] > method._denoising_sigma_list[1:])

    method._select_schedule_stage(20)
    with torch.no_grad():
        trajectory = method._student_trajectory(batch)
    assert trajectory.timesteps.shape[0] == 2
    assert len(trajectory.sigmas) == 2
    assert method.method_config["dmd_denoising_steps"] == LADDER_STAGES[-1]["denoising_steps"]


@pytest.mark.parametrize(
    "ladder, message",
    [
        ([], "non-empty list"),
        ("not-a-list", "non-empty list"),
        ([{"until_iteration": 5}], "denoising_steps"),
        ([{"denoising_steps": [1000, 500]}, {"denoising_steps": [1000, 250]}], "omit until_iteration"),
        (
            [
                {"denoising_steps": [1000, 750, 500, 250], "until_iteration": 10},
                {"denoising_steps": [1000, 500], "until_iteration": 5},
            ],
            "strictly increasing",
        ),
        (
            [
                {"denoising_steps": [1000, 500], "until_iteration": 5},
                {"denoising_steps": [1000, 750, 500, 250]},
            ],
            "strictly decrease",
        ),
    ],
)
def test_step_ladder_validation(ladder: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_state({"tdm_step_ladder": ladder})


def test_warmup_only_control_never_touches_the_critic() -> None:
    """The standalone warmup-only control: no critic updates at all."""
    method, _, _, _ = _build_state({"warmup_steps": 3})
    data_stream = iter([{} for _ in range(6)])
    for iteration in range(3):
        metrics = method.managed_train_step(data_stream, iteration=iteration)[2]
        assert metrics["tdm/warmup"] == 1.0
    assert len(method._critic_optimizer.state) == 0
    assert len(method._student_optimizer.state) > 0


def test_tdm_phase_updates_both_roles() -> None:
    method, _, _, _ = _build_state({"warmup_steps": 0})
    data_stream = iter([{} for _ in range(6)])
    metrics = method.managed_train_step(data_stream, iteration=0)[2]
    assert metrics["tdm/warmup"] == 0.0
    assert len(method._critic_optimizer.state) > 0
    assert len(method._student_optimizer.state) > 0
