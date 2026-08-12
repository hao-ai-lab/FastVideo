# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method


class _RecordingStudent:

    def __init__(self) -> None:
        self.predict_calls: list[torch.Tensor] = []
        self.add_noise_calls: list[torch.Tensor] = []

    def predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del batch, kwargs
        self.predict_calls.append(timestep.detach().clone())
        return noisy_latents + timestep.to(noisy_latents.dtype)

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self.add_noise_calls.append(timestep.detach().clone())
        return clean_latents + noise


def _rollout_method(seed: int = 1234) -> tuple[DMD2Method, _RecordingStudent]:
    method = object.__new__(DMD2Method)
    torch.nn.Module.__init__(method)
    student = _RecordingStudent()
    method.student = student
    method._rollout_mode = "simulate"
    method._cfg_uncond = None
    method._denoising_step_list = torch.tensor([1000, 750, 500, 250])
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(seed)
    return method, student


def _legacy_full_rollout_reference(
    *,
    seed: int,
    target_idx: int,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the pre-optimization simulate rollout and RNG state."""
    step_list = torch.tensor([1000, 750, 500, 250])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    current = torch.randn(shape, generator=generator)
    initial = current.clone()
    noise_latents: list[torch.Tensor] = []

    for step_idx in range(len(step_list) - 1):
        pred_clean = current + step_list[step_idx].to(current.dtype)
        noise = torch.randn(shape, generator=generator)
        current = pred_clean + noise
        noise_latents.append(current.clone())

    noisy_input: torch.Tensor
    if target_idx == 0:
        noisy_input = initial
    else:
        noisy_input = noise_latents[target_idx - 1]
    output = noisy_input + step_list[target_idx].to(noisy_input.dtype)
    return output, generator.get_state()


@pytest.mark.parametrize("target_idx", range(4))
def test_simulate_rollout_only_runs_required_prefix_forwards(
    monkeypatch: pytest.MonkeyPatch,
    target_idx: int,
) -> None:
    method, student = _rollout_method()
    batch = SimpleNamespace(
        latents=torch.zeros((1, 2)),
        dmd_latent_vis_dict={},
    )

    def _fixed_target(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([target_idx], dtype=torch.long)

    monkeypatch.setattr(torch, "randint", _fixed_target)
    method._student_rollout(batch, with_grad=True)

    # One prediction per required prefix step, plus the differentiable target
    # prediction. Prefix noising only happens for the required prefix steps.
    assert len(student.predict_calls) == target_idx + 1
    assert len(student.add_noise_calls) == target_idx


@pytest.mark.parametrize("target_idx", range(4))
def test_simulate_rollout_preserves_method_generator_progress(
    monkeypatch: pytest.MonkeyPatch,
    target_idx: int,
) -> None:
    seed = 4321
    method, _ = _rollout_method(seed)
    shape = (1, 2)
    batch = SimpleNamespace(
        latents=torch.zeros(shape),
        dmd_latent_vis_dict={},
    )

    def _fixed_target(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([target_idx], dtype=torch.long)

    monkeypatch.setattr(torch, "randint", _fixed_target)
    output = method._student_rollout(batch, with_grad=False)

    # The previous implementation drew the initial latent and one noise tensor
    # for each of the three possible prefix transitions. Keep consuming those
    # draws so subsequent DMD2 randomness remains aligned across the change.
    reference_output, reference_state = _legacy_full_rollout_reference(
        seed=seed,
        target_idx=target_idx,
        shape=shape,
    )
    assert torch.equal(output, reference_output)
    assert torch.equal(
        method.cuda_generator.get_state(),
        reference_state,
    )


@pytest.mark.parametrize("target_idx", range(4))
def test_simulate_rollout_uses_global_max_prefix_without_changing_local_output(
    monkeypatch: pytest.MonkeyPatch,
    target_idx: int,
) -> None:
    seed = 9876
    method, student = _rollout_method(seed)
    shape = (1, 2)
    batch = SimpleNamespace(
        latents=torch.zeros(shape),
        dmd_latent_vis_dict={},
    )

    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([target_idx], dtype=torch.long),
    )
    monkeypatch.setattr(
        method,
        "_max_rollout_target_idx_across_ranks",
        lambda sampled_idx: 3,
    )

    output = method._student_rollout(batch, with_grad=False)
    reference_output, reference_state = _legacy_full_rollout_reference(
        seed=seed,
        target_idx=target_idx,
        shape=shape,
    )

    # Every rank participates in the globally required three prefix forwards,
    # then evaluates its own target. The local result and method-owned RNG
    # sequence still match the legacy full rollout exactly.
    assert len(student.predict_calls) == 4
    assert len(student.add_noise_calls) == 3
    assert torch.equal(output, reference_output)
    assert torch.equal(method.cuda_generator.get_state(), reference_state)


def test_dmd2_role_update_cadence_follows_selected_optimizers() -> None:
    method = object.__new__(DMD2Method)
    torch.nn.Module.__init__(method)
    method.method_config = {"generator_update_interval": 5}
    method._student_optimizer = object()
    method._critic_optimizer = object()

    assert not method.did_update_role("student", iteration=4)
    assert method.did_update_role("student", iteration=5)
    assert method.did_update_role("critic", iteration=4)
    assert not method.did_update_role("teacher", iteration=5)
