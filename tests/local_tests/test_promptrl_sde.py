# SPDX-License-Identifier: Apache-2.0
"""SDE transition math tests for 5D video latents."""

from __future__ import annotations

import math

import torch

from fastvideo.train.methods.rl.promptrl.sde import (
    SDETransition,
    sde_step_from_model_output,
    transition_kl_to_reference,
    transition_log_prob,
)

SIGMA_MAX = 0.9


def _rand_latents(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 3, 6, 8, generator=generator)  # [B, C, T, H, W]


class TestSDEStep:
    def test_zero_noise_scale_reduces_to_euler(self):
        sample = _rand_latents()
        model_output = _rand_latents(seed=1)
        sigma, sigma_next = 0.5, 0.3
        prev, log_prob, mean = sde_step_from_model_output(
            model_output,
            sample,
            sigma=sigma,
            sigma_next=sigma_next,
            noise_scale=1e-12,
            sigma_max=SIGMA_MAX,
        )
        euler = sample + (sigma_next - sigma) * model_output
        assert torch.allclose(prev, euler, atol=1e-4)
        assert torch.allclose(mean, euler, atol=1e-4)
        # Deterministic step: sampled point sits at the mean.
        assert torch.isfinite(log_prob).all()

    def test_shapes_and_reduction(self):
        sample = _rand_latents()
        model_output = _rand_latents(seed=1)
        prev, log_prob, mean = sde_step_from_model_output(
            model_output,
            sample,
            sigma=0.7,
            sigma_next=0.5,
            noise_scale=0.8,
            sigma_max=SIGMA_MAX,
            generator=torch.Generator().manual_seed(7),
        )
        assert prev.shape == sample.shape
        assert mean.shape == sample.shape
        assert log_prob.shape == (sample.shape[0], )
        assert torch.isfinite(log_prob).all()

    def test_log_prob_matches_manual_gaussian(self):
        sample = _rand_latents()
        model_output = _rand_latents(seed=1)
        sigma, sigma_next, noise_scale = 0.6, 0.4, 0.8
        prev, log_prob, mean = sde_step_from_model_output(
            model_output,
            sample,
            sigma=sigma,
            sigma_next=sigma_next,
            noise_scale=noise_scale,
            sigma_max=SIGMA_MAX,
            generator=torch.Generator().manual_seed(3),
        )
        dt = sigma_next - sigma
        std = math.sqrt(sigma / (1 - sigma)) * noise_scale
        step_std = std * math.sqrt(-dt)
        manual = (-((prev - mean) ** 2) / (2 * step_std**2) - math.log(step_std) -
                  math.log(math.sqrt(2 * math.pi)))
        manual = manual.mean(dim=tuple(range(1, manual.ndim)))
        assert torch.allclose(log_prob, manual, atol=1e-4)


class TestTransitionRecomputation:
    def test_recompute_recovers_rollout_log_prob(self):
        sample = _rand_latents()
        model_output = _rand_latents(seed=1)
        kwargs = dict(sigma=0.65, sigma_next=0.45, noise_scale=0.8, sigma_max=SIGMA_MAX)
        prev, rollout_log_prob, _ = sde_step_from_model_output(
            model_output, sample, generator=torch.Generator().manual_seed(11), **kwargs)
        recomputed_log_prob, mean = transition_log_prob(
            model_output, sample, prev, **kwargs)
        assert torch.allclose(recomputed_log_prob, rollout_log_prob, atol=1e-5)
        assert mean.requires_grad is False

    def test_recompute_gradients_flow_to_model_output(self):
        sample = _rand_latents()
        model_output = _rand_latents(seed=1).requires_grad_(True)
        prev = _rand_latents(seed=2)
        log_prob, mean = transition_log_prob(
            model_output,
            sample,
            prev,
            sigma=0.65,
            sigma_next=0.45,
            noise_scale=0.8,
            sigma_max=SIGMA_MAX,
        )
        assert log_prob.requires_grad and mean.requires_grad
        (-log_prob.mean()).backward()
        assert model_output.grad is not None
        assert torch.isfinite(model_output.grad).all()
        assert model_output.grad.abs().sum() > 0

    def test_kl_zero_for_identical_means(self):
        mean = _rand_latents()
        kl = transition_kl_to_reference(
            mean, mean.clone(), sigma=0.6, sigma_next=0.4, noise_scale=0.8, sigma_max=SIGMA_MAX)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_kl_matches_gaussian_formula(self):
        mean_new = _rand_latents()
        mean_ref = _rand_latents(seed=5)
        sigma, sigma_next, noise_scale = 0.6, 0.4, 0.8
        kl = transition_kl_to_reference(
            mean_new, mean_ref,
            sigma=sigma, sigma_next=sigma_next,
            noise_scale=noise_scale, sigma_max=SIGMA_MAX)
        dt = sigma_next - sigma
        std = math.sqrt(sigma / (1 - sigma)) * noise_scale
        step_std_sq = std * std * (-dt)
        manual = ((mean_new - mean_ref) ** 2).sum(dim=(1, 2, 3, 4)) / (2 * step_std_sq)
        assert torch.allclose(kl, manual, atol=1e-3)

    def test_transition_dataclass_detached_storage(self):
        sample = _rand_latents().requires_grad_(True)
        transition = SDETransition(
            sample=sample.detach(),
            prev_sample=sample.detach(),
            timestep=500.0,
            sigma=0.5,
            sigma_next=0.3,
            old_log_prob=torch.zeros(2),
        )
        assert not transition.sample.requires_grad
        assert transition.timestep == 500.0
