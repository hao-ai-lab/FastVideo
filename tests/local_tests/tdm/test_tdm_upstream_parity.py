"""Cross-implementation parity against the upstream-verified TDM math.

The standalone TDM port pins its assembled critic and generator updates
against an independent transcription of the official ``Luo-Yihong/TDM``
``train_tdm_demo.py`` expressions
(``feat/tdm-standalone-wan-port:tdm_standalone/tests/test_upstream_parity.py``).
This file ports that guard to the modular ``TDMMethod``: the assembled
context's noise identities, the critic composition
(clipped-SNR and importance weights), and the generator composition
(stop-gradient boundaries, CFG teacher target, per-sample delta
normalization) are compared against upstream formulas computed directly
from the method's own context tensors and the same tiny role models.

Reference configuration (upstream default): ``separate`` noise
intervals, no randomized midpoint, MSE with delta normalization, no
Huber, ``snr_clip = 5.0``, importance clamped at ``10.0``.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.distribution_matching.tdm import (
    _expand_sigma_for_latents,
    _mean_except_batch,
    flow_effective_noise,
    flow_snr,
    flow_transition_to_noisier_sigma,
)

from tests.local_tests.tdm.test_tdm_method_unit import _build_method


def _build_state():
    method, student, critic = _build_method()
    batch = student.prepare_batch(
        {},
        generator=torch.Generator(device="cpu").manual_seed(0),
        latents_source="zeros",
    )
    trajectory = method._student_trajectory(batch)
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(7)
    context = method._sample_tdm_context(trajectory)
    return method, student, critic, batch, trajectory, context


def _grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters() if parameter.grad is not None
    }


def _assert_grads_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-6)


def test_context_satisfies_upstream_noise_identities() -> None:
    method, _, _, _, trajectory, context = _build_state()

    assert context.sigma_source.shape == (2, )
    assert torch.all(context.sigma_intermediate <= context.sigma_target + 1e-6)
    assert torch.all(context.sigma_target < context.sigma_source - 1e-6)
    assert torch.equal(torch.sort(trajectory.sigmas, descending=True).values, trajectory.sigmas)

    eps_source = flow_effective_noise(
        context.noisy_source,
        context.clean_latents,
        context.sigma_source,
        eps=method._sigma_eps,
    )
    torch.testing.assert_close(context.eps_source, eps_source, rtol=1e-5, atol=1e-7)

    sigma_intermediate = _expand_sigma_for_latents(context.sigma_intermediate, context.clean_latents)
    torch.testing.assert_close(
        context.noisy_intermediate,
        (1.0 - sigma_intermediate) * context.clean_latents + sigma_intermediate * context.eps_source,
        rtol=1e-5,
        atol=1e-7,
    )

    sigma_target = _expand_sigma_for_latents(context.sigma_target, context.clean_latents)
    torch.testing.assert_close(
        context.noisy_target,
        (1.0 - sigma_target) * context.clean_latents + sigma_target * context.mixed_noise,
        rtol=1e-5,
        atol=1e-7,
    )

    alpha_ratio = (1.0 - sigma_target) / (1.0 - sigma_intermediate)
    expected_beta_sq = (sigma_target**2 - (alpha_ratio * sigma_intermediate)**2).clamp_min(0.0)
    torch.testing.assert_close(context.transition_beta**2, expected_beta_sq, rtol=1e-5, atol=1e-7)

    expected_mixed = ((context.eps_source / (1.0 - sigma_intermediate)) * (1.0 - sigma_target) *
                      sigma_intermediate + context.transition_beta * context.proposal_noise) / sigma_target
    torch.testing.assert_close(context.mixed_noise, expected_mixed, rtol=1e-5, atol=1e-7)

    scheduler_sigmas = method.student.noise_scheduler.sigmas
    for sample in range(context.sigma_target.shape[0]):
        assert torch.isclose(context.sigma_target[sample], scheduler_sigmas[
            torch.argmin((scheduler_sigmas - context.sigma_target[sample]).abs())])


def test_critic_loss_matches_upstream_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    method, student, critic, batch, trajectory, context = _build_state()
    monkeypatch.setattr(method, "_sample_tdm_context", lambda _trajectory: context)
    guidance_scale = float(method.method_config["real_score_guidance_scale"])
    assert guidance_scale == 4.5

    critic.transformer.zero_grad(set_to_none=True)
    loss, _, _, _ = method._tdm_fake_score_loss(trajectory, batch)
    loss.backward()
    method_grads = _grads(critic.transformer)

    critic.transformer.zero_grad(set_to_none=True)
    critic_timestep = method._model_timestep_for_sigma(context.sigma_target, critic)
    batch.timesteps = critic_timestep
    fake_x0 = critic.predict_x0(
        context.noisy_target,
        critic_timestep,
        batch,
        conditional=True,
        cfg_uncond=method._cfg_uncond,
        attn_kind="dense",
    )
    per_sample = _mean_except_batch((fake_x0.float() - context.clean_latents.float()).square())
    snr = flow_snr(context.sigma_target, eps=method._sigma_eps)
    snr_weight = torch.minimum(snr, torch.full_like(snr, method._snr_clip)).reshape(-1)
    mixed_sq = _mean_except_batch(context.mixed_noise.float().square())
    proposal_sq = _mean_except_batch(context.proposal_noise.float().square())
    importance = torch.exp((0.5 * (proposal_sq - mixed_sq)).clamp(-20.0, 20.0))
    importance = importance.clamp(max=method._importance_weight_clip)
    reference = (per_sample * (snr_weight * importance)).mean()
    reference.backward()
    reference_grads = _grads(critic.transformer)

    torch.testing.assert_close(loss.detach(), reference.detach(), rtol=1e-5, atol=1e-7)
    _assert_grads_equal(method_grads, reference_grads)


def test_generator_loss_matches_upstream_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    method, student, critic, batch, trajectory, context = _build_state()
    teacher = method.teacher
    monkeypatch.setattr(method, "_sample_tdm_context", lambda _trajectory: context)
    guidance_scale = float(method.method_config["real_score_guidance_scale"])

    student.transformer.zero_grad(set_to_none=True)
    loss, _, _ = method._tdm_generator_loss(trajectory, batch)
    loss.backward()
    method_grads = _grads(student.transformer)

    # The generator phase re-derives the target state from the current
    # (detached) student prediction; with no parameter update in between
    # it must reproduce the context the critic phase used.
    with torch.no_grad():
        source_timestep = method._model_timestep_for_sigma(context.sigma_source, student)
        batch.timesteps = source_timestep
        generator_pred_x0 = student.predict_x0(
            context.noisy_source,
            source_timestep,
            batch,
            conditional=True,
            cfg_uncond=method._cfg_uncond,
            attn_kind="vsa",
        )
        eps_source = flow_effective_noise(
            context.noisy_source,
            generator_pred_x0.detach(),
            context.sigma_source,
            eps=method._sigma_eps,
        )
        sigma_intermediate = _expand_sigma_for_latents(context.sigma_intermediate, generator_pred_x0)
        noisy_intermediate = ((1.0 - sigma_intermediate) * generator_pred_x0.detach() +
                              sigma_intermediate * eps_source)
        target_noisy_latents, _, _ = flow_transition_to_noisier_sigma(
            noisy_from=noisy_intermediate,
            clean_latents=generator_pred_x0.detach(),
            eps_from=eps_source,
            sigma_from=context.sigma_intermediate,
            sigma_to=context.sigma_target,
            proposal_noise=context.proposal_noise,
            eps=method._sigma_eps,
        )
    torch.testing.assert_close(
        target_noisy_latents,
        context.noisy_target,
        rtol=1e-4,
        atol=1e-6,
    )
    torch.testing.assert_close(
        generator_pred_x0.detach(),
        context.clean_latents,
        rtol=1e-4,
        atol=1e-6,
    )

    student.transformer.zero_grad(set_to_none=True)
    student_grad_timestep = method._model_timestep_for_sigma(context.sigma_source, student)
    batch.timesteps = student_grad_timestep
    pred_x0 = student.predict_x0(
        context.noisy_source,
        student_grad_timestep,
        batch,
        conditional=True,
        cfg_uncond=method._cfg_uncond,
        attn_kind="vsa",
    )
    with torch.no_grad():
        critic_timestep = method._model_timestep_for_sigma(context.sigma_target, critic)
        teacher_timestep = method._model_timestep_for_sigma(context.sigma_target, teacher)
        batch.timesteps = critic_timestep
        fake_x0 = critic.predict_x0(
            context.noisy_target,
            critic_timestep,
            batch,
            conditional=True,
            cfg_uncond=method._cfg_uncond,
            attn_kind="dense",
        )
        batch.timesteps = teacher_timestep
        real_cond_x0 = teacher.predict_x0(
            context.noisy_target,
            teacher_timestep,
            batch,
            conditional=True,
            cfg_uncond=method._cfg_uncond,
            attn_kind="dense",
        )
        real_uncond_x0 = teacher.predict_x0(
            context.noisy_target,
            teacher_timestep,
            batch,
            conditional=False,
            cfg_uncond=method._cfg_uncond,
            attn_kind="dense",
        )
        real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale
        delta = real_cfg_x0 - fake_x0
        target = pred_x0.detach() + torch.nan_to_num(delta)
        reduce_dims = tuple(range(1, pred_x0.ndim))
        denominator = torch.abs(pred_x0.detach() - real_cfg_x0).mean(dim=reduce_dims, keepdim=True)
    reference = ((pred_x0.float() - target.float()).square() / denominator.clamp_min(method._sigma_eps)).mean()
    reference.backward()
    reference_grads = _grads(student.transformer)

    torch.testing.assert_close(loss.detach(), reference.detach(), rtol=1e-5, atol=1e-7)
    _assert_grads_equal(method_grads, reference_grads)


def test_default_configuration_is_the_upstream_reference_mode() -> None:
    method, _, _, _, _, _ = _build_state()
    assert method._normalize_generator_delta is True
    assert method._use_huber is False
    assert method._use_pseudo_huber is False
    assert method._snr_clip == 5.0
    assert method._importance_weight_clip == 10.0
    assert method._noise_interval_mode == "separate"
    assert method._use_randmid is False
    assert method._rollout_sample_type == "sde"
    assert method.method_config.get("student_sample_type") == "sde"
