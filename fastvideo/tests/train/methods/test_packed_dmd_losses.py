# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for packed carried-DMD objectives, masks, guidance, and clocks."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.tests.train.methods.test_carried_dmd import _build, _method_config


class _Prediction:

    def __init__(self, value):
        self.value = value
        self.calls = []

    def predict_x0(self, noisy, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        self.calls.append((conditional, cfg_uncond, attn_kind))
        return self.value if conditional else self.value * 0.5


def _batch(padded):
    return SimpleNamespace(
        minimax_h3_dmd_layout=SimpleNamespace(pad_tokens=3, valid=torch.tensor([True] * 8 + [False] * 3))
        if padded else None,
        # Joint transformer rows and scalar latent columns have different
        # coordinates: this mask must never be used for latent reductions.
        minimax_h3_layout=SimpleNamespace(valid=torch.tensor([False, True])),
        dmd_latent_vis_dict={},
        timesteps=None,
        attn_metadata=None,
    )


def _padded(tensor):
    return torch.cat((tensor, torch.tensor([[float("nan"), float("inf"), 1e30]])), dim=-1)


@pytest.mark.parametrize("guided", [False, True])
def test_vsd_loss_and_gradients_ignore_poisoned_padding(guided):
    guidance = 2.0 if guided else 1.0
    method = _build(_method_config(real_score_guidance_scale=guidance,
                                                  fake_score_guidance_scale=guidance,
                                                  modality_loss_weights={"video": 0.5, "audio": 2.0}))
    gen = torch.arange(8, dtype=torch.float32).reshape(1, -1) + 3
    fake = gen - 1
    real = gen - 2
    reference = None
    for padded in (False, True):
        value = (_padded(gen) if padded else gen.clone()).requires_grad_()
        method.critic = _Prediction(_padded(fake) if padded else fake)
        method.teacher = _Prediction(_padded(real) if padded else real)
        loss, metrics = method._dmd_loss(value, _batch(padded))
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(value.grad).all()
        assert len(method.critic.calls) == len(method.teacher.calls) == (2 if guided else 1)
        if reference is None:
            reference = loss.detach(), metrics, value.grad.clone()
        else:
            assert torch.equal(loss, reference[0])
            assert metrics.keys() == reference[1].keys()
            for name in metrics:
                assert torch.equal(metrics[name], reference[1][name])
            assert torch.equal(value.grad[:, :8], reference[2])
            assert torch.count_nonzero(value.grad[:, 8:]) == 0


def test_x0_critic_loss_uses_independent_modality_means_and_excludes_padding():
    method = _build(_method_config(modality_loss_weights={"video": 0.5, "audio": 2.0}))
    target = torch.arange(8, dtype=torch.float32).reshape(1, -1)
    prediction = target + torch.tensor([[2.0] * 6 + [3.0] * 2])
    reference_grad = None
    for padded in (False, True):
        pred = (_padded(prediction) if padded else prediction.clone()).requires_grad_()
        method.critic = _Prediction(pred)
        loss, _, _, metrics = method._critic_flow_matching_loss(
            _batch(padded), generator_pred_x0=_padded(target) if padded else target)
        loss.backward()
        assert loss.item() == 0.5 * 4 + 2.0 * 9
        assert metrics["fake_score_loss_video"].item() == 4
        assert metrics["fake_score_loss_audio"].item() == 9
        if reference_grad is None:
            reference_grad = pred.grad.clone()
        else:
            assert torch.equal(pred.grad[:, :8], reference_grad)
            assert torch.count_nonzero(pred.grad[:, 8:]) == 0
            assert torch.isfinite(pred.grad).all()


def test_continuous_score_sampling_matches_shifted_float64_uniform_draws():
    method = _build(_method_config(score_timestep_continuous=True,
                                                  score_timestep_shift=2.4,
                                                  score_timestep_warp_max=0.999))
    reference_generator = torch.Generator().manual_seed(0)
    u = torch.rand((6,), dtype=torch.float64, generator=reference_generator) * (0.999 - 0.001) + 0.001
    inverse_shift = 1 / 2.4
    expected = 1000 * (u * inverse_shift * 0.999 / (u * (inverse_shift - 1) + 0.999))
    actual = method._sample_score_timestep(torch.device("cpu"), count=6)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_condition_dropout_does_not_consume_an_extra_rng_draw(monkeypatch):
    method = _build(_method_config(fake_score_cfg_dropout_rate=0.0))
    method.critic = _Prediction(torch.ones(1, 8))
    monkeypatch.setattr(method, "_critic_condition_dropped", lambda: pytest.fail("zero dropout must not draw"))
    method._critic_flow_matching_loss(_batch(False), generator_pred_x0=torch.zeros(1, 8))
    assert method.critic.calls == [(True, method._cfg_uncond, "dense")]


def test_dmd_loss_matches_independent_modality_normalizers_and_detached_target():
    method = _build()
    gen = torch.tensor([[1., 2., 3., 4., 5., 6., 10., 14.]], requires_grad=True)
    real = torch.zeros_like(gen)
    fake = torch.tensor([[1.] * 6 + [2.] * 2])
    method.teacher = _Prediction(real)
    method.critic = _Prediction(fake)
    loss, metrics = method._dmd_loss(gen, _batch(False))
    loss.backward()
    expected_grad = torch.empty_like(gen)
    expected_loss = 0.0
    for columns in (slice(0, 6), slice(6, 8)):
        values = gen.detach()[:, columns]
        correction = fake[:, columns] / (values.abs().mean() + 1e-6)
        # Preserve FP32 target subtraction: it is the stop-gradient target,
        # not a scalar surrogate for the mathematically equivalent gradient.
        target = values - correction
        residual = values - target
        expected_loss += 0.5 * residual.square().mean()
        expected_grad[:, columns] = residual / values.numel()
    torch.testing.assert_close(loss, expected_loss, atol=0, rtol=0)
    # The MSE backward kernel multiplies by 2/N before the outer 0.5;
    # the analytic residual/N reference can differ by one FP32 rounding.
    torch.testing.assert_close(gen.grad, expected_grad, atol=0, rtol=1e-7)
    assert torch.equal(loss.detach(), metrics["generator_loss_video"] + metrics["generator_loss_audio"])


@pytest.mark.parametrize("setting", [
    {"fake_score_guidance_scale": 0.0},
    {"fake_score_cfg_dropout_rate": 1.0},
    {"score_timestep_continuous": "true"},
    {"score_timestep_shift": -1.0},
    {"score_timestep_warp_max": 0.5},
])
def test_packed_score_configuration_rejects_invalid_settings(setting):
    with pytest.raises(ValueError):
        _build(_method_config(**setting))
