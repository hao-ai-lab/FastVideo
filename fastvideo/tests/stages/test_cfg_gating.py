# SPDX-License-Identifier: Apache-2.0
"""Shared CFG regression tests; no weights or CUDA execution required."""

import pytest
import torch

from fastvideo.tests.stages._denoising_fixtures import (
    NullProgressBar, TinyDenoiser, TinyScheduler, _patch_denoising_module, _tiny_args, _tiny_batch,
)


def _run_stage(monkeypatch, cfg_gate_step):
    denoising, logger = _patch_denoising_module(monkeypatch, cfg_gate_step)
    model = TinyDenoiser()
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()

    result = stage.forward(_tiny_batch(), _tiny_args())
    return result.latents, model, logger


def _run_legacy_two_pass():
    batch = _tiny_batch()
    model = TinyDenoiser()
    scheduler = TinyScheduler()
    assert batch.latents is not None
    assert batch.timesteps is not None
    latents = batch.latents.clone()

    for timestep in batch.timesteps:
        latent_model_input = scheduler.scale_model_input(latents.to(torch.bfloat16), timestep)
        timestep_expand = timestep.repeat(latent_model_input.shape[0])
        noise_pred_text = model(latent_model_input, batch.prompt_embeds, timestep_expand)
        noise_pred_uncond = model(latent_model_input, batch.negative_prompt_embeds, timestep_expand)
        noise_pred = noise_pred_uncond + batch.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    return latents


@pytest.mark.parametrize("cfg_gate_step", [None, "1.0"])
def test_cfg_gating_default_off_matches_legacy_two_pass(monkeypatch, cfg_gate_step):
    out, model, logger = _run_stage(monkeypatch, cfg_gate_step)
    legacy_out = _run_legacy_two_pass()

    assert torch.equal(out, legacy_out)
    assert model.calls == ["cond", "uncond"] * 4
    assert not any("CFG gating enabled" in msg for msg in logger.infos)
    assert any("gate_step=-1/4" in msg and "reused=0" in msg for msg in logger.infos)


def test_cfg_gating_reuses_cached_delta_after_gate(monkeypatch):
    out, model, logger = _run_stage(monkeypatch, "0.5")
    legacy_out = _run_legacy_two_pass()

    assert model.calls == ["cond", "uncond", "cond", "uncond", "cond", "cond"]
    assert any("CFG gating enabled: fraction=0.500, gate_step=2/4" in msg for msg in logger.infos)
    assert any("fresh_uncond=2 reused=2 invalidations=0" in msg for msg in logger.infos)
    assert torch.allclose(out, legacy_out, atol=1e-3, rtol=0.0)
