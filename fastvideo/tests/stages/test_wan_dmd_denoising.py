# SPDX-License-Identifier: Apache-2.0
"""DMD scheduler ownership, layout, and RNG ordering without checkpoints."""

from contextlib import nullcontext

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.tests.stages._denoising_fixtures import NullProgressBar, _patch_denoising_module
from fastvideo.tests.stages._denoising_fixtures import RecordingDenoiser, _args, _batch


def test_wan_dmd_uses_full_training_table_and_preserves_rng_order(monkeypatch):
    _patch_denoising_module(monkeypatch, "1.0")
    from fastvideo.pipelines.basic.wan.stages import dmd
    monkeypatch.setattr(dmd, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(dmd, "set_forward_context", lambda **kwargs: nullcontext())
    args, batch = _args(), _batch(steps=3, cfg=False)
    args.pipeline_config.dmd_denoising_steps = [1000, 750, 500]
    batch.latents = batch.latents.permute(0, 2, 1, 3, 4)
    batch.generator = [torch.Generator().manual_seed(123)]
    expected_generator = torch.Generator().manual_seed(123)
    expected = batch.latents.clone()
    model, reference = RecordingDenoiser(), RecordingDenoiser()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    expected_scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    table = scheduler.timesteps.clone()
    sigmas = scheduler.sigmas.clone()
    timesteps = args.pipeline_config.dmd_denoising_steps
    for index, timestep in enumerate(timesteps):
        t = torch.tensor([timestep], dtype=torch.long)
        noise = reference(expected.to(torch.bfloat16).permute(0, 2, 1, 3, 4), batch.prompt_embeds, t)
        noise = noise.permute(0, 2, 1, 3, 4)
        clean = pred_noise_to_pred_video(noise.flatten(0, 1), expected.flatten(0, 1), t, expected_scheduler)
        clean = clean.unflatten(0, noise.shape[:2])
        if index + 1 < len(timesteps):
            random = torch.randn(expected.shape, generator=expected_generator, dtype=clean.dtype)
            expected = expected_scheduler.add_noise(
                clean.flatten(0, 1), random.flatten(0, 1), torch.tensor([timesteps[index + 1]], dtype=torch.long),
            ).unflatten(0, noise.shape[:2])
        else:
            expected = clean

    stage = dmd.DmdDenoisingStage(model, scheduler)
    assert stage.scheduler is scheduler
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    result = stage.forward(batch, args)
    torch.testing.assert_close(result.latents, expected.permute(0, 2, 1, 3, 4), atol=0, rtol=0)
    assert torch.equal(batch.generator[0].get_state(), expected_generator.get_state())
    assert torch.equal(scheduler.timesteps, table) and torch.equal(scheduler.sigmas, sigmas)
    assert len(table) == 1000
    assert model.calls == ["cond"] * 3


def test_legacy_sampling_exports_are_canonical():
    from fastvideo.pipelines import stages
    from fastvideo.pipelines.basic.wan.stages import causal_denoising, dmd
    from fastvideo.pipelines.stages import causal_denoising as legacy_causal
    from fastvideo.pipelines.stages import denoising as legacy_dense
    assert stages.DmdDenoisingStage is legacy_dense.DmdDenoisingStage is dmd.DmdDenoisingStage
    for name in ("CausalDMDDenosingStage", "CausalDenoisingStage"):
        assert getattr(stages, name) is getattr(legacy_causal, name) is getattr(causal_denoising, name)
