# SPDX-License-Identifier: Apache-2.0
"""DMD scheduler ownership, layout, and RNG ordering without checkpoints."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.tests.stages._denoising_fixtures import NullProgressBar, _patch_denoising_module
from fastvideo.tests.stages._denoising_fixtures import RecordingDenoiser, _args, _batch


def test_wan_dmd_pins_training_schedule_and_preserves_rng_order(monkeypatch):
    _patch_denoising_module(monkeypatch, "1.0")
    from fastvideo.pipelines.basic.wan.stages import dmd
    monkeypatch.setattr(dmd, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(dmd, "set_forward_context", lambda **kwargs: nullcontext())
    args, batch = _args(), _batch(steps=3, cfg=False)
    timesteps = [1000, 750, 500]
    args.pipeline_config.dmd_denoising_steps = timesteps
    batch.latents = batch.latents.permute(0, 2, 1, 3, 4)
    batch.generator = [torch.Generator().manual_seed(123)]
    expected_generator = torch.Generator().manual_seed(123)
    expected = batch.latents.clone()
    model, reference = RecordingDenoiser(), RecordingDenoiser()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    expected_scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    # Scheduler-space labels pin the denoising table to those exact steps.
    pinned = dmd.DmdDenoisingStage._set_dmd_timesteps(
        SimpleNamespace(scheduler=expected_scheduler),
        torch.tensor(timesteps, dtype=torch.long),
        device=torch.device("cpu"),
        scheduler_space=True,
    )
    for index in range(len(pinned)):
        t = pinned[index].reshape(1)
        noise = reference(expected.to(torch.bfloat16).permute(0, 2, 1, 3, 4), batch.prompt_embeds, t)
        noise = noise.permute(0, 2, 1, 3, 4)
        clean = pred_noise_to_pred_video(noise.flatten(0, 1), expected.flatten(0, 1), t, expected_scheduler)
        clean = clean.unflatten(0, noise.shape[:2])
        if index + 1 < len(pinned):
            random = torch.randn(expected.shape, generator=expected_generator, dtype=clean.dtype)
            expected = expected_scheduler.add_noise(
                clean.flatten(0, 1), random.flatten(0, 1), pinned[index + 1].reshape(1),
            ).unflatten(0, noise.shape[:2])
        else:
            expected = clean

    stage = dmd.DmdDenoisingStage(model, scheduler)
    assert stage.scheduler is scheduler
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    result = stage.forward(batch, args)
    torch.testing.assert_close(result.latents, expected.permute(0, 2, 1, 3, 4), atol=0, rtol=0)
    assert torch.equal(batch.generator[0].get_state(), expected_generator.get_state())
    assert scheduler.timesteps.tolist() == [1000.0, 750.0, 500.0]
    torch.testing.assert_close(scheduler.timesteps, pinned)
    assert model.calls == ["cond"] * 3


def test_default_dmd_labels_match_dmd2_training_sigma_grid():
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.models.wan.definition import DMD_TRAINING_NOISE_SHIFT
    from fastvideo.pipelines.basic.wan.stages.dmd import DmdDenoisingStage

    config = PipelineConfig()
    assert config.dmd_denoising_steps_are_scheduler_space is True

    labels = torch.tensor([1000, 750, 500, 250], dtype=torch.long)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=DMD_TRAINING_NOISE_SHIFT)
    DmdDenoisingStage._set_dmd_timesteps(
        SimpleNamespace(scheduler=scheduler),
        labels,
        device=torch.device("cpu"),
        scheduler_space=config.dmd_denoising_steps_are_scheduler_space,
    )

    # DMD2 training resolves these same labels straight out of the scheduler's
    # full shifted table (argmin over timesteps), which is sigma ~ label / T.
    training_table = FlowMatchEulerDiscreteScheduler(shift=DMD_TRAINING_NOISE_SHIFT)
    ids = torch.argmin((training_table.timesteps.unsqueeze(0) - labels.unsqueeze(1)).abs(), dim=1)
    training_sigmas = training_table.sigmas[ids]
    torch.testing.assert_close(scheduler.sigmas[:-1], training_sigmas, rtol=1e-3, atol=1e-3)

    # The default must not reproduce TDM's flow-shifted grid.
    u = labels.float() / 1000.0
    shifted = DMD_TRAINING_NOISE_SHIFT * u / (1.0 + (DMD_TRAINING_NOISE_SHIFT - 1.0) * u)
    assert not torch.allclose(scheduler.sigmas[:-1], shifted)


def test_legacy_sampling_exports_are_canonical():
    from fastvideo.pipelines import stages
    from fastvideo.pipelines.basic.wan.stages import causal_denoising, dmd
    from fastvideo.pipelines.stages import causal_denoising as legacy_causal
    from fastvideo.pipelines.stages import denoising as legacy_dense
    assert stages.DmdDenoisingStage is legacy_dense.DmdDenoisingStage is dmd.DmdDenoisingStage
    for name in ("CausalDMDDenosingStage", "CausalDenoisingStage"):
        assert getattr(stages, name) is getattr(legacy_causal, name) is getattr(causal_denoising, name)
