# SPDX-License-Identifier: Apache-2.0
"""Wan sampling contracts with tiny CPU tensors and no model weights.

The real UniPC scheduler exercises the complete 50-step loop. These tests cover
orchestration, not transformer numerics; component goldens cover real weights.
"""

import pytest
import torch

from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.tests.stages._denoising_fixtures import (
    NullProgressBar, RecordingDenoiser, TinyScheduler, TinyVAE, _args, _batch, _patch_denoising_module,
)


def _stage(monkeypatch, model, scheduler, *, second=None, gate="1.0"):
    _, logger = _patch_denoising_module(monkeypatch, gate)
    from fastvideo.pipelines.basic.wan.stages import denoising
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    stage = denoising.WanDenoisingStage(model, scheduler, transformer_2=second)
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    return stage, logger


@pytest.mark.parametrize("steps", [3, 50])
@pytest.mark.parametrize("cfg", [False, True])
def test_wan_all_scheduler_steps_match_explicit_loop(monkeypatch, steps, cfg):
    batch = _batch(steps, cfg)
    scheduler = FlowUniPCMultistepScheduler(shift=3.0)
    scheduler.set_timesteps(steps, device="cpu")
    batch.timesteps = scheduler.timesteps.clone()
    reference_scheduler = FlowUniPCMultistepScheduler(shift=3.0)
    reference_scheduler.set_timesteps(steps, device="cpu")
    expected = batch.latents.clone()
    reference = RecordingDenoiser()
    trajectory = []
    for t in reference_scheduler.timesteps:
        x = reference_scheduler.scale_model_input(expected.to(torch.bfloat16), t)
        cond = reference(x, batch.prompt_embeds, t.repeat(x.shape[0]))
        if cfg:
            uncond = reference(x, batch.negative_prompt_embeds, t.repeat(x.shape[0]))
            cond = uncond + batch.guidance_scale * (cond - uncond)
        expected = reference_scheduler.step(cond, t, expected, return_dict=False)[0]
        trajectory.append(expected)

    model = RecordingDenoiser()
    stage, _ = _stage(monkeypatch, model, scheduler)
    result = stage.forward(batch, _args())

    torch.testing.assert_close(result.latents, expected, atol=0, rtol=0)
    torch.testing.assert_close(result.trajectory_latents, torch.stack(trajectory, dim=1), atol=0, rtol=0)
    torch.testing.assert_close(result.trajectory_timesteps, reference_scheduler.timesteps, atol=0, rtol=0)
    assert model.calls == (["cond", "uncond"] if cfg else ["cond"]) * steps
    assert scheduler.step_index == steps


def test_wan_expert_boundary_invalidates_cfg_cache(monkeypatch):
    primary, secondary = RecordingDenoiser(), RecordingDenoiser(offset=0.125)
    stage, logger = _stage(monkeypatch, primary, TinyScheduler(), second=secondary, gate="0.0")
    args, batch = _args(), _batch()
    args.pipeline_config.dit_config.boundary_ratio = 0.9
    batch.boundary_ratio = 0.5  # Per-request override wins; equality still uses the high-noise expert.
    batch.timesteps = torch.tensor([750.0, 500.0, 250.0, 1.0])
    batch.guidance_scale_2 = 3.0
    stage.forward(batch, args)
    assert primary.calls == ["cond", "uncond", "cond"]
    assert secondary.calls == ["cond", "uncond", "cond"]
    assert any("invalidations=1" in line and "fresh_uncond=2" in line for line in logger.infos)


@pytest.mark.parametrize("kind,channels", [("t2v", 2), ("i2v", 4), ("v2v", 6), ("lucy", 4), ("ti2v", 2)])
def test_wan_conditioning_layout_and_first_frame(monkeypatch, kind, channels):
    args, batch = _args(), _batch(cfg=False)
    model, vae = RecordingDenoiser(), TinyVAE()
    if kind == "i2v":
        batch.image_latent = torch.full_like(batch.latents, 0.25)
    if kind in {"v2v", "lucy"}:
        batch.video_latent = torch.full_like(batch.latents, 0.375)
    args.pipeline_config.lucy_edit_task = kind == "lucy"
    args.pipeline_config.ti2v_task = kind == "ti2v"
    if kind == "ti2v":
        batch.pil_image = torch.zeros(1, 3, 1, 16, 32)
    stage, _ = _stage(monkeypatch, model, TinyScheduler())
    from fastvideo.pipelines.basic.wan.stages import conditioning
    monkeypatch.setattr(conditioning, "get_local_torch_device", lambda: torch.device("cpu"))
    conditioning.WanFirstFrameEncodingStage(vae).forward(batch, args)
    result = stage.forward(batch, args)
    assert all(x.shape[1] == channels for x, _ in model.inputs)
    assert result.latents.shape == (1, 2, 3, 2, 4)
    if kind in {"lucy", "ti2v"}:
        assert all(t.shape == (1, 6) for _, t in model.inputs)
    else:
        assert all(t.shape == (1,) for _, t in model.inputs)
    if kind == "v2v":
        assert all(torch.count_nonzero(x[:, 4:]) == 0 for x, _ in model.inputs)
    if kind == "ti2v":
        expected = (torch.full((1, 2, 1, 2, 4), 0.5) - vae.shift_factor) * vae.scaling_factor
        torch.testing.assert_close(result.latents[:, :, :1], expected, atol=0, rtol=0)
        assert vae.encode_calls == 1
        assert model.inputs[0][1][0, :2].eq(0).all()
    else:
        assert vae.encode_calls == 0


def test_wan_request_does_not_reuse_previous_cfg_delta(monkeypatch):
    model = RecordingDenoiser()
    stage, _ = _stage(monkeypatch, model, TinyScheduler(), gate="0.0")
    first = stage.forward(_batch(), _args()).latents.clone()
    second = stage.forward(_batch(), _args()).latents.clone()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert model.calls == ["cond", "uncond", "cond", "cond", "cond"] * 2
