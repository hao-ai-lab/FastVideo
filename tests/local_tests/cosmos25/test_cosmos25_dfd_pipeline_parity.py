# SPDX-License-Identifier: Apache-2.0
"""Real-weight four-step DFD V2W rollout parity against NVIDIA FastGen."""

from __future__ import annotations

import gc

import pytest
import torch

from fastvideo.models.schedulers.scheduling_cosmos25_dfd import Cosmos25DFDScheduler
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import Cosmos25DFDV2WDenoisingStage
from tests.local_tests.cosmos25.test_cosmos25_dfd_transformer_parity import (
    _load_fastvideo_model,
    _load_official_model,
    _load_student_checkpoint,
    distributed_setup,  # noqa: F401 - registers the fixture in this module.
)
from tests.local_tests.cosmos25.test_cosmos25_distilled_transformer_parity import _drift


class _Progress:

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def update(self) -> None:
        pass


def _rollout_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    noise = torch.randn((1, 16, 2, 16, 16), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    conditioning_latents = torch.randn(
        (1, 16, 1, 16, 16),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    condition_mask = torch.zeros((1, 1, 2, 16, 16), dtype=torch.bfloat16)
    condition_mask[:, :, :1] = 1
    return {
        "noise": noise,
        "conditioning_latents": conditioning_latents,
        "condition_mask": condition_mask,
        "text": torch.randn((1, 4, 100352), generator=generator, dtype=torch.float32).to(torch.bfloat16),
        "padding_mask": torch.zeros((1, 1, 16, 16), dtype=torch.bfloat16),
        "fps": torch.tensor([24], dtype=torch.float32),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_dfd_v2w_four_step_rollout_matches_official() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Cosmos Predict2.5 DFD pipeline parity gate")

    device = torch.device("cuda:0")
    student = _load_student_checkpoint()
    inputs = _rollout_inputs()
    schedule = Cosmos25DFDScheduler()
    schedule.set_timesteps(4, device=device)
    initial = schedule.scale_noise(
        torch.zeros_like(inputs["noise"], device=device),
        noise=inputs["noise"].to(device),
    )

    official = _load_official_model(student, device)
    condition = {
        "text_embeds": inputs["text"].to(device),
        "conditioning_latents": inputs["conditioning_latents"].to(device),
        "condition_mask": inputs["condition_mask"].to(device),
    }
    state = initial.clone()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timestep, next_timestep in zip(schedule.sigmas[:-1], schedule.sigmas[1:], strict=True):
            timestep_batch = timestep.expand(state.shape[0])
            prediction = official(
                state,
                timestep_batch,
                condition=condition,
                fwd_pred_type="x0",
                fps=inputs["fps"].to(device),
                padding_mask=inputs["padding_mask"][:, 0].to(device),
            )
            prediction = official.preserve_conditioning(prediction, condition)
            if float(next_timestep) > 0:
                inferred_noise = official.noise_scheduler.x0_to_eps(
                    xt=state,
                    x0=prediction,
                    t=timestep_batch,
                )
                state = official.noise_scheduler.forward_process(
                    prediction,
                    inferred_noise,
                    next_timestep.expand(state.shape[0]),
                )
                state = official.preserve_conditioning(state, condition)
        official_output = prediction.float().cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_model(student, device, rope_enable_fps_modulation=True)
    scheduler = Cosmos25DFDScheduler()
    scheduler.set_timesteps(4, device=device)
    stage = Cosmos25DFDV2WDenoisingStage.__new__(Cosmos25DFDV2WDenoisingStage)
    stage.transformer = fastvideo
    stage.scheduler = scheduler
    stage.pipeline = None
    stage.progress_bar = lambda **_kwargs: _Progress()
    batch = ForwardBatch(
        data_type="video",
        prompt_embeds=[inputs["text"].to(device)],
        latents=initial.clone(),
        timesteps=scheduler.timesteps,
        num_inference_steps=4,
        guidance_scale=1.0,
        fps=24,
    )
    batch.conditioning_latents = inputs["conditioning_latents"].to(device)
    batch.cond_indicator = inputs["condition_mask"][:, :, :, :1, :1].to(device)
    batch.cond_mask = inputs["condition_mask"].to(device)
    batch.padding_mask = inputs["padding_mask"].to(device)

    args = type(
        "Args",
        (),
        {
            "disable_autocast": False,
            "model_loaded": {"transformer": True},
            "model_paths": {},
        },
    )()
    with torch.inference_mode():
        stage.forward(batch, args)
    assert batch.latents is not None
    fastvideo_output = batch.latents.float().cpu()

    mean_abs, max_abs, relative_mean = _drift(official_output, fastvideo_output)
    print(
        "Cosmos25 DFD V2W four-step parity: "
        f"max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}, relative_mean={relative_mean:.8f}"
    )
    assert relative_mean < 0.10
    assert max_abs < 1.0

