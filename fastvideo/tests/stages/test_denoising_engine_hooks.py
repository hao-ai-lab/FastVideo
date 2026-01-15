# SPDX-License-Identifier: Apache-2.0
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
from fastvideo.pipelines.stages.denoising_engine_hooks import PerfLoggingHook
from fastvideo.pipelines.stages.denoising_strategies import (ModelInputs,
                                                             StrategyState)


class DummyStrategy:
    def prepare(self, batch: ForwardBatch,
                args: FastVideoArgs) -> StrategyState:
        latents = torch.zeros(1, 1, 1, 1, 1)
        timesteps = torch.tensor([2, 1], dtype=torch.long)
        return StrategyState(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=len(timesteps),
            prompt_embeds=[torch.zeros(1, 1, 1)],
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_attention_mask=None,
            image_embeds=[],
            guidance_scale=1.0,
            guidance_scale_2=None,
            guidance_rescale=0.0,
            do_cfg=False,
            extra={"batch": batch},
        )

    def make_model_inputs(self, state: StrategyState, t: torch.Tensor,
                          step_idx: int) -> ModelInputs:
        return ModelInputs(
            latent_model_input=state.latents,
            timestep=t,
            prompt_embeds=state.prompt_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
        )

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        return state.latents

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        return state.latents

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        return state.extra["batch"]


def test_perf_logging_hook_records_steps():
    batch = ForwardBatch(data_type="test")
    args = FastVideoArgs(model_path="dummy")
    hook = PerfLoggingHook()
    engine = DenoisingEngine(DummyStrategy(), hooks=[hook])

    engine.run(batch, args)

    info = batch.logging_info.get_stage_info("DenoisingEngine")
    step_times = info.get("denoise_step_times_ms")
    total_ms = info.get("denoise_total_ms")

    assert isinstance(step_times, list)
    assert len(step_times) == 2
    assert total_ms is not None
    assert total_ms >= 0.0
