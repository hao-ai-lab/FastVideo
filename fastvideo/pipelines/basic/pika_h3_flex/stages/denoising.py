# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex's fixed six-forward re-noise denoiser."""

from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import MiniMaxH3DenoisingStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def request_seed(batch: ForwardBatch) -> int:
    """The request seed, falling back to the generator's initial seed."""
    if batch.seed is not None:
        return int(batch.seed)
    generator = batch.generator
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is None:
        return 0
    return int(generator.initial_seed())


class PikaH3FlexDenoisingStage(MiniMaxH3DenoisingStage):
    """Six full-model forwards over Pika's clean-time ladder.

    Pika's adapter is not a PDD student: every rung is an ordinary transformer
    forward.  What differs from MiniMax-H3 is the ladder and the re-noise step,
    which draws fresh Gaussian noise from a generator seeded ``seed + 1``.
    """

    def _configure_schedules(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        device: torch.device,
    ):
        steps = list(fastvideo_args.pipeline_config.pika_denoising_steps)
        if batch.num_inference_steps != len(steps):
            raise ValueError(f"Pika H3 Flex needs {len(steps)} transformer forwards plus the terminal zero, got "
                             f"num_inference_steps={batch.num_inference_steps}.")
        for scheduler in (self.scheduler, self.audio_scheduler):
            if not getattr(scheduler, "stochastic_sampling", False):
                raise ValueError("Pika H3 Flex needs the released stochastic (re-noise) scheduler; "
                                 f"{type(scheduler).__name__} did not load stochastic_sampling=true.")
        self._set_shifted_schedule(steps, len(steps) + 1, device, label="Pika H3 Flex")
        return None

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        generator = torch.Generator(device=get_local_torch_device()).manual_seed(request_seed(batch) + 1)
        self.scheduler.set_noise_generator(generator)
        self.audio_scheduler.set_noise_generator(generator)
        return super().forward(batch, fastvideo_args)


__all__ = ["PikaH3FlexDenoisingStage", "request_seed"]
