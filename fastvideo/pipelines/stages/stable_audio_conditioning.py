# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio conditioning stage: T5 (prompt) + NumberEmbedder (seconds).
"""
from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage


class StableAudioConditioningStage(PipelineStage):
    """Run Stable Audio conditioner: prompt (T5) + seconds_start, seconds_total."""

    def __init__(self, conditioner) -> None:
        super().__init__()
        self.conditioner = conditioner

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if fastvideo_args.text_encoder_cpu_offload:
            device = next(self.conditioner.parameters()).device
        else:
            device = get_local_torch_device()
            self.conditioner = self.conditioner.to(device)

        prompts = batch.prompt
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        seconds_start = batch.seconds_start
        seconds_total = batch.seconds_total
        if seconds_start is None:
            seconds_start = 0.0
        if seconds_total is None:
            seconds_total = batch.duration_seconds or 10.0
        if isinstance(seconds_start, int | float):
            seconds_start = [float(seconds_start)] * batch_size
        if isinstance(seconds_total, int | float):
            seconds_total = [float(seconds_total)] * batch_size

        metadata = [{
            "prompt": p,
            "seconds_start": s0,
            "seconds_total": st
        } for p, s0, st in zip(
            prompts, seconds_start, seconds_total, strict=False)]

        conditioning = self.conditioner(metadata, device)

        batch.extra["stable_audio_conditioning"] = conditioning
        batch.is_prompt_processed = True

        cross_attn = conditioning.get("prompt", (None, None))[0]
        if cross_attn is not None:
            batch.prompt_embeds = [cross_attn]
            mask = conditioning["prompt"][1]
            batch.prompt_attention_mask = [mask] if mask is not None else None

        global_conds = []
        for key in ["seconds_start", "seconds_total"]:
            t, _ = conditioning.get(key, (None, None))
            if t is not None:
                global_conds.append(t.squeeze(1))
        if global_conds:
            batch.extra["stable_audio_global_cond"] = torch.cat(global_conds,
                                                                dim=-1)

        return batch
