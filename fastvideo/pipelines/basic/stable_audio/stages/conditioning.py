# SPDX-License-Identifier: Apache-2.0
"""Stable Audio conditioning stage — native StableAudioMultiConditioner path."""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioConditioningStage(PipelineStage):
    """Run T5 + NumberConditioner via the native MultiConditioner and
    package the result for the denoising loop.

    Stashes on `batch.extra`:
      * `cross_attn_cond`            — cat([prompt, seconds_start, seconds_total], dim=1)
      * `cross_attn_mask`            — same layout, [B, L_total]
      * `global_embed`               — cat([seconds_start, seconds_total], dim=-1)
      * `negative_cross_attn_cond`   — same as positive but for the negative prompt
      * `negative_cross_attn_mask`
      * `do_cfg`                     — bool
      * `audio_start_in_s`, `audio_end_in_s`
    """

    def __init__(self, conditioner) -> None:
        super().__init__()
        self.conditioner = conditioner

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pc = fastvideo_args.pipeline_config
        device = next(self.conditioner.parameters()).device

        audio_start_in_s = float(getattr(batch, "audio_start_in_s", None) or pc.audio_start_in_s)
        audio_end_in_s = float(getattr(batch, "audio_end_in_s", None) or pc.audio_end_in_s)
        guidance_scale = float(batch.guidance_scale or pc.guidance_scale)
        do_cfg = guidance_scale > 1.0

        prompt = batch.prompt if isinstance(batch.prompt, str) else batch.prompt[0]
        cond_meta = [{
            "prompt": prompt,
            "seconds_start": audio_start_in_s,
            "seconds_total": audio_end_in_s,
        }]
        cond = self.conditioner(cond_meta, device)
        cross_attn_cond, cross_attn_mask, global_embed = self.conditioner.get_conditioning_inputs(cond)

        neg_cross_attn_cond = None
        neg_cross_attn_mask = None
        neg_global_embed = None
        if do_cfg:
            neg_prompt = batch.negative_prompt or ""
            if isinstance(neg_prompt, list):
                neg_prompt = neg_prompt[0] if neg_prompt else ""
            neg_meta = [{
                "prompt": neg_prompt,
                "seconds_start": audio_start_in_s,
                "seconds_total": audio_end_in_s,
            }]
            neg = self.conditioner(neg_meta, device)
            neg_cross_attn_cond, neg_cross_attn_mask, neg_global_embed = (self.conditioner.get_conditioning_inputs(neg))

        if batch.extra is None:
            batch.extra = {}
        batch.extra["cross_attn_cond"] = cross_attn_cond
        batch.extra["cross_attn_mask"] = cross_attn_mask
        batch.extra["global_embed"] = global_embed
        batch.extra["negative_cross_attn_cond"] = neg_cross_attn_cond
        batch.extra["negative_cross_attn_mask"] = neg_cross_attn_mask
        batch.extra["negative_global_embed"] = neg_global_embed
        batch.extra["do_cfg"] = do_cfg
        batch.extra["audio_start_in_s"] = audio_start_in_s
        batch.extra["audio_end_in_s"] = audio_end_in_s
        return batch
