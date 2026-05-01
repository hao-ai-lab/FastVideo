# SPDX-License-Identifier: Apache-2.0
"""Stable Audio conditioning stage."""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioConditioningStage(PipelineStage):
    """Run the conditioner over the prompt + duration and stash the
    DiT-ready (cross_attn_cond, cross_attn_mask, global_embed) triple
    on `batch.extra` (plus the negative-prompt triple when CFG is on).
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

        # `0 or default` resolves to default — the user-set 0 still falls
        # through to the config default. That's intentional for end_s
        # (callers always specify it explicitly when they care) but means
        # the start default of 0.0 always wins for start_s.
        audio_start_in_s = float(getattr(batch, "audio_start_in_s", None) or pc.audio_start_in_s)
        audio_end_in_s = float(getattr(batch, "audio_end_in_s", None) or pc.audio_end_in_s)
        max_duration = float(getattr(pc, "max_audio_duration_s", 2097152 / 44100))
        if audio_start_in_s < 0:
            raise ValueError(f"audio_start_in_s must be >= 0, got {audio_start_in_s}.")
        if audio_end_in_s <= audio_start_in_s:
            raise ValueError(f"audio_end_in_s ({audio_end_in_s}) must be > audio_start_in_s "
                             f"({audio_start_in_s}).")
        if audio_end_in_s > max_duration:
            raise ValueError(f"audio_end_in_s ({audio_end_in_s}s) exceeds the model's fixed "
                             f"window of {max_duration:.4f}s. Stable Audio Open 1.0 always "
                             f"samples a 2,097,152-frame latent and slices to "
                             f"[start, end] after decode; values past the window are silently "
                             f"truncated. Lower audio_end_in_s or split the request.")
        guidance_scale = float(batch.guidance_scale or pc.guidance_scale)
        do_cfg = guidance_scale > 1.0

        prompt = batch.prompt if isinstance(batch.prompt, str) else batch.prompt[0]
        # Send only the keys the conditioner declares (per-variant).
        all_cond_values = {
            "prompt": prompt,
            "seconds_start": audio_start_in_s,
            "seconds_total": audio_end_in_s,
        }
        active_ids = self.conditioner.cross_attention_cond_ids
        cond_meta = [{k: all_cond_values[k] for k in active_ids if k in all_cond_values}]
        cond = self.conditioner(cond_meta, device)
        cross_attn_cond, cross_attn_mask, global_embed = self.conditioner.get_conditioning_inputs(cond)

        neg_cross_attn_cond = None
        neg_cross_attn_mask = None
        neg_global_embed = None
        if do_cfg:
            neg_prompt = batch.negative_prompt or ""
            if isinstance(neg_prompt, list):
                neg_prompt = neg_prompt[0] if neg_prompt else ""
            neg_values = dict(all_cond_values, prompt=neg_prompt)
            neg_meta = [{k: neg_values[k] for k in active_ids if k in neg_values}]
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
