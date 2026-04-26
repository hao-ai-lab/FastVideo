# SPDX-License-Identifier: Apache-2.0
"""Stable Audio conditioning stage.

Mirrors `StableAudioPipeline.encode_prompt` + `encode_duration` from
diffusers (lines 155-311 of `pipeline_stable_audio.py`):

  1. T5 text encoding has already happened in the upstream
     `TextEncodingStage`; this stage receives `batch.prompt_embeds` and
     applies the projection model + duration embedding on top.
  2. Builds:
        * `text_audio_duration_embeds` = [text_embeds; start_emb; end_emb]
          → cross-attention input to the DiT
        * `audio_duration_embeds`      = [start_emb; end_emb]
          → global states (prepended) input to the DiT
  3. CFG: if guidance_scale > 1.0 and no negative prompt given, builds
     a zero-prompt unconditional and concatenates `[uncond, cond]` into
     a single batch.

Outputs are stashed on `batch.extra` for the downstream stages.
"""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioConditioningStage(PipelineStage):
    """Run projection + duration embedding + CFG batch construction."""

    def __init__(
        self,
        projection_model,
        sampling_rate: int = 44100,
        sample_size: int = 1024,
    ) -> None:
        super().__init__()
        self.projection_model = projection_model
        self.sampling_rate = sampling_rate
        self.sample_size = sample_size

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pc = fastvideo_args.pipeline_config
        device = batch.prompt_embeds[0].device if batch.prompt_embeds else torch.device("cuda")

        # Pull config values from batch / pipeline config with sensible defaults.
        audio_start_in_s = float(getattr(batch, "audio_start_in_s", None) or pc.audio_start_in_s)
        audio_end_in_s = float(getattr(batch, "audio_end_in_s", None) or pc.audio_end_in_s)
        guidance_scale = float(batch.guidance_scale or pc.guidance_scale)
        do_cfg = guidance_scale > 1.0

        # --- 1. T5 prompt embeds (already produced by TextEncodingStage). ---
        prompt_embeds = batch.prompt_embeds[0]  # [B, L, T5_dim]
        # Attention masks come from `StableAudioTextEncodingStage` (which
        # forces `padding="max_length"` and pre-initializes the mask
        # lists). Fall back to all-ones if missing — keeps the stage
        # usable when prompts are pre-encoded externally.
        if batch.prompt_attention_mask:
            attention_mask = batch.prompt_attention_mask[0]
        else:
            attention_mask = torch.ones(
                prompt_embeds.shape[:2],
                dtype=torch.long,
                device=prompt_embeds.device,
            )

        # Negative prompt embeds (already produced by TextEncodingStage if
        # negative_prompt was provided), else zero-pad to match shape.
        negative_attention_mask = None
        if batch.negative_prompt_embeds:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
            if batch.negative_attention_mask:
                negative_attention_mask = batch.negative_attention_mask[0]
        else:
            negative_prompt_embeds = None

        # --- 2. CFG concat for prompt_embeds (the pipeline expects a
        #     single batch with [uncond; cond] when do_cfg is on). ---
        if do_cfg and negative_prompt_embeds is not None:
            # Mask negative tokens where the negative prompt is padded
            if negative_attention_mask is not None:
                negative_prompt_embeds = torch.where(
                    negative_attention_mask.to(torch.bool).unsqueeze(2),
                    negative_prompt_embeds,
                    0.0,
                )
                if attention_mask is None:
                    attention_mask = torch.ones_like(negative_attention_mask)
                attention_mask = torch.cat([negative_attention_mask, attention_mask])
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # --- 3. Projection. ---
        proj_out = self.projection_model(text_hidden_states=prompt_embeds)
        prompt_embeds = proj_out.text_hidden_states
        if attention_mask is not None:
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

        # --- 4. Duration embedding. ---
        batch_size = prompt_embeds.shape[0] // (2 if (do_cfg and negative_prompt_embeds is not None) else 1)
        start_t = torch.tensor([audio_start_in_s] * batch_size, dtype=torch.float32, device=device)
        end_t = torch.tensor([audio_end_in_s] * batch_size, dtype=torch.float32, device=device)
        duration_out = self.projection_model(start_seconds=start_t, end_seconds=end_t)
        seconds_start_hs = duration_out.seconds_start_hidden_states
        seconds_end_hs = duration_out.seconds_end_hidden_states

        # CFG without explicit negative prompt: build zero-uncond.
        if do_cfg and negative_prompt_embeds is None:
            text_audio_duration_embeds = torch.cat(
                [prompt_embeds, seconds_start_hs, seconds_end_hs],
                dim=1,
            )
            negative_text_audio_duration_embeds = torch.zeros_like(text_audio_duration_embeds)
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds],
                dim=0,
            )
            audio_duration_embeds = torch.cat([seconds_start_hs, seconds_end_hs], dim=2)
            audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)
        elif do_cfg:
            # negative_prompt_embeds was concatenated into prompt_embeds above
            # → duplicate seconds_*_hs to match the doubled batch.
            seconds_start_hs = torch.cat([seconds_start_hs, seconds_start_hs], dim=0)
            seconds_end_hs = torch.cat([seconds_end_hs, seconds_end_hs], dim=0)
            text_audio_duration_embeds = torch.cat(
                [prompt_embeds, seconds_start_hs, seconds_end_hs],
                dim=1,
            )
            audio_duration_embeds = torch.cat([seconds_start_hs, seconds_end_hs], dim=2)
        else:
            text_audio_duration_embeds = torch.cat(
                [prompt_embeds, seconds_start_hs, seconds_end_hs],
                dim=1,
            )
            audio_duration_embeds = torch.cat([seconds_start_hs, seconds_end_hs], dim=2)

        if batch.extra is None:
            batch.extra = {}
        batch.extra["text_audio_duration_embeds"] = text_audio_duration_embeds
        batch.extra["audio_duration_embeds"] = audio_duration_embeds
        batch.extra["do_cfg"] = do_cfg
        batch.extra["audio_start_in_s"] = audio_start_in_s
        batch.extra["audio_end_in_s"] = audio_end_in_s
        return batch
