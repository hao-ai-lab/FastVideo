# SPDX-License-Identifier: Apache-2.0
"""SR video-only denoising stage for daVinci-MagiHuman SR-540p."""
from __future__ import annotations

import copy

import torch
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.pipelines.basic.magi_human.stages.denoising import (
    _dit_forward,
    _overwrite_first_frame,
)
from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
    build_static_packed_inputs, )
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class MagiHumanSRDenoisingStage(PipelineStage):
    """Denoise only the SR video latent; audio passes through unchanged."""

    def __init__(
        self,
        transformer,
        scheduler,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        video_in_channels: int = 192,
        audio_in_channels: int = 64,
        sr_num_inference_steps: int = 5,
        sr_video_txt_guidance_scale: float = 3.5,
        use_cfg_trick: bool = True,
        cfg_trick_start_frame: int = 13,
        cfg_trick_value: float = 2.0,
        cfg_number: int = 2,
        coords_style: str = "v1",
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.patch_size = patch_size
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.sr_num_inference_steps = sr_num_inference_steps
        self.sr_video_txt_guidance_scale = sr_video_txt_guidance_scale
        self.use_cfg_trick = use_cfg_trick
        self.cfg_trick_start_frame = cfg_trick_start_frame
        self.cfg_trick_value = cfg_trick_value
        self.cfg_number = cfg_number
        self.coords_style = coords_style

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = batch.latents.device
        shift = fastvideo_args.pipeline_config.flow_shift
        video_scheduler = copy.deepcopy(self.scheduler)
        video_scheduler.set_timesteps(
            self.sr_num_inference_steps,
            device=device,
            shift=shift,
        )

        video_latent = batch.latents
        audio_latent = batch.audio_latents
        audio_feat_len = int(audio_latent.shape[1])
        image_latent = getattr(batch, "image_latent", None)

        txt_feat = batch.prompt_embeds[0]
        txt_feat_len = int(batch.magi_original_text_lens[0])

        neg_txt_feat: torch.Tensor | None = None
        neg_txt_feat_len = 0
        if self.cfg_number == 2:
            neg_list = batch.negative_prompt_embeds or []
            if not neg_list:
                raise ValueError("SR CFG=2 requires negative prompt embeddings.")
            neg_txt_feat = neg_list[0]
            neg_txt_feat_len = int(batch.magi_original_neg_text_lens[0])

        latent_length = video_latent.shape[2]
        guidance = torch.tensor(
            self.sr_video_txt_guidance_scale,
            device=device,
            dtype=video_latent.dtype,
        ).expand(1, 1, latent_length, 1, 1).clone()
        if self.use_cfg_trick:
            guidance[:, :, :self.cfg_trick_start_frame] = min(
                self.cfg_trick_value,
                self.sr_video_txt_guidance_scale,
            )

        disable_tqdm = not getattr(fastvideo_args, "log_level_progress", True)
        for idx, t in enumerate(tqdm(video_scheduler.timesteps, disable=disable_tqdm)):
            video_latent = _overwrite_first_frame(video_latent, image_latent)
            static_packed = build_static_packed_inputs(
                video_latent=video_latent,
                audio_latent=audio_latent,
                audio_feat_len=audio_feat_len,
                patch_size=self.patch_size,
                coords_style=self.coords_style,
            )
            with trace_step(idx), set_forward_context(
                    current_timestep=int(t.item()) if torch.is_tensor(t) else int(t),
                    attn_metadata=None,
            ):
                v_cond_video, _ = _dit_forward(
                    self.transformer,
                    video_latent=video_latent,
                    audio_feat_len=audio_feat_len,
                    txt_feat=txt_feat,
                    txt_feat_len=txt_feat_len,
                    static_packed=static_packed,
                    coords_style=self.coords_style,
                    video_in_channels=self.video_in_channels,
                    audio_in_channels=self.audio_in_channels,
                    patch_size=self.patch_size,
                )
                if self.cfg_number == 2:
                    assert neg_txt_feat is not None
                    v_uncond_video, _ = _dit_forward(
                        self.transformer,
                        video_latent=video_latent,
                        audio_feat_len=audio_feat_len,
                        txt_feat=neg_txt_feat,
                        txt_feat_len=neg_txt_feat_len,
                        static_packed=static_packed,
                        coords_style=self.coords_style,
                        video_in_channels=self.video_in_channels,
                        audio_in_channels=self.audio_in_channels,
                        patch_size=self.patch_size,
                    )
                    v_video = v_uncond_video + guidance * (v_cond_video - v_uncond_video)
                else:
                    v_video = v_cond_video

            video_latent = video_scheduler.step(
                v_video,
                t,
                video_latent,
                return_dict=False,
            )[0]

        batch.latents = _overwrite_first_frame(video_latent, image_latent)
        batch.audio_latents = audio_latent
        return batch
