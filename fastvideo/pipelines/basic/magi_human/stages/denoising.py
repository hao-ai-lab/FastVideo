# SPDX-License-Identifier: Apache-2.0
"""Joint-modality denoising stage for daVinci-MagiHuman base text-to-AV.

Runs the FlowUniPC denoise loop with CFG=2 over video + audio latents
jointly. Text embeddings are already pad-or-trimmed to `t5_gemma_target_length`
by `MagiHumanLatentPreparationStage`; the original context lengths are
stashed on the batch as `magi_original_text_lens` / `magi_original_neg_text_lens`.
"""
from __future__ import annotations

import copy

import torch
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
    StaticPackedInputs,
    assemble_packed_inputs,
    build_static_packed_inputs,
    unpack_tokens,
)


def _dit_forward(
    dit,
    video_latent: torch.Tensor,
    audio_feat_len: int,
    txt_feat: torch.Tensor,
    txt_feat_len: int,
    static_packed: StaticPackedInputs,
    coords_style: str,
    video_in_channels: int,
    audio_in_channels: int,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    x, coords, mm = assemble_packed_inputs(
        static=static_packed,
        txt_feat=txt_feat,
        txt_feat_len=txt_feat_len,
        coords_style=coords_style,
    )
    video_token_num = static_packed.video_token_num
    out = dit(x, coords, mm)
    return unpack_tokens(
        out,
        video_token_num=video_token_num,
        audio_feat_len=audio_feat_len,
        video_in_channels=video_in_channels,
        audio_in_channels=audio_in_channels,
        latent_shape=tuple(video_latent.shape),
        patch_size=patch_size,
    )


class MagiHumanDenoisingStage(PipelineStage):
    """UniPC-flow joint denoising with CFG=2 over (video, audio) latents."""

    def __init__(
        self,
        transformer,
        scheduler,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        video_in_channels: int = 192,
        audio_in_channels: int = 64,
        video_txt_guidance_scale: float = 5.0,
        audio_txt_guidance_scale: float = 5.0,
        cfg_number: int = 2,
        coords_style: str = "v2",
        video_guidance_high_t_threshold: int = 500,
        video_guidance_low_t_value: float = 2.0,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.patch_size = patch_size
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.video_txt_guidance_scale = video_txt_guidance_scale
        self.audio_txt_guidance_scale = audio_txt_guidance_scale
        self.cfg_number = cfg_number
        self.coords_style = coords_style
        self.video_guidance_high_t_threshold = video_guidance_high_t_threshold
        self.video_guidance_low_t_value = video_guidance_low_t_value

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = batch.latents.device
        shift = fastvideo_args.pipeline_config.flow_shift
        # Video and audio use independent FlowUniPC state (upstream
        # inference/pipeline/video_generate.py:404-407 instantiates two
        # separate schedulers). Sharing one scheduler causes the
        # `model_outputs` buffer for the video step to pollute the audio
        # step's diff calculation (different shapes -> broadcast error).
        video_scheduler = copy.deepcopy(self.scheduler)
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_scheduler.set_timesteps(
            batch.num_inference_steps,
            device=device,
            shift=shift,
        )
        audio_scheduler.set_timesteps(
            batch.num_inference_steps,
            device=device,
            shift=shift,
        )
        timesteps = video_scheduler.timesteps

        video_latent = batch.latents
        audio_latent = batch.audio_latents

        # Expect [1, L, 3584] text embeds plus a list of original lengths.
        txt_feat = batch.prompt_embeds[0]
        txt_feat_len = int(batch.magi_original_text_lens[0])

        neg_txt_feat: torch.Tensor | None = None
        neg_txt_feat_len: int = 0
        if self.cfg_number == 2:
            neg_list = batch.negative_prompt_embeds or []
            if not neg_list:
                raise ValueError("CFG=2 requires negative prompt embeddings; got None. "
                                 "Did the prompt encoding stage run?")
            else:
                neg_txt_feat = neg_list[0]
                neg_txt_feat_len = int(batch.magi_original_neg_text_lens[0])

        audio_feat_len = int(audio_latent.shape[1])

        # Precompute step-invariant packed inputs (video+audio tokens, coords,
        # modality map). Text varies per cond/uncond call so it is assembled
        # inside _dit_forward via assemble_packed_inputs.
        static_packed = build_static_packed_inputs(
            video_latent=video_latent,
            audio_latent=audio_latent,
            audio_feat_len=audio_feat_len,
            patch_size=self.patch_size,
            coords_style=self.coords_style,
        )

        disable_tqdm = not getattr(fastvideo_args, "log_level_progress", True)
        for idx, t in enumerate(tqdm(timesteps, disable=disable_tqdm)):
            with trace_step(idx):
                v_cond_video, v_cond_audio = _dit_forward(
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
                    v_uncond_video, v_uncond_audio = _dit_forward(
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
                else:
                    v_uncond_video = None
                    v_uncond_audio = None

            if self.cfg_number == 2:
                video_guidance = (self.video_txt_guidance_scale
                                  if t > self.video_guidance_high_t_threshold else self.video_guidance_low_t_value)
                assert v_uncond_video is not None and v_uncond_audio is not None
                v_video = v_uncond_video + video_guidance * (v_cond_video - v_uncond_video)
                v_audio = v_uncond_audio + self.audio_txt_guidance_scale * (v_cond_audio - v_uncond_audio)
            else:
                v_video = v_cond_video
                v_audio = v_cond_audio

            # Independent scheduler state per modality (see comment above).
            video_latent = video_scheduler.step(
                v_video,
                t,
                video_latent,
                return_dict=False,
            )[0]
            audio_latent = audio_scheduler.step(
                v_audio,
                t,
                audio_latent,
                return_dict=False,
            )[0]

            static_packed = build_static_packed_inputs(
                video_latent=video_latent,
                audio_latent=audio_latent,
                audio_feat_len=audio_feat_len,
                patch_size=self.patch_size,
                coords_style=self.coords_style,
            )

        batch.latents = video_latent
        batch.audio_latents = audio_latent
        return batch
