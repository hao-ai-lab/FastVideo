# SPDX-License-Identifier: Apache-2.0
"""Preview and refiner denoising stages for MAGI-2."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.pipelines.basic.magi2.stages.preview_data_proxy import (
    Magi2DataProxy,
    Magi2DataProxyConfig,
    ModelInput,
)
from fastvideo.pipelines.basic.magi2.stages.refiner_data_proxy import (
    Magi2RefinerDataProxy,
    Magi2RefinerDataProxyConfig,
    RefinerModelInput,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


def _pad_or_trim(
    tensor: torch.Tensor,
    target_size: int,
    dimension: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Match one tensor dimension to the shared classifier-free-guidance length."""
    current_size = tensor.size(dimension)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding = [0] * (2 * tensor.dim())
        padding_dimension = tensor.dim() - 1 - dimension
        padding[2 * padding_dimension + 1] = padding_amount
        return F.pad(tensor, tuple(padding), "constant", pad_value)
    slices = [slice(None)] * tensor.dim()
    slices[dimension] = slice(0, target_size)
    return tensor[tuple(slices)]


class Magi2PreviewDenoisingStage(PipelineStage):
    """Run the joint video-audio preview transformer with two-way CFG."""

    def __init__(
        self,
        transformer: Any,
        data_proxy_config: Magi2DataProxyConfig,
        flow_shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
    ) -> None:
        """Store the preview transformer and published sampling constants."""
        super().__init__()
        self.transformer = transformer
        self.data_proxy = Magi2DataProxy(data_proxy_config)
        self.flow_shift = flow_shift
        self.video_guidance_scale = video_guidance_scale
        self.audio_guidance_scale = audio_guidance_scale

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record; ``forward`` validates required tensors."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record for denoised video and audio latents."""
        del batch, fastvideo_args
        return VerificationResult()

    @staticmethod
    def _prepare_reference_cfg(
        reference_latent: torch.Tensor | None,
        reference_length: torch.Tensor | None,
        special_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Duplicate I2V conditioning across both classifier-free-guidance halves."""
        if reference_latent is None:
            return None, None, None
        if reference_length is None or special_tokens is None:
            raise ValueError("MAGI-2 I2V conditioning is incomplete")
        return (
            torch.cat([reference_latent, reference_latent], dim=0),
            torch.cat([reference_length, reference_length], dim=0),
            torch.cat([special_tokens, special_tokens], dim=0),
        )

    def _prepare_model_input(
        self,
        batch: ForwardBatch,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> ModelInput:
        """Build the exact two-sample conditional/unconditional preview input."""
        text_context = batch.magi2_text_context
        negative_context = batch.magi2_negative_context
        if text_context is None or negative_context is None:
            raise ValueError("MAGI-2 preview denoising requires text conditioning")
        text_length = text_context.shape[1]
        negative_length = negative_context.shape[1]
        shared_length = max(text_length, negative_length)
        text_context = _pad_or_trim(text_context, shared_length, 1)
        negative_context = _pad_or_trim(negative_context, shared_length, 1)

        reference_video = torch.empty_like(video_latent)
        reference_video = torch.cat(
            [reference_video, torch.zeros_like(reference_video)],
            dim=0,
        )
        reference_video_length = torch.tensor(
            [0, 0],
            device=video_latent.device,
        )
        reference_image, reference_image_length, special_tokens = (
            self._prepare_reference_cfg(
                batch.magi2_ref_image_feat,
                batch.magi2_ref_image_feat_len,
                batch.magi2_ref_image_special_tokens,
            )
        )

        timestep_value = timestep.reshape(-1)[0].to(
            device=video_latent.device,
            dtype=video_latent.dtype,
        )
        normalized_timestep = torch.stack([timestep_value, timestep_value]) / 1000.0
        _, _, video_time, video_height, video_width = video_latent.shape
        audio_time = audio_latent.shape[1]
        per_token_video_t = normalized_timestep.view(-1, 1, 1, 1, 1).expand(
            2,
            1,
            video_time,
            video_height,
            video_width,
        ).clone()
        per_token_audio_t = normalized_timestep.view(-1, 1, 1).expand(
            2,
            audio_time,
            1,
        ).clone()
        reference_audio = torch.zeros(
            1,
            0,
            audio_latent.shape[-1],
            device=video_latent.device,
            dtype=torch.float32,
        )
        return ModelInput(
            x_t=torch.cat([video_latent, video_latent], dim=0),
            audio_x_t=torch.cat([audio_latent, audio_latent], dim=0),
            audio_feat_len=torch.tensor(
                [audio_time, audio_time],
                device=video_latent.device,
            ),
            txt_feat=torch.cat([text_context, negative_context], dim=0),
            txt_feat_len=torch.tensor(
                [text_length, negative_length],
                device=video_latent.device,
            ),
            t=normalized_timestep,
            per_token_video_t=per_token_video_t,
            per_token_audio_t=per_token_audio_t,
            ref_audio_feat=torch.cat(
                [reference_audio, torch.zeros_like(reference_audio)],
                dim=0,
            ),
            ref_audio_feat_len=torch.tensor(
                [0, 0],
                device=video_latent.device,
            ),
            ref_video_feat=reference_video,
            ref_video_feat_len=reference_video_length,
            ref_image_feat=reference_image,
            ref_image_feat_len=reference_image_length,
            ref_image_special_token_embedding=special_tokens,
        )

    def _predict_velocity(
        self,
        model_input: ModelInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack tokens, execute the preview transformer, and restore both modalities."""
        transformer_input = self.data_proxy.process_input(model_input)
        transformer_output = self.transformer(*transformer_input)
        return self.data_proxy.process_output(transformer_output)

    @torch.inference_mode()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Advance the video and audio latents through the published preview schedule."""
        if batch.latents is None or batch.audio_latents is None:
            raise ValueError("MAGI-2 preview denoising requires joint noise latents")
        device = torch.device("cuda", torch.cuda.current_device())
        self.transformer.to(device)
        video_latent = batch.latents.clone()
        audio_latent = batch.audio_latents.clone()
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(
            batch.num_inference_steps,
            device=device,
            shift=self.flow_shift,
        )
        audio_scheduler.set_timesteps(
            batch.num_inference_steps,
            device=device,
            shift=self.flow_shift,
        )
        video_guidance = torch.tensor(
            [self.video_guidance_scale],
            device=device,
        )
        for timestep in video_scheduler.timesteps:
            model_input = self._prepare_model_input(
                batch,
                video_latent,
                audio_latent,
                timestep,
            )
            video_prediction, audio_prediction = self._predict_velocity(model_input)
            conditional_video = video_prediction[0:1]
            unconditional_video = video_prediction[1:2]
            video_velocity = unconditional_video + video_guidance * (
                conditional_video - unconditional_video
            )
            conditional_audio = audio_prediction[0:1]
            unconditional_audio = audio_prediction[1:2]
            audio_velocity = unconditional_audio + self.audio_guidance_scale * (
                conditional_audio - unconditional_audio
            )
            video_latent = video_scheduler.step(
                video_velocity,
                timestep,
                video_latent,
                return_dict=False,
            )[0]
            audio_latent = audio_scheduler.step(
                audio_velocity,
                timestep,
                audio_latent,
                return_dict=False,
            )[0]
        batch.latents = video_latent
        batch.audio_latents = audio_latent
        if fastvideo_args.dit_cpu_offload:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
        return batch


class ZeroSNRDDPMDiscretization:
    """Construct the refiner's published zero-terminal-SNR noise schedule."""

    def __init__(
        self,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        num_timesteps: int = 1000,
    ) -> None:
        """Precompute the cumulative alpha schedule in FP64 NumPy arithmetic."""
        self.num_timesteps = num_timesteps
        betas = torch.linspace(
            linear_start**0.5,
            linear_end**0.5,
            num_timesteps,
            dtype=torch.float64,
        ) ** 2
        self.alphas_cumprod = np.cumprod(1.0 - betas.numpy(), axis=0)

    def __call__(
        self,
        count: int,
        do_append_zero: bool = True,
        device: str = "cpu",
        flip: bool = False,
    ) -> torch.Tensor:
        """Return sampled sigma values with optional terminal zero and reversal."""
        sigmas = self.get_sigmas(count, device=device)
        if do_append_zero:
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return torch.flip(sigmas, (0,)) if flip else sigmas

    def get_sigmas(self, count: int, device: str = "cpu") -> torch.Tensor:
        """Convert cumulative alphas into the refiner's sigma parameterization."""
        if count < self.num_timesteps:
            timesteps = np.linspace(
                self.num_timesteps - 1,
                0,
                count,
                endpoint=False,
            ).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif count == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(
                f"sigma count must be <= {self.num_timesteps}, received {count}"
            )
        alphas = torch.tensor(
            alphas_cumprod,
            dtype=torch.float32,
            device=device,
        ).sqrt()
        first_alpha = alphas[0].clone()
        terminal_alpha = alphas[-1].clone()
        alphas -= terminal_alpha
        alphas *= first_alpha / (first_alpha - terminal_alpha)
        return torch.flip(alphas, (0,))


class Magi2RefinerStage(PipelineStage):
    """Upsample preview latents and run the five-step 1080p refiner."""

    def __init__(
        self,
        transformer: Any,
        data_proxy_config: Magi2RefinerDataProxyConfig,
        latent_height: int,
        latent_width: int,
        noise_index: int,
        flow_shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        audio_channels: int,
    ) -> None:
        """Store the refiner transformer, token proxy, and release constants."""
        super().__init__()
        self.transformer = transformer
        self.data_proxy = Magi2RefinerDataProxy(data_proxy_config)
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.noise_index = noise_index
        self.flow_shift = flow_shift
        self.video_guidance_scale = video_guidance_scale
        self.audio_guidance_scale = audio_guidance_scale
        self.audio_channels = audio_channels
        self.sigmas = ZeroSNRDDPMDiscretization()(
            1000,
            do_append_zero=False,
            flip=True,
        )

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record; ``forward`` validates latent conditioning."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record for the post-refiner video latent."""
        del batch, fastvideo_args
        return VerificationResult()

    def _predict_velocity(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        text_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack one text branch, execute the refiner, and restore both modalities."""
        reference_audio = torch.zeros(
            1,
            0,
            self.audio_channels,
            device=video_latent.device,
            dtype=torch.float32,
        )
        model_input = RefinerModelInput(
            x_t=video_latent,
            audio_x_t=audio_latent,
            audio_feat_len=torch.tensor([audio_latent.shape[1]]),
            txt_feat=text_context.to(torch.float32),
            txt_feat_len=torch.tensor([text_context.shape[1]]),
            ref_audio_feat=reference_audio,
            ref_audio_feat_len=torch.tensor([0]),
            ref_video_feat=torch.empty_like(video_latent),
            ref_video_feat_len=torch.tensor([0]),
        )
        transformer_input = self.data_proxy.process_input(model_input)
        transformer_output = self.transformer(*transformer_input)
        return self.data_proxy.process_output(transformer_output)

    @torch.inference_mode()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Inject refiner noise and advance the upsampled latent for five steps."""
        if batch.latents is None:
            raise ValueError("MAGI-2 refiner requires the preview video latent")
        if batch.magi2_text_context is None or batch.magi2_negative_context is None:
            raise ValueError("MAGI-2 refiner requires positive and negative text context")
        device = torch.device("cuda", torch.cuda.current_device())
        video_latent = F.interpolate(
            batch.latents,
            size=(
                batch.latents.shape[2] * 2 - 1,
                self.latent_height,
                self.latent_width,
            ),
            mode="trilinear",
            align_corners=True,
        )
        noise = torch.randn_like(video_latent, device=video_latent.device)
        sigma = self.sigmas.to(video_latent.device)[self.noise_index]
        video_latent = video_latent * sigma + noise * (1 - sigma**2) ** 0.5
        refiner_audio = torch.zeros(
            1,
            0,
            self.audio_channels,
            device=device,
            dtype=torch.float32,
        )
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(
            batch.num_inference_steps_sr,
            device=device,
            shift=self.flow_shift,
        )
        audio_scheduler.set_timesteps(
            batch.num_inference_steps_sr,
            device=device,
            shift=self.flow_shift,
        )
        video_guidance = torch.tensor(
            self.video_guidance_scale,
            device=device,
        ).expand(1, 1, video_latent.shape[2], 1, 1).clone()
        self.transformer.to(device)
        for timestep in video_scheduler.timesteps:
            conditional_video, conditional_audio = self._predict_velocity(
                video_latent,
                refiner_audio,
                batch.magi2_text_context,
            )
            unconditional_video, unconditional_audio = self._predict_velocity(
                video_latent,
                refiner_audio,
                batch.magi2_negative_context,
            )
            video_velocity = unconditional_video + video_guidance * (
                conditional_video - unconditional_video
            )
            audio_velocity = unconditional_audio + self.audio_guidance_scale * (
                conditional_audio - unconditional_audio
            )
            video_latent = video_scheduler.step(
                video_velocity,
                timestep,
                video_latent,
                return_dict=False,
            )[0]
            if audio_velocity.numel() > 0:
                refiner_audio = audio_scheduler.step(
                    audio_velocity,
                    timestep,
                    refiner_audio,
                    return_dict=False,
                )[0]
        batch.latents = video_latent
        if fastvideo_args.dit_cpu_offload:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
        return batch


__all__ = [
    "Magi2PreviewDenoisingStage",
    "Magi2RefinerStage",
    "ZeroSNRDDPMDiscretization",
]
