# SPDX-License-Identifier: Apache-2.0
"""
Implements GEN3C video diffusion pipeline with 3D cache support for camera-controlled video generation.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        InputValidationStage, TextEncodingStage,
                                        TimestepPreparationStage)
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage

logger = init_logger(__name__)


class Gen3CLatentPreparationStage(LatentPreparationStage):
    """
    Latent preparation stage for GEN3C.
    
    This stage prepares latents and encodes 3D cache buffers through the VAE.
    """

    def __init__(self, scheduler, transformer, vae) -> None:
        super().__init__(scheduler, transformer, vae)

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Prepare latents and encode 3D cache buffers."""
        # Get dimensions from pipeline config
        pipeline_config = fastvideo_args.pipeline_config
        device = get_local_torch_device()

        # Determine batch size (following base LatentPreparationStage pattern)
        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]
        batch_size *= batch.num_videos_per_prompt

        # Initialize latents
        num_channels_latents = getattr(self.transformer, 'num_channels_latents',
                                       16)

        # Get latent dimensions from config or defaults
        latent_frames = getattr(pipeline_config, 'state_t', 16)
        height = getattr(batch, 'height', 720)
        width = getattr(batch, 'width', 1280)

        # Calculate latent spatial dimensions (8x compression)
        latent_height = height // 8
        latent_width = width // 8

        # Generate initial noise latents
        latents = torch.randn(
            batch_size,
            num_channels_latents,
            latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=torch.float32,
        )

        # Scale latents by initial sigma
        if hasattr(self.scheduler, 'init_noise_sigma'):
            latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.batch_size = batch_size
        batch.height = height
        batch.width = width
        batch.latent_height = latent_height
        batch.latent_width = latent_width
        batch.latent_frames = latent_frames

        # Prepare conditioning indicator (for video conditioning)
        # Default: no conditioning (will be set by conditioning stage if needed)
        batch.cond_indicator = torch.zeros(batch_size,
                                           1,
                                           latent_frames,
                                           latent_height,
                                           latent_width,
                                           device=device,
                                           dtype=torch.float32)

        # Prepare condition_video_input_mask (binary mask for conditioning frames)
        batch.condition_video_input_mask = torch.zeros(batch_size,
                                                       1,
                                                       latent_frames,
                                                       latent_height,
                                                       latent_width,
                                                       device=device,
                                                       dtype=torch.float32)

        # Prepare condition_video_pose (3D cache buffers - to be filled by cache rendering)
        frame_buffer_max = getattr(pipeline_config, 'frame_buffer_max', 2)
        channels_per_buffer = getattr(
            pipeline_config, 'dit_config', None)
        if channels_per_buffer is not None:
            channels_per_buffer = getattr(
                channels_per_buffer, 'arch_config', None)
        if channels_per_buffer is not None:
            channels_per_buffer = getattr(
                channels_per_buffer, 'CHANNELS_PER_BUFFER', 32)
        else:
            channels_per_buffer = 32
        buffer_channels = frame_buffer_max * channels_per_buffer

        batch.condition_video_pose = torch.zeros(batch_size,
                                                 buffer_channels,
                                                 latent_frames,
                                                 latent_height,
                                                 latent_width,
                                                 device=device,
                                                 dtype=torch.float32)

        # Prepare augment sigma (for noise augmentation on condition frames)
        batch.condition_video_augment_sigma = torch.zeros(batch_size,
                                                          device=device,
                                                          dtype=torch.float32)

        # Required by base stage verification
        batch.raw_latent_shape = latents.shape

        return batch

    def encode_warped_frames(
        self,
        condition_state: torch.Tensor,
        condition_state_mask: torch.Tensor,
        vae,
        frame_buffer_max: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Encode rendered 3D cache buffers through VAE.
        
        Args:
            condition_state: (B, T, N, 3, H, W) rendered RGB images
            condition_state_mask: (B, T, N, 1, H, W) rendered masks
            vae: VAE encoder
            frame_buffer_max: Maximum number of buffers
            dtype: Target dtype
            
        Returns:
            latent_condition: (B, buffer_channels, T_latent, H_latent, W_latent)
        """
        assert condition_state.dim() == 6

        # Convert mask to [-1, 1] range and repeat to 3 channels
        condition_state_mask = (condition_state_mask * 2 - 1).repeat(
            1, 1, 1, 3, 1, 1)

        latent_condition = []
        num_buffers = condition_state.shape[2]
        for i in range(num_buffers):
            # Batch image and mask into a single VAE encode call per buffer
            img_input = condition_state[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)
            mask_input = condition_state_mask[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)
            batched_input = torch.cat([img_input, mask_input], dim=0)
            batched_latent = vae.encode(batched_input).contiguous()
            current_video_latent, current_mask_latent = batched_latent.chunk(2, dim=0)

            latent_condition.append(current_video_latent)
            latent_condition.append(current_mask_latent)

        # Pad with zeros if fewer buffers than frame_buffer_max
        for _ in range(frame_buffer_max - condition_state.shape[2]):
            latent_condition.append(torch.zeros_like(current_video_latent))
            latent_condition.append(torch.zeros_like(current_mask_latent))

        latent_condition = torch.cat(latent_condition, dim=1)
        return latent_condition


class Gen3CDenoisingStage(DenoisingStage):
    """
    Denoising stage for GEN3C models.
    
    This stage extends the Cosmos denoising stage with support for:
    - condition_video_input_mask: Binary mask indicating conditioning frames
    - condition_video_pose: VAE-encoded 3D cache buffers
    - condition_video_augment_sigma: Noise augmentation sigma
    """

    def __init__(self, transformer, scheduler, pipeline=None) -> None:
        super().__init__(transformer, scheduler, pipeline)

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta
            },
        )

        if hasattr(self.transformer, 'module'):
            transformer_dtype = next(self.transformer.module.parameters()).dtype
        else:
            transformer_dtype = next(self.transformer.parameters()).dtype
        target_dtype = transformer_dtype
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latents = batch.latents
        num_inference_steps = batch.num_inference_steps
        guidance_scale = batch.guidance_scale
        fps = getattr(fastvideo_args.pipeline_config, 'fps', 24)

        sigma_max = 80.0
        sigma_min = 0.002
        sigma_data = 1.0
        final_sigmas_type = "sigma_min"

        if self.scheduler is not None:
            self.scheduler.register_to_config(
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                sigma_data=sigma_data,
                final_sigmas_type=final_sigmas_type,
            )

        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        timesteps = self.scheduler.timesteps

        if (hasattr(self.scheduler.config, 'final_sigmas_type')
                and self.scheduler.config.final_sigmas_type == "sigma_min"
                and len(self.scheduler.sigmas) > 1):
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        conditioning_latents = getattr(batch, 'conditioning_latents', None)

        # Get GEN3C-specific inputs
        condition_video_input_mask = getattr(batch,
                                             'condition_video_input_mask', None)
        condition_video_pose = getattr(batch, 'condition_video_pose', None)
        condition_video_augment_sigma = getattr(
            batch, 'condition_video_augment_sigma', None)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                current_sigma = self.scheduler.sigmas[i]
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t

                timestep = current_t.view(1).expand(latents.size(0))

                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):

                    cond_latent = latents * c_in
                    cond_latent = cond_latent.to(target_dtype)

                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                    ):
                        # Prepare padding mask
                        padding_mask = torch.ones(batch.batch_size,
                                                  1,
                                                  batch.height,
                                                  batch.width,
                                                  device=cond_latent.device,
                                                  dtype=target_dtype)

                        # Call GEN3C transformer with 3D cache inputs
                        noise_pred = self.transformer(
                            hidden_states=cond_latent,
                            timestep=timestep.to(target_dtype),
                            encoder_hidden_states=batch.prompt_embeds[0].to(
                                target_dtype),
                            fps=fps,
                            condition_video_input_mask=condition_video_input_mask
                            .to(target_dtype)
                            if condition_video_input_mask is not None else None,
                            condition_video_pose=condition_video_pose.to(
                                target_dtype)
                            if condition_video_pose is not None else None,
                            condition_video_augment_sigma=
                            condition_video_augment_sigma if
                            condition_video_augment_sigma is not None else None,
                            padding_mask=padding_mask,
                        )

                    # Handle both tuple and tensor returns
                    if isinstance(noise_pred, tuple):
                        noise_pred = noise_pred[0]

                    cond_pred = (c_skip * latents +
                                 c_out * noise_pred.float()).to(target_dtype)

                    # Classifier-free guidance
                    if batch.do_classifier_free_guidance and batch.negative_prompt_embeds is not None:
                        uncond_latent = latents * c_in
                        uncond_latent = uncond_latent.to(target_dtype)

                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                        ):
                            # For unconditioned prediction, zero out the pose buffers
                            uncond_pose = torch.zeros_like(
                                condition_video_pose
                            ) if condition_video_pose is not None else None

                            uncond_noise_pred = self.transformer(
                                hidden_states=uncond_latent,
                                timestep=timestep.to(target_dtype),
                                encoder_hidden_states=batch.
                                negative_prompt_embeds[0].to(target_dtype),
                                fps=fps,
                                condition_video_input_mask=
                                condition_video_input_mask.to(target_dtype)
                                if condition_video_input_mask is not None else
                                None,
                                condition_video_pose=uncond_pose.to(
                                    target_dtype)
                                if uncond_pose is not None else None,
                                condition_video_augment_sigma=
                                condition_video_augment_sigma
                                if condition_video_augment_sigma is not None
                                else None,
                                padding_mask=padding_mask,
                            )

                        if isinstance(uncond_noise_pred, tuple):
                            uncond_noise_pred = uncond_noise_pred[0]

                        uncond_pred = (
                            c_skip * latents +
                            c_out * uncond_noise_pred.float()).to(target_dtype)

                        # Apply guidance
                        pred = uncond_pred + guidance_scale * (cond_pred -
                                                               uncond_pred)
                    else:
                        pred = cond_pred

                    # Step scheduler
                    step_output = self.scheduler.step(
                        pred,
                        t,
                        latents,
                        **extra_step_kwargs,
                    )
                    latents = step_output.prev_sample

                    if hasattr(self, "callback_on_step_end"
                               ) and self.callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in self.callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = self.callback_on_step_end(
                            self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()

        batch.latents = latents
        return batch


class Gen3CPipeline(ComposedPipelineBase):
    """
    GEN3C Video Generation Pipeline.
    
    This pipeline extends Cosmos with 3D cache support for camera-controlled
    video generation.
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            use_karras_sigmas=True)

        sigma_max = 80.0
        sigma_min = 0.002
        sigma_data = 1.0
        final_sigmas_type = "sigma_min"

        if self.modules["scheduler"] is not None:
            scheduler = self.modules["scheduler"]
            scheduler.config.sigma_max = sigma_max
            scheduler.config.sigma_min = sigma_min
            scheduler.config.sigma_data = sigma_data
            scheduler.config.final_sigmas_type = final_sigmas_type
            scheduler.sigma_max = sigma_max
            scheduler.sigma_min = sigma_min
            scheduler.sigma_data = sigma_data

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=Gen3CLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=Gen3CDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = Gen3CPipeline
