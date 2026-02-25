# SPDX-License-Identifier: Apache-2.0
"""
Implements GEN3C video diffusion pipeline with 3D cache support for camera-controlled video generation.
"""

import torch
from diffusers import EDMEulerScheduler

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (DecodingStage, InputValidationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage

logger = init_logger(__name__)


class Gen3CConditioningStage(PipelineStage):
    """
    3D cache conditioning stage for GEN3C.

    This stage performs the core GEN3C innovation:
    1. Loads the input image
    2. Predicts depth via MoGe
    3. Initializes a 3D point cloud cache
    4. Generates a camera trajectory
    5. Renders warped frames from the cache at each target camera pose
    6. Stores rendered warps on the batch for VAE encoding in the latent prep stage
    """

    def __init__(self, vae=None) -> None:
        super().__init__()
        self._moge_model = None
        self._vae = vae

    def _get_moge_model(self, device: torch.device, model_name: str):
        """Lazy-load MoGe model on first use and ensure it is on target device."""
        if self._moge_model is None:
            from fastvideo.pipelines.basic.gen3c.depth_estimation import (
                load_moge_model)
            self._moge_model = load_moge_model(model_name, device)
        else:
            first_param = next(self._moge_model.parameters(), None)
            if first_param is not None and first_param.device != device:
                self._moge_model = self._moge_model.to(device)
        return self._moge_model

    def _offload_moge(self):
        """Move MoGe to CPU to free GPU memory before denoising."""
        if self._moge_model is not None and torch.cuda.is_available():
            self._moge_model = self._moge_model.cpu()
            torch.cuda.empty_cache()
            logger.info("MoGe model offloaded to CPU")

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run 3D cache conditioning pipeline."""
        pipeline_config = fastvideo_args.pipeline_config
        device = get_local_torch_device()
        batch_extra = getattr(batch, "extra", {}) or {}

        # Check if an input image was provided
        image_path = getattr(batch, 'image_path', None) or batch_extra.get(
            "image_path")
        if image_path is None:
            logger.info(
                "No image_path provided — skipping 3D cache conditioning "
                "(will use zero conditioning)")
            return batch

        logger.info("Running 3D cache conditioning with image: %s", image_path)

        # Get config parameters
        height = getattr(batch, 'height', None) or getattr(
            pipeline_config, 'video_resolution', (720, 1280))[0]
        width = getattr(batch, 'width', None) or getattr(
            pipeline_config, 'video_resolution', (720, 1280))[1]
        num_frames = getattr(batch, 'num_frames', None) or getattr(
            pipeline_config, 'num_frames', 121)

        trajectory_type = (
            getattr(batch, 'trajectory_type', None)
            or batch_extra.get("trajectory_type")
            or getattr(pipeline_config, 'default_trajectory_type', 'left'))
        movement_distance = (
            getattr(batch, 'movement_distance', None)
            or batch_extra.get("movement_distance")
            or getattr(pipeline_config, 'default_movement_distance', 0.3))
        camera_rotation = (
            getattr(batch, 'camera_rotation', None)
            or batch_extra.get("camera_rotation")
            or getattr(pipeline_config, 'default_camera_rotation',
                       'center_facing'))

        frame_buffer_max = getattr(pipeline_config, 'frame_buffer_max', 2)
        noise_aug_strength = getattr(pipeline_config, 'noise_aug_strength',
                                     0.0)
        filter_points_threshold = getattr(pipeline_config,
                                          'filter_points_threshold', 0.05)

        moge_model_name = getattr(pipeline_config, 'moge_model_name',
                                  'Ruicheng/moge-vitl')

        # Step 1: Predict depth from input image
        from fastvideo.pipelines.basic.gen3c.depth_estimation import (
            predict_depth_from_path)

        moge_model = self._get_moge_model(device, moge_model_name)

        (
            image_b1chw,
            depth_b11hw,
            mask_b11hw,
            w2c_b144,
            intrinsics_b133,
        ) = predict_depth_from_path(image_path, height, width, device,
                                    moge_model)

        logger.info(
            "Depth prediction complete. Depth range: [%.3f, %.3f]",
            depth_b11hw.min().item(),
            depth_b11hw.max().item(),
        )

        # Step 2: Initialize 3D cache
        from fastvideo.pipelines.basic.gen3c.cache_3d import Cache3DBuffer

        seed = getattr(batch, 'seed', None)
        if seed is None:
            seed = 42
        generator = torch.Generator(device=device).manual_seed(seed)

        cache = Cache3DBuffer(
            frame_buffer_max=frame_buffer_max,
            generator=generator,
            noise_aug_strength=noise_aug_strength,
            input_image=image_b1chw[:, 0].clone(),
            input_depth=depth_b11hw[:, 0],
            input_w2c=w2c_b144[:, 0],
            input_intrinsics=intrinsics_b133[:, 0],
            filter_points_threshold=filter_points_threshold,
        )

        logger.info("3D cache initialized with %d frame buffer(s)",
                     frame_buffer_max)

        # Step 3: Generate camera trajectory
        from fastvideo.pipelines.basic.gen3c.camera_utils import (
            generate_camera_trajectory)

        initial_w2c = w2c_b144[0, 0]
        initial_intrinsics = intrinsics_b133[0, 0]

        generated_w2cs, generated_intrinsics = generate_camera_trajectory(
            trajectory_type=trajectory_type,
            initial_w2c=initial_w2c,
            initial_intrinsics=initial_intrinsics,
            num_frames=num_frames,
            movement_distance=movement_distance,
            camera_rotation=camera_rotation,
            center_depth=1.0,
            device=device.type if isinstance(device, torch.device) else device,
        )

        logger.info(
            "Camera trajectory generated: type=%s, frames=%d, distance=%.3f",
            trajectory_type,
            num_frames,
            movement_distance,
        )

        # Step 4: Render 3D cache to target views
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, :num_frames],
            generated_intrinsics[:, :num_frames],
        )

        logger.info(
            "Cache rendered. Warped images shape: %s, non-zero mask ratio: %.3f",
            list(rendered_warp_images.shape),
            (rendered_warp_masks > 0).float().mean().item(),
        )

        # Store on batch for the latent preparation stage
        batch.rendered_warp_images = rendered_warp_images.to(device)
        batch.rendered_warp_masks = rendered_warp_masks.to(device)
        # Keep the original resized input image as the first-frame latent anchor.
        # Official GEN3C conditions on tokenizer latents from the source image.
        batch.input_image_conditioning = image_b1chw[:, 0].unsqueeze(2).contiguous().to(device)
        batch.cache_3d = cache

        # Offload MoGe to free GPU memory before heavy stages.
        if getattr(pipeline_config, "offload_moge_after_depth", True):
            self._offload_moge()

        return batch


class Gen3CLatentPreparationStage(LatentPreparationStage):
    """
    Latent preparation stage for GEN3C.

    This stage prepares latents and encodes 3D cache buffers through the VAE.
    If rendered warped frames are available on the batch (from Gen3CConditioningStage),
    they are VAE-encoded to produce real conditioning. Otherwise falls back to zeros.
    """

    def __init__(self, scheduler, transformer, vae) -> None:
        super().__init__(scheduler, transformer)
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Prepare latents and encode 3D cache buffers."""
        pipeline_config = fastvideo_args.pipeline_config
        device = get_local_torch_device()

        # Determine batch size
        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]
        batch_size *= batch.num_videos_per_prompt

        num_channels_latents = getattr(self.transformer,
                                       'num_channels_latents', 16)

        num_frames = getattr(batch, 'num_frames', None) or getattr(
            pipeline_config, 'num_frames', 121)
        if hasattr(self.vae, "get_latent_num_frames"):
            latent_frames = int(self.vae.get_latent_num_frames(num_frames))
        else:
            temporal_ratio = getattr(
                pipeline_config.vae_config.arch_config,
                "temporal_compression_ratio",
                4,
            )
            latent_frames = int((num_frames - 1) // temporal_ratio + 1)
        height = getattr(batch, 'height', 720)
        width = getattr(batch, 'width', 1280)

        spatial_ratio = getattr(
            pipeline_config.vae_config.arch_config,
            "spatial_compression_ratio",
            8,
        )
        latent_height = height // spatial_ratio
        latent_width = width // spatial_ratio

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

        if hasattr(self.scheduler, 'init_noise_sigma'):
            latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.batch_size = batch_size
        batch.height = height
        batch.width = width
        batch.latent_height = latent_height
        batch.latent_width = latent_width
        batch.latent_frames = latent_frames
        batch.raw_latent_shape = latents.shape

        # Get buffer config
        frame_buffer_max = getattr(pipeline_config, 'frame_buffer_max', 2)
        CHANNELS_PER_BUFFER = 32
        buffer_channels = frame_buffer_max * CHANNELS_PER_BUFFER

        # Check if we have real rendered warps from the conditioning stage
        rendered_warp_images = getattr(batch, 'rendered_warp_images', None)
        rendered_warp_masks = getattr(batch, 'rendered_warp_masks', None)

        if rendered_warp_images is not None and rendered_warp_masks is not None:
            logger.info(
                "Encoding rendered warped frames through VAE (%d buffers)...",
                rendered_warp_images.shape[2],
            )

            # VAE can be CPU-offloaded by default; move it back for conditioning encode.
            self.vae = self.vae.to(device)

            # Determine VAE dtype
            if hasattr(self.vae, 'module'):
                vae_dtype = next(self.vae.module.parameters()).dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype

            condition_video_pose = self.encode_warped_frames(
                rendered_warp_images,
                rendered_warp_masks,
                self.vae,
                frame_buffer_max,
                vae_dtype,
            )
            batch.condition_video_pose = condition_video_pose.to(device)

            logger.info(
                "condition_video_pose encoded. Shape: %s, non-zero: %.4f",
                list(batch.condition_video_pose.shape),
                (batch.condition_video_pose != 0).float().mean().item(),
            )

            # Build image-conditioning latents from the original source image
            # (official V2W/GEN3C behavior), not from rendered cache buffers.
            source_image = getattr(batch, "input_image_conditioning", None)
            if source_image is None:
                source_image = rendered_warp_images[:, 0, 0].unsqueeze(2)
            first_frame = source_image.to(device=device, dtype=vae_dtype)
            first_latent = self._retrieve_latents(self.vae.encode(first_frame))
            conditioning_latents = torch.zeros(
                batch_size,
                num_channels_latents,
                latent_frames,
                latent_height,
                latent_width,
                device=device,
                dtype=first_latent.dtype,
            )
            conditioning_latents[:, :, :first_latent.shape[2], :, :] = first_latent
            batch.conditioning_latents = conditioning_latents

            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")

            # Set condition_video_input_mask: 1 for first frame (the reference image)
            batch.condition_video_input_mask = torch.zeros(
                batch_size,
                1,
                latent_frames,
                latent_height,
                latent_width,
                device=device,
                dtype=torch.float32,
            )
            # First latent frame = conditioned (reference image)
            batch.condition_video_input_mask[:, :, 0, :, :] = 1.0
        else:
            logger.info(
                "No rendered warps available — using zero conditioning")
            batch.condition_video_pose = torch.zeros(
                batch_size,
                buffer_channels,
                latent_frames,
                latent_height,
                latent_width,
                device=device,
                dtype=torch.float32,
            )
            batch.condition_video_input_mask = torch.zeros(
                batch_size,
                1,
                latent_frames,
                latent_height,
                latent_width,
                device=device,
                dtype=torch.float32,
            )
            batch.conditioning_latents = None

        batch.condition_video_augment_sigma = torch.zeros(batch_size,
                                                          device=device,
                                                          dtype=torch.float32)
        batch.cond_indicator = torch.zeros(
            batch_size,
            1,
            latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=torch.float32,
        )
        batch.cond_indicator[:, :, 0, :, :] = 1.0

        ones_padding = torch.ones_like(batch.cond_indicator)
        zeros_padding = torch.zeros_like(batch.cond_indicator)
        batch.cond_mask = batch.cond_indicator * ones_padding + (
            1 - batch.cond_indicator) * zeros_padding

        if batch.do_classifier_free_guidance:
            batch.uncond_indicator = batch.cond_indicator.clone()
            batch.uncond_mask = batch.cond_mask.clone()
        else:
            batch.uncond_indicator = None
            batch.uncond_mask = None

        return batch

    @staticmethod
    def _retrieve_latents(encoder_output):
        if hasattr(encoder_output, "latent_dist"):
            latent_dist = encoder_output.latent_dist
            if hasattr(latent_dist, "mode"):
                return latent_dist.mode()
            if hasattr(latent_dist, "mean"):
                return latent_dist.mean
            return latent_dist.sample()
        if hasattr(encoder_output, "mode"):
            return encoder_output.mode()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        if hasattr(encoder_output, "sample"):
            return encoder_output.sample()
        if isinstance(encoder_output, torch.Tensor):
            return encoder_output
        raise AttributeError(
            f"Unsupported VAE encoder output type: {type(encoder_output)}")

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
            condition_state: (B, T, N, 3, H, W) rendered RGB images in [-1, 1].
            condition_state_mask: (B, T, N, 1, H, W) rendered masks in [0, 1].
            vae: VAE encoder.
            frame_buffer_max: Maximum number of buffers.
            dtype: Target dtype.

        Returns:
            latent_condition: (B, buffer_channels, T_latent, H_latent, W_latent)
        """
        assert condition_state.dim() == 6

        # Convert mask to [-1, 1] range and repeat to 3 channels for VAE
        condition_state_mask = (condition_state_mask * 2 - 1).repeat(
            1, 1, 1, 3, 1, 1)

        latent_condition = []
        num_buffers = condition_state.shape[2]
        for i in range(num_buffers):
            img_input = condition_state[:, :, i].permute(0, 2, 1, 3,
                                                         4).to(dtype)
            mask_input = condition_state_mask[:, :, i].permute(0, 2, 1, 3,
                                                               4).to(dtype)
            batched_input = torch.cat([img_input, mask_input], dim=0)
            batched_latent = self._retrieve_latents(vae.encode(
                batched_input)).contiguous()
            current_video_latent, current_mask_latent = batched_latent.chunk(
                2, dim=0)

            latent_condition.append(current_video_latent)
            latent_condition.append(current_mask_latent)

        # Pad with zeros if fewer buffers than frame_buffer_max
        for _ in range(frame_buffer_max - num_buffers):
            latent_condition.append(torch.zeros_like(current_video_latent))
            latent_condition.append(torch.zeros_like(current_mask_latent))

        return torch.cat(latent_condition, dim=1)


class Gen3CDenoisingStage(DenoisingStage):
    """
    Denoising stage for GEN3C models.

    This stage extends the base denoising stage with support for:
    - condition_video_input_mask: Binary mask indicating conditioning frames
    - condition_video_pose: VAE-encoded 3D cache buffers
    - condition_video_augment_sigma: Noise augmentation sigma
    """

    def __init__(self, transformer, scheduler, pipeline=None) -> None:
        super().__init__(transformer, scheduler, pipeline)

    @staticmethod
    def _reverse_precondition_input(
        xt: torch.Tensor,
        sigma: torch.Tensor,
        sigma_data: float,
    ) -> torch.Tensor:
        c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)
        return xt / c_in

    @staticmethod
    def _reverse_precondition_output(
        latent: torch.Tensor,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        sigma_data: float,
    ) -> torch.Tensor:
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
        return (latent - c_skip * xt) / c_out

    def _augment_noise_with_latent(
        self,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        latent: torch.Tensor,
        indicator: torch.Tensor,
        condition_augment_sigma: float,
        sigma_data: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_indicator = indicator
        if float(condition_augment_sigma) >= float(sigma.item()):
            active_indicator = torch.zeros_like(indicator)

        try:
            noise = torch.randn_like(latent, generator=generator)
        except TypeError:
            noise = torch.randn_like(latent)

        augment_sigma = torch.tensor(
            [condition_augment_sigma], device=latent.device, dtype=latent.dtype)
        augment_latent = latent + noise * augment_sigma
        augment_latent = self.scheduler.precondition_inputs(
            augment_latent, condition_augment_sigma)
        augment_latent_unscaled = self._reverse_precondition_input(
            augment_latent, sigma=sigma, sigma_data=sigma_data)

        new_xt = active_indicator * augment_latent_unscaled + (
            1 - active_indicator) * xt
        return new_xt, latent, active_indicator

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
        sigma_data = float(getattr(fastvideo_args.pipeline_config, "sigma_data",
                                   0.5))
        condition_augment_sigma = float(
            getattr(fastvideo_args.pipeline_config, "sigma_conditional", 0.001))

        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        timesteps = self.scheduler.timesteps

        # Get GEN3C-specific inputs
        condition_video_input_mask = getattr(batch,
                                             'condition_video_input_mask', None)
        condition_video_pose = getattr(batch, 'condition_video_pose', None)
        condition_video_augment_sigma = getattr(
            batch, 'condition_video_augment_sigma', None)
        conditioning_latents = getattr(batch, 'conditioning_latents', None)
        cond_indicator = getattr(batch, "cond_indicator", None)
        unconditioning_latents = conditioning_latents
        uncond_indicator = getattr(batch, "uncond_indicator", None)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                self.scheduler._init_step_index(t)
                sigma = self.scheduler.sigmas[self.scheduler.step_index].to(
                    device=latents.device, dtype=latents.dtype)

                model_input = latents
                latent_for_replace = conditioning_latents
                indicator_for_replace = cond_indicator
                if (conditioning_latents is not None
                        and cond_indicator is not None):
                    model_input, latent_for_replace, indicator_for_replace = (
                        self._augment_noise_with_latent(
                            latents,
                            sigma=sigma,
                            latent=conditioning_latents,
                            indicator=cond_indicator,
                            condition_augment_sigma=condition_augment_sigma,
                            sigma_data=sigma_data,
                            generator=batch.generator,
                        ))

                timestep = t.flatten().expand(latents.size(0))
                padding_mask = torch.zeros(
                    batch.batch_size,
                    1,
                    batch.height,
                    batch.width,
                    device=model_input.device,
                    dtype=target_dtype,
                )

                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):
                    model_input_scaled = self.scheduler.scale_model_input(
                        model_input, timestep=t).to(target_dtype)

                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                    ):
                        noise_pred = self.transformer(
                            hidden_states=model_input_scaled,
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

                    if isinstance(noise_pred, tuple):
                        noise_pred = noise_pred[0]
                    cond_pred = noise_pred.float()

                    # Classifier-free guidance
                    if batch.do_classifier_free_guidance and batch.negative_prompt_embeds is not None:
                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                        ):
                            uncond_pose = torch.zeros_like(
                                condition_video_pose
                            ) if condition_video_pose is not None else None

                            uncond_noise_pred = self.transformer(
                                hidden_states=model_input_scaled,
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
                        uncond_pred = uncond_noise_pred.float()

                        pred = cond_pred + guidance_scale * (cond_pred -
                                                             uncond_pred)
                    else:
                        pred = cond_pred

                model_output = pred
                if (latent_for_replace is not None
                        and indicator_for_replace is not None):
                    latent_unscaled = self._reverse_precondition_output(
                        latent_for_replace,
                        xt=model_input,
                        sigma=sigma,
                        sigma_data=sigma_data,
                    )
                    model_output = indicator_for_replace * latent_unscaled + (
                        1 - indicator_for_replace) * model_output

                latents = self.scheduler.step(
                    model_output,
                    t,
                    model_input,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

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
    video generation. When an input image is provided, it runs the full
    3D cache conditioning pipeline (depth estimation -> point cloud ->
    camera trajectory -> forward warping -> VAE encoding).
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = EDMEulerScheduler(
            sigma_max=80.0,
            sigma_min=0.0002,
            sigma_data=float(getattr(fastvideo_args.pipeline_config,
                                     "sigma_data", 0.5)),
        )

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
                       stage=Gen3CConditioningStage(
                           vae=self.get_module("vae")))

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
