# SPDX-License-Identifier: Apache-2.0
"""
Flux2 latent preparation stage using packed 2x2 layout.

Flux2 uses packed latents: transformer sees 128 channels (32*4) with half
spatial resolution; after denoising we unpatchify to 32 channels and full
spatial for VAE decode. This stage prepares (B, 128, T, H//2, W//2).
"""

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage


class Flux2LatentPreparationStage(LatentPreparationStage):
    """
    Latent preparation for Flux2: packed layout with half spatial dimensions.

    Matches diffusers Flux2Pipeline.prepare_latents: shape is
    (B, num_channels_latents, T, H_latent//2, W_latent//2) so the transformer
    sees 128 channels and half spatial; after denoising we unpatchify to
    (B, 32, H_latent, W_latent) before VAE.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Prepare latents with Flux2 packed half-spatial shape."""
        from fastvideo.distributed import get_local_torch_device

        latent_num_frames = None
        if hasattr(self, "adjust_video_length"):
            latent_num_frames = self.adjust_video_length(batch, fastvideo_args)

        if not batch.prompt_embeds:
            if batch.keyboard_cond is not None:
                batch_size = batch.keyboard_cond.shape[0]
            elif batch.mouse_cond is not None:
                batch_size = batch.mouse_cond.shape[0]
            elif batch.image_embeds:
                batch_size = batch.image_embeds[0].shape[0]
            else:
                batch_size = 1
        elif isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]

        batch_size *= batch.num_videos_per_prompt

        if not batch.prompt_embeds:
            transformer_dtype = next(self.transformer.parameters()).dtype
            device = get_local_torch_device()
            dummy_prompt = torch.zeros(
                batch_size,
                0,
                self.transformer.hidden_size,
                device=device,
                dtype=transformer_dtype,
            )
            batch.prompt_embeds = [dummy_prompt]
            batch.negative_prompt_embeds = []
            batch.do_classifier_free_guidance = False

        dtype = batch.prompt_embeds[0].dtype
        device = get_local_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = (
            latent_num_frames if latent_num_frames is not None else batch.num_frames
        )
        height = batch.height
        width = batch.width

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        vae_arch = fastvideo_args.pipeline_config.vae_config.arch_config
        scale = vae_arch.spatial_compression_ratio
        # Flux2 packed: half spatial (2x2 patch packing)
        latent_h = (height // scale) // 2
        latent_w = (width // scale) // 2

        if self.use_btchw_layout:
            shape = (
                batch_size,
                num_frames,
                self.transformer.num_channels_latents,
                latent_h,
                latent_w,
            )
            bcthw_shape = tuple(shape[i] for i in [0, 2, 1, 3, 4])
        else:
            shape = (
                batch_size,
                self.transformer.num_channels_latents,
                num_frames,
                latent_h,
                latent_w,
            )
            bcthw_shape = shape

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device)
            is_longcat_refine = (
                batch.refine_from is not None or batch.stage1_video is not None
            )
            if (not is_longcat_refine) and hasattr(
                self.scheduler, "init_noise_sigma"
            ):
                latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.raw_latent_shape = bcthw_shape
        # Flux2 mu depends on image_seq_len; use packed spatial size
        batch.n_tokens = latent_h * latent_w
        return batch
