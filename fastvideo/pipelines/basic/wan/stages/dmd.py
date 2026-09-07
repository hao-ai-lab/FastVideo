# SPDX-License-Identifier: Apache-2.0
"""Wan's non-causal DMD sampler. The pipeline supplies its training-noise scheduler."""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.utils import dict_to_3d_list

try:
    from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionBackend
    vsa_available = True
except ImportError:
    vsa_available = False


class DmdDenoisingStage(DenoisingStage):
    """
    Denoising stage for DMD.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run the denoising loop.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with denoised latents.
        """
        # Setup precision and autocast settings
        # TODO(will): make the precision configurable for inference
        # target_dtype = PRECISION_TO_TYPE[fastvideo_args.precision]
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps

        # TODO(will): remove this once we add input/output validation for stages
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Prepare image latents and embeddings for I2V generation
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [image_embed.to(target_dtype) for image_embed in image_embeds]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(None, t_max=50, l_max=60, h_max=24)
            },
        )

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        # Get latents and embeddings
        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents

        video_raw_latent_shape = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert not torch.isnan(prompt_embeds[0]).any(), "prompt_embeds contains nan"
        timesteps = torch.tensor(fastvideo_args.pipeline_config.dmd_denoising_steps,
                                 dtype=torch.long,
                                 device=get_local_torch_device())

        # Run denoising loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue
                # Expand latents for I2V
                noise_latents = latents.clone()
                latent_model_input = latents.to(target_dtype)

                if batch.image_latent is not None:
                    latent_model_input = torch.cat(
                        [latent_model_input, batch.image_latent.permute(0, 2, 1, 3, 4)], dim=2).to(target_dtype)
                assert not torch.isnan(latent_model_input).any(), "latent_model_input contains nan"

                # Prepare inputs for transformer
                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (torch.tensor(
                    [fastvideo_args.pipeline_config.embedded_cfg_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=get_local_torch_device(),
                ).to(target_dtype) * 1000.0 if fastvideo_args.pipeline_config.embedded_cfg_scale is not None else None)

                # Predict noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    if (vsa_available and self.attn_backend == VideoSparseAttentionBackend):
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls()

                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls()
                            # TODO(will): clean this up
                            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                                current_timestep=i,  # type: ignore
                                raw_latent_shape=batch.raw_latent_shape[2:5],  # type: ignore
                                patch_size=fastvideo_args.pipeline_config.  # type: ignore
                                dit_config.patch_size,  # type: ignore
                                VSA_sparsity=fastvideo_args.VSA_sparsity,  # type: ignore
                                device=get_local_torch_device(),  # type: ignore
                            )  # type: ignore
                            assert attn_metadata is not None, "attn_metadata cannot be None"
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None

                    batch.is_cfg_negative = False
                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                            # fastvideo_args=fastvideo_args
                    ):
                        # Run transformer
                        pred_noise = self.transformer(
                            latent_model_input.permute(0, 2, 1, 3, 4),
                            prompt_embeds,
                            t_expand,
                            guidance=guidance_expand,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        ).permute(0, 2, 1, 3, 4)

                    pred_video = pred_noise_to_pred_video(pred_noise=pred_noise.flatten(0, 1),
                                                          noise_input_latent=noise_latents.flatten(0, 1),
                                                          timestep=t_expand,
                                                          scheduler=self.scheduler).unflatten(0, pred_noise.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones([1], dtype=torch.long, device=pred_video.device)
                        noise = torch.randn(video_raw_latent_shape,
                                            dtype=pred_video.dtype,
                                            generator=batch.generator[0]).to(self.device)
                        latents = self.scheduler.add_noise(pred_video.flatten(0, 1), noise.flatten(0, 1),
                                                           next_timestep).unflatten(0, pred_video.shape[:2])
                    else:
                        latents = pred_video

                    # Update progress bar
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                                   (i + 1) % self.scheduler.order == 0 and progress_bar is not None):
                        progress_bar.update()

        # Gather results if using sequence parallelism
        latents = latents.permute(0, 2, 1, 3, 4)
        # Update batch with final latents
        batch.latents = latents

        return batch
