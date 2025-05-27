# SPDX-License-Identifier: Apache-2.0
"""
Denoising stage for diffusion pipelines.
"""

import importlib.util
import inspect
from typing import Any, Dict, Iterable, List, Optional

import torch
from einops import rearrange
from tqdm.auto import tqdm

from fastvideo.v1.attention import get_attn_backend
from fastvideo.v1.distributed import (get_sequence_model_parallel_rank,
                                      get_sequence_model_parallel_world_size,
                                      get_world_group)
from fastvideo.v1.distributed.communication_op import (
    sequence_model_parallel_all_gather)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.platforms import _Backend
from fastvideo.v1.utils import PRECISION_TO_TYPE

st_attn_available = False
spec = importlib.util.find_spec("st_attn")
if spec is not None:
    st_attn_available = True

    from fastvideo.v1.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)

logger = init_logger(__name__)


class DenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in diffusion pipelines.
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

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
        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta
            },
        )

        # Setup precision and autocast settings
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.precision]
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        # Handle sequence parallelism if enabled
        world_size, rank = get_sequence_model_parallel_world_size(
        ), get_sequence_model_parallel_rank()
        sp_group = world_size > 1
        if sp_group:
            latents = rearrange(batch.latents,
                                "b t (n s) h w -> b t n s h w",
                                n=world_size).contiguous()
            latents = latents[:, :, rank, :, :, :]
            batch.latents = latents
            if batch.image_latent is not None:
                image_latent = rearrange(batch.image_latent,
                                         "b t (n s) h w -> b t n s h w",
                                         n=world_size).contiguous()
                image_latent = image_latent[:, :, rank, :, :, :]
                batch.image_latent = image_latent

        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        # TODO(will): remove this once we add input/output validation for stages
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order

        # Create 3D list for mask strategy
        def dict_to_3d_list(mask_strategy,
                            t_max=50,
                            l_max=60,
                            h_max=24) -> List:
            result = [[[None for _ in range(h_max)] for _ in range(l_max)]
                      for _ in range(t_max)]
            if mask_strategy is None:
                return result
            for key, value in mask_strategy.items():
                t, layer, h = map(int, key.split('_'))
                result[t][layer][h] = value
            return result

        # Prepare image latents and embeddings for I2V generation
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(None)
            },
        )

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        neg_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_neg,
                "encoder_attention_mask": batch.negative_attention_mask,
            },
        )

        # Get latents and embeddings
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
            assert neg_prompt_embeds is not None
            assert torch.isnan(neg_prompt_embeds[0]).sum() == 0

        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                # Expand latents for I2V
                latent_model_input = latents.to(target_dtype)
                if batch.image_latent is not None:
                    latent_model_input = torch.cat(
                        [latent_model_input, batch.image_latent],
                        dim=1).to(target_dtype)
                assert torch.isnan(latent_model_input).sum() == 0
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # Prepare inputs for transformer
                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (torch.tensor(
                    [fastvideo_args.embedded_cfg_scale] *
                    latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=fastvideo_args.device,
                ).to(target_dtype) * 1000.0 if fastvideo_args.embedded_cfg_scale
                                   is not None else None)

                # Predict noise residual
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):

                    # TODO(will-refactor): all of this should be in the stage's init
                    attn_head_size = self.transformer.hidden_size // self.transformer.num_attention_heads
                    self.attn_backend = get_attn_backend(
                        head_size=attn_head_size,
                        dtype=torch.float16,  # TODO(will): hack
                        supported_attention_backends=(
                            _Backend.SLIDING_TILE_ATTN, _Backend.FLASH_ATTN,
                            _Backend.TORCH_SDPA)  # hack
                    )
                    if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            # TODO(will): clean this up
                            attn_metadata = self.attn_metadata_builder.build(
                                current_timestep=i,
                                forward_batch=batch,
                                fastvideo_args=fastvideo_args,
                            )
                            assert attn_metadata is not None, "attn_metadata cannot be None"
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None
                    # TODO(will): finalize the interface. vLLM uses this to
                    # support torch dynamo compilation. They pass in
                    # attn_metadata, vllm_config, and num_tokens. We can pass in
                    # fastvideo_args or training_args, and attn_metadata.
                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                            # fastvideo_args=fastvideo_args
                    ):
                        # Run transformer
                        noise_pred = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            t_expand,
                            guidance=guidance_expand,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    # Apply guidance
                    if batch.do_classifier_free_guidance:
                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=attn_metadata,
                                forward_batch=batch,
                                # fastvideo_args=fastvideo_args
                        ):
                            # Run transformer
                            noise_pred_uncond = self.transformer(
                                latent_model_input,
                                neg_prompt_embeds,
                                t_expand,
                                guidance=guidance_expand,
                                **image_kwargs,
                                **neg_cond_kwargs,
                            )
                        noise_pred_text = noise_pred
                        noise_pred = noise_pred_uncond + batch.guidance_scale * (
                            noise_pred_text - noise_pred_uncond)

                        # Apply guidance rescale if needed
                        if batch.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = self.rescale_noise_cfg(
                                noise_pred,
                                noise_pred_text,
                                guidance_rescale=batch.guidance_rescale,
                            )

                    # Compute the previous noisy sample
                    latents = self.scheduler.step(noise_pred,
                                                  t,
                                                  latents,
                                                  **extra_step_kwargs,
                                                  return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                        and progress_bar is not None):
                    progress_bar.update()

        # Gather results if using sequence parallelism
        if sp_group:
            latents = sequence_model_parallel_all_gather(latents, dim=2)

        # Update batch with final latents
        batch.latents = latents

        if fastvideo_args.use_cpu_offload:
            self.transformer.to('cpu')
            torch.cuda.empty_cache()

        return batch

    def prepare_extra_func_kwargs(self, func, kwargs) -> Dict[str, Any]:
        """
        Prepare extra kwargs for the scheduler step / denoise step.
        
        Args:
            func: The function to prepare kwargs for.
            kwargs: The kwargs to prepare.
            
        Returns:
            The prepared kwargs.
        """
        extra_step_kwargs = {}
        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def progress_bar(self,
                     iterable: Optional[Iterable] = None,
                     total: Optional[int] = None) -> tqdm:
        """
        Create a progress bar for the denoising process.
        
        Args:
            iterable: The iterable to iterate over.
            total: The total number of items.
            
        Returns:
            A tqdm progress bar.
        """
        local_rank = get_world_group().local_rank
        if local_rank == 0:
            return tqdm(iterable=iterable, total=total)
        else:
            return tqdm(iterable=iterable, total=total, disable=True)

    def rescale_noise_cfg(self,
                          noise_cfg,
                          noise_pred_text,
                          guidance_rescale=0.0) -> torch.Tensor:
        """
        Rescale noise prediction according to guidance_rescale.
        
        Based on findings of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        (https://arxiv.org/pdf/2305.08891.pdf), Section 3.4.
        
        Args:
            noise_cfg: The noise prediction with guidance.
            noise_pred_text: The text-conditioned noise prediction.
            guidance_rescale: The guidance rescale factor.
            
        Returns:
            The rescaled noise prediction.
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)),
                                       keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)),
                                keepdim=True)
        # Rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # Mix with the original results from guidance by factor guidance_rescale
        noise_cfg = (guidance_rescale * noise_pred_rescaled +
                     (1 - guidance_rescale) * noise_cfg)
        return noise_cfg
