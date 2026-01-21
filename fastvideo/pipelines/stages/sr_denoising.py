# SPDX-License-Identifier: Apache-2.0
"""
Denoising stage for diffusion pipelines.
"""

import inspect
import weakref
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from tqdm.auto import tqdm

from fastvideo.attention import get_attn_backend
from fastvideo.configs.pipelines.base import STA_Mode
from fastvideo.distributed import (get_local_torch_device, get_world_group)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader, UpsamplerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.platforms import AttentionBackendEnum

try:
    from fastvideo.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False

try:
    from fastvideo.attention.backends.vmoba import VMOBAAttentionBackend
    from fastvideo.utils import is_vmoba_available
    vmoba_attn_available = is_vmoba_available()
except ImportError:
    vmoba_attn_available = False

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False

logger = init_logger(__name__)


class SRDenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in SR diffusion pipelines. Used by Hunyuan15 SR pipeline.
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """

    def __init__(self,
                 transformer,
                 scheduler,
                 upsampler,
                 pipeline=None,
                 vae=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.upsampler = upsampler
        self.pipeline = weakref.ref(pipeline) if pipeline else None
        attn_head_size = self.transformer.hidden_size // self.transformer.num_attention_heads
        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,  # TODO(will): hack
            supported_attention_backends=(
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.VMOBA_ATTN,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.SAGE_ATTN_THREE)  # hack
        )

    def add_noise_to_lq(self, lq_latents, strength=0.7):

        def expand_dims(tensor: torch.Tensor, ndim: int):
            shape = tensor.shape + (1, ) * (ndim - tensor.ndim)
            return tensor.reshape(shape)

        noise = torch.randn_like(lq_latents)
        timestep = torch.tensor([1000.0],
                                device=get_local_torch_device()) * strength
        t = expand_dims(timestep, lq_latents.ndim)
        return (1 - t / 1000.0) * lq_latents + (t / 1000.0) * noise

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
        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        if not fastvideo_args.model_loaded["upsampler"]:
            loader = UpsamplerLoader()
            self.upsampler = loader.load(
                fastvideo_args.model_paths["upsampler"], fastvideo_args)
            if pipeline:
                pipeline.add_module("upsampler", self.upsampler)
            fastvideo_args.model_loaded["upsampler"] = True

        # Setup precision and autocast settings
        # TODO(will): make the precision configurable for inference
        # target_dtype = PRECISION_TO_TYPE[fastvideo_args.precision]
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        self.scheduler.set_shift(fastvideo_args.pipeline_config.flow_shift_sr)
        sigmas = np.linspace(1.0, 0.0, batch.num_inference_steps_sr + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas,
                                     device=get_local_torch_device())
        timesteps = self.scheduler.timesteps
        logger.info("timesteps: %s", timesteps)
        num_inference_steps = len(timesteps)
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        assert num_inference_steps == batch.num_inference_steps_sr, "num_inference_steps_sr must match the number of timesteps"

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert not torch.isnan(
                image_embeds[0]).any(), "image_embeds contains nan"
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
            },
        )

        # Prepare STA parameters
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        # Get latents and embeddings
        prompt_embeds = batch.prompt_embeds
        assert not torch.isnan(
            prompt_embeds[0]).any(), "prompt_embeds contains nan"

        latents = batch.latents
        lq_latents = batch.lq_latents
        logger.info("lq_latents: %s", lq_latents.shape)
        logger.info("latents: %s", latents.shape)
        tgt_shape = latents.shape[-2:]  # (h w)
        bsz = lq_latents.shape[0]
        lq_latents = rearrange(lq_latents, "b c f h w -> (b f) c h w")
        lq_latents = F.interpolate(lq_latents,
                                   size=tgt_shape,
                                   mode="bilinear",
                                   align_corners=False)
        lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=bsz)
        lq_latents = self.upsampler(
            lq_latents.to(dtype=torch.float32, device=get_local_torch_device()))
        lq_latents = lq_latents.to(dtype=latents.dtype)
        lq_latents = self.add_noise_to_lq(lq_latents, 0.7)
        b, c, f, h, w = lq_latents.shape
        mask_ones = torch.ones(b, 1, f, h, w).to(lq_latents.device)
        lq_cond_latents = torch.concat([lq_latents, mask_ones],
                                       dim=1).to(target_dtype)
        cond_latents = torch.cat(
            [batch.video_latent, torch.zeros_like(latents)],
            dim=1).to(target_dtype)
        condition = torch.concat([cond_latents, lq_cond_latents], dim=1)
        zero_lq_condition = condition.clone()
        zero_lq_condition[:, c + 1:2 * c + 1] = torch.zeros_like(lq_latents)
        zero_lq_condition[:, 2 * c + 1] = 0

        latent_model_input = latents.to(target_dtype)
        assert latent_model_input.shape[0] == 1, "only support batch size 1"

        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                latent_model_input = latents.to(target_dtype)
                if t < 1000 * 0.7:
                    condition = zero_lq_condition

                latent_model_input = torch.concat([latents, condition], dim=1)

                assert not torch.isnan(
                    latent_model_input).any(), "latent_model_input contains nan"
                t_expand = t.repeat(latent_model_input.shape[0])

                if i == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0],
                                               device=get_local_torch_device())
                else:
                    timesteps_r = timesteps[i + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # Prepare inputs for transformer
                guidance_expand = (
                    torch.tensor(
                        [fastvideo_args.pipeline_config.embedded_cfg_scale] *
                        latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=get_local_torch_device(),
                    ).to(target_dtype) *
                    1000.0 if fastvideo_args.pipeline_config.embedded_cfg_scale
                    is not None else None)

                # Predict noise residual
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):
                    if (st_attn_available
                            and self.attn_backend == SlidingTileAttentionBackend
                        ) or (vsa_available and self.attn_backend
                              == VideoSparseAttentionBackend):
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )

                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            # TODO(will): clean this up
                            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                                current_timestep=i,  # type: ignore
                                raw_latent_shape=batch.
                                raw_latent_shape[2:5],  # type: ignore
                                patch_size=fastvideo_args.
                                pipeline_config.  # type: ignore
                                dit_config.patch_size,  # type: ignore
                                STA_param=batch.STA_param,  # type: ignore
                                VSA_sparsity=fastvideo_args.
                                VSA_sparsity,  # type: ignore
                                device=get_local_torch_device(),
                            )
                            assert attn_metadata is not None, "attn_metadata cannot be None"
                        else:
                            attn_metadata = None
                    elif (vmoba_attn_available
                          and self.attn_backend == VMOBAAttentionBackend):
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            # Prepare V-MoBA parameters from config
                            moba_params = fastvideo_args.moba_config.copy()
                            moba_params.update({
                                "current_timestep":
                                i,
                                "raw_latent_shape":
                                batch.raw_latent_shape[2:5],
                                "patch_size":
                                fastvideo_args.pipeline_config.dit_config.
                                patch_size,
                                "device":
                                get_local_torch_device(),
                            })
                            attn_metadata = self.attn_metadata_builder.build(
                                **moba_params)
                            assert attn_metadata is not None, "attn_metadata cannot be None"
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None
                    # TODO(will): finalize the interface. vLLM uses this to
                    # support torch dynamo compilation. They pass in
                    # attn_metadata, vllm_config, and num_tokens. We can pass in
                    # fastvideo_args or training_args, and attn_metadata.
                    batch.is_cfg_negative = False
                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                            # fastvideo_args=fastvideo_args
                    ):
                        # Run transformer
                        noise_pred = self.transformer(latent_model_input,
                                                      prompt_embeds,
                                                      t_expand,
                                                      guidance=guidance_expand,
                                                      timestep_r=timesteps_r,
                                                      **pos_cond_kwargs,
                                                      **image_kwargs)

                    # Compute the previous noisy sample
                    latents = self.scheduler.step(noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                        and progress_bar is not None):
                    progress_bar.update()

        # Update batch with final latents
        batch.latents = latents

        # Save STA mask search results if needed
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend and fastvideo_args.STA_mode == STA_Mode.STA_SEARCHING:
            self.save_sta_search_results(batch)

        # deallocate transformer if on mps
        if torch.backends.mps.is_available():
            logger.info("Memory before deallocating transformer: %s",
                        torch.mps.current_allocated_memory())
            del self.transformer
            if pipeline is not None and "transformer" in pipeline.modules:
                del pipeline.modules["transformer"]
            fastvideo_args.model_loaded["transformer"] = False
            logger.info("Memory after deallocating transformer: %s",
                        torch.mps.current_allocated_memory())

        return batch

    def prepare_extra_func_kwargs(self, func, kwargs) -> dict[str, Any]:
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
                     iterable: Iterable | None = None,
                     total: int | None = None) -> tqdm:
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

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps,
                         [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check("image_latent", batch.image_latent,
                         V.none_or_tensor_with_dims(5))
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale,
                         V.positive_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, lambda x:
            not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        return result
