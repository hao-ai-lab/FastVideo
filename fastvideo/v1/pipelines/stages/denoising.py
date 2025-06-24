# SPDX-License-Identifier: Apache-2.0
"""
Denoising stage for diffusion pipelines.
"""

import inspect
from typing import Any, Dict, Iterable, Optional

import torch
from einops import rearrange
from tqdm.auto import tqdm

from fastvideo.v1.attention import get_attn_backend
from fastvideo.v1.configs.pipelines.base import STA_Mode
from fastvideo.v1.distributed import (get_sp_parallel_rank, get_sp_world_size,
                                      get_torch_device, get_world_group)
from fastvideo.v1.distributed.communication_op import (
    sequence_model_parallel_all_gather)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.validators import StageValidators as V
from fastvideo.v1.pipelines.stages.validators import VerificationResult
from fastvideo.v1.platforms import AttentionBackendEnum
from fastvideo.v1.utils import dict_to_3d_list

try:
    from fastvideo.v1.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False

try:
    from fastvideo.v1.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False

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
        attn_head_size = self.transformer.hidden_size // self.transformer.num_attention_heads
        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,  # TODO(will): hack
            supported_attention_backends=(
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA
            )  # hack
        )

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
        # TODO(will): make the precision configurable for inference
        # target_dtype = PRECISION_TO_TYPE[fastvideo_args.precision]
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        # Handle sequence parallelism if enabled
        sp_world_size, rank_in_sp_group = get_sp_world_size(
        ), get_sp_parallel_rank()
        sp_group = sp_world_size > 1
        if sp_group:
            latents = rearrange(batch.latents,
                                "b t (n s) h w -> b t n s h w",
                                n=sp_world_size).contiguous()
            latents = latents[:, :, rank_in_sp_group, :, :, :]
            batch.latents = latents
            if batch.image_latent is not None:
                image_latent = rearrange(batch.image_latent,
                                         "b t (n s) h w -> b t n s h w",
                                         n=sp_world_size).contiguous()
                image_latent = image_latent[:, :, rank_in_sp_group, :, :, :]
                batch.image_latent = image_latent
        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        # TODO(will): remove this once we add input/output validation for stages
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order

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
                "mask_strategy": dict_to_3d_list(
                    None, t_max=50, l_max=60, h_max=24)
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

        # Prepare STA parameters
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

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
                guidance_expand = (
                    torch.tensor(
                        [fastvideo_args.pipeline_config.embedded_cfg_scale] *
                        latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=get_torch_device(),
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
                    batch.is_cfg_negative = False
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
                        batch.is_cfg_negative = True
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

        # Save STA mask search results if needed
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend and fastvideo_args.STA_mode == STA_Mode.STA_SEARCHING:
            self.save_sta_search_results(batch)

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

    def prepare_sta_param(self, batch: ForwardBatch,
                          fastvideo_args: FastVideoArgs):
        """
        Prepare Sliding Tile Attention (STA) parameters and settings.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
        """
        # TODO(kevin): STA mask search, currently only support Wan2.1 with 69x768x1280
        from fastvideo.v1.STA_configuration import configure_sta
        STA_mode = fastvideo_args.STA_mode
        skip_time_steps = fastvideo_args.skip_time_steps
        if batch.timesteps is None:
            raise ValueError("Timesteps must be provided")
        timesteps_num = batch.timesteps.shape[0]

        logger.info("STA_mode: %s", STA_mode)
        if (batch.num_frames, batch.height,
                batch.width) != (69, 768, 1280) and STA_mode != "STA_inference":
            raise NotImplementedError(
                "STA mask search/tuning is not supported for this resolution")

        if STA_mode == STA_Mode.STA_SEARCHING or STA_mode == STA_Mode.STA_TUNING or STA_mode == STA_Mode.STA_TUNING_CFG:
            size = (batch.width, batch.height)
            if size == (1280, 768):
                # TODO: make it configurable
                sparse_mask_candidates_searching = [
                    "3, 1, 10", "1, 5, 7", "3, 3, 3", "1, 6, 5", "1, 3, 10",
                    "3, 6, 1"
                ]
                sparse_mask_candidates_tuning = [
                    "3, 1, 10", "1, 5, 7", "3, 3, 3", "1, 6, 5", "1, 3, 10",
                    "3, 6, 1"
                ]
                full_mask = ["3,6,10"]
            else:
                raise NotImplementedError(
                    "STA mask search is not supported for this resolution")
        layer_num = self.transformer.config.num_layers
        # specific for HunyuanVideo
        if hasattr(self.transformer.config, "num_single_layers"):
            layer_num += self.transformer.config.num_single_layers
        head_num = self.transformer.config.num_attention_heads

        if STA_mode == STA_Mode.STA_SEARCHING:
            STA_param = configure_sta(
                mode=STA_Mode.STA_SEARCHING,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_candidates=sparse_mask_candidates_searching +
                full_mask,  # last is full mask; Can add more sparse masks while keep last one as full mask
            )
        elif STA_mode == STA_Mode.STA_TUNING:
            STA_param = configure_sta(
                mode=STA_Mode.STA_TUNING,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_search_files_path=
                f'output/mask_search_result_pos_{size[0]}x{size[1]}/',
                mask_candidates=sparse_mask_candidates_tuning,
                full_attention_mask=[int(x) for x in full_mask[0].split(',')],
                skip_time_steps=
                skip_time_steps,  # Use full attention for first 12 steps
                save_dir=
                f'output/mask_search_strategy_{size[0]}x{size[1]}/',  # Custom save directory
                timesteps=timesteps_num)
        elif STA_mode == STA_Mode.STA_TUNING_CFG:
            STA_param = configure_sta(
                mode=STA_Mode.STA_TUNING_CFG,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_search_files_path_pos=
                f'output/mask_search_result_pos_{size[0]}x{size[1]}/',
                mask_search_files_path_neg=
                f'output/mask_search_result_neg_{size[0]}x{size[1]}/',
                mask_candidates=sparse_mask_candidates_tuning,
                full_attention_mask=[int(x) for x in full_mask[0].split(',')],
                skip_time_steps=skip_time_steps,
                save_dir=f'output/mask_search_strategy_{size[0]}x{size[1]}/',
                timesteps=timesteps_num)
        elif STA_mode == STA_Mode.STA_INFERENCE:
            import fastvideo.v1.envs as envs
            config_file = envs.FASTVIDEO_ATTENTION_CONFIG
            if config_file is None:
                raise ValueError("FASTVIDEO_ATTENTION_CONFIG is not set")
            STA_param = configure_sta(mode=STA_Mode.STA_INFERENCE,
                                      layer_num=layer_num,
                                      head_num=head_num,
                                      time_step_num=timesteps_num,
                                      load_path=config_file)

        batch.STA_param = STA_param
        batch.mask_search_final_result_pos = [[] for _ in range(timesteps_num)]
        batch.mask_search_final_result_neg = [[] for _ in range(timesteps_num)]

    def save_sta_search_results(self, batch: ForwardBatch):
        """
        Save the STA mask search results.
        
        Args:
            batch: The current batch information.
        """
        size = (batch.width, batch.height)
        if size == (1280, 768):
            # TODO: make it configurable
            sparse_mask_candidates_searching = [
                "3, 1, 10", "1, 5, 7", "3, 3, 3", "1, 6, 5", "1, 3, 10",
                "3, 6, 1"
            ]
        else:
            raise NotImplementedError(
                "STA mask search is not supported for this resolution")

        from fastvideo.v1.STA_configuration import save_mask_search_results
        if batch.mask_search_final_result_pos is not None and batch.prompt is not None:
            save_mask_search_results(
                [
                    dict(layer_data)
                    for layer_data in batch.mask_search_final_result_pos
                ],
                prompt=str(batch.prompt),
                mask_strategies=sparse_mask_candidates_searching,
                output_dir=f'output/mask_search_result_pos_{size[0]}x{size[1]}/'
            )
        if batch.mask_search_final_result_neg is not None and batch.prompt is not None:
            save_mask_search_results(
                [
                    dict(layer_data)
                    for layer_data in batch.mask_search_final_result_neg
                ],
                prompt=str(batch.prompt),
                mask_strategies=sparse_mask_candidates_searching,
                output_dir=f'output/mask_search_result_neg_{size[0]}x{size[1]}/'
            )

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
