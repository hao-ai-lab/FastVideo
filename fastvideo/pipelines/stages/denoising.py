# SPDX-License-Identifier: Apache-2.0
"""
Denoising stage for diffusion pipelines.
"""

import inspect
import math
import weakref
from collections.abc import Iterable
from typing import Any

import torch
from einops import rearrange
from tqdm.auto import tqdm

from fastvideo.attention import get_attn_backend
from fastvideo.configs.pipelines.base import STA_Mode
from fastvideo.distributed import (get_local_torch_device, get_sp_parallel_rank,
                                   get_sp_world_size, get_world_group)
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import dict_to_3d_list, masks_like

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


class DenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in diffusion pipelines.
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """

    def __init__(self,
                 transformer,
                 scheduler,
                 pipeline=None,
                 transformer_2=None,
                 vae=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.scheduler = scheduler
        self.vae = vae
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
                                "b c (n t) h w -> b c n t h w",
                                n=sp_world_size).contiguous()
            latents = latents[:, :, rank_in_sp_group, :, :, :]
            batch.latents = latents
            if batch.image_latent is not None:
                image_latent = rearrange(batch.image_latent,
                                         "b c (n t) h w -> b c n t h w",
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
            assert not torch.isnan(
                image_embeds[0]).any(), "image_embeds contains nan"
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
        assert not torch.isnan(
            prompt_embeds[0]).any(), "prompt_embeds contains nan"
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
            assert neg_prompt_embeds is not None
            assert not torch.isnan(
                neg_prompt_embeds[0]).any(), "neg_prompt_embeds contains nan"

        # (Wan2.2) Calculate timestep to switch from high noise expert to low noise expert
        boundary_ratio = fastvideo_args.pipeline_config.dit_config.boundary_ratio
        if batch.boundary_ratio is not None:
            logger.info("Overriding boundary ratio from %s to %s",
                        boundary_ratio, batch.boundary_ratio)
            boundary_ratio = batch.boundary_ratio

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
        else:
            boundary_timestep = None
        latent_model_input = latents.to(target_dtype)
        assert latent_model_input.shape[0] == 1, "only support batch size 1"

        if fastvideo_args.pipeline_config.ti2v_task and batch.pil_image is not None:
            # TI2V directly replaces the first frame of the latent with
            # the image latent instead of appending along the channel dim
            assert batch.image_latent is None, "TI2V task should not have image latents"
            assert self.vae is not None, "VAE is not provided for TI2V task"
            z = self.vae.encode(batch.pil_image).mean.float()
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    z -= self.vae.shift_factor.to(z.device, z.dtype)
                else:
                    z -= self.vae.shift_factor

            if isinstance(self.vae.scaling_factor, torch.Tensor):
                z = z * self.vae.scaling_factor.to(z.device, z.dtype)
            else:
                z = z * self.vae.scaling_factor

            latent_model_input = latent_model_input.squeeze(0)
            _, mask2 = masks_like([latent_model_input], zero=True)

            latent_model_input = (1. -
                                  mask2[0]) * z + mask2[0] * latent_model_input
            # latent_model_input = latent_model_input.unsqueeze(0)
            latent_model_input = latent_model_input.to(get_local_torch_device())
            latents = latent_model_input
            F = batch.num_frames
            temporal_scale = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
            spatial_scale = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
            patch_size = fastvideo_args.pipeline_config.dit_config.arch_config.patch_size
            seq_len = ((F - 1) // temporal_scale +
                       1) * (batch.height // spatial_scale) * (
                           batch.width // spatial_scale) // (patch_size[1] *
                                                             patch_size[2])
            seq_len = int(math.ceil(seq_len / sp_world_size)) * sp_world_size

        # Initialize lists for ODE trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []

        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                if boundary_timestep is None or t >= boundary_timestep:
                    if (fastvideo_args.dit_cpu_offload
                            and self.transformer_2 is not None and next(
                                self.transformer_2.parameters()).device.type
                            == 'cuda'):
                        self.transformer_2.to('cpu')
                    current_model = self.transformer
                    current_guidance_scale = batch.guidance_scale
                else:
                    # low-noise stage in wan2.2
                    if fastvideo_args.dit_cpu_offload and next(
                            self.transformer.parameters(
                            )).device.type == 'cuda':
                        self.transformer.to('cpu')
                    current_model = self.transformer_2
                    current_guidance_scale = batch.guidance_scale_2
                assert current_model is not None, "current_model is None"

                # Expand latents for V2V/I2V
                latent_model_input = latents.to(target_dtype)
                if batch.video_latent is not None:
                    latent_model_input = torch.cat([
                        latent_model_input, batch.video_latent,
                        torch.zeros_like(latents)
                    ],
                                                   dim=1).to(target_dtype)
                elif batch.image_latent is not None:
                    assert not fastvideo_args.pipeline_config.ti2v_task, "image latents should not be provided for TI2V task"
                    latent_model_input = torch.cat(
                        [latent_model_input, batch.image_latent],
                        dim=1).to(target_dtype)

                assert not torch.isnan(
                    latent_model_input).any(), "latent_model_input contains nan"
                if fastvideo_args.pipeline_config.ti2v_task and batch.pil_image is not None:
                    timestep = torch.stack([t]).to(get_local_torch_device())
                    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                    ])
                    timestep = temp_ts.unsqueeze(0)
                    t_expand = timestep.repeat(latent_model_input.shape[0], 1)
                else:
                    t_expand = t.repeat(latent_model_input.shape[0])

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
                        noise_pred = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expand,
                            guidance=guidance_expand,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    if batch.do_classifier_free_guidance:
                        batch.is_cfg_negative = True
                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=attn_metadata,
                                forward_batch=batch,
                        ):
                            noise_pred_uncond = current_model(
                                latent_model_input,
                                neg_prompt_embeds,
                                t_expand,
                                guidance=guidance_expand,
                                **image_kwargs,
                                **neg_cond_kwargs,
                            )

                        noise_pred_text = noise_pred
                        noise_pred = noise_pred_uncond + current_guidance_scale * (
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
                    if fastvideo_args.pipeline_config.ti2v_task and batch.pil_image is not None:
                        latents = latents.squeeze(0)
                        latents = (1. - mask2[0]) * z + mask2[0] * latents
                        # latents = latents.unsqueeze(0)

                # save trajectory latents if needed
                if batch.return_trajectory_latents:
                    trajectory_timesteps.append(t)
                    trajectory_latents.append(latents)

                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                        and progress_bar is not None):
                    progress_bar.update()

        # Gather results if using sequence parallelism
        trajectory_tensor: torch.Tensor | None = None
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps,
                                                      dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        # Gather results if using sequence parallelism
        if sp_group:
            latents = sequence_model_parallel_all_gather(latents, dim=2)
            if batch.return_trajectory_latents:
                trajectory_tensor = trajectory_tensor.to(
                    get_local_torch_device())
                trajectory_tensor = sequence_model_parallel_all_gather(
                    trajectory_tensor, dim=3)

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

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
        from fastvideo.attention.backends.STA_configuration import configure_sta
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
            import fastvideo.envs as envs
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

        from fastvideo.attention.backends.STA_configuration import save_mask_search_results
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


class CosmosDenoisingStage(DenoisingStage):
    """
    Denoising stage for Cosmos models using FlowMatchEulerDiscreteScheduler.
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
        unconditioning_latents = conditioning_latents

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                current_sigma = self.scheduler.sigmas[i]
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t

                timestep = current_t.view(1, 1, 1, 1,
                                          1).expand(latents.size(0), -1,
                                                    latents.size(2), -1,
                                                    -1)  # [B, 1, T, 1, 1]

                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):

                    cond_latent = latents * c_in

                    if hasattr(
                            batch, 'cond_indicator'
                    ) and batch.cond_indicator is not None and conditioning_latents is not None:
                        cond_latent = batch.cond_indicator * conditioning_latents + (
                            1 - batch.cond_indicator) * cond_latent
                    else:
                        logger.warning(
                            "Step %s: Missing conditioning data - cond_indicator: %s, conditioning_latents: %s",
                            i, hasattr(batch, 'cond_indicator'),
                            conditioning_latents is not None)

                    cond_latent = cond_latent.to(target_dtype)

                    cond_timestep = timestep
                    if hasattr(batch, 'cond_indicator'
                               ) and batch.cond_indicator is not None:
                        sigma_conditioning = 0.0001
                        t_conditioning = sigma_conditioning / (
                            sigma_conditioning + 1)
                        cond_timestep = batch.cond_indicator * t_conditioning + (
                            1 - batch.cond_indicator) * timestep
                        cond_timestep = cond_timestep.to(target_dtype)

                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                    ):
                        # Use conditioning masks from CosmosLatentPreparationStage
                        condition_mask = batch.cond_mask.to(
                            target_dtype) if hasattr(batch,
                                                     'cond_mask') else None
                        padding_mask = torch.zeros(1,
                                                   1,
                                                   batch.height,
                                                   batch.width,
                                                   device=cond_latent.device,
                                                   dtype=target_dtype)

                        # Fallback if masks not available
                        if condition_mask is None:
                            batch_size, num_channels, num_frames, height, width = cond_latent.shape
                            condition_mask = torch.zeros(
                                batch_size,
                                1,
                                num_frames,
                                height,
                                width,
                                device=cond_latent.device,
                                dtype=target_dtype)

                        noise_pred = self.transformer(
                            hidden_states=cond_latent,
                            timestep=cond_timestep.to(target_dtype),
                            encoder_hidden_states=batch.prompt_embeds[0].to(
                                target_dtype),
                            fps=24,  # TODO: get fps from batch or config
                            condition_mask=condition_mask,
                            padding_mask=padding_mask,
                            return_dict=False,
                        )[0]

                    cond_pred = (c_skip * latents +
                                 c_out * noise_pred.float()).to(target_dtype)

                    if hasattr(
                            batch, 'cond_indicator'
                    ) and batch.cond_indicator is not None and conditioning_latents is not None:
                        cond_pred = batch.cond_indicator * conditioning_latents + (
                            1 - batch.cond_indicator) * cond_pred

                    if batch.do_classifier_free_guidance and batch.negative_prompt_embeds is not None:
                        uncond_latent = latents * c_in

                        if hasattr(
                                batch, 'uncond_indicator'
                        ) and batch.uncond_indicator is not None and unconditioning_latents is not None:
                            uncond_latent = batch.uncond_indicator * unconditioning_latents + (
                                1 - batch.uncond_indicator) * uncond_latent

                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                        ):
                            uncond_condition_mask = batch.uncond_mask.to(
                                target_dtype
                            ) if hasattr(
                                batch, 'uncond_mask'
                            ) and batch.uncond_mask is not None else condition_mask

                            uncond_timestep = timestep
                            if hasattr(batch, 'uncond_indicator'
                                       ) and batch.uncond_indicator is not None:
                                sigma_conditioning = 0.0001
                                t_conditioning = sigma_conditioning / (
                                    sigma_conditioning + 1)
                                uncond_timestep = batch.uncond_indicator * t_conditioning + (
                                    1 - batch.uncond_indicator) * timestep
                                uncond_timestep = uncond_timestep.to(
                                    target_dtype)

                            noise_pred_uncond = self.transformer(
                                hidden_states=uncond_latent.to(target_dtype),
                                timestep=uncond_timestep.to(target_dtype),
                                encoder_hidden_states=batch.
                                negative_prompt_embeds[0].to(target_dtype),
                                fps=24,  # TODO: get fps from batch or config
                                condition_mask=uncond_condition_mask,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]

                        uncond_pred = (
                            c_skip * latents +
                            c_out * noise_pred_uncond.float()).to(target_dtype)

                        if hasattr(
                                batch, 'uncond_indicator'
                        ) and batch.uncond_indicator is not None and unconditioning_latents is not None:
                            uncond_pred = batch.uncond_indicator * unconditioning_latents + (
                                1 - batch.uncond_indicator) * uncond_pred

                        guidance_diff = cond_pred - uncond_pred
                        final_pred = cond_pred + guidance_scale * guidance_diff
                    else:
                        final_pred = cond_pred

                # Convert to noise for scheduler step
                if current_sigma > 1e-8:
                    noise_for_scheduler = (latents - final_pred) / current_sigma
                else:
                    logger.warning(
                        "Step %s: current_sigma too small (%s), using final_pred directly",
                        i, current_sigma)
                    noise_for_scheduler = final_pred

                if torch.isnan(noise_for_scheduler).sum() > 0:
                    logger.error(
                        "Step %s: NaN detected in noise_for_scheduler, sum: %s",
                        i,
                        noise_for_scheduler.float().sum().item())
                    logger.error(
                        "Step %s: latents sum: %s, final_pred sum: %s, current_sigma: %s",
                        i,
                        latents.float().sum().item(),
                        final_pred.float().sum().item(), current_sigma)

                latents = self.scheduler.step(noise_for_scheduler,
                                              t,
                                              latents,
                                              **extra_step_kwargs,
                                              return_dict=False)[0]

                progress_bar.update()

        batch.latents = latents

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify Cosmos denoising stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale,
                         V.positive_float)
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, lambda x:
            not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify Cosmos denoising stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        return result


class DmdDenoisingStage(DenoisingStage):
    """
    Denoising stage for DMD.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)

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
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

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

        # Prepare STA parameters
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        # Get latents and embeddings
        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents
        latents = latents.permute(0, 2, 1, 3, 4)

        video_raw_latent_shape = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert not torch.isnan(
            prompt_embeds[0]).any(), "prompt_embeds contains nan"
        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device())

        # Handle sequence parallelism if enabled
        sp_world_size, rank_in_sp_group = get_sp_world_size(
        ), get_sp_parallel_rank()
        sp_group = sp_world_size > 1
        if sp_group:
            latents = rearrange(latents,
                                "b (n t) c h w -> b n t c h w",
                                n=sp_world_size).contiguous()
            latents = latents[:, rank_in_sp_group, :, :, :, :]
            if batch.image_latent is not None:
                image_latent = rearrange(batch.image_latent,
                                         "b c (n t) h w -> b c n t h w",
                                         n=sp_world_size).contiguous()

                image_latent = image_latent[:, :, rank_in_sp_group, :, :, :]
                batch.image_latent = image_latent

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
                    latent_model_input = torch.cat([
                        latent_model_input,
                        batch.image_latent.permute(0, 2, 1, 3, 4)
                    ],
                                                   dim=2).to(target_dtype)
                assert not torch.isnan(
                    latent_model_input).any(), "latent_model_input contains nan"

                # Prepare inputs for transformer
                t_expand = t.repeat(latent_model_input.shape[0])
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
                    if (vsa_available and self.attn_backend
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

                    pred_video = pred_noise_to_pred_video(
                        pred_noise=pred_noise.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler).unflatten(
                            0, pred_noise.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1], dtype=torch.long, device=pred_video.device)
                        noise = torch.randn(video_raw_latent_shape,
                                            dtype=pred_video.dtype,
                                            generator=batch.generator[0]).to(
                                                self.device)
                        if sp_group:
                            noise = rearrange(noise,
                                              "b (n t) c h w -> b n t c h w",
                                              n=sp_world_size).contiguous()
                            noise = noise[:, rank_in_sp_group, :, :, :, :]
                        latents = self.scheduler.add_noise(
                            pred_video.flatten(0, 1), noise.flatten(0, 1),
                            next_timestep).unflatten(0, pred_video.shape[:2])
                    else:
                        latents = pred_video

                    # Update progress bar
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and
                        (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None):
                        progress_bar.update()

        # Gather results if using sequence parallelism
        if sp_group:
            latents = sequence_model_parallel_all_gather(latents, dim=1)
        latents = latents.permute(0, 2, 1, 3, 4)
        # Update batch with final latents
        batch.latents = latents

        return batch
