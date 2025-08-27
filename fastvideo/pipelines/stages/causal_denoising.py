import torch  # type: ignore

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage

try:
    from fastvideo.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False
    SlidingTileAttentionBackend = None  # type: ignore

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False
    VideoSparseAttentionBackend = None  # type: ignore

# class CausalDenoisingStage(LoRAPipeline, ComposedPipelineBase):
#     """
#     Flow Causal denoising stage.
#     """

#     def forward(
#         self,
#         batch: ForwardBatch,
#         fastvideo_args: FastVideoArgs,
#     ) -> ForwardBatch:
#         # Setup precision and autocast settings
#         target_dtype = torch.bfloat16
#         autocast_enabled = (target_dtype != torch.float32
#                             ) and not fastvideo_args.disable_autocast

#         # Gather args for causal block-wise processing
#         try:
#             num_frame_per_block = getattr(fastvideo_args.pipeline_config,
#                                           "num_frame_per_block", 1)
#         except Exception:
#             num_frame_per_block = 1
#         try:
#             independent_first_frame = getattr(fastvideo_args.pipeline_config,
#                                               "independent_first_frame", False)
#         except Exception:
#             independent_first_frame = False

#         # Required fields
#         timesteps = batch.timesteps
#         if timesteps is None:
#             raise ValueError("Timesteps must be provided")

#         # Prepare image latents and embeddings for I2V generation
#         image_embeds = batch.image_embeds
#         if len(image_embeds) > 0:
#             assert torch.isnan(image_embeds[0]).sum() == 0
#             image_embeds = [
#                 image_embed.to(target_dtype) for image_embed in image_embeds
#             ]

#         image_kwargs = self.prepare_extra_func_kwargs(
#             self.transformer.forward,
#             {
#                 "encoder_hidden_states_image": image_embeds,
#                 "mask_strategy": dict_to_3d_list(
#                     None, t_max=50, l_max=60, h_max=24)
#             },
#         )

#         pos_cond_kwargs = self.prepare_extra_func_kwargs(
#             self.transformer.forward,
#             {
#                 "encoder_hidden_states_2": batch.clip_embedding_pos,
#                 "encoder_attention_mask": batch.prompt_attention_mask,
#             },
#         )

#         neg_cond_kwargs = self.prepare_extra_func_kwargs(
#             self.transformer.forward,
#             {
#                 "encoder_hidden_states_2": batch.clip_embedding_neg,
#                 "encoder_attention_mask": batch.negative_attention_mask,
#             },
#         )

#         # Prepare STA parameters
#         if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
#             self.prepare_sta_param(batch, fastvideo_args)

#         # Get latents and embeddings
#         latents = batch.latents
#         assert latents is not None, "latents must be provided"
#         prompt_embeds = batch.prompt_embeds
#         assert torch.isnan(prompt_embeds[0]).sum() == 0

#         # Determine blocks
#         b, c, t, h, w = latents.shape
#         if not independent_first_frame or (independent_first_frame
#                                            and batch.image_latent is not None):
#             if t % num_frame_per_block != 0:
#                 raise ValueError(
#                     "num_frames must be divisible by num_frame_per_block for causal denoising"
#                 )
#             num_blocks = t // num_frame_per_block
#             block_sizes = [num_frame_per_block] * num_blocks
#             start_index = 0
#         else:
#             if (t - 1) % num_frame_per_block != 0:
#                 raise ValueError(
#                     "(num_frames - 1) must be divisible by num_frame_per_block when independent_first_frame=True"
#                 )
#             num_blocks = (t - 1) // num_frame_per_block
#             block_sizes = [1] + [num_frame_per_block] * num_blocks
#             start_index = 0

#         # Run causal block-wise denoising loop
#         with self.progress_bar(total=len(block_sizes) *
#                                len(timesteps)) as progress_bar:
#             for current_num_frames in block_sizes:
#                 # Build the latent input for this block
#                 current_latents = latents[:, :, start_index:start_index +
#                                           current_num_frames, :, :]

#                 # Run spatial denoising loop over timesteps
#                 for i, t_cur in enumerate(timesteps):
#                     # Expand latents for I2V
#                     latent_model_input = current_latents.to(target_dtype)
#                     if batch.image_latent is not None and independent_first_frame and start_index == 0:
#                         # Prepend the image latent only for the first block when required
#                         latent_model_input = torch.cat([
#                             batch.image_latent.to(target_dtype),
#                             latent_model_input
#                         ],
#                                                        dim=2)

#                     latent_model_input = self.scheduler.scale_model_input(
#                         latent_model_input, t_cur)

#                     # Prepare inputs for transformer
#                     t_expand = t_cur.repeat(latent_model_input.shape[0])
#                     guidance_expand = (torch.tensor(
#                         [fastvideo_args.pipeline_config.embedded_cfg_scale] *
#                         latent_model_input.shape[0],
#                         dtype=torch.float32,
#                         device=get_local_torch_device(),
#                     ).to(target_dtype) *
#                                        1000.0 if fastvideo_args.pipeline_config.
#                                        embedded_cfg_scale is not None else None)

#                     # Predict noise residual
#                     with torch.autocast(device_type="cuda",
#                                         dtype=target_dtype,
#                                         enabled=autocast_enabled):
#                         if (st_attn_available and self.attn_backend == SlidingTileAttentionBackend) or \
#                            (vsa_available and self.attn_backend == VideoSparseAttentionBackend):
#                             self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
#                             )
#                             if self.attn_metadata_builder_cls is not None:
#                                 self.attn_metadata_builder = self.attn_metadata_builder_cls(
#                                 )
#                                 attn_metadata = self.attn_metadata_builder.build(  # type: ignore
#                                     current_timestep=i,  # type: ignore
#                                     raw_latent_shape=(current_num_frames, h,
#                                                       w),  # type: ignore
#                                     patch_size=fastvideo_args.pipeline_config.
#                                     dit_config.patch_size,  # type: ignore
#                                     STA_param=batch.STA_param,  # type: ignore
#                                     VSA_sparsity=fastvideo_args.
#                                     VSA_sparsity,  # type: ignore
#                                     device=get_local_torch_device(),
#                                 )
#                                 assert attn_metadata is not None, "attn_metadata cannot be None"
#                             else:
#                                 attn_metadata = None
#                         else:
#                             attn_metadata = None

#                         batch.is_cfg_negative = False
#                         with set_forward_context(current_timestep=i,
#                                                  attn_metadata=attn_metadata,
#                                                  forward_batch=batch):
#                             noise_pred = self.transformer(
#                                 latent_model_input,
#                                 prompt_embeds,
#                                 t_expand,
#                                 guidance=guidance_expand,
#                                 **image_kwargs,
#                                 **pos_cond_kwargs,
#                             )

#                         if batch.do_classifier_free_guidance:
#                             batch.is_cfg_negative = True
#                             with set_forward_context(
#                                     current_timestep=i,
#                                     attn_metadata=attn_metadata,
#                                     forward_batch=batch):
#                                 noise_pred_uncond = self.transformer(
#                                     latent_model_input,
#                                     batch.negative_prompt_embeds,
#                                     t_expand,
#                                     guidance=guidance_expand,
#                                     **image_kwargs,
#                                     **neg_cond_kwargs,
#                                 )
#                             noise_pred_text = noise_pred
#                             noise_pred = noise_pred_uncond + batch.guidance_scale * (
#                                 noise_pred_text - noise_pred_uncond)

#                             if batch.guidance_rescale > 0.0:
#                                 noise_pred = self.rescale_noise_cfg(
#                                     noise_pred,
#                                     noise_pred_text,
#                                     guidance_rescale=batch.guidance_rescale,
#                                 )

#                         # Compute the previous noisy sample for this block only
#                         current_latents = self.scheduler.step(
#                             noise_pred,
#                             t_cur,
#                             current_latents,
#                             return_dict=False)[0]

#                     if progress_bar is not None:
#                         progress_bar.update()

#                 # After finishing timesteps, write denoised block back
#                 latents[:, :, start_index:start_index +
#                         current_num_frames, :, :] = current_latents

#                 # Advance to next block
#                 start_index += current_num_frames

#         # Update batch with final latents
#         batch.latents = latents
#         return batch


class CausalDMDDenosingStage(DenoisingStage):
    """
    Denoising stage for causal diffusion.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
        # KV and cross-attention cache state (initialized on first forward)
        self.kv_cache1 = None
        self.crossattn_cache = None
        # Model-dependent constants (aligned with causal_inference.py assumptions)
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        try:
            self.local_attn_size = getattr(self.transformer.model,
                                           "local_attn_size",
                                           -1)  # type: ignore
        except Exception:
            self.local_attn_size = -1

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        # Args
        try:
            num_frame_per_block = getattr(fastvideo_args.pipeline_config,
                                          "num_frame_per_block", 1)
        except Exception:
            num_frame_per_block = 1
        try:
            independent_first_frame = getattr(fastvideo_args.pipeline_config,
                                              "independent_first_frame", False)
        except Exception:
            independent_first_frame = False

        # Timesteps for DMD
        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device())

        # Image kwargs (kept empty unless caller provides compatible args)
        image_kwargs = {}

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                # "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        # STA
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        # Latents and prompts
        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents  # [B, C, T, H, W]
        b, c, t, h, w = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        # Initialize or reset caches
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=latents.shape[0],
                                      dtype=latents.dtype,
                                      device=latents.device)
            self._initialize_crossattn_cache(batch_size=latents.shape[0],
                                             dtype=latents.dtype,
                                             device=latents.device)
        else:
            # reset cross-attention cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index][
                    "is_init"] = False  # type: ignore
            # reset kv cache pointers
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index][
                    "global_end_index"] = torch.tensor(  # type: ignore
                        [0],
                        dtype=torch.long,
                        device=latents.device)
                self.kv_cache1[block_index][
                    "local_end_index"] = torch.tensor(  # type: ignore
                        [0],
                        dtype=torch.long,
                        device=latents.device)

        # Optional: cache context features from provided image latents prior to generation
        current_start_frame = 0
        if getattr(batch, "image_latent", None) is not None:
            image_latent = batch.image_latent
            assert image_latent is not None
            input_frames = image_latent.shape[2]
            # timestep zero (or configured context noise) for cache warm-up
            t_zero = torch.zeros([latents.shape[0]],
                                 device=latents.device,
                                 dtype=torch.long)
            if independent_first_frame and input_frames >= 1:
                # warm-up with the very first frame independently
                image_first_btchw = image_latent[:, :, :1, :, :].to(
                    target_dtype).permute(0, 2, 1, 3, 4)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):
                    _ = self.transformer(
                        image_first_btchw,
                        prompt_embeds,
                        t_zero,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame *
                        self.frame_seq_length,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            # process remaining input frames in blocks of num_frame_per_block
            while remaining_frames > 0:
                block = min(num_frame_per_block, remaining_frames)
                ref_btchw = image_latent[:, :, current_start_frame:
                                         current_start_frame +
                                         block, :, :].to(target_dtype).permute(
                                             0, 2, 1, 3, 4)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):
                    _ = self.transformer(
                        ref_btchw,
                        prompt_embeds,
                        t_zero,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame *
                        self.frame_seq_length,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                current_start_frame += block
                remaining_frames -= block

        # Base position offset from any cache warm-up
        pos_start_base = current_start_frame

        # Determine block sizes
        if not independent_first_frame or (independent_first_frame
                                           and batch.image_latent is not None):
            if t % num_frame_per_block != 0:
                raise ValueError(
                    "num_frames must be divisible by num_frame_per_block for causal DMD denoising"
                )
            num_blocks = t // num_frame_per_block
            block_sizes = [num_frame_per_block] * num_blocks
            start_index = 0
        else:
            if (t - 1) % num_frame_per_block != 0:
                raise ValueError(
                    "(num_frames - 1) must be divisible by num_frame_per_block when independent_first_frame=True"
                )
            num_blocks = (t - 1) // num_frame_per_block
            block_sizes = [1] + [num_frame_per_block] * num_blocks
            start_index = 0

        # DMD loop in causal blocks
        with self.progress_bar(total=len(block_sizes) *
                               len(timesteps)) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[:, :, start_index:start_index +
                                          current_num_frames, :, :]
                # use BTCHW for DMD conversion routines
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(timesteps):
                    # Copy for pred conversion
                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)

                    if batch.image_latent is not None and independent_first_frame and start_index == 0:
                        latent_model_input = torch.cat([
                            latent_model_input,
                            batch.image_latent.to(target_dtype)
                        ],
                                                       dim=2)

                    # Prepare inputs
                    t_expand = t_cur.repeat(latent_model_input.shape[0])

                    # Attention metadata if needed
                    if (vsa_available and self.attn_backend
                            == VideoSparseAttentionBackend):
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                                current_timestep=i,  # type: ignore
                                raw_latent_shape=(current_num_frames, h,
                                                  w),  # type: ignore
                                patch_size=fastvideo_args.pipeline_config.
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

                    with torch.autocast(device_type="cuda",
                                        dtype=target_dtype,
                                        enabled=autocast_enabled):
                        with set_forward_context(current_timestep=i,
                                                 attn_metadata=attn_metadata,
                                                 forward_batch=batch):
                            # Run transformer; follow DMD stage pattern
                            pred_noise_btchw = self.transformer(
                                latent_model_input.permute(0, 2, 1, 3, 4),
                                prompt_embeds,
                                t_expand,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=(pos_start_base + start_index) *
                                self.frame_seq_length,
                                **image_kwargs,
                                **pos_cond_kwargs,
                            ).permute(0, 2, 1, 3, 4)

                    # Convert pred noise to pred video with FM Euler scheduler utilities
                    from fastvideo.training.training_utils import (
                        pred_noise_to_pred_video)
                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler).unflatten(
                            0, pred_noise_btchw.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1],
                            dtype=torch.long,
                            device=pred_video_btchw.device)
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=(batch.generator[0] if isinstance(
                                batch.generator, list) else
                                       batch.generator)).to(self.device)
                        noise_btchw = noise
                        current_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0, 1),
                            noise_btchw.flatten(0, 1),
                            next_timestep).unflatten(0,
                                                     pred_video_btchw.shape[:2])
                        current_latents = current_latents_btchw.permute(
                            0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(
                            0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                # Write back and advance
                latents[:, :, start_index:start_index +
                        current_num_frames, :, :] = current_latents

                # Re-run with context timestep to update KV cache using clean context
                context_noise = getattr(fastvideo_args.pipeline_config,
                                        "context_noise", 0)
                t_context = torch.ones([latents.shape[0]],
                                       device=latents.device,
                                       dtype=torch.long) * int(context_noise)
                context_btchw = current_latents.to(target_dtype).permute(
                    0, 2, 1, 3, 4)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):
                    _ = self.transformer(
                        context_btchw,
                        prompt_embeds,
                        t_context,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                start_index += current_num_frames

        batch.latents = latents
        return batch

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache aligned with the Wan model assumptions.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k":
                torch.zeros([batch_size, kv_cache_size, 12, 128],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, kv_cache_size, 12, 128],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache aligned with the Wan model assumptions.
        """
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([batch_size, 512, 12, 128],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, 512, 12, 128],
                            dtype=dtype,
                            device=device),
                "is_init":
                False,
            })
        self.crossattn_cache = crossattn_cache
