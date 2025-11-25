import torch  # type: ignore
import gc

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video, pred_noise_to_x_bound
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

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

logger = init_logger(__name__)


class CausalDMDDenosingStage(DenoisingStage):
    """
    Denoising stage for causal diffusion.
    """

    def __init__(self, transformer, scheduler, transformer_2=None, vae=None) -> None:
        super().__init__(transformer, scheduler, transformer_2, vae)
        # KV and cross-attention cache state (initialized on first forward)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.vae = vae
        # self.kv_cache1: list | None = None
        # self.crossattn_cache: list | None = None
        # Model-dependent constants (aligned with causal_inference.py assumptions)
        self.num_transformer_blocks = len(self.transformer.blocks)
        self.num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        self.sliding_window_num_frames = self.transformer.config.arch_config.sliding_window_num_frames

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

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_ratio = self.transformer.config.arch_config.patch_size[
            -1] * self.transformer.config.arch_config.patch_size[-2]
        self.frame_seq_length = latent_seq_length // patch_ratio
        # TODO(will): make this a parameter once we add i2v support
        independent_first_frame = self.transformer.independent_first_frame if hasattr(
            self.transformer, 'independent_first_frame') else False
        # Timesteps for DMD
        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long).cpu()
        if fastvideo_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat((self.scheduler.timesteps.cpu(),
                                             torch.tensor([0],
                                                          dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())
        logger.info("timesteps: %s", timesteps)

        # Image kwargs (kept empty unless caller provides compatible args)
        image_kwargs: dict = {}

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
        kv_cache1 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                      dtype=target_dtype,
                                      device=latents.device)
        crossattn_cache = self._initialize_crossattn_cache(
                batch_size=latents.shape[0],
                max_text_len=fastvideo_args.pipeline_config.
                text_encoder_configs[0].arch_config.text_len,
                dtype=target_dtype,
                device=latents.device)
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(batch_size=latents.shape[0],
        #                               dtype=target_dtype,
        #                               device=latents.device)
        #     self._initialize_crossattn_cache(
        #         batch_size=latents.shape[0],
        #         max_text_len=fastvideo_args.pipeline_config.
        #         text_encoder_configs[0].arch_config.text_len,
        #         dtype=target_dtype,
        #         device=latents.device)
        # else:
        #     assert self.crossattn_cache is not None
        #     # reset cross-attention cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index][
        #             "is_init"] = False  # type: ignore
        #     # reset kv cache pointers
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index][
        #             "global_end_index"] = torch.tensor(  # type: ignore
        #                 [0],
        #                 dtype=torch.long,
        #                 device=latents.device)
        #         self.kv_cache1[block_index][
        #             "local_end_index"] = torch.tensor(  # type: ignore
        #                 [0],
        #                 dtype=torch.long,
        #                 device=latents.device)

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
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
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
                block = min(self.num_frames_per_block, remaining_frames)
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
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
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
        block_sizes = [self.num_frames_per_block] * 7
        start_index = 0

        if fastvideo_args.override_boundary_ratio is not None:
            boundary_timestep = fastvideo_args.override_boundary_ratio * self.scheduler.num_train_timesteps
            logger.info("Overriding boundary timestep to %s", boundary_timestep)
        else:
            if fastvideo_args.pipeline_config.dit_config.boundary_ratio is not None:
                boundary_timestep = fastvideo_args.pipeline_config.dit_config.boundary_ratio * self.scheduler.num_train_timesteps
            else:
                boundary_timestep = None

        high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        assert len(high_noise_timesteps) == len(timesteps) // 2, f"There are incorrect number of {len(high_noise_timesteps)} high noise timesteps"

        first_frame_latent = None
        if batch.pil_image is not None:
            # Causal video gen directly replaces the first frame of the latent with
            # the image latent instead of appending along the channel dim
            assert self.vae is not None, "VAE is not provided for causal video gen task"
            self.vae = self.vae.to(get_local_torch_device())
            first_frame_latent = self.vae.encode(batch.pil_image).mean.float()
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    first_frame_latent -= self.vae.shift_factor.to(first_frame_latent.device, first_frame_latent.dtype)
                else:
                    first_frame_latent -= self.vae.shift_factor

            if isinstance(self.vae.scaling_factor, torch.Tensor):
                first_frame_latent = first_frame_latent * self.vae.scaling_factor.to(first_frame_latent.device, first_frame_latent.dtype)
            else:
                first_frame_latent = first_frame_latent * self.vae.scaling_factor

            # latents[:, :, :3, :, :] = first_frame_latent.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            # start_index += 3
            # block_sizes.pop(0)
            
            if fastvideo_args.vae_cpu_offload:
                self.vae = self.vae.to("cpu")
            else:
                raise ValueError("vae_cpu_offload must be True")

            # if boundary_timestep is not None:
                # first_frame_latent_bound = self.scheduler.add_noise(
                #     first_frame_latent,
                #     torch.randn_like(first_frame_latent),
                #     boundary_timestep * torch.ones([1], device=first_frame_latent.device, dtype=torch.long))

        # DMD loop in causal blocks
        with self.progress_bar(total=len(block_sizes) *
                               len(timesteps)) as progress_bar:
            for block_idx, current_num_frames in enumerate(block_sizes):
                current_latents = latents[:, :, start_index:start_index +
                                          current_num_frames, :, :]

                if block_idx == 0 and first_frame_latent is not None:
                    # previous_frame = latents[:, :, start_index - 1:start_index, :, :].clone()
                    first_frame_mask = torch.ones_like(current_latents)
                    first_frame_mask[:, :, 0] = 0
                    current_latents = (1 - first_frame_mask) * first_frame_latent + first_frame_mask * current_latents
                    # if boundary_timestep is not None:
                    #     current_latents = first_frame_latent_bound.repeat(1, 1, current_num_frames, 1, 1)
                    # else:
                    #     current_latents = first_frame_latent.repeat(1, 1, current_num_frames, 1, 1)
                
                # use BTCHW for DMD conversion routines
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(timesteps):
                    if boundary_timestep is not None and t_cur < boundary_timestep:
                        current_model = self.transformer_2
                    else:
                        current_model = self.transformer
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
                                        enabled=autocast_enabled), \
                        set_forward_context(current_timestep=i,
                                            attn_metadata=attn_metadata,
                                            forward_batch=batch):
                        # Run transformer; follow DMD stage pattern
                        t_expanded_noise = t_cur * torch.ones(
                            (latent_model_input.shape[0], current_num_frames),
                            device=latent_model_input.device,
                            dtype=torch.long)

                        if block_idx == 0 and first_frame_latent is not None:
                            t_expanded_noise[:, 0] = 0

                        pred_noise_btchw = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        ).permute(0, 2, 1, 3, 4)

                    # Convert pred noise to pred video with FM Euler scheduler utilities
                    if boundary_timestep is not None and fastvideo_args.use_add_noise_high and t_cur >= boundary_timestep:
                        pred_video_btchw = pred_noise_to_x_bound(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expanded_noise.flatten(0, 1),
                            boundary_timestep=torch.ones_like(t_expanded_noise.flatten(0, 1)) * boundary_timestep,
                            scheduler=self.scheduler).unflatten(0, pred_noise_btchw.shape[:2])
                    else:
                        pred_video_btchw = pred_noise_to_pred_video(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expanded_noise.flatten(0, 1),
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
                        if boundary_timestep is not None and fastvideo_args.use_add_noise_high and i < len(high_noise_timesteps) - 1:
                            noise_latents_btchw = self.scheduler.add_noise_high(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1),
                                next_timestep,
                                torch.ones_like(next_timestep) * boundary_timestep).unflatten(0, pred_video_btchw.shape[:2])
                        elif boundary_timestep is not None and fastvideo_args.use_add_noise_high and i == len(high_noise_timesteps) - 1:
                            noise_latents_btchw = pred_video_btchw
                        else:
                            noise_latents_btchw = self.scheduler.add_noise(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1),
                                next_timestep).unflatten(0,
                                                        pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(
                            0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(
                            0, 2, 1, 3, 4)

                    if block_idx == 0 and first_frame_latent is not None:
                        noise_latents_btchw = (1 - first_frame_mask) * first_frame_latent + first_frame_mask * noise_latents_btchw.permute(0, 2, 1, 3, 4)
                        noise_latents_btchw = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                        current_latents = (1 - first_frame_mask) * first_frame_latent + first_frame_mask * current_latents

                    if progress_bar is not None:
                        progress_bar.update()

                # Write back and advance
                latents[:, :, start_index:start_index +
                        current_num_frames, :, :] = current_latents

                # Decode the latent
                # first_frame = block_idx == 0
                # pixels, self.decode_vae_cache = self.decoding_stage.decode(current_latents, fastvideo_args, *self.decode_vae_cache, first_frame=first_frame)
                # if batch.output is None:
                #     batch.output = pixels.cpu().float()
                # else:
                #     batch.output = torch.cat([batch.output, pixels.cpu().float()], dim=2)

                # Re-run with context timestep to update KV cache using clean context
                context_noise = getattr(fastvideo_args.pipeline_config,
                                        "context_noise", 0)
                t_context = torch.ones([latents.shape[0]],
                                       device=latents.device,
                                       dtype=torch.long) * int(context_noise)
                context_bcthw = current_latents.to(target_dtype)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=attn_metadata,
                                        forward_batch=batch):
                    t_expanded_context = t_context.unsqueeze(1)

                    if boundary_timestep is not None:
                        _ = self.transformer(
                            context_bcthw,
                            prompt_embeds,
                            t_expanded_context,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )
                    
                    _ = current_model(
                        context_bcthw,
                        prompt_embeds,
                        t_expanded_context,
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )

                start_index += current_num_frames

        # remove_indices = torch.arange(3, end=latents.shape[2], step=self.num_frames_per_block)
        # mask = torch.ones(latents.shape[2], device=latents.device, dtype=torch.bool)
        # mask[remove_indices] = False
        # latents = latents[:, :, mask]
        # latents = latents[:, :, 2:, :, :]
        # logger.info("latents: %s", latents.shape)

        del kv_cache1
        del crossattn_cache
        gc.collect()
        torch.cuda.empty_cache()

        batch.latents = latents
        return batch

    def _initialize_kv_cache(self, batch_size, dtype, device) -> None:
        """
        Initialize a Per-GPU KV cache aligned with the Wan model assumptions.
        """
        kv_cache1 = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * self.sliding_window_num_frames
        # kv_cache_size = 1560 * 19

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        # self.kv_cache1 = kv_cache1
        return kv_cache1

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype,
                                    device) -> None:
        """
        Initialize a Per-GPU cross-attention cache aligned with the Wan model assumptions.
        """
        crossattn_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([
                    batch_size, max_text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, max_text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "is_init":
                False,
            })
        # self.crossattn_cache = crossattn_cache
        return crossattn_cache

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
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
