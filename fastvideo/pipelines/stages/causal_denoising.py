from __future__ import annotations
from typing import Any

import torch  # type: ignore

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

    def __init__(self,
                 transformer,
                 scheduler,
                 transformer_2=None,
                 vae=None) -> None:
        super().__init__(transformer, scheduler, transformer_2)
        # KV and cross-attention cache state (initialized on first forward)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.vae = vae
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

        if fastvideo_args.pipeline_config.dit_config.boundary_ratio is not None:
            boundary_timestep = fastvideo_args.pipeline_config.dit_config.boundary_ratio * self.scheduler.num_train_timesteps
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

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
        kv_cache2 = None
        if boundary_timestep is not None:
            # Initialize the low noise kv cache
            kv_cache2 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                                  dtype=target_dtype,
                                                  device=latents.device)

        def _get_kv_cache(timestep: float) -> list[dict]:
            if boundary_timestep is not None:
                if timestep >= boundary_timestep:
                    return kv_cache1
                else:
                    assert kv_cache2 is not None, "kv_cache2 is not initialized"
                    return kv_cache2
            return kv_cache1

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=fastvideo_args.pipeline_config.text_encoder_configs[0].
            arch_config.text_len,
            dtype=target_dtype,
            device=latents.device)

        pos_start_base = 0

        # Determine block sizes
        if t % self.num_frames_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frames_per_block for causal DMD denoising"
            )
        num_blocks = t // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0

        # For now hardcode the first block to be 1 frame assuming the model is Wan2.2-MoE
        if boundary_timestep is not None:
            block_sizes[0] = 1

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
                    first_frame_latent -= self.vae.shift_factor.to(
                        first_frame_latent.device, first_frame_latent.dtype)
                else:
                    first_frame_latent -= self.vae.shift_factor

            if isinstance(self.vae.scaling_factor, torch.Tensor):
                first_frame_latent = first_frame_latent * self.vae.scaling_factor.to(
                    first_frame_latent.device, first_frame_latent.dtype)
            else:
                first_frame_latent = first_frame_latent * self.vae.scaling_factor

            if fastvideo_args.vae_cpu_offload:
                self.vae = self.vae.to("cpu")

            # Fill the low noise and high noise kv cache with first_frame_latent and timestep 0
            t_zero = torch.zeros([latents.shape[0], 1],
                                 device=latents.device,
                                 dtype=torch.long)
            with torch.autocast(device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled), \
                set_forward_context(current_timestep=0,
                                    attn_metadata=None,
                                    forward_batch=batch):
                self.transformer(
                    first_frame_latent.to(target_dtype),
                    prompt_embeds,
                    t_zero,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    self.frame_seq_length,
                    start_frame=start_index,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )
                if boundary_timestep is not None:
                    self.transformer_2(
                        first_frame_latent.to(target_dtype),
                        prompt_embeds,
                        t_zero,
                        kv_cache=kv_cache2,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )

            start_index += 1
            block_sizes.pop(0)
            latents[:, :, :1, :, :] = first_frame_latent

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
                    if boundary_timestep is not None and t_cur < boundary_timestep:
                        current_model = self.transformer_2
                    else:
                        current_model = self.transformer
                    # Copy for pred conversion
                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)

                    # Prepare inputs
                    t_expand = t_cur.repeat(latent_model_input.shape[0])

                    # Attention metadata if needed
                    if (
                        vsa_available
                        and self.attn_backend == VideoSparseAttentionBackend
                    ):
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls()
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls()
                            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                                current_num_frames=current_num_frames,
                                video_raw_latent_shape=video_raw_latent_shape,
                                frame_seq_length=self.frame_seq_length,
                                batch_size=batch.batch_size,
                                device=latent_model_input.device,
                            )
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
                            (latent_model_input.shape[0], 1),
                            device=latent_model_input.device,
                            dtype=torch.long)
                        pred_noise_btchw = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            kv_cache=_get_kv_cache(t_cur),
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        ).permute(0, 2, 1, 3, 4)

                    # Convert pred noise to pred video with FM Euler scheduler utilities
                    if boundary_timestep is not None and t_cur >= boundary_timestep:
                        pred_video_btchw = pred_noise_to_x_bound(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expand,
                            boundary_timestep=torch.ones_like(t_expand) *
                            boundary_timestep,
                            scheduler=self.scheduler).unflatten(
                                0, pred_noise_btchw.shape[:2])
                    else:
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
                        if boundary_timestep is not None and i < len(
                                high_noise_timesteps) - 1:
                            noise_latents_btchw = self.scheduler.add_noise_high(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1), next_timestep,
                                torch.ones_like(next_timestep) *
                                boundary_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        elif boundary_timestep is not None and i == len(
                                high_noise_timesteps) - 1:
                            noise_latents_btchw = pred_video_btchw
                        else:
                            noise_latents_btchw = self.scheduler.add_noise(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1),
                                next_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(
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
                context_bcthw = current_latents.to(target_dtype)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=attn_metadata,
                                        forward_batch=batch):
                    t_expanded_context = t_context.unsqueeze(1)

                    if boundary_timestep is not None:
                        self.transformer_2(
                            context_bcthw,
                            prompt_embeds,
                            t_expanded_context,
                            kv_cache=kv_cache2,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    self.transformer(
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

        if boundary_timestep is not None:
            num_frames_to_remove = self.num_frames_per_block - 1
            latents = latents[:, :, :-num_frames_to_remove, :, :]

        batch.latents = latents
        return batch

    def _initialize_kv_cache(self, batch_size, dtype, device) -> list[dict]:
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

        return kv_cache1

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype,
                                    device) -> list[dict]:
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


class MatrixGameCausalDenoisingStage(DenoisingStage):

    def __init__(self,
                 transformer,
                 scheduler,
                 pipeline=None,
                 transformer_2=None,
                 vae=None) -> None:
        super().__init__(transformer, scheduler, pipeline, transformer_2, vae)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.vae = vae
        self.num_transformer_blocks = len(self.transformer.blocks)

        if hasattr(self.transformer, 'config') and hasattr(
                self.transformer.config, 'arch_config'):
            self.num_frame_per_block = getattr(
                self.transformer.config.arch_config, 'num_frames_per_block',
                getattr(self.transformer, 'num_frame_per_block', 1))
            self.sliding_window_num_frames = getattr(
                self.transformer.config.arch_config,
                'sliding_window_num_frames', 15)
        else:
            self.num_frame_per_block = getattr(self.transformer,
                                               'num_frame_per_block', 1)
            self.sliding_window_num_frames = 15

        try:
            self.local_attn_size = getattr(self.transformer, "local_attn_size",
                                           -1)
        except Exception:
            self.local_attn_size = -1

        assert self.local_attn_size != -1, (
            f"local_attn_size must be set for Matrix-Game causal inference, "
            f"got {self.local_attn_size}. Check MatrixGameWanVideoArchConfig.")
        assert self.num_frame_per_block > 0, (
            f"num_frame_per_block must be positive, got {self.num_frame_per_block}"
        )

        logger.info(
            "MatrixGame causal inference initialized: "
            "local_attn_size=%s, num_frame_per_block=%s", self.local_attn_size,
            self.num_frame_per_block)

        self.action_config = getattr(self.transformer, 'action_config', {})
        self.use_action_module = len(self.action_config) > 0

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_size = self.transformer.patch_size
        patch_ratio = patch_size[-1] * patch_size[-2]
        self.frame_seq_length = latent_seq_length // patch_ratio

        independent_first_frame = getattr(self.transformer,
                                          'independent_first_frame', False)
        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long).cpu()
        if fastvideo_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat((self.scheduler.timesteps.cpu(),
                                             torch.tensor([0],
                                                          dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())

        boundary_ratio = getattr(fastvideo_args.pipeline_config.dit_config,
                                 'boundary_ratio', None)
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        # directly set the kwarg.
        image_kwargs = {"encoder_hidden_states_image": image_embeds}
        pos_cond_kwargs: dict[str, Any] = {}

        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents
        b, c, t, h, w = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        kv_cache1 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                              dtype=target_dtype,
                                              device=latents.device)
        kv_cache2 = None
        if boundary_timestep is not None:
            kv_cache2 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                                  dtype=target_dtype,
                                                  device=latents.device)

        kv_cache_mouse = None
        kv_cache_keyboard = None
        if self.use_action_module:
            kv_cache_mouse, kv_cache_keyboard = self._initialize_action_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device)

        def _get_kv_cache(timestep_val: float) -> list[dict]:
            if boundary_timestep is not None:
                if timestep_val >= boundary_timestep:
                    return kv_cache1
                else:
                    assert kv_cache2 is not None, "kv_cache2 is not initialized"
                    return kv_cache2
            return kv_cache1

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=257,  # 1 CLS + 256 patch tokens
            dtype=target_dtype,
            device=latents.device)

        pos_start_base = 0

        if t % self.num_frame_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frame_per_block for causal denoising"
            )
        num_blocks = t // self.num_frame_per_block
        block_sizes = [self.num_frame_per_block] * num_blocks
        start_index = 0

        if boundary_timestep is not None:
            block_sizes[0] = 1

        # NOTE: MatrixGame does NOT process the first frame separately.
        # The first frame information is already encoded in batch.image_latent (cond_concat)
        # and will be used by the model via channel concatenation: torch.cat([x, cond_concat], dim=1)

        with self.progress_bar(total=len(block_sizes) *
                               len(timesteps)) as progress_bar:
            for block_idx, current_num_frames in enumerate(block_sizes):
                current_latents = latents[:, :, start_index:start_index +
                                          current_num_frames, :, :]
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                _video_raw_latent_shape = noise_latents_btchw.shape  # noqa: F841

                # NOTE: crossattn_cache should NOT be reset between blocks!

                action_kwargs = self._prepare_action_kwargs(
                    batch, start_index, current_num_frames)

                for i, t_cur in enumerate(timesteps):
                    if boundary_timestep is not None and t_cur < boundary_timestep:
                        current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
                    else:
                        current_model = self.transformer

                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)

                    if batch.image_latent is not None and independent_first_frame and start_index == 0:
                        latent_model_input = torch.cat([
                            latent_model_input,
                            batch.image_latent.to(target_dtype)
                        ],
                                                       dim=2)

                    # t_expand needs to be [batch * frames] to match flattened pred_noise/noise_latents
                    t_expand = t_cur.repeat(latent_model_input.shape[0] *
                                            current_num_frames)

                    if vsa_available and self.attn_backend == VideoSparseAttentionBackend:
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            attn_metadata = self.attn_metadata_builder.build(
                                current_timestep=i,
                                raw_latent_shape=(current_num_frames, h, w),
                                patch_size=fastvideo_args.pipeline_config.
                                dit_config.patch_size,
                                STA_param=batch.STA_param,
                                VSA_sparsity=fastvideo_args.VSA_sparsity,
                                device=get_local_torch_device(),
                            )
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
                        # Expand timestep to per-frame format [batch, num_frames] for causal model
                        t_expanded_noise = t_cur * torch.ones(
                            (latent_model_input.shape[0], current_num_frames),
                            device=latent_model_input.device,
                            dtype=torch.long)

                        model_kwargs = {
                            "kv_cache": _get_kv_cache(t_cur),
                            "crossattn_cache": crossattn_cache,
                            "current_start": (pos_start_base + start_index) *
                            self.frame_seq_length,
                            "start_frame": start_index,
                        }

                        if self.use_action_module and current_model == self.transformer:
                            model_kwargs.update({
                                "kv_cache_mouse":
                                kv_cache_mouse,
                                "kv_cache_keyboard":
                                kv_cache_keyboard,
                            })
                            model_kwargs.update(action_kwargs)

                        pred_noise_btchw = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            **image_kwargs,
                            **pos_cond_kwargs,
                            **model_kwargs,
                        ).permute(0, 2, 1, 3, 4)

                    if boundary_timestep is not None and t_cur >= boundary_timestep:
                        pred_video_btchw = pred_noise_to_x_bound(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expand,
                            boundary_timestep=torch.ones_like(t_expand) *
                            boundary_timestep,
                            scheduler=self.scheduler).unflatten(
                                0, pred_noise_btchw.shape[:2])
                    else:
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
                        # Use global RNG like MatrixGame (torch.randn_like without generator)
                        noise_btchw = torch.randn_like(pred_video_btchw)
                        if boundary_timestep is not None and high_noise_timesteps is not None and i < len(
                                high_noise_timesteps) - 1:
                            noise_latents_btchw = self.scheduler.add_noise_high(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1), next_timestep,
                                torch.ones_like(next_timestep) *
                                boundary_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        elif boundary_timestep is not None and high_noise_timesteps is not None and i == len(
                                high_noise_timesteps) - 1:
                            noise_latents_btchw = pred_video_btchw
                        else:
                            noise_latents_btchw = self.scheduler.add_noise(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1),
                                next_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(
                            0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(
                            0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                latents[:, :, start_index:start_index +
                        current_num_frames, :, :] = current_latents

                context_noise = getattr(fastvideo_args.pipeline_config,
                                        "context_noise", 0)
                # Expand context timestep to per-frame format [batch, num_frames] for causal model
                t_context = torch.ones([latents.shape[0], current_num_frames],
                                       device=latents.device,
                                       dtype=torch.long) * int(context_noise)
                context_bcthw = current_latents.to(target_dtype)

                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=attn_metadata,
                                        forward_batch=batch):

                    context_model_kwargs = {
                        "kv_cache": kv_cache1,
                        "crossattn_cache": crossattn_cache,
                        "current_start":
                        (pos_start_base + start_index) * self.frame_seq_length,
                        "start_frame": start_index,
                    }

                    if self.use_action_module:
                        context_model_kwargs.update({
                            "kv_cache_mouse":
                            kv_cache_mouse,
                            "kv_cache_keyboard":
                            kv_cache_keyboard,
                        })
                        context_model_kwargs.update(action_kwargs)

                    if boundary_timestep is not None and self.transformer_2 is not None:
                        self.transformer_2(
                            context_bcthw,
                            prompt_embeds,
                            t_context,
                            kv_cache=kv_cache2,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    self.transformer(
                        context_bcthw,
                        prompt_embeds,
                        t_context,
                        **image_kwargs,
                        **pos_cond_kwargs,
                        **context_model_kwargs,
                    )

                start_index += current_num_frames

        if boundary_timestep is not None:
            num_frames_to_remove = self.num_frame_per_block - 1
            if num_frames_to_remove > 0:
                latents = latents[:, :, :-num_frames_to_remove, :, :]

        batch.latents = latents
        return batch

    def _prepare_action_kwargs(self, batch: ForwardBatch, start_index: int,
                               num_frames: int) -> dict[str, Any]:
        action_kwargs: dict[str, Any] = {}
        if not self.use_action_module:
            return action_kwargs

        vae_time_compression_ratio = 4
        end_frame_idx = 1 + vae_time_compression_ratio * (start_index +
                                                          num_frames - 1)
        if hasattr(batch, 'mouse_cond') and batch.mouse_cond is not None:
            action_kwargs['mouse_cond'] = batch.mouse_cond[:, :end_frame_idx]
        if hasattr(batch, 'keyboard_cond') and batch.keyboard_cond is not None:
            action_kwargs[
                'keyboard_cond'] = batch.keyboard_cond[:, :end_frame_idx]

        # CRITICAL: Pass num_frame_per_block to model - this should be num_frames (current block size)
        action_kwargs['num_frame_per_block'] = num_frames
        return action_kwargs

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype,
                             device: torch.device) -> list[dict]:
        kv_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = getattr(
            self.transformer, 'attention_head_dim',
            self.transformer.hidden_size // num_attention_heads)
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * self.sliding_window_num_frames

        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
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

        return kv_cache

    def _initialize_action_kv_cache(self, batch_size: int, dtype: torch.dtype,
                                    device: torch.device):
        kv_cache_mouse = []
        kv_cache_keyboard = []

        action_heads = self.action_config.get('heads_num', 16)
        mouse_head_dim = self.action_config.get('mouse_hidden_dim',
                                                1024) // action_heads
        keyboard_head_dim = self.action_config.get('keyboard_hidden_dim',
                                                   1024) // action_heads

        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15

        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, action_heads, keyboard_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, action_heads, keyboard_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })
            kv_cache_mouse.append({
                "k":
                torch.zeros([
                    batch_size * self.frame_seq_length, kv_cache_size,
                    action_heads, mouse_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size * self.frame_seq_length, kv_cache_size,
                    action_heads, mouse_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        return kv_cache_mouse, kv_cache_keyboard

    def _initialize_crossattn_cache(self, batch_size: int, max_text_len: int,
                                    dtype: torch.dtype,
                                    device: torch.device) -> list[dict]:
        crossattn_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = getattr(
            self.transformer, 'attention_head_dim',
            self.transformer.hidden_size // num_attention_heads)
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
        return crossattn_cache

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
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
