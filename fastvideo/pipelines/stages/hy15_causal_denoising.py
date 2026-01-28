import math
import torch  # type: ignore

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.causal_denoising import CausalDMDDenosingStage
from fastvideo.utils import PRECISION_TO_TYPE

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


class Hy15CausalDMDDenosingStage(CausalDMDDenosingStage):
    """
    Denoising stage for causal diffusion.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        if isinstance(self.transformer.config.patch_size, tuple):
            patch_ratio = self.transformer.config.patch_size[
                1] * self.transformer.config.patch_size[2]
        elif isinstance(self.transformer.config.patch_size, int):
            patch_ratio = self.transformer.config.patch_size**2
        else:
            raise ValueError(
                f"Unsupported patch size type: {type(self.transformer.config.patch_size)}"
            )

        self.frame_seq_length = latent_seq_length // patch_ratio
        # TODO(will): make this a parameter once we add i2v support
        independent_first_frame = self.transformer.independent_first_frame if hasattr(
            self.transformer, 'independent_first_frame') else False
        # Timesteps for DMD
        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long).cpu()
        self.scheduler.set_timesteps(num_inference_steps=1000,
                                     extra_one_step=True,
                                     device=get_local_torch_device())
        if fastvideo_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat((self.scheduler.timesteps.cpu(),
                                             torch.tensor([0],
                                                          dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())
        logger.info("[causal_denoising] timesteps: %s", timesteps)

        # Image kwargs (kept empty unless caller provides compatible args)
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert not torch.isnan(
                image_embeds[0]).any(), "image_embeds contains nan"
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

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

        pos_start_base = 0
        num_blocks = math.ceil(t / self.num_frames_per_block)
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0

        # Initialize txt kv cache
        with torch.autocast(device_type="cuda",
                                        dtype=target_dtype,
                                        enabled=autocast_enabled), \
            set_forward_context(current_timestep=0,
                                attn_metadata=None,
                                forward_batch=batch):
            txt_kv_cache = self.transformer(
                txt_inference=True,
                vision_inference=False,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                encoder_attention_mask=batch.prompt_attention_mask,
                timestep=torch.zeros([latents.shape[0]], device=latents.device),
                cache_txt=True,
            )

        first_frame_latent = None
        if batch.pil_image is not None:
            # Causal video gen directly replaces the first frame of the latent with
            # the image latent instead of appending along the channel dim
            assert self.vae is not None, "VAE is not provided for causal video gen task"
            self.vae = self.vae.to(get_local_torch_device())
            vae_dtype = PRECISION_TO_TYPE[
                fastvideo_args.pipeline_config.vae_precision]
            first_frame_latent = self.vae.encode(
                batch.pil_image.to(vae_dtype)).mean.float()
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
            if batch.video_latent is not None:
                video_latent_chunk = batch.video_latent[:, :, start_index:
                                                        start_index + 1, :, :]
                first_frame_input = torch.cat([
                    first_frame_latent,
                    video_latent_chunk,
                    torch.zeros_like(first_frame_latent),
                ],
                                              dim=1)
            else:
                first_frame_input = first_frame_latent.clone()
            with torch.autocast(device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled), \
                set_forward_context(current_timestep=0,
                                    attn_metadata=None,
                                    forward_batch=batch):
                self.transformer(
                    txt_inference=False,
                    vision_inference=True,
                    hidden_states=first_frame_input.to(target_dtype),
                    timestep=t_zero,
                    kv_cache=kv_cache1,
                    txt_kv_cache=txt_kv_cache,
                    current_start=(pos_start_base + start_index) *
                    self.frame_seq_length,
                    rope_start_idx=start_index,
                )

            start_index += 1
            block_sizes.pop(0)
            latents[:, :, :1, :, :] = first_frame_latent

        vision_input_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "vision_inference": True,
                "txt_inference": False,
            },
        )

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

                    if batch.video_latent is not None:
                        video_latent_chunk = batch.video_latent[:, :,
                                                                start_index:
                                                                start_index +
                                                                current_num_frames, :, :]
                        latent_model_input = torch.cat([
                            latent_model_input,
                            video_latent_chunk,
                            torch.zeros_like(current_latents),
                        ],
                                                       dim=1)
                    elif batch.image_latent is not None and independent_first_frame and start_index == 0:
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
                        pred_noise_btchw, kv_cache1 = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expanded_noise,
                            kv_cache=kv_cache1,
                            txt_kv_cache=txt_kv_cache,
                            current_start=(pos_start_base + start_index) *
                            self.frame_seq_length,
                            rope_start_idx=start_index,
                            **vision_input_kwargs,
                        )
                        pred_noise_btchw = pred_noise_btchw.permute(
                            0, 2, 1, 3, 4)

                    # Convert pred noise to pred video with FM Euler scheduler utilities
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

                    if progress_bar is not None:
                        progress_bar.update()

                # Write back and advance
                latents[:, :, start_index:start_index +
                        current_num_frames, :, :] = current_latents

                # Re-run with context timestep to update KV cache using clean context
                context_noise = getattr(fastvideo_args.pipeline_config,
                                        "context_noise", 0)
                t_context = torch.ones([latents.shape[0], current_num_frames],
                                       device=latents.device,
                                       dtype=torch.long) * int(context_noise)
                context_bcthw = current_latents.to(target_dtype)
                if batch.video_latent is not None:
                    video_latent_chunk = batch.video_latent[:, :, start_index:
                                                            start_index +
                                                            current_num_frames, :, :]
                    context_bcthw = torch.cat([
                        context_bcthw,
                        video_latent_chunk,
                        torch.zeros_like(current_latents),
                    ],
                                              dim=1)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=attn_metadata,
                                        forward_batch=batch):

                    _, kv_cache1 = self.transformer(
                        hidden_states=context_bcthw,
                        timestep=t_context,
                        kv_cache=kv_cache1,
                        txt_kv_cache=txt_kv_cache,
                        current_start=(pos_start_base + start_index) *
                        self.frame_seq_length,
                        rope_start_idx=start_index,
                        **vision_input_kwargs,
                    )
                start_index += current_num_frames

        batch.latents = latents
        return batch
