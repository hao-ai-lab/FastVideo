# SPDX-License-Identifier: Apache-2.0
"""
Causal block denoising strategy (DMD).
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.utils import pred_noise_to_pred_video, pred_noise_to_x_bound
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_strategies import (
    BlockContext,
    BlockDenoisingStrategy,
    BlockPlan,
    BlockPlanItem,
    ModelInputs,
    StrategyState,
)

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


class CausalBlockStrategy(BlockDenoisingStrategy):

    def __init__(self, stage: Any) -> None:
        self.stage = stage
        self.num_transformer_blocks = 0
        self.num_frames_per_block = 0
        self.sliding_window_num_frames = 0
        self.local_attn_size = -1
        self.frame_seq_length = 0

    def _ensure_model_constants(self) -> None:
        transformer = self.stage.transformer
        self.num_transformer_blocks = len(transformer.blocks)
        arch_config = transformer.config.arch_config
        self.num_frames_per_block = arch_config.num_frames_per_block
        self.sliding_window_num_frames = arch_config.sliding_window_num_frames
        try:
            self.local_attn_size = getattr(transformer.model, "local_attn_size",
                                           -1)
        except Exception:
            self.local_attn_size = -1

    def _initialize_kv_cache(self, batch_size, dtype, device) -> list[dict]:
        kv_cache1 = []
        num_attention_heads = self.stage.transformer.num_attention_heads
        attention_head_dim = self.stage.transformer.attention_head_dim
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
        crossattn_cache = []
        num_attention_heads = self.stage.transformer.num_attention_heads
        attention_head_dim = self.stage.transformer.attention_head_dim
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

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        self._ensure_model_constants()
        latents = batch.latents
        if latents is None:
            raise ValueError("latents must be provided")

        latent_seq_length = latents.shape[-1] * latents.shape[-2]
        patch_ratio = (
            self.stage.transformer.config.arch_config.patch_size[-1] *
            self.stage.transformer.config.arch_config.patch_size[-2])
        self.frame_seq_length = latent_seq_length // patch_ratio

        independent_first_frame = getattr(self.stage.transformer,
                                          "independent_first_frame", False)

        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long).cpu()
        if getattr(fastvideo_args.pipeline_config, "warp_denoising_step",
                   False):
            scheduler_timesteps = torch.cat((
                self.stage.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32),
            ))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())

        boundary_ratio = fastvideo_args.pipeline_config.dit_config.boundary_ratio
        if boundary_ratio is not None:
            boundary_timestep = (boundary_ratio *
                                 self.stage.scheduler.num_train_timesteps)
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

        image_kwargs: dict[str, Any] = {}
        pos_cond_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        if (st_attn_available
                and self.stage.attn_backend == SlidingTileAttentionBackend):
            self.stage.prepare_sta_param(batch, fastvideo_args)

        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        kv_cache1 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                              dtype=target_dtype,
                                              device=latents.device)
        kv_cache2 = None
        if boundary_timestep is not None:
            kv_cache2 = self._initialize_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device,
            )

        text_len = None
        if fastvideo_args.pipeline_config.text_encoder_configs:
            text_len = getattr(
                fastvideo_args.pipeline_config.text_encoder_configs[0].
                arch_config, "text_len", None)
        if not text_len:
            if batch.prompt_attention_mask:
                text_len = batch.prompt_attention_mask[0].shape[-1]
            elif batch.prompt_embeds:
                text_len = batch.prompt_embeds[0].shape[1]
            else:
                text_len = 0

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=text_len,
            dtype=target_dtype,
            device=latents.device,
        )

        num_frames = latents.shape[2]
        if num_frames % self.num_frames_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frames_per_block for "
                "causal DMD denoising")
        num_blocks = num_frames // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0

        if boundary_timestep is not None:
            block_sizes[0] = 1

        pos_start_base = 0
        if batch.pil_image is not None:
            assert self.stage.vae is not None, (
                "VAE is not provided for causal video gen task")
            self.stage.vae = self.stage.vae.to(get_local_torch_device())
            first_frame_latent = self.stage.vae.encode(
                batch.pil_image).mean.float()
            if (hasattr(self.stage.vae, "shift_factor")
                    and self.stage.vae.shift_factor is not None):
                if isinstance(self.stage.vae.shift_factor, torch.Tensor):
                    first_frame_latent -= self.stage.vae.shift_factor.to(
                        first_frame_latent.device, first_frame_latent.dtype)
                else:
                    first_frame_latent -= self.stage.vae.shift_factor

            if isinstance(self.stage.vae.scaling_factor, torch.Tensor):
                first_frame_latent = (
                    first_frame_latent * self.stage.vae.scaling_factor.to(
                        first_frame_latent.device, first_frame_latent.dtype))
            else:
                first_frame_latent = (first_frame_latent *
                                      self.stage.vae.scaling_factor)

            if fastvideo_args.vae_cpu_offload:
                self.stage.vae = self.stage.vae.to("cpu")

            t_zero = torch.zeros([latents.shape[0], 1],
                                 device=latents.device,
                                 dtype=torch.long)
            with torch.autocast(device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled), \
                set_forward_context(current_timestep=0,
                                    attn_metadata=None,
                                    forward_batch=batch):
                self.stage.transformer(
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
                    self.stage.transformer_2(
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

        progress_bar = self.stage.progress_bar(total=len(block_sizes) *
                                               len(timesteps))

        extra: dict[str, Any] = {
            "batch":
            batch,
            "fastvideo_args":
            fastvideo_args,
            "target_dtype":
            target_dtype,
            "autocast_enabled":
            autocast_enabled,
            "boundary_timestep":
            boundary_timestep,
            "high_noise_timesteps":
            high_noise_timesteps,
            "image_kwargs":
            image_kwargs,
            "pos_cond_kwargs":
            pos_cond_kwargs,
            "kv_cache1":
            kv_cache1,
            "kv_cache2":
            kv_cache2,
            "crossattn_cache":
            crossattn_cache,
            "block_sizes":
            block_sizes,
            "start_index":
            start_index,
            "progress_bar":
            progress_bar,
            "independent_first_frame":
            independent_first_frame,
            "context_noise":
            getattr(fastvideo_args.pipeline_config, "context_noise", 0),
            "pos_start_base":
            pos_start_base,
        }

        return StrategyState(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=len(timesteps),
            prompt_embeds=batch.prompt_embeds,
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=batch.prompt_attention_mask,
            negative_attention_mask=batch.negative_attention_mask,
            image_embeds=batch.image_embeds,
            guidance_scale=batch.guidance_scale,
            guidance_scale_2=batch.guidance_scale_2,
            guidance_rescale=batch.guidance_rescale,
            do_cfg=batch.do_classifier_free_guidance,
            extra=extra,
        )

    def block_plan(self, state: StrategyState) -> BlockPlan:
        block_sizes = state.extra["block_sizes"]
        start_index = state.extra["start_index"]
        items: list[BlockPlanItem] = []
        for block_size in block_sizes:
            items.append(
                BlockPlanItem(
                    start_index=start_index,
                    num_frames=block_size,
                    use_kv_cache=True,
                    model_selector="default",
                ))
            start_index += block_size
        return BlockPlan(items=items)

    def init_block_context(self, state: StrategyState,
                           block_item: BlockPlanItem,
                           block_idx: int) -> BlockContext:
        return BlockContext(
            kv_cache=state.extra["kv_cache1"],
            kv_cache_2=state.extra["kv_cache2"],
            crossattn_cache=state.extra["crossattn_cache"],
            action_cache=None,
            extra={
                "block_idx": block_idx,
                "start_index": block_item.start_index,
                "num_frames": block_item.num_frames,
            },
        )

    def process_block(self, state: StrategyState, block_ctx: BlockContext,
                      block_item: BlockPlanItem) -> None:
        batch = state.extra["batch"]
        fastvideo_args = state.extra["fastvideo_args"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        boundary_timestep = state.extra["boundary_timestep"]
        high_noise_timesteps = state.extra["high_noise_timesteps"]
        image_kwargs = state.extra["image_kwargs"]
        pos_cond_kwargs = state.extra["pos_cond_kwargs"]
        progress_bar = state.extra["progress_bar"]
        independent_first_frame = state.extra["independent_first_frame"]

        start_index = block_item.start_index
        current_num_frames = block_item.num_frames

        kv_cache1 = block_ctx.kv_cache
        kv_cache2 = block_ctx.kv_cache_2
        crossattn_cache = block_ctx.crossattn_cache

        def _get_kv_cache(timestep_val: float) -> list[dict]:
            if boundary_timestep is not None:
                if timestep_val >= boundary_timestep:
                    return kv_cache1
                if kv_cache2 is None:
                    raise ValueError("kv_cache2 is not initialized")
                return kv_cache2
            return kv_cache1

        current_latents = state.latents[:, :, start_index:start_index +
                                        current_num_frames, :, :]
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        video_raw_latent_shape = noise_latents_btchw.shape
        h, w = current_latents.shape[-2:]

        attn_metadata = None
        for i, t_cur in enumerate(state.timesteps):
            if boundary_timestep is not None and t_cur < boundary_timestep:
                current_model = self.stage.transformer_2
            else:
                current_model = self.stage.transformer
            noise_latents = noise_latents_btchw.clone()
            latent_model_input = current_latents.to(target_dtype)

            if (batch.image_latent is not None and independent_first_frame
                    and start_index == 0):
                latent_model_input = torch.cat(
                    [latent_model_input,
                     batch.image_latent.to(target_dtype)],
                    dim=2)

            t_expand = t_cur.repeat(latent_model_input.shape[0])

            if (vsa_available
                    and self.stage.attn_backend == VideoSparseAttentionBackend):
                self.attn_metadata_builder_cls = (
                    self.stage.attn_backend.get_builder_cls())
                if self.attn_metadata_builder_cls is not None:
                    self.attn_metadata_builder = (
                        self.attn_metadata_builder_cls())
                    attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                        current_timestep=i,  # type: ignore
                        raw_latent_shape=(current_num_frames, h,
                                          w),  # type: ignore
                        patch_size=fastvideo_args.pipeline_config.dit_config.
                        patch_size,  # type: ignore
                        STA_param=batch.STA_param,  # type: ignore
                        VSA_sparsity=fastvideo_args.
                        VSA_sparsity,  # type: ignore
                        device=get_local_torch_device(),  # type: ignore
                    )
                    assert attn_metadata is not None, (
                        "attn_metadata cannot be None")
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
                t_expanded_noise = t_cur * torch.ones(
                    (latent_model_input.shape[0], 1),
                    device=latent_model_input.device,
                    dtype=torch.long)
                pred_noise_btchw = current_model(
                    latent_model_input,
                    batch.prompt_embeds,
                    t_expanded_noise,
                    kv_cache=_get_kv_cache(t_cur),
                    crossattn_cache=crossattn_cache,
                    current_start=start_index * self.frame_seq_length,
                    start_frame=start_index,
                    **image_kwargs,
                    **pos_cond_kwargs,
                ).permute(0, 2, 1, 3, 4)

            if boundary_timestep is not None and t_cur >= boundary_timestep:
                pred_video_btchw = pred_noise_to_x_bound(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=t_expand,
                    boundary_timestep=torch.ones_like(t_expand) *
                    boundary_timestep,
                    scheduler=self.stage.scheduler,
                ).unflatten(0, pred_noise_btchw.shape[:2])
            else:
                pred_video_btchw = pred_noise_to_pred_video(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=t_expand,
                    scheduler=self.stage.scheduler,
                ).unflatten(0, pred_noise_btchw.shape[:2])

            if i < len(state.timesteps) - 1:
                next_timestep = state.timesteps[i + 1] * torch.ones(
                    [1],
                    dtype=torch.long,
                    device=pred_video_btchw.device,
                )
                noise = torch.randn(
                    video_raw_latent_shape,
                    dtype=pred_video_btchw.dtype,
                    generator=(batch.generator[0] if isinstance(
                        batch.generator, list) else batch.generator),
                ).to(self.stage.device)
                noise_btchw = noise
                if (boundary_timestep is not None
                        and high_noise_timesteps is not None
                        and i < len(high_noise_timesteps) - 1):
                    noise_latents_btchw = self.stage.scheduler.add_noise_high(
                        pred_video_btchw.flatten(0, 1),
                        noise_btchw.flatten(0, 1),
                        next_timestep,
                        torch.ones_like(next_timestep) * boundary_timestep,
                    ).unflatten(0, pred_video_btchw.shape[:2])
                elif (boundary_timestep is not None
                      and high_noise_timesteps is not None
                      and i == len(high_noise_timesteps) - 1):
                    noise_latents_btchw = pred_video_btchw
                else:
                    noise_latents_btchw = self.stage.scheduler.add_noise(
                        pred_video_btchw.flatten(0, 1),
                        noise_btchw.flatten(0, 1),
                        next_timestep,
                    ).unflatten(0, pred_video_btchw.shape[:2])
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
            else:
                current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

            if progress_bar is not None:
                progress_bar.update()

        block_ctx.extra["attn_metadata"] = attn_metadata
        state.latents[:, :, start_index:start_index +
                      current_num_frames, :, :] = (current_latents)

    def update_context(self, state: StrategyState, block_ctx: BlockContext,
                       block_item: BlockPlanItem) -> None:
        batch = state.extra["batch"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        boundary_timestep = state.extra["boundary_timestep"]
        kv_cache1 = state.extra["kv_cache1"]
        kv_cache2 = state.extra["kv_cache2"]
        crossattn_cache = state.extra["crossattn_cache"]
        image_kwargs = state.extra["image_kwargs"]
        pos_cond_kwargs = state.extra["pos_cond_kwargs"]
        context_noise = state.extra["context_noise"]

        start_index = block_item.start_index
        current_num_frames = block_item.num_frames

        current_latents = state.latents[:, :, start_index:start_index +
                                        current_num_frames, :, :]
        latents_device = current_latents.device

        t_context = torch.ones([current_latents.shape[0]],
                               device=latents_device,
                               dtype=torch.long) * int(context_noise)
        context_bcthw = current_latents.to(target_dtype)

        attn_metadata = block_ctx.extra.get("attn_metadata")
        with torch.autocast(device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled), \
            set_forward_context(current_timestep=0,
                                attn_metadata=attn_metadata,
                                forward_batch=batch):
            t_expanded_context = t_context.unsqueeze(1)

            if boundary_timestep is not None:
                self.stage.transformer_2(
                    context_bcthw,
                    batch.prompt_embeds,
                    t_expanded_context,
                    kv_cache=kv_cache2,
                    crossattn_cache=crossattn_cache,
                    current_start=start_index * self.frame_seq_length,
                    start_frame=start_index,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )

            self.stage.transformer(
                context_bcthw,
                batch.prompt_embeds,
                t_expanded_context,
                kv_cache=kv_cache1,
                crossattn_cache=crossattn_cache,
                current_start=start_index * self.frame_seq_length,
                start_frame=start_index,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        progress_bar = state.extra.get("progress_bar")
        if progress_bar is not None:
            progress_bar.close()

        batch = state.extra["batch"]
        boundary_timestep = state.extra["boundary_timestep"]

        latents = state.latents
        if boundary_timestep is not None:
            num_frames_to_remove = self.num_frames_per_block - 1
            latents = latents[:, :, :-num_frames_to_remove, :, :]

        batch.latents = latents
        return batch

    def make_model_inputs(self, state: StrategyState, t: torch.Tensor,
                          step_idx: int) -> ModelInputs:
        raise NotImplementedError

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        raise NotImplementedError

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
