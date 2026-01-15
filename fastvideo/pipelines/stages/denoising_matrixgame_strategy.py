# SPDX-License-Identifier: Apache-2.0
"""
MatrixGame causal block denoising strategy.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_strategies import (
    BlockContext,
    BlockDenoisingStrategy,
    BlockPlan,
    BlockPlanItem,
    ModelInputs,
    StrategyState,
)
from fastvideo.pipelines.stages.matrixgame_denoising import BlockProcessingContext

try:
    from fastvideo.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False
    SlidingTileAttentionBackend = None  # type: ignore


class MatrixGameBlockStrategy(BlockDenoisingStrategy):

    def __init__(self, stage: Any) -> None:
        self.stage = stage

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latents = batch.latents
        if latents is None:
            raise ValueError("latents must be provided")

        latent_seq_length = latents.shape[-1] * latents.shape[-2]
        patch_size = self.stage.transformer.patch_size
        patch_ratio = patch_size[-1] * patch_size[-2]
        self.stage.frame_seq_length = latent_seq_length // patch_ratio

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

        boundary_ratio = getattr(fastvideo_args.pipeline_config.dit_config,
                                 "boundary_ratio", None)
        if boundary_ratio is not None:
            boundary_timestep = (boundary_ratio *
                                 self.stage.scheduler.num_train_timesteps)
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

        image_kwargs = {"encoder_hidden_states_image": image_embeds}
        pos_cond_kwargs: dict[str, Any] = {}

        if (st_attn_available
                and self.stage.attn_backend == SlidingTileAttentionBackend):
            self.stage.prepare_sta_param(batch, fastvideo_args)

        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        kv_cache1 = self.stage._initialize_kv_cache(batch_size=latents.shape[0],
                                                    dtype=target_dtype,
                                                    device=latents.device)
        kv_cache2 = None
        if boundary_timestep is not None:
            kv_cache2 = self.stage._initialize_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device,
            )

        kv_cache_mouse = None
        kv_cache_keyboard = None
        if self.stage.use_action_module:
            kv_cache_mouse, kv_cache_keyboard = (
                self.stage._initialize_action_kv_cache(
                    batch_size=latents.shape[0],
                    dtype=target_dtype,
                    device=latents.device,
                ))

        crossattn_cache = self.stage._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=257,
            dtype=target_dtype,
            device=latents.device,
        )

        num_frames = latents.shape[2]
        if num_frames % self.stage.num_frame_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frame_per_block for "
                "causal denoising")
        num_blocks = num_frames // self.stage.num_frame_per_block
        block_sizes = [self.stage.num_frame_per_block] * num_blocks
        start_index = 0

        if boundary_timestep is not None:
            block_sizes[0] = 1

        ctx = BlockProcessingContext(
            batch=batch,
            block_idx=0,
            start_index=0,
            kv_cache1=kv_cache1,
            kv_cache2=kv_cache2,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            crossattn_cache=crossattn_cache,
            timesteps=timesteps,
            block_sizes=block_sizes,
            noise_pool=None,
            fastvideo_args=fastvideo_args,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            boundary_timestep=boundary_timestep,
            high_noise_timesteps=high_noise_timesteps,
            context_noise=getattr(fastvideo_args.pipeline_config,
                                  "context_noise", 0),
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
        )

        progress_bar = self.stage.progress_bar(total=len(block_sizes) *
                                               len(timesteps))

        extra: dict[str, Any] = {
            "batch": batch,
            "fastvideo_args": fastvideo_args,
            "ctx": ctx,
            "block_sizes": block_sizes,
            "start_index": start_index,
            "progress_bar": progress_bar,
            "boundary_timestep": boundary_timestep,
        }

        return StrategyState(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=len(timesteps),
            prompt_embeds=batch.prompt_embeds,
            negative_prompt_embeds=batch.negative_prompt_embeds,
            prompt_attention_mask=batch.prompt_attention_mask,
            negative_attention_mask=batch.negative_attention_mask,
            image_embeds=image_embeds,
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
        ctx = state.extra["ctx"]
        ctx.block_idx = block_idx
        ctx.start_index = block_item.start_index

        action_kwargs = self.stage._prepare_action_kwargs(
            state.extra["batch"],
            block_item.start_index,
            block_item.num_frames,
        )

        return BlockContext(
            kv_cache=ctx.kv_cache1,
            kv_cache_2=ctx.kv_cache2,
            crossattn_cache=ctx.crossattn_cache,
            action_cache=None,
            extra={
                "ctx": ctx,
                "action_kwargs": action_kwargs,
                "start_index": block_item.start_index,
                "num_frames": block_item.num_frames,
            },
        )

    def process_block(self, state: StrategyState, block_ctx: BlockContext,
                      block_item: BlockPlanItem) -> None:
        ctx = block_ctx.extra["ctx"]
        batch = state.extra["batch"]
        progress_bar = state.extra["progress_bar"]

        start_index = block_item.start_index
        current_num_frames = block_item.num_frames
        action_kwargs = block_ctx.extra["action_kwargs"]

        current_latents = state.latents[:, :, start_index:start_index +
                                        current_num_frames, :, :]

        noise_generator = None
        if ctx.noise_pool is not None:
            latents_device = state.latents.device

            def noise_generator(shape: tuple, dtype: torch.dtype,
                                step_idx: int) -> torch.Tensor:
                if step_idx < len(ctx.noise_pool):
                    noise = ctx.noise_pool[step_idx]
                    if noise.shape != shape:
                        noise = noise[:, :shape[1], :, :, :]
                    return noise.to(device=latents_device, dtype=dtype)

                generator = batch.generator
                if isinstance(generator, list):
                    generator = generator[0] if generator else None
                return torch.randn(shape, dtype=dtype,
                                   generator=generator).to(latents_device)

        current_latents = self.stage._process_single_block(
            current_latents=current_latents,
            batch=batch,
            start_index=start_index,
            current_num_frames=current_num_frames,
            timesteps=state.timesteps,
            ctx=ctx,
            action_kwargs=action_kwargs,
            progress_bar=progress_bar,
            noise_generator=noise_generator,
        )

        state.latents[:, :, start_index:start_index +
                      current_num_frames, :, :] = (current_latents)

    def update_context(self, state: StrategyState, block_ctx: BlockContext,
                       block_item: BlockPlanItem) -> None:
        ctx = block_ctx.extra["ctx"]
        batch = state.extra["batch"]
        action_kwargs = block_ctx.extra["action_kwargs"]
        start_index = block_item.start_index
        current_num_frames = block_item.num_frames

        current_latents = state.latents[:, :, start_index:start_index +
                                        current_num_frames, :, :]

        self.stage._update_context_cache(
            current_latents=current_latents,
            batch=batch,
            start_index=start_index,
            current_num_frames=current_num_frames,
            ctx=ctx,
            action_kwargs=action_kwargs,
            context_noise=ctx.context_noise,
        )

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        progress_bar = state.extra.get("progress_bar")
        if progress_bar is not None:
            progress_bar.close()

        batch = state.extra["batch"]
        boundary_timestep = state.extra["boundary_timestep"]

        latents = state.latents
        if boundary_timestep is not None:
            num_frames_to_remove = self.stage.num_frame_per_block - 1
            if num_frames_to_remove > 0:
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
