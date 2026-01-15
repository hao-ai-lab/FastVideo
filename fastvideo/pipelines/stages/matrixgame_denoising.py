from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
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
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False
    VideoSparseAttentionBackend = None  # type: ignore

logger = init_logger(__name__)


@dataclass
class BlockProcessingContext:
    """Dataclass contains for block processing."""

    batch: ForwardBatch

    block_idx: int
    start_index: int

    kv_cache1: list[dict[Any, Any]]
    kv_cache2: list[dict[Any, Any]] | None
    kv_cache_mouse: list[dict[Any, Any]] | None
    kv_cache_keyboard: list[dict[Any, Any]] | None
    crossattn_cache: list[dict[Any, Any]]

    timesteps: torch.Tensor
    block_sizes: list[int]
    noise_pool: list[torch.Tensor] | None

    fastvideo_args: FastVideoArgs
    target_dtype: torch.dtype
    autocast_enabled: bool
    boundary_timestep: float | None
    high_noise_timesteps: torch.Tensor | None
    context_noise: float

    image_kwargs: dict[str, Any]
    pos_cond_kwargs: dict[str, Any]

    def get_kv_cache(self, timestep_val: float) -> list[dict[Any, Any]]:
        if self.boundary_timestep is not None:
            if timestep_val >= self.boundary_timestep:
                return self.kv_cache1
            else:
                assert self.kv_cache2 is not None, "kv_cache2 is not initialized"
                return self.kv_cache2
        return self.kv_cache1


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

        self._streaming_initialized: bool = False
        self._streaming_ctx: BlockProcessingContext | None = None
        self._streaming_engine = None
        self._streaming_state = None
        self._streaming_block_plan = None

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_matrixgame_strategy import (
            MatrixGameBlockStrategy)

        engine = DenoisingEngine(MatrixGameBlockStrategy(self),
                                 hooks=self._build_engine_hooks())
        return engine.run(batch, fastvideo_args)

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

    def _process_single_block(
        self,
        current_latents: torch.Tensor,
        batch: ForwardBatch,
        start_index: int,
        current_num_frames: int,
        timesteps: torch.Tensor,
        ctx: BlockProcessingContext,
        action_kwargs: dict[str, Any],
        noise_generator: Callable[[tuple, torch.dtype, int], torch.Tensor]
        | None = None,
        progress_bar: Any | None = None,
    ) -> torch.Tensor:
        prompt_embeds = batch.prompt_embeds
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)

        for i, t_cur in enumerate(timesteps):
            if ctx.boundary_timestep is not None and t_cur < ctx.boundary_timestep:
                current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
            else:
                current_model = self.transformer

            noise_latents = noise_latents_btchw.clone()
            latent_model_input = current_latents.to(ctx.target_dtype)

            independent_first_frame = getattr(self.transformer,
                                              'independent_first_frame', False)
            if batch.image_latent is not None and independent_first_frame and start_index == 0:
                latent_model_input = torch.cat([
                    latent_model_input,
                    batch.image_latent.to(ctx.target_dtype)
                ],
                                               dim=2)

            # t_expand needs to be [batch * frames] to match flattened pred_noise/noise_latents
            t_expand = t_cur.repeat(latent_model_input.shape[0] *
                                    current_num_frames)

            # Build attention metadata if VSA is available
            if vsa_available and self.attn_backend == VideoSparseAttentionBackend:
                self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                )
                if self.attn_metadata_builder_cls is not None:
                    self.attn_metadata_builder = self.attn_metadata_builder_cls(
                    )
                    h, w = current_latents.shape[-2:]
                    attn_metadata = self.attn_metadata_builder.build(
                        current_timestep=i,
                        raw_latent_shape=(current_num_frames, h, w),
                        patch_size=ctx.fastvideo_args.pipeline_config.
                        dit_config.patch_size,
                        STA_param=batch.STA_param,
                        VSA_sparsity=ctx.fastvideo_args.VSA_sparsity,
                        device=get_local_torch_device(),
                    )
                    assert attn_metadata is not None, "attn_metadata cannot be None"
                else:
                    attn_metadata = None
            else:
                attn_metadata = None

            with torch.autocast(device_type="cuda",
                                dtype=ctx.target_dtype,
                                enabled=ctx.autocast_enabled), \
                set_forward_context(current_timestep=i,
                                    attn_metadata=attn_metadata,
                                    forward_batch=batch):
                # Expand timestep to per-frame format [batch, num_frames] for causal model
                t_expanded_noise = t_cur * torch.ones(
                    (latent_model_input.shape[0], current_num_frames),
                    device=latent_model_input.device,
                    dtype=torch.long)

                model_kwargs = {
                    "kv_cache": ctx.get_kv_cache(t_cur),
                    "crossattn_cache": ctx.crossattn_cache,
                    "current_start": start_index * self.frame_seq_length,
                    "start_frame": start_index,
                }

                if self.use_action_module and current_model == self.transformer:
                    model_kwargs.update({
                        "kv_cache_mouse":
                        ctx.kv_cache_mouse,
                        "kv_cache_keyboard":
                        ctx.kv_cache_keyboard,
                    })
                    model_kwargs.update(action_kwargs)

                pred_noise_btchw = current_model(
                    latent_model_input,
                    prompt_embeds,
                    t_expanded_noise,
                    **ctx.image_kwargs,
                    **ctx.pos_cond_kwargs,
                    **model_kwargs,
                ).permute(0, 2, 1, 3, 4)

            if ctx.boundary_timestep is not None and t_cur >= ctx.boundary_timestep:
                pred_video_btchw = pred_noise_to_x_bound(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=t_expand,
                    boundary_timestep=torch.ones_like(t_expand) *
                    ctx.boundary_timestep,
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
                    [1], dtype=torch.long, device=pred_video_btchw.device)

                # Use custom noise generator if provided (for streaming), else generate
                if noise_generator is not None:
                    noise = noise_generator(pred_video_btchw.shape,
                                            pred_video_btchw.dtype, i)
                else:
                    noise = torch.randn(
                        pred_video_btchw.shape,
                        dtype=pred_video_btchw.dtype,
                        generator=(batch.generator[0] if isinstance(
                            batch.generator, list) else batch.generator)).to(
                                pred_video_btchw.device)

                noise_btchw = noise
                if ctx.boundary_timestep is not None and ctx.high_noise_timesteps is not None and i < len(
                        ctx.high_noise_timesteps) - 1:
                    noise_latents_btchw = self.scheduler.add_noise_high(
                        pred_video_btchw.flatten(0, 1),
                        noise_btchw.flatten(0, 1), next_timestep,
                        torch.ones_like(next_timestep) *
                        ctx.boundary_timestep).unflatten(
                            0, pred_video_btchw.shape[:2])
                elif ctx.boundary_timestep is not None and ctx.high_noise_timesteps is not None and i == len(
                        ctx.high_noise_timesteps) - 1:
                    noise_latents_btchw = pred_video_btchw
                else:
                    noise_latents_btchw = self.scheduler.add_noise(
                        pred_video_btchw.flatten(0,
                                                 1), noise_btchw.flatten(0, 1),
                        next_timestep).unflatten(0, pred_video_btchw.shape[:2])
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
            else:
                current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

            if progress_bar is not None:
                progress_bar.update()

        return current_latents

    def _update_context_cache(
        self,
        current_latents: torch.Tensor,
        batch: ForwardBatch,
        start_index: int,
        current_num_frames: int,
        ctx: BlockProcessingContext,
        action_kwargs: dict[str, Any],
        context_noise: float,
    ) -> None:
        prompt_embeds = batch.prompt_embeds
        latents_device = current_latents.device

        # Expand context timestep to per-frame format [batch, num_frames] for causal model
        t_context = torch.ones([current_latents.shape[0], current_num_frames],
                               device=latents_device,
                               dtype=torch.long) * int(context_noise)
        context_bcthw = current_latents.to(ctx.target_dtype)

        with torch.autocast(device_type="cuda",
                            dtype=ctx.target_dtype,
                            enabled=ctx.autocast_enabled), \
            set_forward_context(current_timestep=0,
                                attn_metadata=None,
                                forward_batch=batch):

            context_model_kwargs = {
                "kv_cache": ctx.kv_cache1,
                "crossattn_cache": ctx.crossattn_cache,
                "current_start": start_index * self.frame_seq_length,
                "start_frame": start_index,
            }

            if self.use_action_module:
                context_model_kwargs.update({
                    "kv_cache_mouse":
                    ctx.kv_cache_mouse,
                    "kv_cache_keyboard":
                    ctx.kv_cache_keyboard,
                })
                context_model_kwargs.update(action_kwargs)

            if ctx.boundary_timestep is not None and self.transformer_2 is not None:
                self.transformer_2(
                    context_bcthw,
                    prompt_embeds,
                    t_context,
                    kv_cache=ctx.kv_cache2,
                    crossattn_cache=ctx.crossattn_cache,
                    current_start=start_index * self.frame_seq_length,
                    start_frame=start_index,
                    **ctx.image_kwargs,
                    **ctx.pos_cond_kwargs,
                )

            self.transformer(
                context_bcthw,
                prompt_embeds,
                t_context,
                **ctx.image_kwargs,
                **ctx.pos_cond_kwargs,
                **context_model_kwargs,
            )

    def streaming_reset(self, batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_matrixgame_strategy import (
            MatrixGameBlockStrategy)

        strategy = MatrixGameBlockStrategy(self)
        engine = DenoisingEngine(strategy, hooks=self._build_engine_hooks())
        state = strategy.prepare(batch, fastvideo_args)
        for hook in engine.hooks:
            hook.on_init(engine, batch, fastvideo_args)
        for hook in engine.hooks:
            hook.pre_run(state)
        block_plan = strategy.block_plan(state)

        progress_bar = state.extra.get("progress_bar")
        if progress_bar is not None:
            progress_bar.close()
        state.extra["progress_bar"] = None

        ctx = state.extra["ctx"]
        latents = state.latents
        assert latents is not None, "latents must be provided"
        b, c, t, h, w = latents.shape

        num_denoising_steps = len(state.timesteps)
        noise_shape = (b, self.num_frame_per_block, c, h, w)
        noise_pool = [
            torch.randn(
                noise_shape,
                dtype=ctx.target_dtype,
                device=latents.device,
            ) for _ in range(max(num_denoising_steps - 1, 0))
        ]
        ctx.noise_pool = noise_pool

        self._streaming_engine = engine
        self._streaming_state = state
        self._streaming_block_plan = block_plan
        self._streaming_ctx = ctx
        self._streaming_initialized = True
        return batch

    def streaming_step(
            self,
            keyboard_action: torch.Tensor | None = None,
            mouse_action: torch.Tensor | None = None) -> ForwardBatch:
        if (not self._streaming_initialized or self._streaming_ctx is None
                or self._streaming_engine is None
                or self._streaming_state is None
                or self._streaming_block_plan is None):
            raise RuntimeError(
                "Streaming not initialized! Call streaming_reset first.")

        ctx = self._streaming_ctx
        block_plan = self._streaming_block_plan
        if ctx.block_idx >= len(block_plan.items):
            return ctx.batch

        batch = ctx.batch
        latents = batch.latents
        assert latents is not None, "latents must be set in batch"

        block_item = block_plan.items[ctx.block_idx]
        start_index = block_item.start_index
        current_num_frames = block_item.num_frames

        # Update batch with new actions for this block
        if keyboard_action is not None or mouse_action is not None:
            vae_ratio = 4
            start_frame = 0 if start_index == 0 else 1 + vae_ratio * (
                start_index - 1)

            if keyboard_action is not None:
                n = keyboard_action.shape[1]
                batch.keyboard_cond[:, start_frame:start_frame +
                                    n] = keyboard_action.to(
                                        batch.keyboard_cond.device)
            if mouse_action is not None:
                n = mouse_action.shape[1]
                batch.mouse_cond[:, start_frame:start_frame +
                                 n] = mouse_action.to(batch.mouse_cond.device)

        self._streaming_engine.run_blocks(
            self._streaming_state,
            block_plan=block_plan,
            start_block=ctx.block_idx,
            num_blocks=1,
        )

        # Advance streaming state
        ctx.start_index = start_index + current_num_frames
        ctx.block_idx += 1

        return batch

    def streaming_clear(self) -> None:
        if (self._streaming_engine is not None
                and self._streaming_state is not None):
            batch = (self._streaming_ctx.batch if self._streaming_ctx
                     is not None else self._streaming_state.extra.get("batch"))
            if batch is not None:
                for hook in self._streaming_engine.hooks:
                    hook.post_run(self._streaming_state, batch)
        self._streaming_initialized = False
        self._streaming_ctx = None
        self._streaming_engine = None
        self._streaming_state = None
        self._streaming_block_plan = None

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
