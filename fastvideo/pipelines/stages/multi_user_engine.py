"""Multi-user engine implementing ORCA-style batching for streaming generation.

Groups users at the same (block_idx, denoising_step) and batches their forward
passes together, sharing model weights while maintaining per-user KV caches.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.matrixgame_denoising import (
    BlockProcessingContext,
    MatrixGameCausalDenoisingStage,
)

logger = init_logger(__name__)


@dataclass
class CompletedResult:
    """Result for a user whose block generation finished."""
    user_id: str
    output_batch: ForwardBatch


@dataclass
class UserSession:
    """Per-user state for multi-user streaming."""
    user_id: str
    ctx: BlockProcessingContext
    vae_cache: list[torch.Tensor | None] | None
    batch: ForwardBatch
    fastvideo_args: FastVideoArgs

    # Denoising state (set when a step is in progress)
    denoising_step: int = -1  # -1 means no pending work
    current_latents: torch.Tensor | None = None
    noise_latents_btchw: torch.Tensor | None = None
    action_kwargs: dict[str, Any] = field(default_factory=dict)
    block_start_index: int = 0
    block_num_frames: int = 0
    dit_elapsed_ms: float = 0.0  # accumulated DiT time across denoising steps


class MultiUserEngine:
    """ORCA-style scheduler that batches multiple users on a single GPU."""

    def __init__(self, pipeline, disable_batching: bool = False):
        self.pipeline = pipeline
        self.disable_batching = disable_batching
        if not pipeline.post_init_called:
            pipeline.post_init()
        self.denoiser: MatrixGameCausalDenoisingStage = (
            pipeline._stage_name_mapping["denoising_stage"]
        )
        self.decoder = pipeline._stage_name_mapping["decoding_stage"]
        self.users: dict[str, UserSession] = {}

    def add_user(
        self,
        user_id: str,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Initialize a new user session (runs preprocessing + streaming_reset)."""
        def _gpu_mem():
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved() / 1024**3
            return f"alloc={a:.2f}GiB, reserved={r:.2f}GiB"

        logger.info("add_user(%s) start: %s", user_id, _gpu_mem())
        logger.info("add_user(%s) memory summary:\n%s",
                     user_id, torch.cuda.memory_summary())

        if user_id in self.users:
            logger.warning("User %s already exists, removing first", user_id)
            self.remove_user(user_id)

        # Free any existing streaming state FIRST to make room on GPU.
        # After warmup, the denoiser holds KV caches + noise pool.
        import gc
        logger.info("add_user(%s) denoiser._streaming_initialized=%s, ctx=%s",
                     user_id, self.denoiser._streaming_initialized,
                     self.denoiser._streaming_ctx is not None)
        if self.denoiser._streaming_initialized:
            ctx = self.denoiser._streaming_ctx
            if ctx is not None:
                # Explicitly delete large tensors to break any reference cycles
                ctx.kv_cache1 = None
                ctx.kv_cache2 = None
                ctx.kv_cache_mouse = None
                ctx.kv_cache_keyboard = None
                ctx.crossattn_cache = None
                ctx.noise_pool = None
                if ctx.batch is not None:
                    ctx.batch.latents = None
                    ctx.batch.image_latent = None
                    ctx.batch.prompt_embeds = None
                ctx.batch = None
            self.denoiser._streaming_ctx = None
            self.denoiser._streaming_initialized = False
            logger.info("add_user(%s) after ctx cleanup: %s", user_id, _gpu_mem())
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("add_user(%s) after gc+empty_cache: %s", user_id, _gpu_mem())
            logger.info("add_user(%s) post-cleanup summary:\n%s",
                         user_id, torch.cuda.memory_summary())

        # Run preprocessing stages
        if not self.pipeline.post_init_called:
            self.pipeline.post_init()

        # Ensure VAE temporal tiling is enabled to avoid OOM when encoding
        # the full video_condition [1,3,num_frames,H,W] through 3D conv layers.
        # Enable on both the pipeline's VAE and the stage's VAE reference.
        vae = self.pipeline.get_module("vae", None)
        if vae is not None:
            vae.enable_tiling(use_temporal_tiling=True)
            logger.info("Pipeline VAE tiling: use_tiling=%s, temporal=%s, min_frames=%s, id=%s",
                        vae.use_tiling, vae.use_temporal_tiling,
                        getattr(vae, 'tile_sample_min_num_frames', 'N/A'), id(vae))
        img_stage = self.pipeline._stage_name_mapping.get("image_latent_preparation_stage")
        if img_stage is not None and hasattr(img_stage, 'vae'):
            img_stage.vae.enable_tiling(use_temporal_tiling=True)
            logger.info("Stage VAE tiling: use_tiling=%s, temporal=%s, min_frames=%s, id=%s",
                        img_stage.vae.use_tiling, img_stage.vae.use_temporal_tiling,
                        getattr(img_stage.vae, 'tile_sample_min_num_frames', 'N/A'),
                        id(img_stage.vae))

        stages_to_run = [
            "input_validation_stage", "prompt_encoding_stage",
            "image_encoding_stage", "conditioning_stage",
            "latent_preparation_stage", "image_latent_preparation_stage",
        ]
        for stage_name in stages_to_run:
            if stage_name in self.pipeline._stage_name_mapping:
                batch = self.pipeline._stage_name_mapping[stage_name].forward(
                    batch, fastvideo_args)
                logger.info("add_user(%s) after %s: %s",
                            user_id, stage_name, _gpu_mem())

        # Initialize denoiser state for this user (creates KV caches etc.)
        # We call streaming_reset to set up the context, then steal it
        logger.info("add_user(%s) before streaming_reset: %s", user_id, _gpu_mem())
        self.denoiser.streaming_reset(batch, fastvideo_args)
        ctx = self.denoiser._streaming_ctx
        assert ctx is not None
        # Detach from denoiser so it doesn't interfere with other users
        self.denoiser._streaming_ctx = None
        self.denoiser._streaming_initialized = False

        self.users[user_id] = UserSession(
            user_id=user_id,
            ctx=ctx,
            vae_cache=None,
            batch=batch,
            fastvideo_args=fastvideo_args,
        )
        logger.info("Added user %s (total: %d)", user_id, len(self.users))

    def remove_user(self, user_id: str) -> None:
        """Remove a user session and free GPU memory."""
        session = self.users.pop(user_id, None)
        if session is not None:
            # Let GC handle tensor cleanup
            session.ctx = None  # type: ignore
            session.vae_cache = None
            session.current_latents = None
            session.noise_latents_btchw = None
            logger.info("Removed user %s (remaining: %d)",
                        user_id, len(self.users))

    def submit_step(
        self,
        user_id: str,
        keyboard_action: torch.Tensor | None,
        mouse_action: torch.Tensor | None,
    ) -> None:
        """Queue a block generation request for a user."""
        session = self.users.get(user_id)
        if session is None:
            raise KeyError(f"User {user_id} not found")

        ctx = session.ctx
        if ctx.block_idx >= len(ctx.block_sizes):
            logger.warning("User %s has no more blocks to generate", user_id)
            return

        batch = session.batch
        latents = batch.latents
        assert latents is not None

        current_num_frames = ctx.block_sizes[ctx.block_idx]
        start_index = ctx.start_index

        current_latents = latents[:, :, start_index:start_index +
                                  current_num_frames, :, :]

        # Update batch with new actions
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

        action_kwargs = self.denoiser._prepare_action_kwargs(
            batch, start_index, current_num_frames)

        # Set up denoising state
        session.denoising_step = 0
        session.dit_elapsed_ms = 0.0
        session.current_latents = current_latents
        session.noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        session.action_kwargs = action_kwargs
        session.block_start_index = start_index
        session.block_num_frames = current_num_frames

    def has_pending_work(self) -> bool:
        """Check if any user has pending denoising work."""
        return any(s.denoising_step >= 0 for s in self.users.values())

    def run_iteration(self) -> Iterator[CompletedResult]:
        """Run one denoising step for all ready users (ORCA iteration-level scheduling).

        Groups users by (block_idx, denoising_step) and batches compatible groups.
        Yields completed results immediately after each user's VAE decode finishes,
        so responses can be sent without waiting for all users in the batch.
        """
        # Group users with pending work by (block_idx, denoising_step)
        groups: dict[tuple[int, int], list[UserSession]] = {}
        for session in self.users.values():
            if session.denoising_step < 0:
                continue
            key = (session.ctx.block_idx, session.denoising_step)
            groups.setdefault(key, []).append(session)

        for (block_idx, step_idx), sessions in groups.items():
            timesteps = sessions[0].ctx.timesteps
            t_cur = timesteps[step_idx]
            next_timestep = timesteps[step_idx + 1] if step_idx < len(timesteps) - 1 else None

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            user_ids = [s.user_id[:8] for s in sessions]
            if len(sessions) == 1 or self.disable_batching:
                # Process each user individually
                for s in sessions:
                    logger.info("ORCA step: block=%d step=%d single user=%s",
                                block_idx, step_idx, s.user_id[:8])
                    self._run_single_user_step(s, t_cur, step_idx, next_timestep)
            else:
                # Multiple users at same state - batch them
                logger.info("ORCA step: block=%d step=%d BATCHED %d users=%s",
                            block_idx, step_idx, len(sessions), user_ids)
                self._run_batched_step(sessions, t_cur, step_idx, next_timestep)

            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000

            # Advance step counters and finalize completed users
            for s in sessions:
                s.dit_elapsed_ms += step_ms
                s.denoising_step += 1
                if s.denoising_step >= len(timesteps):
                    # Denoising complete — VAE decode and yield immediately
                    result = self._finalize_block(s)
                    if result is not None:
                        yield result

    def _run_single_user_step(
        self,
        session: UserSession,
        timestep: torch.Tensor,
        step_idx: int,
        next_timestep: torch.Tensor | None,
    ) -> None:
        """Run one denoising step for a single user (no batching)."""
        noise_gen = self._make_noise_generator(session)

        current_latents, noise_latents_btchw = self.denoiser._denoise_one_step(
            current_latents=session.current_latents,
            noise_latents_btchw=session.noise_latents_btchw,
            batch=session.batch,
            start_index=session.block_start_index,
            current_num_frames=session.block_num_frames,
            timestep=timestep,
            step_idx=step_idx,
            next_timestep=next_timestep,
            ctx=session.ctx,
            action_kwargs=session.action_kwargs,
            noise_generator=noise_gen,
        )
        session.current_latents = current_latents
        session.noise_latents_btchw = noise_latents_btchw

    def _run_batched_step(
        self,
        sessions: list[UserSession],
        timestep: torch.Tensor,
        step_idx: int,
        next_timestep: torch.Tensor | None,
    ) -> None:
        """Run one denoising step for multiple users batched together."""
        n = len(sessions)

        # Concatenate along batch dimension
        batched_latents = torch.cat(
            [s.current_latents for s in sessions], dim=0)
        batched_noise = torch.cat(
            [s.noise_latents_btchw for s in sessions], dim=0)

        # Build a merged ForwardBatch with concatenated per-user tensors
        merged_batch = self._merge_batches(sessions)

        # Build merged context with concatenated KV caches
        merged_ctx = self._merge_contexts(sessions, merged_batch)

        # Merge action kwargs
        merged_action_kwargs = self._merge_action_kwargs(sessions)

        # Create noise generator that concatenates per-user noise
        device = batched_latents.device

        def batched_noise_gen(shape, dtype, si):
            noises = []
            for s in sessions:
                gen = self._make_noise_generator(s)
                if gen is not None:
                    per_user_shape = (1,) + shape[1:]
                    noises.append(gen(per_user_shape, dtype, si))
                else:
                    noises.append(torch.randn(
                        (1,) + shape[1:], dtype=dtype).to(device))
            return torch.cat(noises, dim=0)

        current_latents, noise_latents_btchw = self.denoiser._denoise_one_step(
            current_latents=batched_latents,
            noise_latents_btchw=batched_noise,
            batch=merged_batch,
            start_index=sessions[0].block_start_index,
            current_num_frames=sessions[0].block_num_frames,
            timestep=timestep,
            step_idx=step_idx,
            next_timestep=next_timestep,
            ctx=merged_ctx,
            action_kwargs=merged_action_kwargs,
            noise_generator=batched_noise_gen,
        )

        # Unbatch results back to per-user tensors
        latents_list = current_latents.split(1, dim=0)
        noise_list = noise_latents_btchw.split(1, dim=0)

        # Restore per-user KV caches from the batched caches
        self._split_contexts(merged_ctx, sessions)

        for i, s in enumerate(sessions):
            s.current_latents = latents_list[i]
            s.noise_latents_btchw = noise_list[i]

    def _merge_batches(self, sessions: list[UserSession]) -> ForwardBatch:
        """Create a merged ForwardBatch with concatenated user tensors."""
        ref = sessions[0].batch
        # prompt_embeds and image_embeds are shared (same game)
        # latents, keyboard_cond, mouse_cond differ per user
        merged = ForwardBatch.__new__(ForwardBatch)
        merged.__dict__.update(ref.__dict__)

        # image_latent: cat along batch dim if present
        if ref.image_latent is not None:
            merged.image_latent = torch.cat(
                [s.batch.image_latent for s in sessions], dim=0)

        return merged

    def _merge_contexts(
        self, sessions: list[UserSession],
        merged_batch: ForwardBatch | None = None,
    ) -> BlockProcessingContext:
        """Create a merged BlockProcessingContext with concatenated KV caches."""
        ref = sessions[0].ctx
        n = len(sessions)

        # Concatenate KV caches along batch dimension
        merged_kv1 = self._cat_kv_caches([s.ctx.kv_cache1 for s in sessions])
        merged_kv2 = None
        if ref.kv_cache2 is not None:
            merged_kv2 = self._cat_kv_caches(
                [s.ctx.kv_cache2 for s in sessions])

        merged_crossattn = self._cat_crossattn_caches(
            [s.ctx.crossattn_cache for s in sessions])

        merged_kv_mouse = None
        merged_kv_keyboard = None
        if ref.kv_cache_mouse is not None:
            merged_kv_mouse = self._cat_action_kv_caches(
                [s.ctx.kv_cache_mouse for s in sessions],
                is_mouse=True)
        if ref.kv_cache_keyboard is not None:
            merged_kv_keyboard = self._cat_kv_caches(
                [s.ctx.kv_cache_keyboard for s in sessions])

        # image_kwargs: cat image embeds
        merged_image_kwargs = dict(ref.image_kwargs)
        if "encoder_hidden_states_image" in ref.image_kwargs:
            embeds = ref.image_kwargs["encoder_hidden_states_image"]
            if isinstance(embeds, list) and len(embeds) > 0 and torch.is_tensor(embeds[0]):
                # Cat each tensor in the list across users
                merged_embeds = []
                for idx in range(len(embeds)):
                    merged_embeds.append(torch.cat(
                        [s.ctx.image_kwargs["encoder_hidden_states_image"][idx]
                         for s in sessions], dim=0))
                merged_image_kwargs["encoder_hidden_states_image"] = merged_embeds

        if merged_batch is None:
            merged_batch = self._merge_batches(sessions)

        return BlockProcessingContext(
            batch=merged_batch,
            block_idx=ref.block_idx,
            start_index=ref.start_index,
            kv_cache1=merged_kv1,
            kv_cache2=merged_kv2,
            kv_cache_mouse=merged_kv_mouse,
            kv_cache_keyboard=merged_kv_keyboard,
            crossattn_cache=merged_crossattn,
            timesteps=ref.timesteps,
            block_sizes=ref.block_sizes,
            noise_pool=None,  # noise handled per-user
            fastvideo_args=ref.fastvideo_args,
            target_dtype=ref.target_dtype,
            autocast_enabled=ref.autocast_enabled,
            boundary_timestep=ref.boundary_timestep,
            high_noise_timesteps=ref.high_noise_timesteps,
            context_noise=ref.context_noise,
            image_kwargs=merged_image_kwargs,
            pos_cond_kwargs=ref.pos_cond_kwargs,
        )

    def _cat_kv_caches(
        self, caches_list: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Concatenate KV caches along batch dimension."""
        num_layers = len(caches_list[0])
        merged = []
        for layer_idx in range(num_layers):
            merged.append({
                "k": torch.cat(
                    [c[layer_idx]["k"] for c in caches_list], dim=0),
                "v": torch.cat(
                    [c[layer_idx]["v"] for c in caches_list], dim=0),
                "global_end_index": caches_list[0][layer_idx]["global_end_index"],
                "local_end_index": caches_list[0][layer_idx]["local_end_index"],
            })
        return merged

    def _cat_action_kv_caches(
        self,
        caches_list: list[list[dict[str, Any]]],
        is_mouse: bool = False,
    ) -> list[dict[str, Any]]:
        """Concatenate action KV caches.

        Mouse caches have shape [B*frame_seq_length, ...] so we need
        to concatenate correctly.
        """
        num_layers = len(caches_list[0])
        merged = []
        for layer_idx in range(num_layers):
            if is_mouse:
                # Mouse: [B*F, cache_size, heads, dim] -> [N*F, cache_size, heads, dim]
                # Each user has [1*F, ...], so simple cat along dim 0 works
                merged.append({
                    "k": torch.cat(
                        [c[layer_idx]["k"] for c in caches_list], dim=0),
                    "v": torch.cat(
                        [c[layer_idx]["v"] for c in caches_list], dim=0),
                    "global_end_index": caches_list[0][layer_idx]["global_end_index"],
                    "local_end_index": caches_list[0][layer_idx]["local_end_index"],
                })
            else:
                merged.append({
                    "k": torch.cat(
                        [c[layer_idx]["k"] for c in caches_list], dim=0),
                    "v": torch.cat(
                        [c[layer_idx]["v"] for c in caches_list], dim=0),
                    "global_end_index": caches_list[0][layer_idx]["global_end_index"],
                    "local_end_index": caches_list[0][layer_idx]["local_end_index"],
                })
        return merged

    def _cat_crossattn_caches(
        self, caches_list: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Concatenate cross-attention caches along batch dimension."""
        num_layers = len(caches_list[0])
        merged = []
        for layer_idx in range(num_layers):
            merged.append({
                "k": torch.cat(
                    [c[layer_idx]["k"] for c in caches_list], dim=0),
                "v": torch.cat(
                    [c[layer_idx]["v"] for c in caches_list], dim=0),
                "is_init": caches_list[0][layer_idx]["is_init"],
            })
        return merged

    def _split_contexts(
        self,
        merged_ctx: BlockProcessingContext,
        sessions: list[UserSession],
    ) -> None:
        """Split merged KV caches back into per-user caches after forward pass."""
        n = len(sessions)

        self._split_kv_caches(merged_ctx.kv_cache1,
                              [s.ctx.kv_cache1 for s in sessions])
        if merged_ctx.kv_cache2 is not None:
            self._split_kv_caches(merged_ctx.kv_cache2,
                                  [s.ctx.kv_cache2 for s in sessions])

        self._split_crossattn_caches(merged_ctx.crossattn_cache,
                                     [s.ctx.crossattn_cache for s in sessions])

        if merged_ctx.kv_cache_mouse is not None:
            frame_seq_len = self.denoiser.frame_seq_length
            self._split_action_kv_caches(
                merged_ctx.kv_cache_mouse,
                [s.ctx.kv_cache_mouse for s in sessions],
                is_mouse=True, frame_seq_length=frame_seq_len)
        if merged_ctx.kv_cache_keyboard is not None:
            self._split_kv_caches(merged_ctx.kv_cache_keyboard,
                                  [s.ctx.kv_cache_keyboard for s in sessions])

    def _split_kv_caches(
        self,
        merged: list[dict[str, Any]],
        per_user: list[list[dict[str, Any]]],
    ) -> None:
        """Copy data from merged KV cache back to per-user caches."""
        n = len(per_user)
        for layer_idx in range(len(merged)):
            k_split = merged[layer_idx]["k"].split(1, dim=0)
            v_split = merged[layer_idx]["v"].split(1, dim=0)
            for i in range(n):
                per_user[i][layer_idx]["k"].copy_(k_split[i])
                per_user[i][layer_idx]["v"].copy_(v_split[i])
                per_user[i][layer_idx]["global_end_index"].copy_(
                    merged[layer_idx]["global_end_index"])
                per_user[i][layer_idx]["local_end_index"].copy_(
                    merged[layer_idx]["local_end_index"])

    def _split_action_kv_caches(
        self,
        merged: list[dict[str, Any]],
        per_user: list[list[dict[str, Any]]],
        is_mouse: bool = False,
        frame_seq_length: int = 1,
    ) -> None:
        """Split action KV caches back to per-user."""
        n = len(per_user)
        for layer_idx in range(len(merged)):
            if is_mouse:
                # [N*F, ...] -> N chunks of [F, ...]
                k_split = merged[layer_idx]["k"].split(frame_seq_length, dim=0)
                v_split = merged[layer_idx]["v"].split(frame_seq_length, dim=0)
            else:
                k_split = merged[layer_idx]["k"].split(1, dim=0)
                v_split = merged[layer_idx]["v"].split(1, dim=0)
            for i in range(n):
                per_user[i][layer_idx]["k"].copy_(k_split[i])
                per_user[i][layer_idx]["v"].copy_(v_split[i])
                per_user[i][layer_idx]["global_end_index"].copy_(
                    merged[layer_idx]["global_end_index"])
                per_user[i][layer_idx]["local_end_index"].copy_(
                    merged[layer_idx]["local_end_index"])

    def _split_crossattn_caches(
        self,
        merged: list[dict[str, Any]],
        per_user: list[list[dict[str, Any]]],
    ) -> None:
        """Split cross-attention caches back to per-user."""
        n = len(per_user)
        for layer_idx in range(len(merged)):
            k_split = merged[layer_idx]["k"].split(1, dim=0)
            v_split = merged[layer_idx]["v"].split(1, dim=0)
            for i in range(n):
                per_user[i][layer_idx]["k"].copy_(k_split[i])
                per_user[i][layer_idx]["v"].copy_(v_split[i])
                per_user[i][layer_idx]["is_init"] = merged[layer_idx]["is_init"]

    def _merge_action_kwargs(
        self, sessions: list[UserSession]
    ) -> dict[str, Any]:
        """Merge action kwargs by concatenating tensors along batch dim."""
        ref = sessions[0].action_kwargs
        if not ref:
            return {}

        merged: dict[str, Any] = {}
        for key in ref:
            vals = [s.action_kwargs[key] for s in sessions]
            if torch.is_tensor(vals[0]):
                merged[key] = torch.cat(vals, dim=0)
            else:
                # Scalars (like num_frame_per_block) - use ref value
                merged[key] = vals[0]
        return merged

    def _make_noise_generator(self, session: UserSession):
        """Create a noise generator using the user's pre-allocated noise pool."""
        ctx = session.ctx
        latents = session.batch.latents

        def noise_gen(shape, dtype, step_idx):
            if ctx.noise_pool is not None and step_idx < len(ctx.noise_pool):
                return ctx.noise_pool[step_idx][:, :shape[1], :, :, :].to(
                    latents.device)
            else:
                return torch.randn(shape, dtype=dtype).to(latents.device)

        return noise_gen

    def _finalize_block(self, session: UserSession) -> CompletedResult | None:
        """Finalize a completed block: update context cache + VAE decode."""
        ctx = session.ctx
        batch = session.batch
        latents = batch.latents
        start_index = session.block_start_index
        current_num_frames = session.block_num_frames

        # Write denoised latents back
        latents[:, :, start_index:start_index +
                current_num_frames, :, :] = session.current_latents

        # Update KV caches with clean context
        self.denoiser._update_context_cache(
            current_latents=session.current_latents,
            batch=batch,
            start_index=start_index,
            current_num_frames=current_num_frames,
            ctx=ctx,
            action_kwargs=session.action_kwargs,
            context_noise=ctx.context_noise,
        )

        # Advance streaming state
        old_start = ctx.start_index
        ctx.start_index += current_num_frames
        ctx.block_idx += 1

        # VAE decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        current_latents_for_vae = latents[:, :, start_index:start_index +
                                          current_num_frames, :, :]
        decoded_frames, session.vae_cache = self.decoder.streaming_decode(
            current_latents_for_vae,
            session.fastvideo_args,
            cache=session.vae_cache,
            is_first_chunk=(start_index == 0),
        )
        torch.cuda.synchronize()
        vae_ms = (time.perf_counter() - t0) * 1000

        batch.output = decoded_frames
        batch.stage_timings = {"dit_ms": session.dit_elapsed_ms, "vae_ms": vae_ms}

        # Reset denoising state
        session.denoising_step = -1
        session.current_latents = None
        session.noise_latents_btchw = None

        return CompletedResult(user_id=session.user_id, output_batch=batch)

