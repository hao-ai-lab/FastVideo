# SPDX-License-Identifier: Apache-2.0
"""Shared bidirectional and causal WanTrack sampling helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import CausalModelBase, ModelBase

_Branch = Literal["full", "no_text", "no_motion"]


def prepare_wantrack_batch(
    model: ModelBase,
    raw_batch: dict[str, Any],
    *,
    seed: int,
    latents_source: Literal["data", "zeros"] = "zeros",
) -> TrainingBatch:
    """Build inference conditions through the same path used by training."""
    augmentation = getattr(model, "track_augmentation", None)
    if augmentation is None:
        raise TypeError("WanTrack inference requires a WanTrack training model")

    generator = torch.Generator(device=model.device).manual_seed(int(seed))
    model.track_augmentation = replace(
        augmentation,
        track_dropout_probability=0.0,
        temporal_mask_probability=0.0,
        motion_dropout_probability=0.0,
        text_dropout_probability=0.0,
    )
    try:
        batch = model.prepare_batch(
            raw_batch,
            generator=generator,
            latents_source=latents_source,
        )
    finally:
        model.track_augmentation = augmentation

    # Streaming inference owns its cache/mask geometry. Training attention
    # metadata must not leak into validation or standalone sampling.
    batch.attn_metadata = None
    batch.attn_metadata_vsa = None
    return batch


def _branch_args(branch: _Branch) -> tuple[bool, dict[str, Any] | None]:
    if branch == "full":
        return True, None
    if branch == "no_text":
        return False, {
            "text": "zero",
            "track": "keep",
            "on_missing": "ignore",
        }
    if branch == "no_motion":
        return False, {
            "text": "keep",
            "track": "drop",
            "on_missing": "ignore",
        }
    raise ValueError(f"Unknown WanTrack CFG branch: {branch!r}")


def _predict(
    model: ModelBase,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    batch: TrainingBatch,
    *,
    branch: _Branch,
    cache_tag: str,
    start_frame: int,
    store_kv: bool,
) -> torch.Tensor | None:
    conditional, cfg_uncond = _branch_args(branch)
    if isinstance(model, CausalModelBase):
        return model.predict_noise_streaming(
            latents,
            timestep,
            batch,
            conditional=conditional,
            cache_tag=cache_tag,
            store_kv=store_kv,
            cur_start_frame=start_frame,
            cfg_uncond=cfg_uncond,
            attn_kind="dense",
        )
    if store_kv:
        return None
    return model.predict_noise(
        latents,
        timestep,
        batch,
        conditional=conditional,
        cfg_uncond=cfg_uncond,
        attn_kind="dense",
    )


def _guided_prediction(
    model: ModelBase,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    batch: TrainingBatch,
    *,
    start_frame: int,
    text_guidance_scale: float,
    motion_guidance_scale: float,
    motion_cfg: bool,
) -> tuple[torch.Tensor, tuple[_Branch, ...]]:
    full = _predict(
        model,
        latents,
        timestep,
        batch,
        branch="full",
        cache_tag="wantrack_full",
        start_frame=start_frame,
        store_kv=False,
    )
    if full is None:
        raise RuntimeError("WanTrack prediction unexpectedly returned None")

    text_scale = float(text_guidance_scale)
    motion_scale = float(motion_guidance_scale)
    if text_scale == 1.0 and motion_scale == 1.0:
        return full, ("full", )

    no_text = _predict(
        model,
        latents,
        timestep,
        batch,
        branch="no_text",
        cache_tag="wantrack_no_text",
        start_frame=start_frame,
        store_kv=False,
    )
    if no_text is None:
        raise RuntimeError("WanTrack no-text prediction returned None")

    if not motion_cfg:
        return no_text + text_scale * (full - no_text), (
            "full",
            "no_text",
        )

    no_motion = _predict(
        model,
        latents,
        timestep,
        batch,
        branch="no_motion",
        cache_tag="wantrack_no_motion",
        start_frame=start_frame,
        store_kv=False,
    )
    if no_motion is None:
        raise RuntimeError("WanTrack no-motion prediction returned None")

    denominator = text_scale + motion_scale
    alpha = text_scale / denominator if denominator > 0 else 0.5
    base = alpha * no_text + (1.0 - alpha) * no_motion
    guided = (base + text_scale * (full - no_text) + motion_scale * (full - no_motion))
    return guided, ("full", "no_text", "no_motion")


def _store_causal_context(
    model: CausalModelBase,
    latents: torch.Tensor,
    batch: TrainingBatch,
    *,
    start_frame: int,
    branches: tuple[_Branch, ...],
) -> None:
    timestep = torch.zeros(
        latents.shape[:2],
        device=latents.device,
        dtype=torch.float32,
    )
    batch.timesteps = timestep
    for branch in branches:
        _predict(
            model,
            latents,
            timestep,
            batch,
            branch=branch,
            cache_tag=f"wantrack_{branch}",
            start_frame=start_frame,
            store_kv=True,
        )


def _sample_block(
    model: ModelBase,
    latents: torch.Tensor,
    batch: TrainingBatch,
    *,
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int,
    start_frame: int,
    text_guidance_scale: float,
    motion_guidance_scale: float,
    motion_cfg: bool,
) -> tuple[torch.Tensor, tuple[_Branch, ...]]:
    scheduler.set_timesteps(int(num_inference_steps), device=latents.device)
    branches: tuple[_Branch, ...] = ("full", )
    for current_timestep in scheduler.timesteps:
        timestep = torch.full(
            latents.shape[:2],
            float(current_timestep.item()),
            device=latents.device,
            dtype=torch.float32,
        )
        batch.timesteps = timestep
        prediction, branches = _guided_prediction(
            model,
            latents,
            timestep,
            batch,
            start_frame=start_frame,
            text_guidance_scale=text_guidance_scale,
            motion_guidance_scale=motion_guidance_scale,
            motion_cfg=motion_cfg,
        )
        latents = scheduler.step(
            prediction.float(),
            current_timestep,
            latents.float(),
            return_dict=False,
        )[0]
    return latents, branches


@torch.no_grad()
def sample_wantrack(
    model: ModelBase,
    batch: TrainingBatch,
    *,
    num_inference_steps: int = 30,
    seed: int = 0,
    text_guidance_scale: float = 1.0,
    motion_guidance_scale: float = 1.0,
    motion_cfg: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Generate normalized latents in ``[B, T, C, H, W]`` layout.

    Bidirectional models denoise the complete clip. Causal models reuse the
    same streaming prediction and cache API as RobotWM.
    """
    if batch.latents is None or batch.latents.ndim != 5:
        raise ValueError("WanTrack inference requires [B, T, C, H, W] latents")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn(
        tuple(batch.latents.shape),
        generator=generator,
        dtype=torch.float32,
    ).to(device=batch.latents.device, dtype=batch.latents.dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(getattr(model, "timestep_shift", 5.0)), )

    if not isinstance(model, CausalModelBase):
        sampled, _ = _sample_block(
            model,
            latents,
            batch,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            start_frame=0,
            text_guidance_scale=text_guidance_scale,
            motion_guidance_scale=motion_guidance_scale,
            motion_cfg=motion_cfg,
        )
        return sampled

    if chunk_size is None:
        transformer = getattr(model, "transformer", None)
        chunk_size = int(
            getattr(
                transformer,
                "num_frame_per_block",
                getattr(transformer.config.arch_config, "num_frames_per_block", 3),
            ))
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    num_frames = int(latents.shape[1])
    if num_frames % chunk_size != 0:
        raise ValueError("Causal WanTrack inference requires latent frames "
                         f"divisible by chunk_size; got {num_frames} and "
                         f"{chunk_size}")

    for branch in ("full", "no_text", "no_motion"):
        model.clear_caches(cache_tag=f"wantrack_{branch}")

    sampled_blocks: list[torch.Tensor] = []
    try:
        for start_frame in range(0, num_frames, chunk_size):
            block = latents[:, start_frame:start_frame + chunk_size]
            block, branches = _sample_block(
                model,
                block,
                batch,
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                start_frame=start_frame,
                text_guidance_scale=text_guidance_scale,
                motion_guidance_scale=motion_guidance_scale,
                motion_cfg=motion_cfg,
            )
            sampled_blocks.append(block)
            _store_causal_context(
                model,
                block,
                batch,
                start_frame=start_frame,
                branches=branches,
            )
    finally:
        for branch in ("full", "no_text", "no_motion"):
            model.clear_caches(cache_tag=f"wantrack_{branch}")

    return torch.cat(sampled_blocks, dim=1)
