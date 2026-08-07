# SPDX-License-Identifier: Apache-2.0
"""Shared bidirectional and causal WanTrack sampling helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import CausalModelBase, ModelBase

_Branch = Literal["full", "no_text", "no_motion"]
_CACHE_BRANCHES: tuple[_Branch, ...] = (
    "full",
    "no_text",
    "no_motion",
)


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


def resolve_dmd_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    dmd_denoising_steps: list[int] | tuple[int, ...],
    *,
    warp_denoising_step: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map Self-Forcing DMD indices onto the flow-matching schedule."""
    if not dmd_denoising_steps:
        raise ValueError("dmd_denoising_steps must be a non-empty sequence")
    steps = torch.tensor([int(s) for s in dmd_denoising_steps], dtype=torch.long)
    if bool(warp_denoising_step):
        # Match WanTrackCausalDenoisingStage / Self-Forcing training warp.
        scheduler.set_timesteps(1000, device="cpu")
        schedule = torch.cat(
            (scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        num_train = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        steps = schedule[num_train - steps]
    if device is None:
        return steps
    return steps.to(device=device)


def _sample_block_euler(
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
    latent_dtype = latents.dtype
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
        )[0].to(dtype=latent_dtype)
    return latents, branches


def _sample_block_dmd(
    model: ModelBase,
    latents: torch.Tensor,
    batch: TrainingBatch,
    *,
    scheduler: FlowMatchEulerDiscreteScheduler,
    dmd_timesteps: torch.Tensor,
    start_frame: int,
    text_guidance_scale: float,
    motion_guidance_scale: float,
    motion_cfg: bool,
) -> tuple[torch.Tensor, tuple[_Branch, ...]]:
    """Self-Forcing / DMD multistep: predict x0, then re-noise to the next t."""
    latent_dtype = latents.dtype
    branches: tuple[_Branch, ...] = ("full", )
    for step_idx, current_timestep in enumerate(dmd_timesteps):
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
        pred_x0 = pred_noise_to_pred_video(
            pred_noise=prediction.flatten(0, 1).float(),
            noise_input_latent=latents.flatten(0, 1).float(),
            timestep=timestep,
            scheduler=scheduler,
        ).unflatten(0, prediction.shape[:2]).to(dtype=latent_dtype)

        if step_idx + 1 >= len(dmd_timesteps):
            latents = pred_x0
            break

        next_timestep = dmd_timesteps[step_idx + 1]
        next_t = torch.full(
            latents.shape[:2],
            float(next_timestep.item()),
            device=latents.device,
            dtype=torch.float32,
        )
        noise = torch.randn_like(pred_x0, dtype=torch.float32)
        latents = scheduler.add_noise(
            pred_x0.flatten(0, 1).float(),
            noise.flatten(0, 1),
            next_t,
        ).unflatten(0, pred_x0.shape[:2]).to(dtype=latent_dtype)
    return latents, branches


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
    dmd_denoising_steps: list[int] | tuple[int, ...] | None = None,
    warp_denoising_step: bool = True,
) -> tuple[torch.Tensor, tuple[_Branch, ...]]:
    if dmd_denoising_steps is not None:
        # Ensure the scheduler owns a 1000-step grid before x0 / add_noise.
        scheduler.set_timesteps(1000, device=latents.device)
        dmd_timesteps = resolve_dmd_timesteps(
            scheduler,
            dmd_denoising_steps,
            warp_denoising_step=warp_denoising_step,
            device=latents.device,
        )
        return _sample_block_dmd(
            model,
            latents,
            batch,
            scheduler=scheduler,
            dmd_timesteps=dmd_timesteps,
            start_frame=start_frame,
            text_guidance_scale=text_guidance_scale,
            motion_guidance_scale=motion_guidance_scale,
            motion_cfg=motion_cfg,
        )
    return _sample_block_euler(
        model,
        latents,
        batch,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        start_frame=start_frame,
        text_guidance_scale=text_guidance_scale,
        motion_guidance_scale=motion_guidance_scale,
        motion_cfg=motion_cfg,
    )


def clear_wantrack_caches(model: ModelBase) -> None:
    """Clear every CFG-tagged causal cache owned by WanTrack sampling."""
    if not isinstance(model, CausalModelBase):
        return
    for branch in _CACHE_BRANCHES:
        model.clear_caches(cache_tag=f"wantrack_{branch}")


@torch.no_grad()
def sample_wantrack_block(
    model: CausalModelBase,
    batch: TrainingBatch,
    latents: torch.Tensor,
    *,
    start_frame: int,
    num_inference_steps: int = 30,
    text_guidance_scale: float = 1.0,
    motion_guidance_scale: float = 1.0,
    motion_cfg: bool = True,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
    commit: bool = True,
    dmd_denoising_steps: list[int] | tuple[int, ...] | None = None,
    warp_denoising_step: bool = True,
) -> torch.Tensor:
    """Denoise one causal block and optionally commit its context once."""
    if latents.ndim != 5:
        raise ValueError("WanTrack block latents must be [B, T, C, H, W]")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if scheduler is None:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=float(getattr(model, "timestep_shift", 5.0)), )
    sampled, branches = _sample_block(
        model,
        latents,
        batch,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        start_frame=start_frame,
        text_guidance_scale=text_guidance_scale,
        motion_guidance_scale=motion_guidance_scale,
        motion_cfg=motion_cfg,
        dmd_denoising_steps=dmd_denoising_steps,
        warp_denoising_step=warp_denoising_step,
    )
    if commit:
        _store_causal_context(
            model,
            sampled,
            batch,
            start_frame=start_frame,
            branches=branches,
        )
    return sampled


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
    dmd_denoising_steps: list[int] | tuple[int, ...] | None = None,
    warp_denoising_step: bool = True,
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
            dmd_denoising_steps=dmd_denoising_steps,
            warp_denoising_step=warp_denoising_step,
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
    if num_frames % chunk_size == 0:
        block_sizes = [chunk_size] * (num_frames // chunk_size)
    elif (num_frames - 1) % chunk_size == 0:
        # Wan I2V clips have one leading latent frame followed by regular
        # causal blocks (for example, 31 = 1 + 10 * 3).
        block_sizes = [1] + [chunk_size] * ((num_frames - 1) // chunk_size)
    else:
        raise ValueError("Causal WanTrack inference requires latent frames "
                         "to form complete blocks, optionally after one "
                         f"leading I2V frame; got {num_frames} and "
                         f"{chunk_size}")

    clear_wantrack_caches(model)

    sampled_blocks: list[torch.Tensor] = []
    try:
        start_frame = 0
        for block_size in block_sizes:
            block = latents[:, start_frame:start_frame + block_size]
            block = sample_wantrack_block(
                model,
                batch,
                block,
                start_frame=start_frame,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale,
                motion_guidance_scale=motion_guidance_scale,
                motion_cfg=motion_cfg,
                scheduler=scheduler,
                commit=True,
                dmd_denoising_steps=dmd_denoising_steps,
                warp_denoising_step=warp_denoising_step,
            )
            sampled_blocks.append(block)
            start_frame += block_size
    finally:
        clear_wantrack_caches(model)

    return torch.cat(sampled_blocks, dim=1)
