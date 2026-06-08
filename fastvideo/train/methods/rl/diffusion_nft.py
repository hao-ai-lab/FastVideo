# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT RL method for FastVideo train.

This ports the DiffusionNFT forward-process optimization pattern while
keeping the standard FastVideo ``TrainingMethod`` contract. It supports
normal video rollouts; setting ``num_frames=1``/``num_latent_t=1`` is
just the cheap image/debug case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist

from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.rl.objectives.diffusion_nft import (
    AdvMode,
    compute_diffusion_nft_loss,
)
from fastvideo.train.methods.rl.reward.diffusion_nft import (
    build_diffusion_nft_reward_fn, )
from fastvideo.train.methods.rl.utils.rewards import move_reward_models
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
    require_bool,
    require_choice,
    require_non_negative_float,
    require_positive_int,
)
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )

logger = init_logger(__name__)


@dataclass(slots=True)
class _NFTQueuedBatch:
    clean_latents: torch.Tensor
    noisy_latents: torch.Tensor
    old_prediction: torch.Tensor
    timestep: torch.Tensor
    advantages: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    raw_latent_shape: tuple[int, ...]


class DiffusionNFTMethod(TrainingMethod):
    """Online DiffusionNFT training for Wan image/video rollouts.

    Required roles:
    - ``student``: trainable policy model.

    Optional roles:
    - ``reference``: frozen model for a prediction-space KL penalty.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if not self.student._trainable:
            raise ValueError("DiffusionNFTMethod requires student to be trainable")
        self.reference = role_models.get("reference")
        if self.reference is not None and self.reference._trainable:
            raise ValueError("DiffusionNFTMethod requires reference to be frozen")

        self._attn_kind: Literal["dense", "vsa"] = self._infer_attn_kind()
        self._media_type = self._parse_media_type()
        self._validate_rollout_setup()
        self._is_main = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

        self.student.init_preprocessors(self.training_config)
        self._sampling_scheduler = self._build_sampling_scheduler()

        self._sample_num_steps = get_optional_int(
            self.method_config,
            "sample_num_steps",
            where="method.sample_num_steps",
        )
        if self._sample_num_steps is not None and self._sample_num_steps <= 0:
            raise ValueError("method.sample_num_steps must be > 0")
        explicit_sample_timesteps = self.method_config.get("sample_timesteps")
        self._sample_timesteps_are_explicit = (explicit_sample_timesteps is not None)
        self._sample_timesteps = (self._parse_timestep_list("sample_timesteps", default=[])
                                  if self._sample_timesteps_are_explicit else self._scheduler_timesteps_for_num_steps(
                                      int(self._sample_num_steps or 16)))
        self._train_timesteps = self._build_train_timestep_list()
        self._sample_guidance_scale = require_non_negative_float(
            self.method_config,
            "sample_guidance_scale",
            default=4.5,
            where="method.sample_guidance_scale",
        )
        self._configure_student_negative_conditioning()

        samples_per_prompt = self.method_config.get(
            "num_samples_per_prompt",
            self.method_config.get("num_images_per_prompt", 2),
        )
        self._num_samples_per_prompt = int(samples_per_prompt)
        if self._num_samples_per_prompt <= 0:
            raise ValueError("method.num_samples_per_prompt must be > 0")
        self._collection_batch_size = require_positive_int(
            self.method_config,
            "collection_batch_size",
            default=self._num_samples_per_prompt,
            where="method.collection_batch_size",
        )
        self._inner_epochs = require_positive_int(
            self.method_config,
            "inner_epochs",
            default=1,
            where="method.inner_epochs",
        )
        self._nft_beta = require_non_negative_float(
            self.method_config,
            "nft_beta",
            default=1.0,
            where="method.nft_beta",
        )
        if self._nft_beta <= 0.0:
            raise ValueError("method.nft_beta must be > 0")
        self._adv_clip_max = require_non_negative_float(
            self.method_config,
            "adv_clip_max",
            default=5.0,
            where="method.adv_clip_max",
        )
        if self._adv_clip_max <= 0.0:
            raise ValueError("method.adv_clip_max must be > 0")
        self._adv_mode: AdvMode = require_choice(
            self.method_config,
            "adv_mode",
            {
                "all",
                "positive_only",
                "negative_only",
                "one_only",
                "binary",
            },
            default="all",
            where="method.adv_mode",
        )  # type: ignore[assignment]
        self._global_std = require_bool(
            self.method_config,
            "global_std",
            default=True,
            where="method.global_std",
        )
        self._kl_weight = require_non_negative_float(
            self.method_config,
            "kl_weight",
            default=0.0,
            where="method.kl_weight",
        )
        if self._kl_weight > 0.0 and self.reference is None:
            raise ValueError("method.kl_weight > 0 requires a frozen reference role")

        train_batch_size = get_optional_int(
            self.method_config,
            "train_batch_size",
            where="method.train_batch_size",
        )
        self._train_batch_size = int(train_batch_size or self.training_config.data.train_batch_size or 1)
        if self._train_batch_size <= 0:
            raise ValueError("method.train_batch_size must be > 0")
        if self._train_batch_size > self._num_samples_per_prompt:
            raise ValueError("method.train_batch_size must be <= "
                             "method.num_samples_per_prompt")
        log_sample_max_videos = get_optional_int(
            self.method_config,
            "log_sample_max_videos",
            where="method.log_sample_max_videos",
        )
        self._log_sample_max_videos = (4 if log_sample_max_videos is None else max(0, int(log_sample_max_videos)))

        self._queue: list[_NFTQueuedBatch] = []
        self._collection_round = 0
        self._pending_collection_metrics: dict[str, LogScalar] = {}
        self._latest_sample_videos: torch.Tensor | None = None
        self._latest_sample_prompts: list[str] | None = None
        self._reward_model_cfg: dict[str, float] = {}

        self._init_optimizer()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._lr_scheduler}

    def on_train_start(self) -> None:
        super().on_train_start()
        self._reward_model_cfg = self._normalize_reward_model_cfg(self.method_config.get("reward_fn"))
        self._reward_fn = build_diffusion_nft_reward_fn(
            self.method_config.get("reward_fn"),
            device=self.student.device,
            backend=str(self.method_config.get("reward_backend", "auto")),
        )

    def _normalize_reward_model_cfg(
        self,
        reward_config: Any,
    ) -> dict[str, float]:
        raw_rewards = reward_config or {}
        if isinstance(raw_rewards, dict) and "rewards" in raw_rewards:
            raw_rewards = raw_rewards["rewards"]
        if not isinstance(raw_rewards, dict):
            return {}
        return {str(key): float(value) for key, value in raw_rewards.items()}

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        del iteration
        step_t0 = time.perf_counter()
        if not self._queue:
            self._collect_round(batch)
        collect_done = time.perf_counter()

        queued = self._queue.pop(0)
        training_batch = self._make_training_batch(queued)
        self.student.transformer.train()
        self._sync_cuda()
        loss_t0 = time.perf_counter()
        loss_map, step_metrics = self._compute_nft_loss(queued, training_batch)
        self._sync_cuda()
        loss_t1 = time.perf_counter()
        step_metrics["time/nft_loss_sec"] = loss_t1 - loss_t0
        step_metrics["time/queue_wait_or_collect_sec"] = (collect_done - step_t0)
        step_metrics["queue/remaining_batches"] = float(len(self._queue))
        if self._pending_collection_metrics:
            step_metrics.update(self._pending_collection_metrics)
            self._pending_collection_metrics = {}
        outputs: dict[str, Any] = {"_fv_backward": (queued.timestep, training_batch.attn_metadata)}
        if self._is_main and self._latest_sample_videos is not None:
            outputs["sample_videos"] = self._latest_sample_videos
            outputs["sample_prompts"] = self._latest_sample_prompts or []
        return loss_map, outputs, step_metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        ctx = outputs.get("_fv_backward")
        if ctx is None:
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            return
        self.student.backward(
            loss_map["total_loss"],
            ctx,
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._lr_scheduler]

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        del iteration
        return {"student": self.student.transformer}

    def _init_optimizer(self) -> None:
        trainable_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("DiffusionNFT student transformer has no trainable "
                             "parameters.")
        tc = self.training_config
        (
            self._optimizer,
            self._lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=trainable_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=float(tc.optimizer.learning_rate),
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.student.device)

    def _configure_student_negative_conditioning(self) -> None:
        setter = getattr(self.student, "set_requires_negative_conditioning", None)
        if setter is not None:
            setter(self._sample_guidance_scale != 1.0)

    def _parse_media_type(self) -> Literal["image", "video"]:
        raw = str(self.method_config.get("media_type", "video")).strip().lower()
        if raw not in {"image", "video"}:
            raise ValueError("method.media_type must be one of image or video, got "
                             f"{raw!r}")
        return raw  # type: ignore[return-value]

    def _validate_rollout_setup(self) -> None:
        data_type = str(getattr(
            self.training_config.data,
            "preprocessed_data_type",
            "t2v",
        )).strip().lower()
        if data_type != "text_only":
            raise ValueError("DiffusionNFTMethod expects "
                             "training.data.preprocessed_data_type='text_only'")
        num_latent_t = int(self.training_config.data.num_latent_t or 0)
        num_frames = int(self.training_config.data.num_frames or 0)
        if num_latent_t <= 0:
            raise ValueError("DiffusionNFTMethod expects training.data.num_latent_t > 0")
        if num_frames <= 0:
            raise ValueError("DiffusionNFTMethod expects training.data.num_frames > 0")
        if self._media_type == "image" and (num_latent_t != 1 or num_frames != 1):
            raise ValueError("method.media_type='image' requires "
                             "training.data.num_latent_t=1 and "
                             "training.data.num_frames=1. Use media_type='video' "
                             "for multi-frame DiffusionNFT.")

    def _parse_timestep_list(
        self,
        key: str,
        *,
        default: list[int],
    ) -> torch.Tensor:
        raw = self.method_config.get(key, default)
        if not isinstance(raw, list | tuple) or not raw:
            raise ValueError(f"method.{key} must be a non-empty list")
        return torch.tensor(
            [float(x) for x in raw],
            dtype=torch.float32,
            device=self.student.device,
        )

    def _build_sampling_scheduler(self) -> FlowUniPCMultistepScheduler:
        flow_shift = get_optional_float(
            self.method_config,
            "sample_flow_shift",
            where="method.sample_flow_shift",
        )
        if flow_shift is None:
            flow_shift = float(getattr(
                self.training_config.pipeline_config,
                "flow_shift",
                3.0,
            ) or 3.0)
        return FlowUniPCMultistepScheduler(shift=float(flow_shift))

    def _scheduler_timesteps_for_num_steps(
        self,
        num_steps: int,
    ) -> torch.Tensor:
        scheduler = self._sampling_scheduler
        original_timesteps = scheduler.timesteps
        original_sigmas = scheduler.sigmas
        original_step_index = scheduler.step_index
        original_begin_index = scheduler.begin_index
        original_num_inference_steps = getattr(scheduler, "num_inference_steps", None)
        try:
            scheduler.set_timesteps(
                int(num_steps),
                device=self.student.device,
            )
            return scheduler.timesteps.detach().clone().to(
                device=self.student.device,
                dtype=torch.float32,
            )
        finally:
            scheduler.timesteps = original_timesteps
            scheduler.sigmas = original_sigmas
            scheduler._step_index = original_step_index
            scheduler._begin_index = original_begin_index
            if original_num_inference_steps is not None:
                scheduler.num_inference_steps = (original_num_inference_steps)

    def _build_train_timestep_list(self) -> torch.Tensor:
        explicit = self.method_config.get("train_timesteps")
        if explicit is not None:
            return self._nearest_training_timesteps(self._parse_timestep_list("train_timesteps", default=[]))
        fraction = get_optional_float(
            self.method_config,
            "train_timestep_fraction",
            where="method.train_timestep_fraction",
        )
        fraction = 1.0 if fraction is None else float(fraction)
        if fraction <= 0.0:
            raise ValueError("method.train_timestep_fraction must be > 0")
        count = max(
            1,
            min(
                len(self._sample_timesteps),
                int(len(self._sample_timesteps) * fraction),
            ),
        )
        return self._nearest_training_timesteps(self._sample_timesteps[:count])

    def _nearest_training_timesteps(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        base_timesteps = self.student.noise_scheduler.timesteps.to(
            device=self.student.device,
            dtype=torch.float32,
        )
        requested = timesteps.to(
            device=self.student.device,
            dtype=torch.float32,
        )
        indices = torch.argmin(
            (base_timesteps.unsqueeze(0) - requested.unsqueeze(1)).abs(),
            dim=1,
        )
        return base_timesteps.index_select(0, indices)

    def _repeat_raw_batch(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
        repeat = self._num_samples_per_prompt
        result: dict[str, Any] = {}
        for key, value in raw_batch.items():
            if torch.is_tensor(value) and value.shape[:1]:
                result[key] = value.repeat_interleave(repeat, dim=0)
            elif key == "info_list" and isinstance(value, list):
                result[key] = [dict(item) for item in value for _ in range(repeat)]
            elif key == "caption_text" and isinstance(value, list):
                result[key] = [str(item) for item in value for _ in range(repeat)]
            else:
                result[key] = value

        info_list = result.get("info_list") or []
        if not isinstance(info_list, list):
            raise ValueError("text-only dataloader must provide info_list")
        prompts = [str(info.get("prompt") or info.get("caption") or "") for info in info_list]
        metadata = [dict(info) for info in info_list]
        if any(not prompt for prompt in prompts):
            raise ValueError("text-only dataloader rows must include caption/prompt "
                             "metadata")
        return result, prompts, metadata

    @torch.no_grad()
    def _collect_round(self, raw_batch: dict[str, Any]) -> None:
        self._collection_round += 1
        timings: dict[str, float] = {}
        self._sync_cuda()
        round_t0 = time.perf_counter()

        t0 = time.perf_counter()
        repeated_batch, prompts, metadata = self._repeat_raw_batch(raw_batch)
        self._sync_cuda()
        timings["time/collection_prepare_batch_sec"] = (time.perf_counter() - t0)

        sample_batches: list[TrainingBatch] = []
        clean_chunks: list[torch.Tensor] = []
        reward_chunks: dict[str, list[torch.Tensor]] = {}
        media_means: list[torch.Tensor] = []
        media_stds: list[torch.Tensor] = []
        clean_latent_stds: list[torch.Tensor] = []
        timings["time/collection_sample_sec"] = 0.0
        timings["time/collection_decode_sec"] = 0.0
        timings["time/collection_reward_sec"] = 0.0
        captured_samples = False
        collection_batch_size = min(self._collection_batch_size, len(prompts))
        for start in range(0, len(prompts), collection_batch_size):
            end = min(start + collection_batch_size, len(prompts))
            chunk_batch = self._slice_raw_batch(repeated_batch, start, end)
            sample_batch = self.student.prepare_batch(
                chunk_batch,
                generator=self.cuda_generator,
                latents_source="zeros",
            )
            sample_batches.append(sample_batch)

            t0 = time.perf_counter()
            clean = self._sample_clean_latents(sample_batch)
            self._sync_cuda()
            timings["time/collection_sample_sec"] += (time.perf_counter() - t0)
            clean_chunks.append(clean)
            clean_latent_stds.append(clean.detach().float().std())

            t0 = time.perf_counter()
            media = self._decode_media(clean)
            self._sync_cuda()
            timings["time/collection_decode_sec"] += (time.perf_counter() - t0)
            media_float = media.detach().float()
            media_means.append(media_float.mean())
            media_stds.append(media_float.std())
            if (self._is_main and self._log_sample_max_videos > 0 and not captured_samples):
                self._capture_sample_media(
                    media,
                    prompts[start:end],
                )
                captured_samples = True

            try:
                move_reward_models(
                    self._reward_model_cfg,
                    self.student.device,
                )
                t0 = time.perf_counter()
                chunk_rewards = self._reward_fn(
                    media,
                    prompts[start:end],
                    metadata[start:end],
                )
                self._sync_cuda()
                timings["time/collection_reward_sec"] += (time.perf_counter() - t0)
                for key, value in chunk_rewards.items():
                    reward_chunks.setdefault(key, []).append(value.detach().to(device=self.student.device))
            finally:
                move_reward_models(self._reward_model_cfg, "cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        sample_batch = self._concat_sample_batches(sample_batches)
        clean_latents = torch.cat(clean_chunks, dim=0)
        reward_details = {key: torch.cat(values, dim=0) for key, values in reward_chunks.items()}

        t0 = time.perf_counter()
        advantages, reward_metrics = self._compute_advantages(prompts, reward_details)
        self._sync_cuda()
        timings["time/collection_advantages_sec"] = (time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._queue = self._build_training_queue(
            sample_batch,
            clean_latents,
            advantages,
        )
        self._sync_cuda()
        timings["time/collection_queue_build_sec"] = (time.perf_counter() - t0)
        timings["time/collection_total_sec"] = (time.perf_counter() - round_t0)
        self._pending_collection_metrics = {
            "collection/round": float(self._collection_round),
            "collection/queued_batches": float(len(self._queue)),
            "collection/sample_batches": float(len(sample_batches)),
            "sample/media_mean": self._mean_metric(media_means),
            "sample/media_std": self._mean_metric(media_stds),
            "sample/clean_latent_std": self._mean_metric(clean_latent_stds),
            **reward_metrics,
            **timings,
        }
        logger.info(
            "DiffusionNFT collection %d queued %d batches: "
            "reward/avg=%.6f reward/avg_std=%.6f total=%.1fs "
            "sample=%.1fs decode=%.1fs reward=%.1fs queue=%.1fs",
            self._collection_round,
            len(self._queue),
            float(reward_metrics.get("reward/avg", 0.0)),
            float(reward_metrics.get("reward/avg_std", 0.0)),
            timings["time/collection_total_sec"],
            timings["time/collection_sample_sec"],
            timings["time/collection_decode_sec"],
            timings["time/collection_reward_sec"],
            timings["time/collection_queue_build_sec"],
        )

    def _mean_metric(self, values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        stacked = torch.stack([value.detach().float() for value in values])
        mean, _std = self._distributed_mean_std(stacked)
        return mean

    def _capture_sample_media(
        self,
        media: torch.Tensor,
        prompts: list[str],
    ) -> None:
        n = min(int(media.shape[0]), self._log_sample_max_videos)
        if n <= 0:
            return
        samples = media[:n].detach().float().clamp(0, 1)
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        if samples.ndim != 5:
            raise ValueError("DiffusionNFT sample logging expected media with shape "
                             f"(B,C,H,W) or (B,C,F,H,W), got {tuple(samples.shape)}")
        self._latest_sample_videos = ((samples * 255.0).round().to(torch.uint8).cpu())
        self._latest_sample_prompts = [str(p) for p in prompts[:n]]

    def _slice_raw_batch(
        self,
        raw_batch: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in raw_batch.items():
            if torch.is_tensor(value) and value.shape[:1] or isinstance(value, list) and len(value) >= end:
                result[key] = value[start:end]
            else:
                result[key] = value
        return result

    def _concat_sample_batches(self, batches: list[TrainingBatch]) -> TrainingBatch:
        if not batches:
            raise RuntimeError("No sample batches were collected")
        raw_shape = batches[0].raw_latent_shape
        embeds = [batch.encoder_hidden_states for batch in batches]
        masks = [batch.encoder_attention_mask for batch in batches]
        if raw_shape is None or any(batch.raw_latent_shape is None or batch.raw_latent_shape[1:] != raw_shape[1:]
                                    for batch in batches):
            raise RuntimeError("Collected sample batches changed latent shape")
        if any(embed is None for embed in embeds) or any(mask is None for mask in masks):
            raise RuntimeError("Collected sample batch missing text embeds")
        batch = TrainingBatch()
        batch.encoder_hidden_states = torch.cat([embed for embed in embeds if embed is not None], dim=0)
        batch.encoder_attention_mask = torch.cat([mask for mask in masks if mask is not None], dim=0)
        batch.raw_latent_shape = (
            batch.encoder_hidden_states.shape[0],
            *raw_shape[1:],
        )
        return batch

    @torch.no_grad()
    def _sample_clean_latents(self, batch: TrainingBatch) -> torch.Tensor:
        if batch.latents is None:
            raise RuntimeError("prepare_batch() did not create latents")
        raw_latent_shape = batch.raw_latent_shape
        if raw_latent_shape is None:
            raise RuntimeError("prepare_batch() did not set raw_latent_shape")

        model_dtype = batch.latents.dtype
        latents = torch.randn(
            batch.latents.shape,
            device=batch.latents.device,
            dtype=torch.float32,
            generator=self.cuda_generator,
        )

        scheduler = self._sampling_scheduler
        original_timesteps = scheduler.timesteps
        original_sigmas = scheduler.sigmas
        original_step_index = scheduler.step_index
        original_begin_index = scheduler.begin_index
        original_num_inference_steps = getattr(scheduler, "num_inference_steps", None)

        try:
            if self._sample_timesteps_are_explicit:
                raise ValueError("method.sample_timesteps is not supported with "
                                 "Wan UniPC sampling. Use method.sample_num_steps "
                                 "instead.")
            scheduler.set_timesteps(
                int(self._sample_num_steps or len(self._sample_timesteps)),
                device=latents.device,
            )
            scheduler.set_begin_index(0)
            timesteps = scheduler.timesteps.to(device=latents.device)

            self.student.transformer.eval()
            for step_index, timestep in enumerate(timesteps):
                timestep_batch = timestep.expand(latents.shape[0]).to(device=latents.device)
                batch.timesteps = timestep_batch
                batch.raw_latent_shape = raw_latent_shape
                pred = self._guided_predict_noise_for_sampling(
                    latents.to(model_dtype),
                    timestep_batch,
                    batch,
                    step_index=step_index,
                    guidance_scale=self._sample_guidance_scale,
                )
                latents = scheduler.step(
                    pred.float(),
                    timestep,
                    latents.float(),
                    return_dict=False,
                )[0]
            return latents.detach()
        finally:
            scheduler.timesteps = original_timesteps
            scheduler.sigmas = original_sigmas
            scheduler._step_index = original_step_index
            scheduler._begin_index = original_begin_index
            if original_num_inference_steps is not None:
                scheduler.num_inference_steps = (original_num_inference_steps)

    @torch.no_grad()
    def _guided_predict_noise_for_sampling(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        step_index: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        if self._attn_kind != "dense":
            return self._guided_predict_noise(
                self.student,
                latents,
                timestep,
                batch,
                guidance_scale=guidance_scale,
            )

        cond = batch.conditional_dict
        if cond is None:
            raise RuntimeError("Missing conditional_dict in TrainingBatch")
        dtype = latents.dtype
        device_type = self.student.device.type
        hidden_states = latents.permute(0, 2, 1, 3, 4)

        def predict(text_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            with (
                    torch.autocast(device_type, dtype=dtype),
                    set_forward_context(
                        current_timestep=step_index,
                        attn_metadata=None,
                    ),
            ):
                out = self.student.transformer(
                    hidden_states.to(dtype),
                    text_dict["encoder_hidden_states"].to(dtype),
                    timestep,
                )
            return out.permute(0, 2, 1, 3, 4)

        if guidance_scale == 1.0:
            return predict(cond)

        uncond = getattr(batch, "unconditional_dict", None)
        if uncond is None:
            raise RuntimeError("Missing unconditional_dict; negative conditioning "
                               "may have failed")
        cond_pred = predict(cond)
        uncond_pred = predict(uncond)
        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    @torch.no_grad()
    def _guided_predict_noise(
        self,
        model: ModelBase,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return model.predict_noise(
                latents,
                timestep,
                batch,
                conditional=True,
                attn_kind=self._attn_kind,
            )
        cond = model.predict_noise(
            latents,
            timestep,
            batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )
        uncond = model.predict_noise(
            latents,
            timestep,
            batch,
            conditional=False,
            attn_kind=self._attn_kind,
        )
        return uncond + guidance_scale * (cond - uncond)

    @torch.no_grad()
    def _decode_media(self, clean_latents: torch.Tensor) -> torch.Tensor:
        """Decode clean latents to image or video reward inputs.

        Returns:
            ``media_type=image``: ``(B, C, H, W)`` first-frame tensor.
            ``media_type=video``: ``(B, C, F, H, W)`` video tensor.
        """
        latents = clean_latents.permute(0, 2, 1, 3, 4).to(self.student.device)
        decoded = self.student.decode_latents(latents)
        if self._media_type == "image":
            return decoded[:, :, 0].contiguous()
        return decoded.contiguous()

    def _compute_advantages(
        self,
        prompts: list[str],
        reward_details: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, LogScalar]]:
        local_rewards = reward_details["avg"].detach().float()
        rewards = local_rewards
        all_prompts = prompts
        rank = 0
        rank_lengths = [len(prompts)]
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_prompts: list[list[str] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_prompts, prompts)
            gathered_rewards: list[list[float] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered_rewards,
                local_rewards.detach().cpu().tolist(),
            )
            all_prompts = [prompt for rank_prompts in gathered_prompts for prompt in (rank_prompts or [])]
            rank_lengths = [len(rank_prompts or []) for rank_prompts in gathered_prompts]
            rewards = torch.tensor(
                [score for rank_rewards in gathered_rewards for score in (rank_rewards or [])],
                device=local_rewards.device,
                dtype=torch.float32,
            )
        advantages = torch.zeros_like(rewards)
        prompt_array = np.array(all_prompts)
        prompt_stds: list[float] = []
        global_std = rewards.std(unbiased=False).clamp_min(1e-4)
        for prompt in np.unique(prompt_array):
            indices_np = np.nonzero(prompt_array == prompt)[0]
            indices = torch.as_tensor(indices_np, device=rewards.device, dtype=torch.long)
            prompt_rewards = rewards.index_select(0, indices)
            mean = prompt_rewards.mean()
            std = (global_std if self._global_std else prompt_rewards.std(unbiased=False).clamp_min(1e-4))
            prompt_stds.append(float(prompt_rewards.std(unbiased=False).detach().cpu()))
            advantages.index_copy_(0, indices, (prompt_rewards - mean) / std)

        start = sum(rank_lengths[:rank])
        end = start + rank_lengths[rank]
        local_advantages = advantages[start:end].to(local_rewards.device)
        return local_advantages, self._reward_metrics(reward_details, prompt_stds)

    def _reward_metrics(
        self,
        reward_details: dict[str, torch.Tensor],
        prompt_stds: list[float],
    ) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}
        for key, value in reward_details.items():
            mean, std = self._distributed_mean_std(value.detach().float())
            metrics[f"reward/{key}"] = mean
            metrics[f"reward/{key}_std"] = std

        prompt_stds_np = np.asarray(prompt_stds, dtype=np.float64)
        if prompt_stds_np.size:
            metrics["reward/zero_std_ratio"] = float(np.count_nonzero(prompt_stds_np == 0.0) / prompt_stds_np.size)
            metrics["reward/prompt_std_mean"] = float(prompt_stds_np.mean())
        else:
            metrics["reward/zero_std_ratio"] = 0.0
            metrics["reward/prompt_std_mean"] = 0.0
        return metrics

    def _distributed_mean_std(self, value: torch.Tensor) -> tuple[float, float]:
        value = value.float()
        local = torch.stack([
            value.sum(),
            (value * value).sum(),
            torch.tensor(
                float(value.numel()),
                device=value.device,
                dtype=value.dtype,
            ),
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        count = local[2].clamp_min(1.0)
        mean = local[0] / count
        var = (local[1] / count - mean * mean).clamp_min(0.0)
        return (
            float(mean.detach().cpu()),
            float(var.sqrt().detach().cpu()),
        )

    @torch.no_grad()
    def _build_training_queue(
        self,
        sample_batch: TrainingBatch,
        clean_latents: torch.Tensor,
        advantages: torch.Tensor,
    ) -> list[_NFTQueuedBatch]:
        if (sample_batch.encoder_hidden_states is None or sample_batch.encoder_attention_mask is None):
            raise RuntimeError("sample_batch is missing text conditioning")
        raw_shape = sample_batch.raw_latent_shape
        if raw_shape is None:
            raise RuntimeError("sample_batch is missing raw_latent_shape")

        queue: list[_NFTQueuedBatch] = []
        total = clean_latents.shape[0]
        train_timesteps = self._train_timesteps.to(device=clean_latents.device)
        for _inner_epoch in range(self._inner_epochs):
            perm = torch.randperm(
                total,
                device=clean_latents.device,
                generator=self.cuda_generator,
            )
            timestep_perm = torch.randperm(
                len(train_timesteps),
                device=clean_latents.device,
                generator=self.cuda_generator,
            )
            for start in range(0, total, self._train_batch_size):
                idx = perm[start:start + self._train_batch_size]
                clean = clean_latents.index_select(0, idx).detach()
                embeds = (sample_batch.encoder_hidden_states.index_select(0, idx).detach())
                mask = (sample_batch.encoder_attention_mask.index_select(0, idx).detach())
                clean = clean.to(dtype=embeds.dtype)
                adv = advantages.index_select(0, idx).detach()
                for timestep_idx in timestep_perm:
                    timestep_value = train_timesteps[timestep_idx]
                    timestep = timestep_value.expand(clean.shape[0])
                    noise = torch.randn(
                        clean.shape,
                        device=clean.device,
                        dtype=clean.dtype,
                        generator=self.cuda_generator,
                    )
                    noisy = self.student.add_noise(clean, noise, timestep)
                    old_batch = self._make_training_batch_from_tensors(
                        embeds,
                        mask,
                        timestep,
                        raw_shape,
                    )
                    old_pred = self.student.predict_noise(
                        noisy,
                        timestep,
                        old_batch,
                        conditional=True,
                        attn_kind=self._attn_kind,
                    ).detach()
                    queue.append(
                        _NFTQueuedBatch(
                            clean_latents=clean,
                            noisy_latents=noisy.detach(),
                            old_prediction=old_pred,
                            timestep=timestep.detach(),
                            advantages=adv,
                            encoder_hidden_states=embeds,
                            encoder_attention_mask=mask,
                            raw_latent_shape=raw_shape,
                        ))
        return queue

    def _make_training_batch(self, queued: _NFTQueuedBatch) -> TrainingBatch:
        return self._make_training_batch_from_tensors(
            queued.encoder_hidden_states,
            queued.encoder_attention_mask,
            queued.timestep,
            queued.raw_latent_shape,
        )

    def _make_training_batch_from_tensors(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep: torch.Tensor,
        raw_latent_shape: tuple[int, ...],
    ) -> TrainingBatch:
        batch = TrainingBatch()
        batch.encoder_hidden_states = encoder_hidden_states
        batch.encoder_attention_mask = encoder_attention_mask
        batch.timesteps = timestep
        batch.raw_latent_shape = (
            encoder_hidden_states.shape[0],
            *raw_latent_shape[1:],
        )
        batch.conditional_dict = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        batch = self.student._build_attention_metadata(batch)  # type: ignore[attr-defined]
        batch.attn_metadata_vsa = batch.attn_metadata
        return batch

    def _compute_nft_loss(
        self,
        queued: _NFTQueuedBatch,
        training_batch: TrainingBatch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
        forward_prediction = self.student.predict_noise(
            queued.noisy_latents,
            queued.timestep,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )

        reference_prediction = None
        if self.reference is not None and self._kl_weight > 0.0:
            with torch.no_grad():
                reference_prediction = self.reference.predict_noise(
                    queued.noisy_latents,
                    queued.timestep,
                    training_batch,
                    conditional=True,
                    attn_kind=self._attn_kind,
                )

        return compute_diffusion_nft_loss(
            student=self.student,
            forward_prediction=forward_prediction,
            old_prediction=queued.old_prediction,
            noisy_latents=queued.noisy_latents,
            clean_latents=queued.clean_latents,
            timestep=queued.timestep,
            advantages=queued.advantages,
            adv_clip_max=self._adv_clip_max,
            adv_mode=self._adv_mode,
            nft_beta=self._nft_beta,
            reference_prediction=reference_prediction,
            kl_weight=self._kl_weight,
        )
