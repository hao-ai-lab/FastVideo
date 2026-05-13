# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT method for FastVideo train.

This ports the forward-process optimization pattern from DiffusionNFT while
keeping the FastVideo trainer contract intact: ``training.loop.max_train_steps``
counts optimizer updates, and DiffusionNFT data collection is an internal
method-level collection round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.diffusion_nft.rewards import build_reward_fn
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
    """Online DiffusionNFT training for text-only Wan image rollouts.

    Required roles:
    - ``student``: trainable policy model.

    Optional roles:
    - ``reference``: frozen model for a KL penalty. If omitted, KL is disabled.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("DiffusionNFTMethod requires role 'student'")
        if not self.student._trainable:
            raise ValueError("DiffusionNFTMethod requires student to be trainable")

        self.reference = role_models.get("reference")
        if self.reference is not None and self.reference._trainable:
            raise ValueError("DiffusionNFTMethod requires reference to be non-trainable")

        self._attn_kind: Literal["dense", "vsa"] = "dense"
        self._validate_text_only_image_setup()

        self.student.init_preprocessors(self.training_config)

        self._sample_timesteps = self._parse_timestep_list("sample_timesteps", default=[1000, 750, 500, 250])
        self._train_timesteps = self._build_train_timestep_list()
        self._sample_guidance_scale = require_non_negative_float(
            self.method_config,
            "sample_guidance_scale",
            default=1.0,
            where="method.sample_guidance_scale",
        )
        self._configure_student_negative_conditioning()
        self._num_images_per_prompt = require_positive_int(
            self.method_config,
            "num_images_per_prompt",
            default=2,
            where="method.num_images_per_prompt",
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
        self._adv_mode = require_choice(
            self.method_config,
            "adv_mode",
            {"all", "positive_only", "negative_only", "one_only", "binary"},
            default="all",
            where="method.adv_mode",
        )
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
            raise ValueError("method.kl_weight > 0 requires a frozen 'reference' role")

        train_batch_size = get_optional_int(
            self.method_config,
            "train_batch_size",
            where="method.train_batch_size",
        )
        self._train_batch_size = int(train_batch_size or self.training_config.data.train_batch_size or 1)
        if self._train_batch_size <= 0:
            raise ValueError("method.train_batch_size must be > 0")

        self._queue: list[_NFTQueuedBatch] = []
        self._collection_round = 0
        self._pending_collection_metrics: dict[str, LogScalar] = {}

        self._init_optimizers_and_schedulers()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def on_train_start(self) -> None:
        super().on_train_start()
        self._reward_fn = build_reward_fn(
            self.method_config.get("reward_fn"),
            device=self.student.device,
        )

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
        if not self._queue:
            self._collect_round(batch)

        queued = self._queue.pop(0)
        training_batch = self._make_training_batch(queued)
        self.student.transformer.train()
        loss_map, step_metrics = self._compute_nft_loss(queued, training_batch)
        if self._pending_collection_metrics:
            step_metrics.update(self._pending_collection_metrics)
            self._pending_collection_metrics = {}
        return loss_map, {"_fv_backward": (queued.timestep, training_batch.attn_metadata)}, step_metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        ctx = outputs.get("_fv_backward")
        if ctx is None:
            super().backward(loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return
        self.student.backward(
            loss_map["total_loss"],
            ctx,
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def get_grad_clip_targets(self, iteration: int) -> dict[str, torch.nn.Module]:
        del iteration
        return {"student": self.student.transformer}

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.optimizer.learning_rate must be > 0 for DiffusionNFT")
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    def _configure_student_negative_conditioning(self) -> None:
        setter = getattr(self.student, "set_requires_negative_conditioning", None)
        if setter is not None:
            setter(self._sample_guidance_scale != 1.0)

    def _validate_text_only_image_setup(self) -> None:
        data_type = str(getattr(
            self.training_config.data,
            "preprocessed_data_type",
            "t2v",
        )).strip().lower()
        if data_type != "text_only":
            raise ValueError("DiffusionNFTMethod expects training.data.preprocessed_data_type='text_only'")
        if int(self.training_config.data.num_latent_t or 0) != 1:
            raise ValueError("DiffusionNFTMethod uses Wan as an image generator; set training.data.num_latent_t=1")
        if int(self.training_config.data.num_frames or 0) != 1:
            raise ValueError("DiffusionNFTMethod uses Wan as an image generator; set training.data.num_frames=1")

    def _parse_timestep_list(
        self,
        key: str,
        *,
        default: list[int],
    ) -> torch.Tensor:
        raw = self.method_config.get(key, default)
        if not isinstance(raw, list | tuple) or not raw:
            raise ValueError(f"method.{key} must be a non-empty list")
        return torch.tensor([int(x) for x in raw], dtype=torch.long, device=self.student.device)

    def _build_train_timestep_list(self) -> torch.Tensor:
        explicit = self.method_config.get("train_timesteps")
        if explicit is not None:
            return self._parse_timestep_list("train_timesteps", default=[])
        fraction = get_optional_float(
            self.method_config,
            "train_timestep_fraction",
            where="method.train_timestep_fraction",
        )
        if fraction is None:
            fraction = 1.0
        if fraction <= 0.0:
            raise ValueError("method.train_timestep_fraction must be > 0")
        count = max(1, min(len(self._sample_timesteps), int(len(self._sample_timesteps) * float(fraction))))
        return self._sample_timesteps[:count]

    def _repeat_raw_batch(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
        repeat = self._num_images_per_prompt
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
            raise ValueError("text-only dataloader rows must include caption/prompt metadata")
        return result, prompts, metadata

    @torch.no_grad()
    def _collect_round(self, raw_batch: dict[str, Any]) -> None:
        self._collection_round += 1
        repeated_batch, prompts, metadata = self._repeat_raw_batch(raw_batch)
        sample_batch = self.student.prepare_batch(
            repeated_batch,
            generator=self.cuda_generator,
            latents_source="zeros",
        )

        clean_latents = self._sample_clean_latents(sample_batch)
        images = self._decode_first_frame(clean_latents)
        reward_details = self._reward_fn(images, prompts, metadata)
        advantages, reward_metrics = self._compute_advantages(prompts, reward_details)

        self._queue = self._build_training_queue(
            sample_batch,
            clean_latents,
            advantages,
        )
        self._pending_collection_metrics = {
            "collection/round": float(self._collection_round),
            "collection/queued_batches": float(len(self._queue)),
            **reward_metrics,
        }
        logger.info(
            "DiffusionNFT collection %d queued %d batches: reward/avg=%.6f reward/avg_std=%.6f",
            self._collection_round,
            len(self._queue),
            float(reward_metrics.get("reward/avg", 0.0)),
            float(reward_metrics.get("reward/avg_std", 0.0)),
        )

    @torch.no_grad()
    def _sample_clean_latents(self, batch: TrainingBatch) -> torch.Tensor:
        if batch.latents is None:
            raise RuntimeError("prepare_batch() did not create latents")
        latents = torch.randn(
            batch.latents.shape,
            device=batch.latents.device,
            dtype=batch.latents.dtype,
            generator=self.cuda_generator,
        )
        raw_latent_shape = batch.raw_latent_shape
        if raw_latent_shape is None:
            raise RuntimeError("prepare_batch() did not set raw_latent_shape")

        self.student.transformer.eval()
        for i, timestep in enumerate(self._sample_timesteps):
            timestep_batch = timestep.expand(latents.shape[0]).to(device=latents.device)
            batch.timesteps = timestep_batch
            batch.raw_latent_shape = raw_latent_shape
            pred = self._guided_predict_noise(
                self.student,
                latents,
                timestep_batch,
                batch,
                guidance_scale=self._sample_guidance_scale,
            )
            pred_x0 = self._pred_noise_to_x0(pred, latents, timestep_batch)
            if i < len(self._sample_timesteps) - 1:
                next_timestep = self._sample_timesteps[i + 1].expand(latents.shape[0]).to(device=latents.device)
                noise = torch.randn(
                    pred_x0.shape,
                    device=pred_x0.device,
                    dtype=pred_x0.dtype,
                    generator=self.cuda_generator,
                )
                latents = self.student.add_noise(pred_x0, noise, next_timestep)
            else:
                latents = pred_x0
        return latents.detach()

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

    def _pred_noise_to_x0(
        self,
        pred_noise: torch.Tensor,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latents.flatten(0, 1),
            timestep=timestep,
            scheduler=self.student.noise_scheduler,
        ).unflatten(0, pred_noise.shape[:2])

    @torch.no_grad()
    def _decode_first_frame(self, clean_latents: torch.Tensor) -> torch.Tensor:
        vae = self.student.vae
        latents = clean_latents.permute(0, 2, 1, 3, 4).to(self.student.device)
        cfg = getattr(vae, "config", None)
        if not bool(getattr(vae, "handles_latent_denorm", False)):
            if cfg is not None and hasattr(cfg, "latents_mean") and hasattr(cfg, "latents_std"):
                latents_mean = torch.tensor(cfg.latents_mean, device=latents.device,
                                            dtype=latents.dtype).view(1, -1, 1, 1, 1)
                latents_std = torch.tensor(cfg.latents_std, device=latents.device,
                                           dtype=latents.dtype).view(1, -1, 1, 1, 1)
                latents = latents * latents_std + latents_mean
            elif hasattr(vae, "scaling_factor"):
                scaling_factor = vae.scaling_factor
                if torch.is_tensor(scaling_factor):
                    latents = latents / scaling_factor.to(latents.device, latents.dtype)
                else:
                    latents = latents / scaling_factor
                shift_factor = getattr(vae, "shift_factor", None)
                if shift_factor is not None:
                    if torch.is_tensor(shift_factor):
                        latents = latents + shift_factor.to(latents.device, latents.dtype)
                    else:
                        latents = latents + shift_factor

        vae = vae.to(self.student.device)
        with torch.autocast(device_type=self.student.device.type, dtype=torch.bfloat16):
            decoded = vae.decode(latents)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        return decoded[:, :, 0].contiguous()

    def _compute_advantages(
        self,
        prompts: list[str],
        reward_details: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, LogScalar]]:
        rewards = reward_details["avg"].detach().float()
        advantages = torch.zeros_like(rewards)
        prompt_array = np.array(prompts)
        unique_prompts = np.unique(prompt_array)
        prompt_stds: list[float] = []
        global_std = rewards.std(unbiased=False).clamp_min(1e-4)
        for prompt in unique_prompts:
            indices_np = np.nonzero(prompt_array == prompt)[0]
            indices = torch.as_tensor(indices_np, device=rewards.device, dtype=torch.long)
            prompt_rewards = rewards.index_select(0, indices)
            mean = prompt_rewards.mean()
            std = global_std if self._global_std else prompt_rewards.std(unbiased=False).clamp_min(1e-4)
            prompt_stds.append(float(prompt_rewards.std(unbiased=False).detach().cpu()))
            advantages.index_copy_(0, indices, (prompt_rewards - mean) / std)

        metrics = self._reward_metrics(reward_details, prompt_stds)
        return advantages, metrics

    def _reward_metrics(
        self,
        reward_details: dict[str, torch.Tensor],
        prompt_stds: list[float],
    ) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}
        for key, value in reward_details.items():
            value = value.detach().float()
            mean, std = self._distributed_mean_std(value)
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
        local = torch.stack([
            value.mean(),
            value.var(unbiased=False),
            torch.ones((), device=value.device, dtype=value.dtype),
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            world = float(dist.get_world_size())
            mean = local[0] / world
            var = local[1] / world
        else:
            mean = local[0]
            var = local[1]
        return float(mean.detach().cpu()), float(var.clamp_min(0).sqrt().detach().cpu())

    @torch.no_grad()
    def _build_training_queue(
        self,
        sample_batch: TrainingBatch,
        clean_latents: torch.Tensor,
        advantages: torch.Tensor,
    ) -> list[_NFTQueuedBatch]:
        if sample_batch.encoder_hidden_states is None or sample_batch.encoder_attention_mask is None:
            raise RuntimeError("sample_batch is missing text conditioning")
        raw_shape = sample_batch.raw_latent_shape
        if raw_shape is None:
            raise RuntimeError("sample_batch is missing raw_latent_shape")

        queue: list[_NFTQueuedBatch] = []
        total = clean_latents.shape[0]
        for _inner_epoch in range(self._inner_epochs):
            perm = torch.randperm(
                total,
                device=clean_latents.device,
                generator=self.cuda_generator,
            )
            for start in range(0, total, self._train_batch_size):
                idx = perm[start:start + self._train_batch_size]
                clean = clean_latents.index_select(0, idx).detach()
                embeds = sample_batch.encoder_hidden_states.index_select(0, idx).detach()
                mask = sample_batch.encoder_attention_mask.index_select(0, idx).detach()
                adv = advantages.index_select(0, idx).detach()
                timestep = self._sample_train_timestep(clean.shape[0], clean.device)
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

    def _sample_train_timestep(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        idx = torch.randint(
            0,
            len(self._train_timesteps),
            (batch_size, ),
            device=device,
            generator=self.cuda_generator,
        )
        return self._train_timesteps.to(device=device).index_select(0, idx)

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
        batch.raw_latent_shape = raw_latent_shape
        batch.conditional_dict = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        batch = self.student._build_attention_metadata(batch)  # type: ignore[attr-defined]
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
        old_prediction = queued.old_prediction

        advantages = self._shape_advantages(queued.advantages)
        r = ((advantages / self._adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

        positive_prediction = self._nft_beta * forward_prediction + (1.0 - self._nft_beta) * old_prediction.detach()
        negative_prediction = (1.0 + self._nft_beta) * old_prediction.detach() - self._nft_beta * forward_prediction

        positive_x0 = self._pred_noise_to_x0(positive_prediction, queued.noisy_latents, queued.timestep)
        negative_x0 = self._pred_noise_to_x0(negative_prediction, queued.noisy_latents, queued.timestep)

        reduce_dims = tuple(range(1, queued.clean_latents.ndim))
        with torch.no_grad():
            positive_weight = (positive_x0.float() - queued.clean_latents.float()).abs().mean(
                dim=reduce_dims,
                keepdim=True,
            ).clamp_min(1e-5)
            negative_weight = (negative_x0.float() - queued.clean_latents.float()).abs().mean(
                dim=reduce_dims,
                keepdim=True,
            ).clamp_min(1e-5)

        positive_loss = ((positive_x0 - queued.clean_latents)**2 / positive_weight).mean(dim=reduce_dims)
        negative_loss = ((negative_x0 - queued.clean_latents)**2 / negative_weight).mean(dim=reduce_dims)
        unweighted_policy_loss = (r * positive_loss / self._nft_beta +
                                  (1.0 - r) * negative_loss / self._nft_beta)
        policy_loss = (unweighted_policy_loss * self._adv_clip_max).mean()

        kl_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
        if self.reference is not None and self._kl_weight > 0.0:
            with torch.no_grad():
                ref_prediction = self.reference.predict_noise(
                    queued.noisy_latents,
                    queued.timestep,
                    training_batch,
                    conditional=True,
                    attn_kind=self._attn_kind,
                )
            kl_loss = F.mse_loss(forward_prediction.float(), ref_prediction.float())

        total_loss = policy_loss + self._kl_weight * kl_loss
        loss_map = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
        }
        metrics: dict[str, LogScalar] = {
            "nft/unweighted_policy_loss": unweighted_policy_loss.mean().detach(),
            "nft/old_deviate": ((forward_prediction - old_prediction)**2).mean().detach(),
            "nft/advantage_abs_mean": advantages.abs().mean().detach(),
            "nft/r_mean": r.mean().detach(),
            "nft/timestep_mean": queued.timestep.float().mean().detach(),
        }
        return loss_map, metrics

    def _shape_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        clipped = advantages.float().clamp(-self._adv_clip_max, self._adv_clip_max)
        if self._adv_mode == "positive_only":
            clipped = clipped.clamp(0.0, self._adv_clip_max)
        elif self._adv_mode == "negative_only":
            clipped = clipped.clamp(-self._adv_clip_max, 0.0)
        elif self._adv_mode == "one_only":
            clipped = torch.where(clipped > 0, torch.ones_like(clipped), torch.zeros_like(clipped))
        elif self._adv_mode == "binary":
            clipped = torch.sign(clipped)
        return clipped
