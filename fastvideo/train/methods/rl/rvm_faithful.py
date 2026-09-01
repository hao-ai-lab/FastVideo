# SPDX-License-Identifier: Apache-2.0
"""Paper-faithful Reward-based Velocity Matching for FastH3.

This module keeps the tested FastH3 rollout/reward/checkpoint implementation in
``rvm.py`` and replaces the two choices that differed from the published RVM
video recipe:

1. rewards are centered per prompt group but divided by one standard deviation
   computed globally across every sample in the rollout batch; and
2. the flow-matching state is sampled continuously with ``t ~ Uniform(0, 1)``
   by default instead of only at the four deployment timesteps.

The deployment-grid training-time sampler remains available as an explicit H3
ablation. Rollout generation itself always uses the released four-step FastH3
VSA sampler.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common import (
    detached_rvm_surrogate,
    partition_indices,
    standardize_group_rewards,
    visual_text_from_h3_prompt,
)
from fastvideo.train.methods.rl.rvm import RVMMethod, _RVMItem
from fastvideo.train.utils.lora_context import temporarily_disable_lora
from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed


@dataclass(slots=True)
class _CollectedRVMGroup:
    raw_batch: dict[str, Any]
    endpoints: list[torch.Tensor]
    prompt: str
    rewards: dict[str, torch.Tensor]


class RVMFaithfulMethod(RVMMethod):
    """RVM with paper-faithful batch-global reward scaling and time sampling."""

    _TIMESTEP_SCALE = 1000.0

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        raw_timestep = self.method_config.get("training_timestep") or {}
        if not isinstance(raw_timestep, Mapping):
            raise ValueError("method.training_timestep must be a mapping")
        self._training_timestep_mode = str(
            raw_timestep.get("mode", "continuous_uniform")
        ).strip().lower()
        if self._training_timestep_mode not in {
            "continuous_uniform",
            "deployment_grid",
        }:
            raise ValueError(
                "method.training_timestep.mode must be continuous_uniform "
                "or deployment_grid"
            )
        self._training_timestep_min = float(raw_timestep.get("min", 0.0))
        self._training_timestep_max = float(raw_timestep.get("max", 1.0))
        if not (
            0.0
            <= self._training_timestep_min
            < self._training_timestep_max
            <= 1.0
        ):
            raise ValueError(
                "method.training_timestep requires 0 <= min < max <= 1"
            )
        if "dynamic_tracking" in self._reward_names:
            self._reward_keys = [
                *self._reward_names,
                "dynamic_tracking_raw",
                "dynamic_tracking_saturation",
                "avg",
            ]

    def on_train_start(self) -> None:
        super().on_train_start()
        self._log_progress(
            "RVM fidelity settings: reward_center=per_prompt, "
            "reward_std=batch_global, "
            f"training_timestep={self._training_timestep_mode}"
        )

    def _collect_rollouts(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> None:
        self.student.transformer.eval()
        self._buffer = []
        local_groups = self._prompt_groups_per_rollout // self._dp_world_size
        pending: list[_CollectedRVMGroup] = []
        reward_sums: dict[str, float] = defaultdict(float)
        reward_count = 0

        self._log_progress(
            f"RVM step {iteration}: collecting {local_groups} local prompt "
            f"groups x {self._samples_per_prompt} samples"
        )
        with torch.no_grad():
            for _ in range(local_groups):
                raw_batch = self._clone_raw_batch(next(data_stream))
                full_prompt = self._extract_prompt(raw_batch)
                visual_prompt = visual_text_from_h3_prompt(full_prompt)
                prepared = self.student.prepare_batch(
                    raw_batch,
                    generator=self._require_generator(),
                    latents_source="zeros",
                )
                endpoints: list[torch.Tensor] = []
                for _sample_index in range(self._samples_per_prompt):
                    result = self._sampler.sample(
                        self.student,
                        prepared,
                        generator=self._require_generator(),
                    )
                    endpoints.append(result.endpoint.detach().cpu())

                reward_dict = self._score_endpoint_group(
                    endpoints,
                    visual_prompt,
                )
                for key, values in reward_dict.items():
                    reward_sums[key] += float(values.float().sum())
                reward_count += self._samples_per_prompt
                pending.append(
                    _CollectedRVMGroup(
                        raw_batch=raw_batch,
                        endpoints=endpoints,
                        prompt=full_prompt,
                        rewards=reward_dict,
                    )
                )

        if torch.cuda.is_available():
            # Release VAE/reward allocations before collective statistics ask
            # NCCL for communication buffers.
            torch.cuda.empty_cache()

        reward_stats = {
            key: self._global_reward_stats(
                [group.rewards[key] for group in pending]
            )
            for key in self._reward_keys
        }
        global_reward_mean, global_reward_std = reward_stats["avg"]
        positive_only = iteration <= self._positive_only_steps
        zero_std_count = 0
        group_std_sum = 0.0
        clipped_count = 0
        advantage_abs_sum = 0.0
        advantage_count = 0

        for group in pending:
            rewards = group.rewards["avg"]
            centered = rewards.detach().float() - rewards.detach().float().mean()
            raw_normalized = centered / (
                float(global_reward_std) + self._advantage_eps
            )
            clipped_count += int(
                (raw_normalized.abs() > self._advantage_clip).sum().item()
            )
            advantages, group_std = standardize_group_rewards(
                rewards,
                scale=self._advantage_scale,
                eps=self._advantage_eps,
                clip=self._advantage_clip,
                positive_only=positive_only,
                normalization_std=global_reward_std,
            )
            group_std_value = float(group_std)
            group_std_sum += group_std_value
            if group_std_value <= self._advantage_eps:
                zero_std_count += 1
            advantage_abs_sum += float(advantages.abs().sum())
            advantage_count += int(advantages.numel())

            for sample_index, endpoint in enumerate(group.endpoints):
                self._buffer.append(
                    _RVMItem(
                        raw_batch=group.raw_batch,
                        endpoint=endpoint,
                        advantage=float(advantages[sample_index]),
                        prompt=group.prompt,
                    )
                )

        self._collection_count += 1
        cpu_generator = torch.Generator(device="cpu").manual_seed(
            int(self.training_config.data.seed)
            + 100_000
            + self._collection_count
        )
        self._buffer_partitions = partition_indices(
            len(self._buffer),
            self._updates_per_rollout,
            generator=cpu_generator,
            device="cpu",
        )
        self._buffer_cursor = 0

        metrics: dict[str, LogScalar] = {
            "rvm/rollout_samples": float(
                self._prompt_groups_per_rollout
                * self._samples_per_prompt
            ),
            "rvm/reward_global_mean": global_reward_mean,
            "rvm/reward_global_std": global_reward_std,
            "rvm/group_reward_std_mean": self._global_leader_mean(
                group_std_sum,
                float(local_groups),
            ),
            "rvm/zero_std_group_ratio": self._global_leader_mean(
                float(zero_std_count),
                float(local_groups),
            ),
            "rvm/advantage_abs_mean": self._global_leader_mean(
                advantage_abs_sum,
                float(advantage_count),
            ),
            "rvm/advantage_clip_ratio": self._global_leader_mean(
                float(clipped_count),
                float(advantage_count),
            ),
            "rvm/positive_only": float(positive_only),
        }
        for key, value in reward_sums.items():
            metrics[f"reward/{key}"] = self._global_leader_mean(
                value,
                float(reward_count),
            )
            metrics[f"reward_std/{key}"] = reward_stats[key][1]
        self._collection_metrics = metrics

    def _global_reward_stats(
        self,
        local_groups: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global mean/std of aggregate rewards over DP prompt groups.

        Only sequence-parallel leaders contribute sufficient statistics, so
        each generated sample is counted once even though every SP rank holds
        the same scalar rewards.
        """
        if not local_groups:
            raise RuntimeError("RVM collected no reward groups")
        local_values = torch.cat(
            [value.detach().float().reshape(-1) for value in local_groups]
        )
        stats = torch.zeros(
            3,
            device=self.student.device,
            dtype=torch.float64,
        )
        if self._is_sp_leader:
            values = local_values.to(
                device=self.student.device,
                dtype=torch.float64,
            )
            stats[0] = float(values.numel())
            stats[1] = values.sum()
            stats[2] = values.square().sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        count = stats[0].clamp_min(1.0)
        mean = stats[1] / count
        variance = (stats[2] / count - mean.square()).clamp_min(0.0)
        return mean.to(torch.float32), variance.sqrt().to(torch.float32)

    def _train_partition(
        self,
        partition: torch.Tensor,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
        self.student.transformer.train()
        self._optimizer.zero_grad(set_to_none=True)
        count = int(partition.numel())
        if count <= 0:
            raise RuntimeError("RVM received an empty optimizer partition")

        totals: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((), device=self.student.device)
        )
        timestep_counts = torch.zeros(
            len(self._sampling_config.denoising_steps),
            device=self.student.device,
            dtype=torch.float32,
        )
        timestep_sum = torch.zeros(
            (),
            device=self.student.device,
            dtype=torch.float32,
        )
        timestep_min = torch.full_like(timestep_sum, float("inf"))
        timestep_max = torch.full_like(timestep_sum, float("-inf"))

        for item_index in partition.tolist():
            item = self._buffer[int(item_index)]
            (
                losses,
                backward_context,
                timestep_index,
                base_timestep,
            ) = self._faithful_endpoint_loss(item)
            self.student.backward(
                losses["total_loss"],
                backward_context,
                grad_accum_rounds=count,
            )
            for key, value in losses.items():
                totals[key] = totals[key] + value.detach()
            timestep_counts[timestep_index] += 1.0
            timestep_sum += base_timestep
            timestep_min = torch.minimum(timestep_min, base_timestep)
            timestep_max = torch.maximum(timestep_max, base_timestep)

        grad_norm = clip_grad_norm_if_needed(
            self.student.transformer,
            self._max_grad_norm,
        )
        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)

        averaged = {
            key: value / count
            for key, value in totals.items()
        }
        reduced = {
            key: self._mean_scalar_across_ranks(value)
            for key, value in averaged.items()
        }
        reduced.setdefault(
            "total_loss",
            torch.zeros((), device=self.student.device),
        )
        metrics: dict[str, LogScalar] = {
            "rvm/optimizer_step": float(iteration),
            "rvm/local_train_samples": float(count),
            "rvm/grad_norm": float(grad_norm),
            "rvm/grad_clipped": float(
                self._max_grad_norm > 0.0
                and grad_norm > self._max_grad_norm
            ),
            "rvm/learning_rate": float(
                self._optimizer.param_groups[0]["lr"]
            ),
        }

        total_timestep_counts = timestep_counts.clone()
        timestep_stats = torch.stack(
            [
                timestep_sum,
                torch.tensor(
                    float(count),
                    device=self.student.device,
                    dtype=torch.float32,
                ),
                timestep_min,
                timestep_max,
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                total_timestep_counts,
                op=dist.ReduceOp.SUM,
            )
            # Sum and count use SUM; extrema need their own reductions.
            sum_and_count = timestep_stats[:2].clone()
            min_value = timestep_stats[2].clone()
            max_value = timestep_stats[3].clone()
            dist.all_reduce(sum_and_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
            timestep_stats = torch.stack(
                [
                    sum_and_count[0],
                    sum_and_count[1],
                    min_value,
                    max_value,
                ]
            )

        denominator = total_timestep_counts.sum().clamp_min(1.0)
        for index, step in enumerate(
            self._sampling_config.denoising_steps
        ):
            metrics[f"rvm/timestep_fraction_{step}"] = (
                total_timestep_counts[index] / denominator
            )
        metrics["rvm/training_timestep_mean"] = (
            timestep_stats[0] / timestep_stats[1].clamp_min(1.0)
        )
        metrics["rvm/training_timestep_min"] = timestep_stats[2]
        metrics["rvm/training_timestep_max"] = timestep_stats[3]
        return reduced, metrics

    def _faithful_endpoint_loss(
        self,
        item: _RVMItem,
    ) -> tuple[
        dict[str, torch.Tensor],
        tuple[torch.Tensor, Any],
        int,
        torch.Tensor,
    ]:
        batch = self.student.prepare_batch(
            item.raw_batch,
            generator=self._require_generator(),
            latents_source="zeros",
        )
        if batch.latents is None:
            raise RuntimeError(
                "H3 batch preparation returned no packed latent geometry"
            )
        endpoint = item.endpoint.to(
            device=self.student.device,
            dtype=batch.latents.dtype,
        )
        timestep, timestep_index, base_timestep = (
            self._sample_training_timestep()
        )
        self.student.refresh_vsa_metadata(
            batch,
            current_timestep=timestep_index,
        )
        noise = torch.randn(
            endpoint.shape,
            device=endpoint.device,
            dtype=endpoint.dtype,
            generator=self._require_generator(),
        )
        noisy = self.student.add_noise(endpoint, noise, timestep)
        target = noise - endpoint
        prediction = self.student.predict_noise(
            noisy,
            timestep,
            batch,
            conditional=True,
            attn_kind=self._sampling_config.attn_kind,
        )
        backward_context = (
            batch.timesteps,
            batch.attn_metadata_vsa,
        )

        reference: torch.Tensor | None = None
        if (
            self._video_anchor_beta != 0.0
            or self._audio_anchor_beta != 0.0
        ):
            with (
                temporarily_disable_lora(self.student.transformer),
                torch.no_grad(),
            ):
                reference = self.student.predict_noise(
                    noisy,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind=self._sampling_config.attn_kind,
                )

        slices = dict(self.student.modality_slices())
        video_slice = slices["video"]
        audio_slice = slices["audio"]
        video_reference = (
            None
            if reference is None
            else reference[:, video_slice]
        )
        audio_reference = (
            None
            if reference is None
            else reference[:, audio_slice]
        )

        video_loss = detached_rvm_surrogate(
            prediction[:, video_slice],
            target[:, video_slice],
            coefficient=float(item.advantage),
            reference=video_reference,
            anchor_beta=self._video_anchor_beta,
        )
        if self._audio_anchor_beta != 0.0:
            audio_loss = detached_rvm_surrogate(
                prediction[:, audio_slice],
                target[:, audio_slice],
                coefficient=0.0,
                reference=audio_reference,
                anchor_beta=self._audio_anchor_beta,
            )
        else:
            audio_loss = (
                prediction[:, audio_slice].sum() * 0.0
            )
        total_loss = (
            video_loss
            + self._audio_loss_weight * audio_loss
        )
        return (
            {
                "total_loss": total_loss,
                "video_rvm_loss": video_loss,
                "audio_anchor_loss": audio_loss,
                "advantage": torch.tensor(
                    float(item.advantage),
                    device=self.student.device,
                ),
            },
            backward_context,
            timestep_index,
            base_timestep,
        )

    def _sample_training_timestep(
        self,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        steps = torch.tensor(
            self._sampling_config.denoising_steps,
            device=self.student.device,
            dtype=torch.float32,
        )
        if self._training_timestep_mode == "deployment_grid":
            index_tensor = torch.randint(
                0,
                len(self._sampling_config.denoising_steps),
                (1,),
                device=self.student.device,
                generator=self._require_generator(),
            )
            timestep_index = int(index_tensor.item())
            timestep = steps[timestep_index].reshape(1)
            return (
                timestep,
                timestep_index,
                timestep / self._TIMESTEP_SCALE,
            )

        unit = torch.rand(
            (1,),
            device=self.student.device,
            dtype=torch.float32,
            generator=self._require_generator(),
        )
        unit = (
            self._training_timestep_min
            + (
                self._training_timestep_max
                - self._training_timestep_min
            )
            * unit
        )
        timestep = unit * self._TIMESTEP_SCALE
        timestep_index = int(
            torch.argmin(
                torch.abs(steps - timestep[0])
            ).item()
        )
        return timestep, timestep_index, unit[0]


__all__ = ["RVMFaithfulMethod"]
