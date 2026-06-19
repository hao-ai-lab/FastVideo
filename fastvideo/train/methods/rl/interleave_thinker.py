# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker-style RL method for critic/prompt-refinement actors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.rl.rewards import (
    InterleaveThinkerRewardScorer, )
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
    parse_betas,
)
from fastvideo.train.utils.instantiate import (
    instantiate, )
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )

logger = init_logger(__name__)


class InterleaveThinkerRLMethod(TrainingMethod):
    """GRPO-style InterleaveThinker critic RL.

    One ``Trainer`` step performs a complete InterleaveThinker RL outer step:
    actor rollout generation, reward scoring, group-normalized advantage
    computation, and an actor-owned policy update. The method intentionally
    delegates tokenizer/VLM/logprob details to the student model wrapper via two
    hooks:

    - ``generate_interleave_responses(batch, **kwargs)`` returns rollout dicts.
    - ``train_interleave_rollouts(rollouts=..., advantages=..., **kwargs)``
      performs the policy update and returns loss/metric dictionaries.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if not self.student._trainable:
            raise ValueError("InterleaveThinkerRLMethod requires a trainable student")

        self.reference = role_models.get("reference")
        if self.reference is not None:
            if self.reference._trainable:
                raise ValueError("InterleaveThinkerRLMethod requires models.reference.trainable=false")
            self._freeze_reference_model()

        self.student.init_preprocessors(self.training_config)
        self._num_generations = self._read_int("num_generations", 8)
        self._num_batches_per_step = self._read_int("num_batches_per_step", 1)
        self._max_new_tokens = get_optional_int(
            self.method_config,
            "max_new_tokens",
            where="method.max_new_tokens",
        )
        self._temperature = self._read_float("temperature", 1.0)
        self._top_p = self._read_float("top_p", 1.0)
        self._advantage_eps = self._read_float("advantage_eps", 1.0e-4)
        self._advantage_clip = get_optional_float(
            self.method_config,
            "advantage_clip",
            where="method.advantage_clip",
        )
        self._clip_range = self._read_float("clip_range", 0.2)
        if self._clip_range < 0.0:
            raise ValueError("method.clip_range must be non-negative")
        self._kl_coef = self._read_float("kl_coef", 0.0)
        if self._kl_coef < 0.0:
            raise ValueError("method.kl_coef must be non-negative")
        self._update_micro_batch_size = self._read_optional_int_alias(
            "micro_batch_size_per_device_for_update",
            "update_micro_batch_size",
        )
        self._max_grad_norm = self._read_float("max_grad_norm", 0.0)
        self._terminal_progress = bool(self.method_config.get("terminal_progress", True))
        self._reward_scorer = InterleaveThinkerRewardScorer(
            format_weight=self._read_float("format_weight", 0.5),
            judge_accuracy_weight=self._read_float("judge_accuracy_weight", 0.2),
            semantic_weight=self._read_float("semantic_weight", 0.6),
            quality_weight=self._read_float("quality_weight", 0.2),
            fallback_edit_reward=self._read_float("fallback_edit_reward", 0.5),
            edit_scorer=self._build_edit_scorer(self.method_config.get("edit_scorer")),
        )
        self._student_optimizer: torch.optim.Optimizer | None = None
        self._student_lr_scheduler: Any | None = None
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer} if self._student_optimizer is not None else {}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler} if self._student_lr_scheduler is not None else {}

    def manages_optimization(self) -> bool:
        return True

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del batch, iteration
        raise RuntimeError("InterleaveThinkerRLMethod uses managed_train_step()")

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        self._log_progress(f"[InterleaveThinkerRL] step {iteration}: generating rollouts")
        rollouts: list[dict[str, Any]] = []
        for batch_idx in range(self._num_batches_per_step):
            batch = next(data_stream)
            rollouts.extend(self._generate_rollouts(batch, iteration=iteration, batch_idx=batch_idx))
        if not rollouts:
            raise RuntimeError("InterleaveThinkerRLMethod generated no rollouts")

        reference_logprob_count = self._attach_reference_logprobs(rollouts)

        self._log_progress(f"[InterleaveThinkerRL] step {iteration}: scoring {len(rollouts)} rollouts")
        reward_inputs = self._make_reward_inputs(rollouts)
        reward_tensors = self._reward_scorer.as_tensors(reward_inputs, device=self.student.device)

        self._log_progress(f"[InterleaveThinkerRL] step {iteration}: computing advantages")
        group_keys = [self._rollout_group_key(rollout, idx) for idx, rollout in enumerate(rollouts)]
        advantages = self._compute_group_advantages(
            rewards=reward_tensors["overall"],
            group_keys=group_keys,
            eps=self._advantage_eps,
            clip=self._advantage_clip,
        )

        self._log_progress(f"[InterleaveThinkerRL] step {iteration}: actor update")
        loss_map, train_metrics = self._train_actor(
            rollouts,
            advantages=advantages,
            rewards=reward_tensors,
            iteration=iteration,
        )
        metrics: dict[str, LogScalar] = {}
        metrics.update(self._reward_metrics(reward_tensors))
        metrics.update(self._advantage_metrics(advantages, group_keys))
        metrics.update(train_metrics)
        metrics["interleave/num_rollouts"] = float(len(rollouts))
        metrics["interleave/num_groups"] = float(len(set(group_keys)))
        metrics["interleave/num_batches_per_step"] = float(self._num_batches_per_step)
        metrics["interleave/reference_logprob_rollouts"] = float(reference_logprob_count)
        return loss_map, {}, metrics

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer] if self._student_optimizer is not None else []

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler] if self._student_lr_scheduler is not None else []

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        del iteration
        transformer = getattr(self.student, "transformer", None)
        if isinstance(transformer, torch.nn.Module):
            return {"student": transformer}
        return {}

    def on_train_start(self) -> None:
        super().on_train_start()
        self._freeze_reference_model()

    def _init_optimizer_and_scheduler(self) -> None:
        transformer = getattr(self.student, "transformer", None)
        if not isinstance(transformer, torch.nn.Module):
            return
        params = [p for p in transformer.parameters() if p.requires_grad]
        if not params:
            return
        betas = self.training_config.optimizer.betas
        betas_raw = self.method_config.get("betas", None)
        if betas_raw is not None:
            betas = parse_betas(betas_raw, where="method.betas")
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=float(self.training_config.optimizer.learning_rate),
            betas=betas,
            scheduler_name=str(self.training_config.optimizer.lr_scheduler),
        )

    def _build_edit_scorer(
        self,
        raw: Any,
    ) -> Any:
        if raw is None:
            return None
        if callable(raw):
            return raw
        if isinstance(raw, Mapping):
            return instantiate(dict(raw))
        raise TypeError("method.edit_scorer must be a callable or a mapping with _target_")

    def _freeze_reference_model(self) -> None:
        if self.reference is None:
            return
        transformer = getattr(self.reference, "transformer", None)
        if isinstance(transformer, torch.nn.Module):
            transformer.requires_grad_(False)
            transformer.eval()

    def _attach_reference_logprobs(
        self,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> int:
        if self.reference is None:
            return 0
        pending: list[dict[str, Any]] = []
        for rollout in rollouts:
            if self._has_reference_logprobs(rollout):
                continue
            if not isinstance(rollout, dict):
                raise TypeError("InterleaveThinker reference logprobs require mutable rollout dictionaries")
            pending.append(rollout)
        if not pending:
            return 0

        hook = getattr(self.reference, "reference_logprobs_for_interleave_rollouts", None)
        if not callable(hook):
            raise RuntimeError("models.reference must implement reference_logprobs_for_interleave_rollouts()")

        self._log_progress(f"[InterleaveThinkerRL] computing reference logprobs for {len(pending)} rollouts")
        self._freeze_reference_model()
        with torch.no_grad():
            rows = hook(pending)
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            raise TypeError("reference_logprobs_for_interleave_rollouts() must return a sequence")
        reference_rows = list(rows)
        if len(reference_rows) != len(pending):
            raise ValueError("reference logprob row count must match rollout count")
        for rollout, row in zip(pending, reference_rows, strict=True):
            rollout["reference_logprobs"] = self._coerce_reference_logprobs(row)
        return len(pending)

    @staticmethod
    def _has_reference_logprobs(rollout: Mapping[str, Any]) -> bool:
        return rollout.get("reference_logprobs") is not None or rollout.get("ref_logprobs") is not None

    @staticmethod
    def _coerce_reference_logprobs(row: Any) -> list[float]:
        if torch.is_tensor(row):
            values = row.detach().cpu().float().flatten().tolist()
        elif isinstance(row, Sequence) and not isinstance(row, str | bytes):
            values = [float(value) for value in row]
        else:
            raise TypeError("reference logprob rows must be tensors or sequences of floats")
        if not values:
            raise ValueError("reference logprob rows must be non-empty")
        return values

    def _generate_rollouts(
        self,
        batch: dict[str, Any],
        *,
        iteration: int,
        batch_idx: int,
    ) -> list[dict[str, Any]]:
        generate = getattr(self.student, "generate_interleave_responses", None)
        if callable(generate):
            generated = generate(
                batch,
                num_generations=self._num_generations,
                temperature=self._temperature,
                top_p=self._top_p,
                max_new_tokens=self._max_new_tokens,
                generator=self.cuda_generator,
                iteration=iteration,
                batch_idx=batch_idx,
            )
            return self._normalize_generated_rollouts(generated, batch=batch)
        return self._offline_rollouts_from_batch(batch)

    def _offline_rollouts_from_batch(
        self,
        batch: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        items = self._batch_to_items(batch)
        rollouts: list[dict[str, Any]] = []
        for item_idx, item in enumerate(items):
            responses = item["responses"] if "responses" in item else item.get("response")
            if responses is None:
                raise RuntimeError("Student model must implement generate_interleave_responses(), "
                                   "or batches must contain response/responses for offline rollouts")
            if isinstance(responses, str):
                response_list = [responses]
            elif isinstance(responses, Sequence):
                response_list = list(responses)
            else:
                response_list = [responses]
            for response_idx, response in enumerate(response_list):
                rollout = dict(item)
                for per_response_key in ("edit_score", "edit_scores"):
                    per_response_value = rollout.get(per_response_key)
                    if (isinstance(per_response_value, Sequence) and not isinstance(per_response_value, str)
                            and len(per_response_value) == len(response_list)):
                        rollout[per_response_key] = per_response_value[response_idx]
                rollout.pop("responses", None)
                rollout["response"] = str(response)
                rollout.setdefault("sample_index", item_idx)
                rollout.setdefault("generation_index", response_idx)
                rollout.setdefault("group_key", self._rollout_group_key(rollout, item_idx))
                rollouts.append(rollout)
        return rollouts

    def _normalize_generated_rollouts(
        self,
        generated: Any,
        *,
        batch: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if isinstance(generated, Mapping) and "rollouts" in generated:
            generated = generated["rollouts"]
        if isinstance(generated, str):
            generated = [generated]
        if not isinstance(generated, Sequence):
            raise TypeError("generate_interleave_responses() must return a sequence of rollout mappings")

        batch_items = self._batch_to_items(batch)
        rollouts: list[dict[str, Any]] = []
        for idx, raw_rollout in enumerate(generated):
            if isinstance(raw_rollout, str):
                base = dict(batch_items[min(idx // max(1, self._num_generations), len(batch_items) - 1)])
                rollout = {**base, "response": raw_rollout}
            elif isinstance(raw_rollout, Mapping):
                rollout = dict(raw_rollout)
            else:
                raise TypeError(f"Rollout must be a mapping or string, got {type(raw_rollout).__name__}")
            rollout.setdefault("sample_index", idx // max(1, self._num_generations))
            rollout.setdefault("generation_index", idx % max(1, self._num_generations))
            if "group_key" not in rollout:
                sample_index = int(rollout["sample_index"])
                if 0 <= sample_index < len(batch_items):
                    base_item = batch_items[sample_index]
                    for key, value in base_item.items():
                        rollout.setdefault(key, value)
                rollout["group_key"] = self._rollout_group_key(rollout, idx)
            rollouts.append(rollout)
        return rollouts

    def _make_reward_inputs(
        self,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        reward_inputs: list[dict[str, Any]] = []
        for rollout in rollouts:
            if "response" not in rollout:
                raise ValueError("Each InterleaveThinker rollout must include 'response'")
            item = dict(rollout)
            item["response"] = str(item["response"])
            reward_inputs.append(item)
        return reward_inputs

    def _train_actor(
        self,
        rollouts: Sequence[Mapping[str, Any]],
        *,
        advantages: torch.Tensor,
        rewards: Mapping[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
        train = getattr(self.student, "train_interleave_rollouts", None)
        if not callable(train):
            raise RuntimeError("InterleaveThinkerRLMethod requires student.train_interleave_rollouts() "
                               "to perform the actor policy update")
        result = train(
            rollouts=rollouts,
            advantages=advantages,
            rewards=rewards,
            iteration=iteration,
            optimizer=self._student_optimizer,
            lr_scheduler=self._student_lr_scheduler,
            gradient_accumulation_steps=max(1, int(self.training_config.loop.gradient_accumulation_steps or 1)),
            clip_range=self._clip_range,
            kl_coef=self._kl_coef,
            update_micro_batch_size=self._update_micro_batch_size,
            max_grad_norm=self._max_grad_norm,
        )
        return self._coerce_train_result(result)

    def _coerce_train_result(
        self,
        result: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
        if isinstance(result, tuple) and len(result) == 2:
            loss_map_raw, metrics_raw = result
        elif isinstance(result, Mapping):
            loss_map_raw = result.get("loss_map")
            metrics_raw = result.get("metrics", {})
            if loss_map_raw is None and "loss" in result:
                loss_map_raw = {"total_loss": result["loss"]}
        else:
            raise TypeError("train_interleave_rollouts() must return (loss_map, metrics) or a mapping")
        if not isinstance(loss_map_raw, Mapping):
            raise TypeError("train_interleave_rollouts() result must include a loss_map mapping")
        if not isinstance(metrics_raw, Mapping):
            raise TypeError("train_interleave_rollouts() metrics must be a mapping")
        loss_map = {str(k): self._coerce_tensor(v) for k, v in loss_map_raw.items()}
        if "total_loss" not in loss_map:
            if len(loss_map) != 1:
                raise ValueError("loss_map must include total_loss when multiple losses are returned")
            only_value = next(iter(loss_map.values()))
            loss_map["total_loss"] = only_value
        metrics = {str(k): self._coerce_log_scalar(v) for k, v in metrics_raw.items()}
        return loss_map, metrics

    def _coerce_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.tensor(float(value), device=self.student.device, dtype=torch.float32)

    def _coerce_log_scalar(self, value: Any) -> LogScalar:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar metric tensor, got shape={tuple(value.shape)}")
            return value.detach()
        if isinstance(value, float | int):
            return float(value)
        raise TypeError(f"Expected scalar metric, got {type(value).__name__}")

    @staticmethod
    def _batch_to_items(batch: Mapping[str, Any], ) -> list[dict[str, Any]]:
        if "items" in batch and isinstance(batch["items"], Sequence):
            return [dict(item) for item in batch["items"]]

        batch_size = 1
        for value in batch.values():
            if isinstance(value, list | tuple):
                batch_size = len(value)
                break
            if torch.is_tensor(value) and value.ndim > 0:
                batch_size = int(value.shape[0])
                break

        items: list[dict[str, Any]] = []
        for idx in range(batch_size):
            item: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, list | tuple) and len(value) == batch_size or torch.is_tensor(
                        value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
                    item[key] = value[idx]
                else:
                    item[key] = value
            items.append(item)
        return items

    @staticmethod
    def _rollout_group_key(
        rollout: Mapping[str, Any],
        index: int,
    ) -> str:
        for key in ("group_key", "problem_id", "sample_index", "origin_prompt", "prompt"):
            value = rollout.get(key)
            if value is not None:
                return str(value)
        return str(index)

    @staticmethod
    def _compute_group_advantages(
        *,
        rewards: torch.Tensor,
        group_keys: Sequence[str],
        eps: float,
        clip: float | None,
    ) -> torch.Tensor:
        if rewards.ndim != 1:
            raise ValueError(f"rewards must have shape [N], got {tuple(rewards.shape)}")
        if int(rewards.shape[0]) != len(group_keys):
            raise ValueError("reward count must match group key count")
        advantages = torch.empty_like(rewards, dtype=torch.float32)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, group_key in enumerate(group_keys):
            groups[str(group_key)].append(idx)
        for indices in groups.values():
            index_tensor = torch.tensor(indices, device=rewards.device, dtype=torch.long)
            group_rewards = rewards[index_tensor].detach().float()
            group_std = group_rewards.std(unbiased=False)
            advantages[index_tensor] = (group_rewards - group_rewards.mean()) / (group_std + float(eps))
        if clip is not None:
            advantages = advantages.clamp(-float(clip), float(clip))
        return advantages

    def _reward_metrics(
        self,
        rewards: Mapping[str, torch.Tensor],
    ) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}
        for key, value in rewards.items():
            if value.numel() > 0:
                metrics[f"interleave/reward/{key}"] = value.detach().float().mean()
        return metrics

    def _advantage_metrics(
        self,
        advantages: torch.Tensor,
        group_keys: Sequence[str],
    ) -> dict[str, LogScalar]:
        group_sizes: dict[str, int] = defaultdict(int)
        for key in group_keys:
            group_sizes[key] += 1
        group_size_values = torch.tensor(list(group_sizes.values()), device=advantages.device, dtype=torch.float32)
        return {
            "interleave/advantage_mean": advantages.detach().float().mean(),
            "interleave/advantage_std": advantages.detach().float().std(unbiased=False),
            "interleave/group_size_mean": group_size_values.mean(),
        }

    def _read_int(
        self,
        key: str,
        default: int,
    ) -> int:
        value = get_optional_int(self.method_config, key, where=f"method.{key}")
        if value is None:
            value = default
        if value <= 0:
            raise ValueError(f"method.{key} must be a positive integer")
        return int(value)

    def _read_optional_int_alias(
        self,
        *keys: str,
    ) -> int | None:
        for key in keys:
            value = get_optional_int(self.method_config, key, where=f"method.{key}")
            if value is None:
                continue
            if value <= 0:
                raise ValueError(f"method.{key} must be a positive integer")
            return int(value)
        return None

    def _read_float(
        self,
        key: str,
        default: float,
    ) -> float:
        value = get_optional_float(self.method_config, key, where=f"method.{key}")
        if value is None:
            value = default
        return float(value)

    def _log_progress(self, message: str) -> None:
        if not self._terminal_progress:
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            logger.info(message)


__all__ = ["InterleaveThinkerRLMethod"]
