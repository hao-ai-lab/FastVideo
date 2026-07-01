# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT multi-reward policy optimization method."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from tqdm.auto import tqdm

from fastvideo.dataset.parquet_dataset_map_style import (
    get_parquet_files_and_length,
    read_row_from_parquet_file,
)
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.methods.rl.rewards import (
    GENRL_REWARD_NAMES,
    build_multi_reward_scorer,
    normalize_reward_weights,
)
from fastvideo.train.methods.rl.common import (
    DiffusionSampler,
    RLValidationConfig,
    SamplingConfig,
    distributed_k_repeat_indices,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
    parse_betas,
)
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.training.training_utils import (
    EMA_FSDP,
    clip_grad_norm_while_handling_failing_dtensor_cases,
)

logger = init_logger(__name__)


class _DiffusionNFTEMAState:
    """DCP state wrapper for method-owned DiffusionNFT EMA."""

    def __init__(self, method: DiffusionNFTMethod) -> None:
        self._method = method

    def state_dict(self) -> dict[str, Any]:
        ema = self._method._student_ema
        return {
            "shadow": ema.state_dict() if ema is not None else {},
            "update_count": torch.tensor(self._method._ema_update_count, dtype=torch.long),
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        ema = self._method._student_ema
        shadow = state_dict.get("shadow", {})
        if ema is not None and isinstance(shadow, dict):
            ema.load_state_dict(shadow)
        update_count = state_dict.get("update_count", 0)
        if torch.is_tensor(update_count):
            update_count = int(update_count.item())
        self._method._ema_update_count = int(update_count)


class _FullModelState(Stateful):
    """DCP wrapper that saves frozen model parameters too.

    The shared ``ModelWrapper`` intentionally filters to ``requires_grad``
    parameters. DiffusionNFT's old policy is frozen but must be restored on
    resume, so it needs full model state.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)  # type: ignore[no-any-return]

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


class DiffusionNFTMethod(TrainingMethod):
    """DiffusionNFT-style RL for diffusion models.

    This method owns the algorithm's sample-then-inner-train loop. One
    ``Trainer`` step corresponds to one DiffusionNFT outer epoch.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "old" not in role_models:
            raise ValueError("DiffusionNFTMethod requires role 'old'")
        if "reference" not in role_models:
            raise ValueError("DiffusionNFTMethod requires role 'reference'")
        if not self.student._trainable:
            raise ValueError("DiffusionNFTMethod requires a trainable student")

        self.old = role_models["old"]
        self.reference = role_models["reference"]
        self.student.init_preprocessors(self.training_config)

        self._sampling_config = self._parse_sampling_config()
        self._sampler = DiffusionSampler(self._sampling_config)
        self._validation_config = RLValidationConfig.from_mapping(self.method_config.get("validation"))
        self._validation_sampling_config = self._parse_validation_sampling_config()
        self._validation_sampler = DiffusionSampler(self._validation_sampling_config)
        self._validation_items: list[tuple[int, bool, dict[str, Any]]] | None = None
        self._sample_steps = int(self._sampling_config.num_steps)
        self._sample_train_batch_size = self._read_int(
            "sample_train_batch_size",
            max(1, int(self.training_config.data.train_batch_size or 1)),
        )
        self._train_batch_size = self._read_int("train_batch_size", self._sample_train_batch_size)
        self._num_batches_per_epoch = self._read_int("num_batches_per_epoch", 48)
        self._num_inner_epochs = self._read_int("num_inner_epochs", 1)
        self._num_video_per_prompt = self._read_int("num_video_per_prompt", 24)
        self._adv_clip_max = self._read_float("adv_clip_max", 5.0)
        self._timestep_fraction = self._read_float("timestep_fraction", 0.99)
        self._kl_beta = self._read_float("kl_beta", 0.0001)
        self._nft_beta = self._read_float("beta", 0.1)
        self._max_grad_norm = self._read_float("max_grad_norm", 1.0)
        self._decay_type = self._read_int("decay_type", 1)
        self._adv_mode = str(self.method_config.get("adv_mode", "all") or "all").strip().lower()
        self._terminal_progress = bool(self.method_config.get("terminal_progress", True))
        ema_config = self._parse_ema_config()
        self._ema_enabled = bool(ema_config["enabled"])
        self._ema_decay = float(ema_config["decay"])
        self._ema_update_after_step = int(ema_config["update_after_step"])
        self._validation_use_ema = bool(ema_config["validation"])
        self._student_ema: EMA_FSDP | None = None
        self._ema_update_count = 0
        self._trained_prompt_hashes: set[int] = set()
        if self._adv_mode not in {"all", "positive_only", "negative_only", "one_only", "binary"}:
            raise ValueError("method.adv_mode must be one of "
                             "{all, positive_only, negative_only, one_only, binary}")

        reward_fn = self.method_config.get("reward_fn", None)
        self._reward_fn_config, reward_backend = normalize_reward_weights(reward_fn)
        self._reward_backend = str(
            self.method_config.get(
                "reward_backend",
                reward_backend or "auto",
            ) or "auto").strip().lower()
        if self._reward_backend not in {"auto", "diffusion_nft", "genrl"}:
            raise ValueError("method.reward_backend must be one of auto, diffusion_nft, "
                             f"or genrl, got {self._reward_backend!r}")
        if self._reward_backend == "genrl" and not any(name in GENRL_REWARD_NAMES for name in self._reward_fn_config):
            raise ValueError("method.reward_backend='genrl' requires at least one GenRL reward "
                             f"from {sorted(GENRL_REWARD_NAMES)}")

        self._reward_scorer: Any | None = None
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self, ) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def manages_optimization(self) -> bool:
        return True

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del batch
        raise RuntimeError("DiffusionNFTMethod uses managed_train_step()")

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        self._log_progress(f"[DiffusionNFT] outer step {iteration}: start sampling "
                           f"{self._num_batches_per_epoch} batches")
        sample_items = self._sample_epoch(data_stream, iteration)
        self._log_progress(f"[DiffusionNFT] outer step {iteration}: scoring rewards")
        rewards = self._score_samples(sample_items)
        self._log_progress(f"[DiffusionNFT] outer step {iteration}: computing advantages")
        advantages = self._compute_advantages(sample_items, rewards)
        self._log_progress(f"[DiffusionNFT] outer step {iteration}: start inner training")
        loss_map, metrics = self._inner_train(sample_items, advantages, iteration)
        self._update_old_model(iteration)
        metrics.update(self._reward_metrics(rewards))
        metrics.update(self._reward_diagnostic_metrics(sample_items, rewards))
        metrics["nft/num_sampled"] = float(sum(item["latents_clean"].shape[0] for item in sample_items))
        return loss_map, {}, metrics

    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        return self._maybe_run_validation(iteration)

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["roles.old.transformer"] = _FullModelState(self.old.transformer)
        if self._ema_enabled:
            states["diffusion_nft.ema"] = _DiffusionNFTEMAState(self)
        return states

    def seed_optimizer_state_for_resume(self) -> None:
        super().seed_optimizer_state_for_resume()

    def on_train_start(self) -> None:
        super().on_train_start()
        self._sync_old_from_student()
        if self._ema_enabled:
            self._student_ema = EMA_FSDP(
                self.student.transformer,
                decay=self._ema_decay,
                mode="local_shard",
            )
            self._log_progress(f"[DiffusionNFT] EMA enabled (decay={self._ema_decay})")
        self._reward_scorer = build_multi_reward_scorer(
            self._reward_fn_config,
            device=self.student.device,
            backend=self._reward_backend,
        )

    def _init_optimizer_and_scheduler(self) -> None:
        lr = float(self.training_config.optimizer.learning_rate)
        betas = self.training_config.optimizer.betas
        betas_raw = self.method_config.get("betas", None)
        if betas_raw is not None:
            betas = parse_betas(betas_raw, where="method.betas")
        params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=lr,
            betas=betas,
            scheduler_name=str(self.training_config.optimizer.lr_scheduler),
        )

    def _sample_epoch(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, Any]]:
        self.student.transformer.eval()
        self.old.transformer.eval()
        sample_items: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch_idx in tqdm(
                    range(self._num_batches_per_epoch),
                    desc=f"DiffusionNFT step {iteration}: sampling",
                    position=1,
                    leave=False,
                    disable=not self._show_terminal_progress(),
            ):
                raw_batch = self._sample_prompt_batch(data_stream, iteration, batch_idx)
                prompts = self._extract_prompts(raw_batch)
                batch = self.student.prepare_batch(
                    raw_batch,
                    generator=self.cuda_generator,
                    latents_source="zeros",
                )
                sampling_result = self._sampler.sample(
                    self.old,
                    batch,
                    generator=self.cuda_generator,
                )
                latents_clean = sampling_result.latents
                train_timesteps = sampling_result.timesteps[:self._num_train_timesteps()]
                media = self.student.decode_latents(latents_clean)
                sample_items.append({
                    "encoder_hidden_states": batch.encoder_hidden_states.detach(),
                    "encoder_attention_mask": batch.encoder_attention_mask.detach(),
                    "latents_clean": latents_clean.detach(),
                    "timesteps": train_timesteps.detach().unsqueeze(0).repeat(latents_clean.shape[0], 1),
                    "media": media.detach().cpu(),
                    "prompts": prompts,
                })
        return sample_items

    def _sample_prompt_batch(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
        batch_idx: int,
    ) -> dict[str, Any]:
        dataset = getattr(getattr(self.student, "dataloader", None), "dataset", None)
        if dataset is None or not all(
                hasattr(dataset, attr) for attr in ("parquet_files", "lengths", "parquet_schema")):
            return self._repeat_first_prompt(next(data_stream), self._sample_train_batch_size)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        total_rows = int(sum(dataset.lengths))
        sample = distributed_k_repeat_indices(
            dataset_length=total_rows,
            batch_size=self._sample_train_batch_size,
            repeats_per_prompt=self._num_video_per_prompt,
            world_size=world_size,
            rank=rank,
            seed=int(self.training_config.data.seed) + (int(iteration) - 1) * self._num_batches_per_epoch + batch_idx,
        )
        rows = []
        for prompt_idx in sample.local_indices:
            row = read_row_from_parquet_file(dataset.parquet_files, prompt_idx, dataset.lengths)
            row["_sample_index"] = prompt_idx
            rows.append(row)
        return self._collate_rows(rows)

    def _score_samples(self, sample_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if self._reward_scorer is None:
            raise RuntimeError("Reward scorer has not been initialized")
        media = torch.cat([item["media"] for item in sample_items], dim=0)
        prompts = [prompt for item in sample_items for prompt in item["prompts"]]
        reward_dict = self._reward_scorer(media, prompts)
        device = self.student.device
        return {k: v.to(device=device, dtype=torch.float32) for k, v in reward_dict.items()}

    @torch.no_grad()
    def _maybe_run_validation(self, iteration: int) -> dict[str, LogScalar]:
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}
        if self._reward_scorer is None:
            raise RuntimeError("Reward scorer has not been initialized")

        use_ema = self._validation_use_ema and self._student_ema is not None
        suffix = " with EMA" if use_ema else ""
        self._log_progress(f"[DiffusionNFT] validation step {iteration}: start "
                           f"{config.num_prompts} prompts, {config.num_steps} sampling steps{suffix}")
        with self._ema_context():
            return self._run_validation(iteration)

    @torch.no_grad()
    def _run_validation(self, iteration: int) -> dict[str, LogScalar]:
        config = self._validation_config
        if self._reward_scorer is None:
            raise RuntimeError("Reward scorer has not been initialized")
        self.student.transformer.eval()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        prepare_generator = torch.Generator(device=self.student.device).manual_seed(config.seed + 100_000 + rank)
        validation_generator = torch.Generator(device=self.student.device).manual_seed(config.seed + rank)

        local_rewards: dict[str, list[torch.Tensor]] = defaultdict(list)
        local_masks: list[torch.Tensor] = []
        local_logs: list[dict[str, Any]] = []
        items = self._get_validation_items()

        total_batches = max(1, (len(items) + config.batch_size - 1) // config.batch_size)
        for start in tqdm(
                range(0, len(items), config.batch_size),
                total=total_batches,
                desc=f"DiffusionNFT validation {iteration}",
                position=1,
                leave=False,
                disable=not self._show_terminal_progress(),
        ):
            batch_items = items[start:start + config.batch_size]
            raw_batch = self._collate_validation_rows([item[2] for item in batch_items])
            prompts = self._extract_prompts(raw_batch)
            batch = self.student.prepare_batch(
                raw_batch,
                generator=prepare_generator,
                latents_source="zeros",
            )
            sampling_result = self._validation_sampler.sample(
                self.student,
                batch,
                generator=validation_generator,
            )
            media = self.student.decode_latents(sampling_result.latents).detach().cpu()

            # DiffusionNFT's original ``train_nft_sd3.py::eval_fn`` scores
            # validation samples and puts reward values in sample captions.
            rewards = self._reward_scorer(media, prompts)
            valid_mask = torch.tensor([item[1] for item in batch_items],
                                      device=self.student.device,
                                      dtype=torch.float32)
            local_masks.append(valid_mask)
            for key, value in rewards.items():
                local_rewards[key].append(value.to(device=self.student.device, dtype=torch.float32))

            if config.log_samples:
                for sample_idx, (global_idx, valid, _) in enumerate(batch_items):
                    if not valid:
                        continue
                    local_logs.append({
                        "index": int(global_idx),
                        "prompt": prompts[sample_idx],
                        "media": media[sample_idx],
                        "rewards": {
                            key: float(value[sample_idx].detach().float().cpu())
                            for key, value in rewards.items()
                        },
                    })

        if not local_rewards or not local_masks:
            return {}

        local_mask = torch.cat(local_masks, dim=0)
        gathered_mask = self._gather_tensor(local_mask).bool()
        metrics: dict[str, LogScalar] = {}
        for key, chunks in local_rewards.items():
            values = torch.cat(chunks, dim=0)
            gathered_values = self._gather_tensor(values.detach().float())
            valid_values = gathered_values[gathered_mask]
            if valid_values.numel() > 0:
                metrics[f"validation/reward/{key}"] = valid_values.mean()
        metrics["validation/num_prompts"] = gathered_mask.float().sum()

        if config.log_samples:
            self._log_progress(f"[DiffusionNFT] validation step {iteration}: logging samples")
            self._log_validation_samples(local_logs, iteration)
        self._log_progress(f"[DiffusionNFT] validation step {iteration}: finished")
        return metrics

    def _get_validation_items(self) -> list[tuple[int, bool, dict[str, Any]]]:
        if self._validation_items is not None:
            return self._validation_items

        dataset = getattr(getattr(self.student, "dataloader", None), "dataset", None)
        if dataset is None:
            raise RuntimeError("DiffusionNFT validation requires the student dataloader dataset")

        data_path = self._validation_config.data_path or getattr(dataset, "path", None)
        if not data_path:
            data_path = self.training_config.data.data_path
        if self._validation_config.data_path is None or data_path == getattr(dataset, "path", None):
            parquet_files = list(dataset.parquet_files)
            lengths = list(dataset.lengths)
        else:
            parquet_files, lengths = get_parquet_files_and_length(data_path)
            parquet_files = list(parquet_files)
            lengths = list(lengths)

        total_rows = int(sum(lengths))
        if total_rows <= 0:
            raise RuntimeError(f"Validation data_path {data_path!r} has no rows")
        num_prompts = min(self._validation_config.num_prompts, total_rows)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        items: list[tuple[int, bool, dict[str, Any]]] = []
        for prompt_idx, valid in validation_shard_indices(num_prompts, rank=rank, world_size=world_size):
            row = read_row_from_parquet_file(parquet_files, prompt_idx, lengths)
            row["_sample_index"] = prompt_idx
            items.append((prompt_idx, valid, row))
        self._validation_items = items
        return items

    def _collate_validation_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._collate_rows(rows)

    def _collate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = getattr(getattr(self.student, "dataloader", None), "dataset", None)
        if dataset is None or not hasattr(dataset, "parquet_schema"):
            raise RuntimeError("DiffusionNFT requires a parquet-backed student dataset")
        return collate_rows_from_parquet_schema(
            rows,
            dataset.parquet_schema,
            int(getattr(dataset, "text_padding_length", 512)),
            cfg_rate=0.0,
            seed=self._validation_config.seed,
        )

    def _log_validation_samples(
        self,
        local_logs: list[dict[str, Any]],
        iteration: int,
    ) -> None:
        logs = self._gather_objects(local_logs)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0 or not logs:
            return
        tracker = getattr(self, "tracker", None)
        if tracker is None:
            return

        artifacts = []
        fps = int(self._validation_config.fps)
        sorted_logs = sorted(logs, key=lambda x: int(x["index"]))
        if self._validation_config.max_samples is not None:
            sorted_logs = sorted_logs[:self._validation_config.max_samples]
        for item in sorted_logs:
            artifact = tracker.video(
                media_to_video_array(item["media"]),
                caption=validation_caption(str(item["prompt"]), item["rewards"]),
                fps=fps,
            )
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            tracker.log_artifacts({"validation/videos": artifacts}, step=iteration)

    def _compute_advantages(
        self,
        sample_items: list[dict[str, Any]],
        rewards: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        avg = rewards["avg"].detach().float()
        local_prompts = [prompt for item in sample_items for prompt in item["prompts"]]
        gathered_rewards = self._gather_tensor(avg)
        gathered_prompts = self._gather_prompts(local_prompts)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        local_count = int(avg.shape[0])
        if len(gathered_prompts) != int(gathered_rewards.shape[0]):
            raise RuntimeError("Gathered prompt count does not match gathered rewards")

        global_advantages = torch.empty_like(gathered_rewards)
        prompt_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, prompt in enumerate(gathered_prompts):
            prompt_to_indices[prompt].append(idx)
        for indices in prompt_to_indices.values():
            index_tensor = torch.tensor(indices, device=gathered_rewards.device, dtype=torch.long)
            group_rewards = gathered_rewards[index_tensor]
            group_std = group_rewards.std(unbiased=False)
            global_advantages[index_tensor] = (group_rewards - group_rewards.mean()) / (group_std + 1e-4)

        start = rank * local_count
        end = start + local_count
        advantages = global_advantages[start:end].to(avg.device)
        num_train_timesteps = self._num_train_timesteps()
        return advantages.unsqueeze(1).repeat(1, num_train_timesteps)

    def _inner_train(
        self,
        sample_items: list[dict[str, Any]],
        advantages: torch.Tensor,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
        self.student.transformer.train()
        self.old.transformer.eval()
        self.reference.transformer.eval()
        self._student_optimizer.zero_grad(set_to_none=True)

        samples = self._collate_samples(sample_items)
        total_samples = int(samples["latents_clean"].shape[0])
        num_train_timesteps = self._num_train_timesteps()
        if total_samples != int(advantages.shape[0]):
            raise RuntimeError("advantages and samples have mismatched batch sizes")

        effective_grad_accum = max(1, int(self.training_config.loop.gradient_accumulation_steps or 1))
        effective_grad_accum *= max(1, num_train_timesteps)
        current_accum = 0
        optimizer_steps = 0
        partial_step_micro_steps = 0
        loss_terms: dict[str, list[torch.Tensor]] = defaultdict(list)
        num_batches = max(1, total_samples // max(1, self._train_batch_size))
        training_batch_size = max(1, total_samples // num_batches)
        total_train_batches = self._num_inner_epochs * num_batches

        with tqdm(
                total=total_train_batches,
                desc=f"DiffusionNFT step {iteration}: training",
                position=1,
                leave=False,
                disable=not self._show_terminal_progress(),
        ) as progress:
            for _ in range(self._num_inner_epochs):
                perm = torch.randperm(
                    total_samples,
                    device=self.student.device,
                    generator=self.cuda_generator,
                )
                shuffled = {k: v[perm] for k, v in samples.items()}
                shuffled_adv = advantages[perm]
                perms_time = torch.stack([
                    torch.randperm(num_train_timesteps, device=self.student.device, generator=self.cuda_generator)
                    for _ in range(total_samples)
                ])
                shuffled["timesteps"] = shuffled["timesteps"][
                    torch.arange(total_samples, device=self.student.device)[:, None],
                    perms_time,
                ]
                shuffled_adv = shuffled_adv[
                    torch.arange(total_samples, device=self.student.device)[:, None],
                    perms_time,
                ]

                for batch_idx in range(num_batches):
                    start = batch_idx * training_batch_size
                    end = total_samples if batch_idx == num_batches - 1 else (batch_idx + 1) * training_batch_size
                    train_sample = {k: v[start:end] for k, v in shuffled.items()}
                    train_adv = shuffled_adv[start:end]
                    for timestep_idx in range(num_train_timesteps):
                        losses, backward_ctx = self._training_timestep_loss(
                            train_sample,
                            train_adv[:, timestep_idx],
                            timestep_idx,
                        )
                        self.student.backward(
                            losses["total_loss"],
                            backward_ctx,
                            grad_accum_rounds=effective_grad_accum,
                        )
                        current_accum += 1
                        for key, value in losses.items():
                            loss_terms[key].append(value.detach())

                        if current_accum % effective_grad_accum == 0:
                            self._clip_student_grads()
                            self._student_optimizer.step()
                            self._student_lr_scheduler.step()
                            self._update_ema()
                            self._student_optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1
                    progress.update(1)

        if current_accum % effective_grad_accum != 0:
            partial_step_micro_steps = current_accum % effective_grad_accum
            self._log_progress("[DiffusionNFT] final optimizer step uses a partial "
                               f"gradient accumulation window "
                               f"({partial_step_micro_steps}/{effective_grad_accum} "
                               "timestep micro-steps)")
            self._clip_student_grads()
            self._student_optimizer.step()
            self._student_lr_scheduler.step()
            self._update_ema()
            self._student_optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        self._log_progress(f"[DiffusionNFT] outer step {iteration}: finished inner training "
                           f"({current_accum} micro-steps, {optimizer_steps} optimizer steps)")

        reduced_local = {
            key: torch.stack(values).mean() if values else torch.zeros((), device=self.student.device)
            for key, values in loss_terms.items()
        }
        reduced = {key: self._mean_scalar_across_ranks(value) for key, value in reduced_local.items()}
        reduced.setdefault("total_loss", torch.zeros((), device=self.student.device))
        metrics: dict[str, LogScalar] = {
            "nft/iteration": float(iteration),
            "nft/num_inner_epochs": float(self._num_inner_epochs),
            "nft/inner_micro_steps": float(current_accum),
            "nft/effective_grad_accum_micro_steps": float(effective_grad_accum),
            "nft/partial_optimizer_step_micro_steps": float(partial_step_micro_steps),
            "nft/partial_optimizer_step_ratio": float(partial_step_micro_steps) / float(effective_grad_accum),
            "nft/optimizer_steps": float(optimizer_steps),
            "ema/update_count": float(self._ema_update_count),
        }
        return reduced, metrics

    @contextlib.contextmanager
    def _ema_context(self) -> Iterator[None]:
        if self._validation_use_ema and self._student_ema is not None:
            with self._student_ema.apply_to_model(self.student.transformer):
                yield
            return
        yield

    def _update_ema(self) -> None:
        if not self._ema_enabled or self._student_ema is None:
            return
        if self._ema_update_count >= self._ema_update_after_step:
            self._student_ema.update(self.student.transformer)
        self._ema_update_count += 1

    def _log_progress(self, message: str) -> None:
        if not self._terminal_progress:
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            logger.info(message)

    def _show_terminal_progress(self) -> bool:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        return bool(self._terminal_progress and rank == 0)

    def _training_timestep_loss(
        self,
        sample: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        timestep_idx: int,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, Any]]:
        # Ported from DiffusionNFT's ``scripts/train_nft_sd3.py`` inner
        # timestep loss, adapted to WanModel's predict_noise interface.
        x0 = sample["latents_clean"]
        timestep = sample["timesteps"][:, timestep_idx].to(device=x0.device)
        t = timestep.float() / float(self.student.num_train_timesteps)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = torch.randn(
            x0.shape,
            device=x0.device,
            dtype=x0.dtype,
            generator=self.cuda_generator,
        )
        xt = ((1 - t_expanded) * x0 + t_expanded * noise).to(dtype=x0.dtype)
        batch = self._make_training_batch(sample, timestep)

        with torch.no_grad():
            old_prediction = self.old.predict_noise(
                xt,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            ).detach()
            ref_forward_prediction = self.reference.predict_noise(
                xt,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            ).detach()

        forward_prediction = self.student.predict_noise(
            xt,
            timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )

        advantages_clip = torch.clamp(advantages, -self._adv_clip_max, self._adv_clip_max)
        if self._adv_mode == "positive_only":
            advantages_clip = torch.clamp(advantages_clip, 0, self._adv_clip_max)
        elif self._adv_mode == "negative_only":
            advantages_clip = torch.clamp(advantages_clip, -self._adv_clip_max, 0)
        elif self._adv_mode == "one_only":
            advantages_clip = torch.where(advantages_clip > 0, torch.ones_like(advantages_clip),
                                          torch.zeros_like(advantages_clip))
        elif self._adv_mode == "binary":
            advantages_clip = torch.sign(advantages_clip)

        normalized_advantages_clip = (advantages_clip / self._adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(normalized_advantages_clip, 0, 1)

        positive_prediction = self._nft_beta * forward_prediction + (1 - self._nft_beta) * old_prediction.detach()
        implicit_negative_prediction = ((1.0 + self._nft_beta) * old_prediction.detach() -
                                        self._nft_beta * forward_prediction)

        x0_prediction = xt - t_expanded * positive_prediction
        with torch.no_grad():
            weight_factor = torch.abs(x0_prediction.double() - x0.double())
            weight_factor = weight_factor.mean(dim=tuple(range(1, x0.ndim)), keepdim=True).clip(min=0.00001)
        positive_loss = ((x0_prediction - x0)**2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))

        negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
        with torch.no_grad():
            negative_weight_factor = torch.abs(negative_x0_prediction.double() - x0.double())
            negative_weight_factor = negative_weight_factor.mean(dim=tuple(range(1, x0.ndim)),
                                                                 keepdim=True).clip(min=0.00001)
        negative_loss = ((negative_x0_prediction - x0)**2 / negative_weight_factor).mean(dim=tuple(range(1, x0.ndim)))

        ori_policy_loss = r * positive_loss / self._nft_beta + (1.0 - r) * negative_loss / self._nft_beta
        policy_loss = (ori_policy_loss * self._adv_clip_max).mean()

        kl_div_loss = ((forward_prediction - ref_forward_prediction)**2).mean(dim=tuple(range(1, x0.ndim))).mean()
        total_loss = policy_loss + self._kl_beta * kl_div_loss
        losses = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "unweighted_policy_loss": ori_policy_loss.mean(),
            "kl_div_loss": kl_div_loss,
            "old_deviate": ((forward_prediction - old_prediction)**2).mean(),
            "old_kl_div": ((old_prediction - ref_forward_prediction)**2).mean(),
            "x0_norm": torch.mean(x0**2),
        }
        return losses, (batch.timesteps, batch.attn_metadata)

    def _repeat_first_prompt(
        self,
        raw_batch: dict[str, Any],
        repeat: int,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in raw_batch.items():
            if torch.is_tensor(value) and value.shape[0] > 0:
                out[key] = value[:1].repeat((repeat, ) + (1, ) * (value.ndim - 1))
            elif key == "info_list" and isinstance(value, list) and value:
                out[key] = [dict(value[0]) for _ in range(repeat)]
            elif isinstance(value, list) and value:
                out[key] = [value[0] for _ in range(repeat)]
            else:
                out[key] = value
        return out

    @staticmethod
    def _extract_prompts(raw_batch: dict[str, Any]) -> list[str]:
        infos = raw_batch.get("info_list")
        if isinstance(infos, list) and infos:
            prompts: list[str] = []
            for info in infos:
                if isinstance(info, dict):
                    prompts.append(str(info.get("prompt") or info.get("caption") or ""))
                else:
                    prompts.append("")
            return prompts
        captions = raw_batch.get("caption_text")
        if isinstance(captions, list):
            return [str(c) for c in captions]
        raise ValueError("Could not find prompts in batch info_list or caption_text")

    def _collate_samples(self, sample_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        keys = ["encoder_hidden_states", "encoder_attention_mask", "latents_clean", "timesteps"]
        return {key: torch.cat([item[key] for item in sample_items], dim=0).to(self.student.device) for key in keys}

    @staticmethod
    def _make_training_batch(
        sample: dict[str, torch.Tensor],
        timestep: torch.Tensor,
    ) -> TrainingBatch:
        batch = TrainingBatch()
        batch.encoder_hidden_states = sample["encoder_hidden_states"]
        batch.encoder_attention_mask = sample["encoder_attention_mask"]
        batch.conditional_dict = {
            "encoder_hidden_states": batch.encoder_hidden_states,
            "encoder_attention_mask": batch.encoder_attention_mask,
        }
        batch.timesteps = timestep
        batch.raw_latent_shape = tuple(sample["latents_clean"].shape)
        return batch

    def _num_train_timesteps(self) -> int:
        schedule_len = self._sample_steps
        if self._sampling_config.timesteps is not None:
            schedule_len = len(self._sampling_config.timesteps)
        elif self._sampling_config.sigmas is not None:
            schedule_len = len(self._sampling_config.sigmas)
        return max(1, min(schedule_len, int(schedule_len * self._timestep_fraction)))

    def _clip_student_grads(self) -> None:
        if self._max_grad_norm <= 0.0:
            return
        clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in self.student.transformer.parameters()],
            self._max_grad_norm,
            foreach=None,
        )

    def _sync_old_from_student(self) -> None:
        with torch.no_grad():
            for src, tgt in zip(self.student.transformer.parameters(), self.old.transformer.parameters(), strict=True):
                tgt.data.copy_(src.detach().data)

    def _update_old_model(self, iteration: int) -> None:
        decay = self._return_decay(iteration, self._decay_type)
        with torch.no_grad():
            for src, tgt in zip(self.student.transformer.parameters(), self.old.transformer.parameters(), strict=True):
                tgt.data.copy_(tgt.detach().data * decay + src.detach().data * (1.0 - decay))

    @staticmethod
    def _return_decay(step: int, decay_type: int) -> float:
        # Ported from DiffusionNFT's ``scripts/train_nft_sd3.py::return_decay``.
        if decay_type == 0:
            flat, uprate, uphold = 0, 0.0, 0.0
        elif decay_type == 1:
            flat, uprate, uphold = 0, 0.001, 0.5
        elif decay_type == 2:
            flat, uprate, uphold = 75, 0.0075, 0.999
        else:
            raise ValueError(f"Unsupported decay_type: {decay_type}")
        if step < flat:
            return 0.0
        return min((step - flat) * uprate, uphold)

    def _reward_metrics(self, rewards: dict[str, torch.Tensor]) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}
        for key, value in rewards.items():
            gathered = self._gather_tensor(value.detach().float())
            metrics[f"reward/{key}"] = gathered.mean()
        return metrics

    def _reward_diagnostic_metrics(
        self,
        sample_items: list[dict[str, Any]],
        rewards: dict[str, torch.Tensor],
    ) -> dict[str, LogScalar]:
        avg = rewards["avg"].detach().float()
        local_prompts = [prompt for item in sample_items for prompt in item["prompts"]]
        gathered_rewards = self._gather_tensor(avg)
        gathered_prompts = self._gather_prompts(local_prompts)
        if len(gathered_prompts) != int(gathered_rewards.shape[0]):
            raise RuntimeError("Gathered prompt count does not match gathered rewards")

        prompt_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, prompt in enumerate(gathered_prompts):
            prompt_to_indices[prompt].append(idx)
            self._trained_prompt_hashes.add(hash(prompt))
        if not prompt_to_indices:
            return {}

        group_sizes = []
        group_stds = []
        grouped_rewards = []
        for indices in prompt_to_indices.values():
            index_tensor = torch.tensor(indices, device=gathered_rewards.device, dtype=torch.long)
            group_rewards = gathered_rewards[index_tensor].float()
            grouped_rewards.append(group_rewards)
            group_sizes.append(float(group_rewards.numel()))
            group_stds.append(group_rewards.std(unbiased=False))

        std_tensor = torch.stack(group_stds)
        metrics: dict[str, LogScalar] = {
            "group_size": sum(group_sizes) / max(1, len(group_sizes)),
            "trained_prompt_num": float(len(self._trained_prompt_hashes)),
            "zero_std_ratio": (std_tensor == 0).float().mean(),
            "reward_std_mean": std_tensor.mean(),
        }
        for top_percentage in (100, 75, 50, 25, 10):
            metrics[f"mean_reward_{top_percentage}"] = self._mean_of_top_group_rewards(
                grouped_rewards,
                top_percentage,
            )
        return metrics

    @staticmethod
    def _mean_of_top_group_rewards(
        grouped_rewards: list[torch.Tensor],
        top_percentage: int,
    ) -> torch.Tensor:
        per_prompt_means = []
        for rewards in grouped_rewards:
            if rewards.numel() == 0:
                continue
            if top_percentage == 100:
                per_prompt_means.append(rewards.mean())
                continue
            threshold = torch.quantile(
                rewards.float(),
                (100 - float(top_percentage)) / 100.0,
            )
            top_rewards = rewards[rewards >= threshold]
            if top_rewards.numel() > 0:
                per_prompt_means.append(top_rewards.mean())
        if not per_prompt_means:
            device = grouped_rewards[0].device if grouped_rewards else torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.stack(per_prompt_means).mean()

    @staticmethod
    def _gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return tensor.detach()
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0).detach()

    @staticmethod
    def _mean_scalar_across_ranks(value: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return value.detach()
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        return reduced

    @staticmethod
    def _gather_prompts(prompts: list[str]) -> list[str]:
        if not dist.is_available() or not dist.is_initialized():
            return list(prompts)
        gathered: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, list(prompts))
        flat: list[str] = []
        for rank_prompts in gathered:
            if rank_prompts is None:
                continue
            flat.extend(rank_prompts)
        return flat

    @staticmethod
    def _gather_objects(items: list[Any]) -> list[Any]:
        if not dist.is_available() or not dist.is_initialized():
            return list(items)
        gathered: list[list[Any] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, list(items))
        flat: list[Any] = []
        for rank_items in gathered:
            if rank_items is None:
                continue
            flat.extend(rank_items)
        return flat

    def _read_int(self, key: str, default: int) -> int:
        value = get_optional_int(self.method_config, key, where=f"method.{key}")
        return int(default if value is None else value)

    def _read_float(self, key: str, default: float) -> float:
        value = get_optional_float(self.method_config, key, where=f"method.{key}")
        return float(default if value is None else value)

    def _parse_sampling_config(self) -> SamplingConfig:
        raw = self.method_config.get("sampling", None)
        if isinstance(raw, dict):
            raw = dict(raw)
            if raw.get("flow_shift", None) in (None, "inherit"):
                pipeline_config = getattr(self.training_config, "pipeline_config", None)
                flow_shift = getattr(pipeline_config, "flow_shift", None)
                if flow_shift is not None:
                    raw["flow_shift"] = float(flow_shift)
        return SamplingConfig.from_mapping(raw)

    def _parse_validation_sampling_config(self) -> SamplingConfig:
        raw = dict(self.method_config.get("sampling", {}) or {})
        validation_sampling = self._validation_config.sampling
        if validation_sampling is not None:
            raw.update(validation_sampling)
        if validation_sampling is None or "num_steps" not in validation_sampling:
            raw["num_steps"] = self._validation_config.num_steps
        if raw.get("flow_shift") in (None, "inherit"):
            pipeline_config = getattr(self.training_config, "pipeline_config", None)
            flow_shift = getattr(pipeline_config, "flow_shift", None)
            if flow_shift is not None:
                raw["flow_shift"] = float(flow_shift)
        return SamplingConfig.from_mapping(raw)

    def _parse_ema_config(self) -> dict[str, bool | float | int]:
        raw = self.method_config.get("ema", None)
        config: dict[str, bool | float | int] = {
            "enabled": True,
            "decay": 0.9,
            "update_after_step": 0,
            "validation": True,
        }
        if raw is None:
            return config
        if isinstance(raw, bool | str | int):
            config["enabled"] = self._coerce_bool(raw, where="method.ema")
            return config
        if not isinstance(raw, dict):
            raise ValueError("method.ema must be a bool or mapping")
        allowed = set(config)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unsupported method.ema key(s): {unknown}")
        if "enabled" in raw:
            config["enabled"] = self._coerce_bool(raw["enabled"], where="method.ema.enabled")
        if "decay" in raw:
            decay = float(raw["decay"])
            if not 0.0 <= decay <= 1.0:
                raise ValueError("method.ema.decay must be in [0, 1]")
            config["decay"] = decay
        if "update_after_step" in raw:
            update_after_step = int(raw["update_after_step"])
            if update_after_step < 0:
                raise ValueError("method.ema.update_after_step must be >= 0")
            config["update_after_step"] = update_after_step
        if "validation" in raw:
            config["validation"] = self._coerce_bool(raw["validation"], where="method.ema.validation")
        return config

    @staticmethod
    def _coerce_bool(value: Any, *, where: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError(f"{where} must be a boolean")
