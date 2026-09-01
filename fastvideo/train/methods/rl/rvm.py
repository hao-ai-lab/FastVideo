# SPDX-License-Identifier: Apache-2.0
"""Reward-based velocity matching for the four-step FastH3 student.

The implementation follows the endpoint-only RVM algorithm: a frozen behavior
snapshot (the current LoRA before an optimizer update) generates endpoints,
black-box video rewards produce group-relative signed coefficients, and each
endpoint is analytically noised once for the ordinary flow-matching regression
update. Only the new quality LoRA is trainable. Audio receives no reward term;
an optional function-space anchor preserves the released FastH3 audio field.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.dataset.parquet_dataset_map_style import (
    get_parquet_files_and_length,
    read_row_from_parquet_file,
)
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import (
    get_dp_rank,
    get_dp_world_size,
    get_sp_group,
    get_world_group,
)
from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.rl.common import (
    H3RVMSamplingConfig,
    MiniMaxH3RVMSampler,
    detached_rvm_surrogate,
    five_percent_interval,
    media_to_video_array,
    partition_indices,
    standardize_group_rewards,
    validation_caption,
    visual_text_from_h3_prompt,
)
from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer
from fastvideo.train.models.minimax_h3.minimax_h3_rvm import MiniMaxH3RVMModel
from fastvideo.train.utils.lora_context import temporarily_disable_lora
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler,
    clip_grad_norm_if_needed,
)

logger = init_logger(__name__)


@dataclass(slots=True)
class _RVMItem:
    raw_batch: dict[str, Any]
    endpoint: torch.Tensor
    advantage: float
    prompt: str


@dataclass(slots=True)
class _ValidationSettings:
    every_steps: int
    num_prompts: int
    seed: int
    data_path: str | None
    run_at_start: bool
    log_samples: bool
    log_sample_limit: int
    save_videos: bool


class RVMMethod(TrainingMethod):
    """On-policy RVM specialized to FastH3's packed video/audio field."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if set(role_models) != {"student"}:
            raise ValueError("RVMMethod requires exactly one role: models.student")
        if not isinstance(self.student, MiniMaxH3RVMModel):
            raise TypeError("RVMMethod requires models.student._target_=MiniMaxH3RVMModel")
        if not self.student._trainable:
            raise ValueError("RVMMethod requires a trainable student")

        self.student.init_preprocessors(self.training_config)
        self._sampling_config = H3RVMSamplingConfig.from_mapping(self.method_config.get("sampling"))
        self._sampler = MiniMaxH3RVMSampler(self._sampling_config)

        self._samples_per_prompt = self._read_positive_int("samples_per_prompt", 8)
        self._prompt_groups_per_rollout = self._read_positive_int("prompt_groups_per_rollout", 32)
        self._updates_per_rollout = self._read_positive_int("optimizer_updates_per_rollout", 2)
        self._advantage_scale = self._read_float("advantage_scale", 0.1)
        self._advantage_eps = self._read_float("advantage_eps", 1e-4)
        self._advantage_clip = self._read_float("advantage_clip", 5.0)
        self._positive_only_steps = max(0, self._read_int("positive_only_steps", 0))
        self._video_anchor_beta = self._read_float("video_anchor_beta", 0.0)
        self._audio_anchor_beta = self._read_float("audio_anchor_beta", 1e-3)
        self._audio_loss_weight = self._read_float("audio_loss_weight", 1.0)
        self._max_grad_norm = self._read_float("max_grad_norm", 1.0)
        self._terminal_progress = bool(self.method_config.get("terminal_progress", True))
        self._reward_device_spec = str(self.method_config.get("reward_device", "cuda") or "cuda")

        if self._samples_per_prompt < 2:
            raise ValueError("samples_per_prompt must be at least two for group-relative RVM")
        if self._updates_per_rollout > self._prompt_groups_per_rollout * self._samples_per_prompt:
            raise ValueError("optimizer_updates_per_rollout cannot exceed the rollout sample count")
        if self._video_anchor_beta < 0 or self._audio_anchor_beta < 0:
            raise ValueError("anchor betas must be nonnegative")
        if self._audio_loss_weight < 0:
            raise ValueError("audio_loss_weight must be nonnegative")

        reward_config = self.method_config.get("reward_fn")
        if not isinstance(reward_config, Mapping) or not reward_config:
            raise ValueError("method.reward_fn must be a nonempty reward mapping")
        self._reward_config = dict(reward_config)
        raw_rewards = self._reward_config.get("rewards", self._reward_config)
        if not isinstance(raw_rewards, Mapping) or not raw_rewards:
            raise ValueError("method.reward_fn.rewards must be a nonempty mapping")
        self._reward_names = [str(name).strip().lower() for name in raw_rewards]
        self._reward_keys = [*self._reward_names, "avg"]

        self._validation = self._parse_validation_settings()
        self._validation_items: list[int] | None = None
        self._reward_scorer: Any | None = None
        self._buffer: list[_RVMItem] = []
        self._buffer_partitions: list[torch.Tensor] = []
        self._buffer_cursor = 0
        self._collection_count = 0
        self._collection_metrics: dict[str, LogScalar] = {}

        self._sp_group = None
        self._is_sp_leader = True
        self._dp_rank = 0
        self._dp_world_size = 1
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._lr_scheduler}

    def manages_optimization(self) -> bool:
        return True

    def get_optimizers(self, iteration: int) -> Sequence[torch.optim.Optimizer]:
        del iteration
        return [self._optimizer]

    def get_lr_schedulers(self, iteration: int) -> Sequence[Any]:
        del iteration
        return [self._lr_scheduler]

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del batch, iteration
        raise RuntimeError("RVMMethod owns its sample-then-update loop; use managed_train_step()")

    def on_train_start(self) -> None:
        super().on_train_start()
        self._sp_group = get_sp_group()
        if self._sp_group is None:
            raise RuntimeError("RVM requires an initialized sequence-parallel process group")
        self._is_sp_leader = int(self._sp_group.rank_in_group) == 0
        self._dp_rank = int(get_dp_rank())
        self._dp_world_size = int(get_dp_world_size())
        if self._prompt_groups_per_rollout % self._dp_world_size != 0:
            raise ValueError("prompt_groups_per_rollout must be divisible by the number of data-parallel H3 replicas "
                             f"({self._prompt_groups_per_rollout} vs {self._dp_world_size})")
        if self._is_sp_leader:
            reward_device: torch.device | str
            if self._reward_device_spec == "cuda":
                reward_device = self.student.device
            else:
                reward_device = torch.device(self._reward_device_spec)
            self._reward_scorer = build_multi_reward_scorer(
                self._reward_config,
                device=reward_device,
            )
        self._log_progress(
            "RVM initialized: "
            f"K={self._samples_per_prompt}, global_prompt_groups={self._prompt_groups_per_rollout}, "
            f"updates_per_rollout={self._updates_per_rollout}, validation_every={self._validation.every_steps}")

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        if not self._buffer_partitions or self._buffer_cursor >= len(self._buffer_partitions):
            self._collect_rollouts(data_stream, iteration)

        partition = self._buffer_partitions[self._buffer_cursor]
        self._buffer_cursor += 1
        loss_map, metrics = self._train_partition(partition, iteration)
        if self._collection_metrics:
            metrics.update(self._collection_metrics)
            self._collection_metrics = {}
        metrics["rvm/buffer_partition"] = float(self._buffer_cursor)
        metrics["rvm/rollout_collection"] = float(self._collection_count)

        if self._buffer_cursor >= len(self._buffer_partitions):
            self._buffer.clear()
            self._buffer_partitions.clear()
            self._buffer_cursor = 0
        return loss_map, {}, metrics

    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        should_run = (iteration == 0
                      and self._validation.run_at_start) or (iteration > 0
                                                             and iteration % self._validation.every_steps == 0)
        if not should_run:
            return {}
        return self._run_validation(iteration)

    def _collect_rollouts(self, data_stream: Iterator[dict[str, Any]], iteration: int) -> None:
        self.student.transformer.eval()
        self._buffer = []
        local_groups = self._prompt_groups_per_rollout // self._dp_world_size
        reward_sums: dict[str, float] = defaultdict(float)
        reward_count = 0
        zero_std_count = 0

        self._log_progress(f"RVM step {iteration}: collecting {local_groups} local prompt groups x "
                           f"{self._samples_per_prompt} samples")
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

                reward_dict = self._score_endpoint_group(endpoints, visual_prompt)
                advantages, group_std = standardize_group_rewards(
                    reward_dict["avg"],
                    scale=self._advantage_scale,
                    eps=self._advantage_eps,
                    clip=self._advantage_clip,
                    positive_only=iteration <= self._positive_only_steps,
                )
                if float(group_std) == 0.0:
                    zero_std_count += 1
                for key, values in reward_dict.items():
                    reward_sums[key] += float(values.float().sum())
                reward_count += self._samples_per_prompt

                for sample_index, endpoint in enumerate(endpoints):
                    self._buffer.append(
                        _RVMItem(
                            raw_batch=raw_batch,
                            endpoint=endpoint,
                            advantage=float(advantages[sample_index]),
                            prompt=full_prompt,
                        ))

        if torch.cuda.is_available():
            # VAE/reward allocations must be released before the metric
            # all-reduces below ask NCCL for its communication buffers.
            torch.cuda.empty_cache()
        self._collection_count += 1
        cpu_generator = torch.Generator(
            device="cpu").manual_seed(int(self.training_config.data.seed) + 100_000 + self._collection_count)
        self._buffer_partitions = partition_indices(
            len(self._buffer),
            self._updates_per_rollout,
            generator=cpu_generator,
            device="cpu",
        )
        self._buffer_cursor = 0

        metrics: dict[str, LogScalar] = {
            "rvm/rollout_samples": float(self._prompt_groups_per_rollout * self._samples_per_prompt),
            "rvm/zero_std_group_ratio": self._global_leader_mean(float(zero_std_count), float(local_groups)),
        }
        for key, value in reward_sums.items():
            metrics[f"reward/{key}"] = self._global_leader_mean(value, float(reward_count))
        self._collection_metrics = metrics

    def _score_endpoint_group(
        self,
        endpoints: list[torch.Tensor],
        visual_prompt: str,
    ) -> dict[str, torch.Tensor]:
        if self._sp_group is None:
            raise RuntimeError("on_train_start() has not initialized the SP group")
        leader_result: dict[str, torch.Tensor] = {}
        if self._is_sp_leader:
            if self._reward_scorer is None:
                raise RuntimeError("SP leader has no reward scorer")
            packed = torch.cat(endpoints, dim=0).to(self.student.device)
            media = self.student.decode_latents(packed).cpu()
            leader_result = self._reward_scorer(
                media,
                [visual_prompt] * self._samples_per_prompt,
            )

        broadcast: dict[str, torch.Tensor] = {}
        for key in self._reward_keys:
            if self._is_sp_leader:
                value = leader_result[key].to(device=self.student.device, dtype=torch.float32)
            else:
                value = torch.zeros(
                    self._samples_per_prompt,
                    device=self.student.device,
                    dtype=torch.float32,
                )
            self._sp_group.broadcast(value, src=0)
            broadcast[key] = value.detach().cpu()
        return broadcast

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

        totals: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=self.student.device))
        timestep_counts = torch.zeros(
            len(self._sampling_config.denoising_steps),
            device=self.student.device,
            dtype=torch.float32,
        )
        for item_index in partition.tolist():
            item = self._buffer[int(item_index)]
            losses, backward_context, timestep_index = self._endpoint_loss(item)
            self.student.backward(
                losses["total_loss"],
                backward_context,
                grad_accum_rounds=count,
            )
            for key, value in losses.items():
                totals[key] = totals[key] + value.detach()
            timestep_counts[timestep_index] += 1.0

        grad_norm = clip_grad_norm_if_needed(self.student.transformer, self._max_grad_norm)
        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)

        averaged = {key: value / count for key, value in totals.items()}
        reduced = {key: self._mean_scalar_across_ranks(value) for key, value in averaged.items()}
        reduced.setdefault("total_loss", torch.zeros((), device=self.student.device))
        metrics: dict[str, LogScalar] = {
            "rvm/optimizer_step": float(iteration),
            "rvm/local_train_samples": float(count),
            "rvm/grad_norm": float(grad_norm),
            "rvm/learning_rate": float(self._optimizer.param_groups[0]["lr"]),
        }
        total_timestep_counts = timestep_counts.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_timestep_counts, op=dist.ReduceOp.SUM)
        denom = total_timestep_counts.sum().clamp_min(1.0)
        for index, step in enumerate(self._sampling_config.denoising_steps):
            metrics[f"rvm/timestep_fraction_{step}"] = total_timestep_counts[index] / denom
        return reduced, metrics

    def _endpoint_loss(
        self,
        item: _RVMItem,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, Any], int]:
        batch = self.student.prepare_batch(
            item.raw_batch,
            generator=self._require_generator(),
            latents_source="zeros",
        )
        if batch.latents is None:
            raise RuntimeError("H3 batch preparation returned no packed latent geometry")
        endpoint = item.endpoint.to(device=self.student.device, dtype=batch.latents.dtype)
        timestep_index_tensor = torch.randint(
            0,
            len(self._sampling_config.denoising_steps),
            (1, ),
            device=self.student.device,
            generator=self._require_generator(),
        )
        timestep_index = int(timestep_index_tensor.item())
        timestep = torch.tensor(
            [self._sampling_config.denoising_steps[timestep_index]],
            device=self.student.device,
            dtype=torch.long,
        )
        self.student.refresh_vsa_metadata(batch, current_timestep=timestep_index)
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
        backward_context = (batch.timesteps, batch.attn_metadata_vsa)

        reference: torch.Tensor | None = None
        if self._video_anchor_beta != 0.0 or self._audio_anchor_beta != 0.0:
            with temporarily_disable_lora(self.student.transformer), torch.no_grad():
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
        video_reference = None if reference is None else reference[:, video_slice]
        audio_reference = None if reference is None else reference[:, audio_slice]

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
            audio_loss = prediction[:, audio_slice].sum() * 0.0
        total_loss = video_loss + self._audio_loss_weight * audio_loss
        return (
            {
                "total_loss": total_loss,
                "video_rvm_loss": video_loss,
                "audio_anchor_loss": audio_loss,
                "advantage": torch.tensor(float(item.advantage), device=self.student.device),
            },
            backward_context,
            timestep_index,
        )

    @torch.no_grad()
    def _run_validation(self, iteration: int) -> dict[str, LogScalar]:
        if self._sp_group is None:
            raise RuntimeError("on_train_start() has not initialized distributed groups")
        was_training = self.student.transformer.training
        self.student.transformer.eval()
        local_sums: dict[str, float] = defaultdict(float)
        local_count = 0
        local_artifacts: list[dict[str, Any]] = []
        indices = self._get_validation_indices()
        files, lengths, schema, text_padding_length = self._validation_dataset_metadata()
        output_root = Path(self.training_config.checkpoint.output_dir) / "validation" / f"step-{iteration:06d}"
        if self._is_sp_leader and self._validation.save_videos:
            output_root.mkdir(parents=True, exist_ok=True)

        for global_index in indices:
            row = read_row_from_parquet_file(files, global_index, lengths)
            row["_sample_index"] = global_index
            raw_batch = collate_rows_from_parquet_schema(
                [row],
                schema,
                text_padding_length,
                cfg_rate=0.0,
                seed=self._validation.seed,
            )
            full_prompt = self._extract_prompt(raw_batch)
            visual_prompt = visual_text_from_h3_prompt(full_prompt)
            generator = torch.Generator(device=self.student.device).manual_seed(self._validation.seed +
                                                                                int(global_index))
            prepared = self.student.prepare_batch(
                raw_batch,
                generator=generator,
                latents_source="zeros",
            )
            endpoint = self._sampler.sample(self.student, prepared, generator=generator).endpoint

            rewards: dict[str, torch.Tensor] = {}
            media: torch.Tensor | None = None
            if self._is_sp_leader:
                if self._reward_scorer is None:
                    raise RuntimeError("SP leader has no reward scorer")
                media = self.student.decode_latents(endpoint).cpu()
                rewards = self._reward_scorer(media, [visual_prompt])
            for key in self._reward_keys:
                if self._is_sp_leader:
                    value = rewards[key].to(device=self.student.device, dtype=torch.float32)
                else:
                    value = torch.zeros(1, device=self.student.device, dtype=torch.float32)
                self._sp_group.broadcast(value, src=0)
                if self._is_sp_leader:
                    local_sums[key] += float(value[0])
            if self._is_sp_leader:
                local_count += 1
                assert media is not None
                path: str | None = None
                if self._validation.save_videos:
                    path = str(output_root / f"prompt-{global_index:06d}.mp4")
                    self._write_video(path, media[0])
                if self._validation.log_samples and len(local_artifacts) < self._validation.log_sample_limit:
                    local_artifacts.append({
                        "index": int(global_index),
                        "prompt": full_prompt,
                        "path": path,
                        "media": None if path is not None else media[0],
                        "rewards": {
                            key: float(rewards[key][0])
                            for key in self._reward_keys
                        },
                    })

        metrics: dict[str, LogScalar] = {}
        for key in self._reward_keys:
            metrics[f"validation/reward/{key}"] = self._global_leader_mean(local_sums[key], float(local_count))
        metrics["validation/num_prompts"] = self._global_leader_sum(float(local_count))
        if self._validation.log_samples:
            self._log_validation_artifacts(local_artifacts, iteration)
        if was_training:
            self.student.transformer.train()
        self._log_progress(
            f"RVM validation step {iteration}: evaluated {int(metrics['validation/num_prompts'])} prompts")
        return metrics

    def _get_validation_indices(self) -> list[int]:
        if self._validation_items is None:
            _files, lengths, _schema, _padding = self._validation_dataset_metadata()
            total_rows = int(sum(lengths))
            num_prompts = min(100, self._validation.num_prompts, total_rows)
            generator = torch.Generator(device="cpu").manual_seed(self._validation.seed)
            selected = torch.randperm(total_rows, generator=generator)[:num_prompts].tolist()
            self._validation_items = [int(index) for index in selected]
            if self._validation.data_path is None:
                self._log_progress(
                    "No separate validation parquet was configured; using a deterministic held-out sample "
                    f"of {num_prompts} training prompts.")
        return self._validation_items[self._dp_rank::self._dp_world_size]

    def _validation_dataset_metadata(self) -> tuple[list[str], list[int], Any, int]:
        dataset = getattr(getattr(self.student, "dataloader", None), "dataset", None)
        if dataset is None or not hasattr(dataset, "parquet_schema"):
            raise RuntimeError("RVM requires the student's parquet-backed dataset")
        if self._validation.data_path is None:
            files = list(dataset.parquet_files)
            lengths = list(dataset.lengths)
        else:
            found_files, found_lengths = get_parquet_files_and_length(self._validation.data_path)
            files = list(found_files)
            lengths = list(found_lengths)
        return files, lengths, dataset.parquet_schema, int(getattr(dataset, "text_padding_length", 512))

    def _log_validation_artifacts(self, local: list[dict[str, Any]], iteration: int) -> None:
        gathered = self._gather_objects(local)
        if int(get_world_group().rank) != 0 or not gathered or self.tracker is None:
            return
        artifacts = []
        for item in sorted(gathered, key=lambda value: int(value["index"]))[:self._validation.log_sample_limit]:
            data = item["path"] if item["path"] is not None else media_to_video_array(item["media"])
            artifact = self.tracker.video(
                data,
                caption=validation_caption(str(item["prompt"]), item["rewards"]),
                fps=24,
            )
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            self.tracker.log_artifacts({"validation/videos": artifacts}, step=iteration)

    @staticmethod
    def _write_video(path: str, media: torch.Tensor) -> None:
        import imageio.v2 as imageio

        frames = media_to_video_array(media).transpose(0, 2, 3, 1)
        imageio.mimsave(path, frames, fps=24, codec="libx264", quality=8)

    def _parse_validation_settings(self) -> _ValidationSettings:
        raw = self.method_config.get("validation")
        raw = dict(raw or {})
        every = int(raw.get("every_steps", 0) or 0)
        if every <= 0:
            every = five_percent_interval(int(self.training_config.loop.max_train_steps))
        return _ValidationSettings(
            every_steps=every,
            num_prompts=min(100, max(1, int(raw.get("num_prompts", 100) or 100))),
            seed=int(raw.get("seed", 42) or 42),
            data_path=(None if raw.get("data_path") in (None, "") else str(raw["data_path"])),
            run_at_start=bool(raw.get("run_at_start", True)),
            log_samples=bool(raw.get("log_samples", True)),
            log_sample_limit=min(100, max(0, int(raw.get("log_sample_limit", 16) or 0))),
            save_videos=bool(raw.get("save_videos", True)),
        )

    def _init_optimizer_and_scheduler(self) -> None:
        params = [parameter for parameter in self.student.transformer.parameters() if parameter.requires_grad]
        self._optimizer, self._lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=float(self.training_config.optimizer.learning_rate),
            betas=tuple(self.training_config.optimizer.betas),
            scheduler_name=str(self.training_config.optimizer.lr_scheduler),
        )

    def _require_generator(self) -> torch.Generator:
        if self.cuda_generator is None:
            raise RuntimeError("RVM CUDA generator is not initialized")
        return self.cuda_generator

    @staticmethod
    def _extract_prompt(raw_batch: dict[str, Any]) -> str:
        infos = raw_batch.get("info_list")
        if isinstance(infos, list) and infos:
            info = infos[0]
            if isinstance(info, dict):
                return str(info.get("prompt") or info.get("caption") or "")
        captions = raw_batch.get("caption_text")
        if isinstance(captions, list) and captions:
            return str(captions[0])
        raise ValueError("Could not find an H3 prompt in info_list or caption_text")

    @classmethod
    def _clone_raw_batch(cls, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cls._clone_raw_batch(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._clone_raw_batch(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_raw_batch(item) for item in value)
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _global_leader_mean(self, local_sum: float, local_count: float) -> torch.Tensor:
        values = torch.tensor(
            [local_sum if self._is_sp_leader else 0.0, local_count if self._is_sp_leader else 0.0],
            device=self.student.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return (values[0] / values[1].clamp_min(1.0)).to(torch.float32)

    def _global_leader_sum(self, local_value: float) -> torch.Tensor:
        value = torch.tensor(
            local_value if self._is_sp_leader else 0.0,
            device=self.student.device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @staticmethod
    def _mean_scalar_across_ranks(value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        return reduced

    @staticmethod
    def _gather_objects(items: list[Any]) -> list[Any]:
        if not dist.is_available() or not dist.is_initialized():
            return list(items)
        gathered: list[list[Any] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, list(items))
        flattened: list[Any] = []
        for rank_items in gathered:
            if rank_items:
                flattened.extend(rank_items)
        return flattened

    def _log_progress(self, message: str) -> None:
        if self._terminal_progress and int(get_world_group().rank) == 0:
            logger.info(message)

    def _read_int(self, key: str, default: int) -> int:
        return int(self.method_config.get(key, default) if self.method_config.get(key) is not None else default)

    def _read_positive_int(self, key: str, default: int) -> int:
        value = self._read_int(key, default)
        if value <= 0:
            raise ValueError(f"method.{key} must be positive")
        return value

    def _read_float(self, key: str, default: float) -> float:
        raw = self.method_config.get(key, default)
        return float(default if raw is None else raw)


__all__ = ["RVMMethod"]
