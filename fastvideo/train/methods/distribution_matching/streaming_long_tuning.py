# SPDX-License-Identifier: Apache-2.0
"""LongLive-style multi-stage self-forcing for streaming rollouts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.distribution_matching.self_forcing import (
    SelfForcingMethod, )
from fastvideo.train.models.base import CausalModelBase
from fastvideo.train.utils.config import get_optional_float, get_optional_int


@dataclass(frozen=True, slots=True)
class DistillStage:
    """Resolved multi-stage distillation stage."""

    name: str
    start_step: int
    end_step: int | None
    num_latent_t: int
    streaming_training: bool
    streaming_chunk_size: int | None = None
    streaming_max_length: int | None = None
    streaming_min_new_frame: int | None = None
    streaming_fixed_overlap_latents: int | None = None
    train_first_chunk: bool = True


@dataclass(slots=True)
class _StreamingState:
    stage: DistillStage
    batch: Any
    current_length: int = 0
    previous_latents: torch.Tensor | None = None
    cache_tag: str = "streaming_long"


@dataclass(frozen=True, slots=True)
class _StreamingChunkInfo:
    chunk_start: int
    chunk_end: int
    train_start: int
    train_end: int
    new_frames: int
    overlap: int
    current_length: int
    max_length: int


def _as_bool(raw: Any, *, where: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"Expected bool at {where}, got {type(raw).__name__}")


def _as_int(raw: Any, *, where: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"Expected int at {where}, got bool")
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        return int(raw)
    raise ValueError(f"Expected int at {where}, got {type(raw).__name__}")


def parse_multi_phased_distill_schedule(
    raw: Any,
    *,
    default_num_latent_t: int,
    default_streaming_chunk_size: int | None = None,
    default_streaming_max_length: int | None = None,
) -> list[DistillStage]:
    """Parse a compact/list multi-stage schedule.

    Preferred YAML form:

    ``[{stage: self_forcing, end_step: 700, num_latent_t: 21}, ...]``

    Compact string form accepts ``"700:21,3000:240"`` and treats the
    first stage as non-streaming self-forcing and later stages as streaming.
    """

    if raw is None or raw == "":
        max_length = default_streaming_max_length or default_num_latent_t
        return [
            DistillStage(
                name="streaming_long",
                start_step=0,
                end_step=None,
                num_latent_t=int(max_length),
                streaming_training=True,
                streaming_chunk_size=default_streaming_chunk_size,
                streaming_max_length=int(max_length),
            )
        ]

    stages: list[DistillStage] = []
    previous_end = 0

    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        for idx, part in enumerate(parts):
            fields = [field.strip() for field in part.split(":")]
            end_step: int | None
            if len(fields) == 2:
                start_step = previous_end
                end_step = _as_int(
                    fields[0],
                    where="multi_phased_distill_schedule.end_step",
                )
                num_latent_t = _as_int(
                    fields[1],
                    where="multi_phased_distill_schedule.num_latent_t",
                )
            elif len(fields) == 3:
                start_step = _as_int(
                    fields[0],
                    where="multi_phased_distill_schedule.start_step",
                )
                end_step = _as_int(
                    fields[1],
                    where="multi_phased_distill_schedule.end_step",
                )
                num_latent_t = _as_int(
                    fields[2],
                    where="multi_phased_distill_schedule.num_latent_t",
                )
            else:
                raise ValueError("multi_phased_distill_schedule string entries must be "
                                 "'end_step:num_latent_t' or 'start:end:num_latent_t', "
                                 f"got {part!r}")
            streaming = idx > 0
            stages.append(
                DistillStage(
                    name="streaming_long" if streaming else "self_forcing",
                    start_step=start_step,
                    end_step=end_step,
                    num_latent_t=num_latent_t,
                    streaming_training=streaming,
                    streaming_chunk_size=(default_streaming_chunk_size if streaming else None),
                    streaming_max_length=num_latent_t if streaming else None,
                ))
            previous_end = end_step
    elif isinstance(raw, list | tuple):
        for idx, stage_raw in enumerate(raw):
            if not isinstance(stage_raw, dict):
                raise ValueError("multi_phased_distill_schedule list entries must be dicts")
            name = str(stage_raw.get("stage", "") or stage_raw.get("name", "")).strip()
            streaming_raw = stage_raw.get("streaming_training", None)
            if streaming_raw is None:
                streaming = name in {"streaming_long", "long", "streaming"}
            else:
                streaming = _as_bool(
                    streaming_raw,
                    where=f"multi_phased_distill_schedule[{idx}].streaming_training",
                )
            if not name:
                name = "streaming_long" if streaming else "self_forcing"

            start_step = _as_int(
                stage_raw.get("start_step", previous_end),
                where=f"multi_phased_distill_schedule[{idx}].start_step",
            )
            end_raw = stage_raw.get("end_step", None)
            end_step = (None if end_raw is None else _as_int(
                end_raw,
                where=f"multi_phased_distill_schedule[{idx}].end_step",
            ))
            num_latent_t = _as_int(
                stage_raw.get(
                    "num_latent_t",
                    stage_raw.get(
                        "streaming_max_length",
                        stage_raw.get("max_length", default_num_latent_t),
                    ),
                ),
                where=f"multi_phased_distill_schedule[{idx}].num_latent_t",
            )
            streaming_max = stage_raw.get("streaming_max_length", None)
            chunk_size = stage_raw.get("streaming_chunk_size", None)
            min_new = stage_raw.get("streaming_min_new_frame", None)
            fixed_overlap = stage_raw.get("streaming_fixed_overlap_latents", None)
            train_first = stage_raw.get("train_first_chunk", True)
            stages.append(
                DistillStage(
                    name=name,
                    start_step=start_step,
                    end_step=end_step,
                    num_latent_t=num_latent_t,
                    streaming_training=streaming,
                    streaming_chunk_size=(None if chunk_size is None else _as_int(
                        chunk_size,
                        where=("multi_phased_distill_schedule"
                               f"[{idx}].streaming_chunk_size"),
                    )),
                    streaming_max_length=(None if streaming_max is None else _as_int(
                        streaming_max,
                        where=("multi_phased_distill_schedule"
                               f"[{idx}].streaming_max_length"),
                    )),
                    streaming_min_new_frame=(None if min_new is None else _as_int(
                        min_new,
                        where=("multi_phased_distill_schedule"
                               f"[{idx}].streaming_min_new_frame"),
                    )),
                    streaming_fixed_overlap_latents=(None if fixed_overlap is None else _as_int(
                        fixed_overlap,
                        where=("multi_phased_distill_schedule"
                               f"[{idx}].streaming_fixed_overlap_latents"),
                    )),
                    train_first_chunk=_as_bool(
                        train_first,
                        where=f"multi_phased_distill_schedule[{idx}].train_first_chunk",
                    ),
                ))
            if end_step is not None:
                previous_end = end_step
    else:
        raise ValueError("multi_phased_distill_schedule must be a list, string, or empty")

    if not stages:
        raise ValueError("multi_phased_distill_schedule produced no stages")

    previous_end = 0
    for stage in stages:
        if stage.start_step < previous_end:
            raise ValueError("Distillation stages must be ordered and non-overlapping")
        if stage.end_step is not None and stage.end_step <= stage.start_step:
            raise ValueError("Distillation stage end_step must be > start_step")
        if stage.num_latent_t <= 0:
            raise ValueError("Distillation stage num_latent_t must be positive")
        if stage.streaming_training:
            max_length = stage.streaming_max_length or stage.num_latent_t
            chunk_size = stage.streaming_chunk_size or default_streaming_chunk_size
            if max_length <= 0:
                raise ValueError("streaming_max_length must be positive")
            if chunk_size is None or chunk_size <= 0:
                raise ValueError("streaming_chunk_size must be positive")
            if stage.streaming_fixed_overlap_latents is not None:
                if stage.streaming_fixed_overlap_latents < 0:
                    raise ValueError("streaming_fixed_overlap_latents must be non-negative")
                if stage.streaming_fixed_overlap_latents >= chunk_size:
                    raise ValueError("streaming_fixed_overlap_latents must be smaller than "
                                     "streaming_chunk_size")
        if stage.end_step is not None:
            previous_end = stage.end_step
    return stages


def select_distill_stage(
    stages: list[DistillStage],
    iteration: int,
) -> DistillStage:
    """Select the active stage for ``iteration``."""

    step = int(iteration)
    for stage in stages:
        if stage.end_step is None:
            if step >= stage.start_step:
                return stage
        elif stage.start_step <= step < stage.end_step:
            return stage
    return stages[-1]


class StreamingLongTuningMethod(SelfForcingMethod):
    """Two-stage MatrixGame/LongLive-style self-forcing method.

    Stage 1 uses the existing full self-forcing rollout over a short latent
    horizon. Stage 2 uses a persistent streaming sequence and trains on chunks
    generated with the causal student cache.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        pipeline_config = getattr(cfg.training, "pipeline_config", None)
        vae_config = getattr(pipeline_config, "vae_config", None)
        if vae_config is not None:
            vae_config.load_encoder = True
            vae_config.load_decoder = True

        super().__init__(cfg=cfg, role_models=role_models)
        if not isinstance(self.student, CausalModelBase):
            raise ValueError("StreamingLongTuningMethod requires a causal student")

        mcfg = self.method_config
        default_chunk = get_optional_int(
            mcfg,
            "streaming_chunk_size",
            where="method.streaming_chunk_size",
        )
        default_max = get_optional_int(
            mcfg,
            "streaming_max_length",
            where="method.streaming_max_length",
        )
        default_num_latent_t = int(self.training_config.data.num_latent_t)
        self._distill_stages = parse_multi_phased_distill_schedule(
            mcfg.get("multi_phased_distill_schedule", None),
            default_num_latent_t=default_num_latent_t,
            default_streaming_chunk_size=default_chunk,
            default_streaming_max_length=default_max,
        )
        self._streaming_state: _StreamingState | None = None
        self._streaming_anchor_inject_k = int(
            get_optional_int(
                mcfg,
                "streaming_anchor_inject_k",
                where="method.streaming_anchor_inject_k",
            ) or 1)
        if self._streaming_anchor_inject_k < 1:
            raise ValueError("method.streaming_anchor_inject_k must be >= 1")
        reencode_anchor_raw = mcfg.get("streaming_reencode_overlap_anchor", True)
        if reencode_anchor_raw is None:
            reencode_anchor_raw = True
        self._streaming_reencode_overlap_anchor = _as_bool(
            reencode_anchor_raw,
            where="method.streaming_reencode_overlap_anchor",
        )
        require_full_blocks_raw = mcfg.get("streaming_require_full_blocks", False)
        if require_full_blocks_raw is None:
            require_full_blocks_raw = False
        self._streaming_require_full_blocks = _as_bool(
            require_full_blocks_raw,
            where="method.streaming_require_full_blocks",
        )
        real_score_cfg_raw = mcfg.get("real_score_cfg", True)
        if real_score_cfg_raw is None:
            real_score_cfg_raw = True
        self._real_score_cfg = _as_bool(
            real_score_cfg_raw,
            where="method.real_score_cfg",
        )
        self._score_min_timestep = self._score_timestep_from_ratio("min_timestep_ratio")
        self._score_max_timestep = self._score_timestep_from_ratio("max_timestep_ratio")

        max_stage_latents = max(s.num_latent_t for s in self._distill_stages)
        if max_stage_latents > default_num_latent_t:
            raise ValueError("training.data.num_latent_t must be at least the largest "
                             "multi-stage horizon so prepared conditioning is not trimmed "
                             "before streaming: "
                             f"num_latent_t={default_num_latent_t}, "
                             f"max_stage_latents={max_stage_latents}")

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        stage = select_distill_stage(self._distill_stages, iteration)
        if not stage.streaming_training:
            self._streaming_state = None
            return self._single_train_step_short(batch, iteration, stage)
        return self._single_train_step_streaming(batch, iteration, stage)

    def _single_train_step_short(
        self,
        batch: dict[str, Any],
        iteration: int,
        stage: DistillStage,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        batch = self._with_padded_first_frame_latent(
            batch,
            target_latent_frames=int(self.training_config.data.num_latent_t),
        )
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="zeros",
        )
        self._truncate_batch_latents(training_batch, stage.num_latent_t)
        return self._losses_for_batch(
            training_batch,
            iteration,
            stage=stage,
            chunk_mask=None,
        )

    def _single_train_step_streaming(
        self,
        batch: dict[str, Any],
        iteration: int,
        stage: DistillStage,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        state = self._ensure_streaming_state(batch, stage)
        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros(
            (),
            device=state.batch.latents.device,
            dtype=state.batch.latents.dtype,
        )
        student_ctx = None
        generator_outputs: dict[str, Any] = {}
        pred_x0, chunk_mask, chunk_info = self._generate_streaming_chunk(
            state,
            with_grad=update_student,
        )
        if update_student:
            student_ctx = (
                state.batch.timesteps,
                state.batch.attn_metadata_vsa,
            )
            generator_loss = self._dmd_loss_masked(
                pred_x0,
                state.batch,
                chunk_mask=chunk_mask,
            )
            generator_outputs["dmd_latent_vis_dict"] = {
                "generator_pred_video": pred_x0.detach(),
                "streaming_chunk_mask": chunk_mask.detach(),
            }
            self.student.backward(
                generator_loss,
                student_ctx,
                grad_accum_rounds=self._gradient_accumulation_steps(),
            )
            generator_loss = generator_loss.detach()
            student_ctx = None

        fake_score_loss, critic_ctx, critic_outputs = (self._critic_flow_matching_loss_for_x0(
            pred_x0.detach(),
            state.batch,
            chunk_mask=chunk_mask,
        ))
        self._log_train_chunk(
            iteration=iteration,
            stage=stage,
            update_student=update_student,
            chunk_info=chunk_info,
        )

        total_loss = generator_loss + fake_score_loss
        outputs: dict[str, Any] = dict(generator_outputs)
        outputs.update(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": False,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            "distill_stage_index": float(self._distill_stages.index(stage)),
            "streaming_current_length": float(state.current_length),
            "streaming_max_length": float(self._stage_max_length(stage)),
        }
        return (
            {
                "total_loss": total_loss,
                "generator_loss": generator_loss,
                "fake_score_loss": fake_score_loss,
            },
            outputs,
            metrics,
        )

    def _losses_for_batch(
        self,
        training_batch: Any,
        iteration: int,
        *,
        stage: DistillStage,
        chunk_mask: torch.Tensor | None,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros(
            (),
            device=training_batch.latents.device,
            dtype=training_batch.latents.dtype,
        )
        student_ctx = None
        if update_student:
            generator_pred_x0 = self._student_rollout(
                training_batch,
                with_grad=True,
            )
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss = self._dmd_loss_masked(
                generator_pred_x0,
                training_batch,
                chunk_mask=chunk_mask,
            )
            self.student.backward(
                generator_loss,
                student_ctx,
                grad_accum_rounds=self._gradient_accumulation_steps(),
            )
            generator_loss = generator_loss.detach()
            student_ctx = None

        with torch.no_grad():
            generator_pred_x0 = self._student_rollout(
                training_batch,
                with_grad=False,
            )

        fake_score_loss, critic_ctx, critic_outputs = (self._critic_flow_matching_loss_for_x0(
            generator_pred_x0,
            training_batch,
            chunk_mask=chunk_mask,
        ))

        total_loss = generator_loss + fake_score_loss
        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": False,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            "distill_stage_index": float(self._distill_stages.index(stage)),
            "active_num_latent_t": float(stage.num_latent_t),
        }
        self._log_train_chunk(
            iteration=iteration,
            stage=stage,
            update_student=update_student,
            chunk_info=_StreamingChunkInfo(
                chunk_start=0,
                chunk_end=int(stage.num_latent_t) - 1,
                train_start=0,
                train_end=int(stage.num_latent_t) - 1,
                new_frames=int(stage.num_latent_t),
                overlap=0,
                current_length=int(stage.num_latent_t),
                max_length=int(stage.num_latent_t),
            ),
        )
        return (
            {
                "total_loss": total_loss,
                "generator_loss": generator_loss,
                "fake_score_loss": fake_score_loss,
            },
            outputs,
            metrics,
        )

    def _gradient_accumulation_steps(self) -> int:
        loop_config = getattr(self.training_config, "loop", None)
        return max(1, int(getattr(loop_config, "gradient_accumulation_steps", 1) or 1))

    def _student_rollout_streaming(
        self,
        batch: Any,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        # Short-stage rollout should respect the truncated batch length above.
        return super()._student_rollout_streaming(batch, with_grad=with_grad)

    def _ensure_streaming_state(
        self,
        raw_batch: dict[str, Any],
        stage: DistillStage,
    ) -> _StreamingState:
        state = self._streaming_state
        if (state is None or state.stage != stage or not self._can_generate_more(state)):
            if state is not None:
                self.student.clear_caches(cache_tag=state.cache_tag)
            raw_batch = self._with_padded_first_frame_latent(
                raw_batch,
                target_latent_frames=int(self.training_config.data.num_latent_t),
            )
            training_batch = self.student.prepare_batch(
                raw_batch,
                generator=self.cuda_generator,
                latents_source="zeros",
            )
            self._truncate_batch_latents(
                training_batch,
                self._stage_max_length(stage),
            )
            cache_tag = f"streaming_long_{id(training_batch)}"
            self.student.clear_caches(cache_tag=cache_tag)
            state = _StreamingState(
                stage=stage,
                batch=training_batch,
                current_length=0,
                previous_latents=None,
                cache_tag=cache_tag,
            )
            self._streaming_state = state
        return state

    def _with_padded_first_frame_latent(
        self,
        raw_batch: dict[str, Any],
        *,
        target_latent_frames: int,
    ) -> dict[str, Any]:
        first_frame_latent = raw_batch.get("first_frame_latent")
        if first_frame_latent is None:
            return raw_batch
        if first_frame_latent.ndim != 5:
            raise ValueError("first_frame_latent must have shape [B, C, T, H, W], "
                             f"got {tuple(first_frame_latent.shape)}")
        target_latent_frames = int(target_latent_frames)
        if first_frame_latent.shape[2] >= target_latent_frames:
            return raw_batch

        # Long tuning passes first-frame-only MatrixGame I2V condition as a full-length latent tensor.
        pad_frames = target_latent_frames - int(first_frame_latent.shape[2])
        pad = torch.zeros(
            first_frame_latent.shape[0],
            first_frame_latent.shape[1],
            pad_frames,
            first_frame_latent.shape[3],
            first_frame_latent.shape[4],
            device=first_frame_latent.device,
            dtype=first_frame_latent.dtype,
        )
        batch = dict(raw_batch)
        batch["first_frame_latent"] = torch.cat([first_frame_latent, pad], dim=2)
        return batch

    def _can_generate_more(self, state: _StreamingState) -> bool:
        max_length = self._stage_max_length(state.stage)
        return state.current_length < max_length

    def _generate_streaming_chunk(
        self,
        state: _StreamingState,
        *,
        with_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, _StreamingChunkInfo]:
        assert isinstance(self.student, CausalModelBase)

        batch = state.batch
        latents = batch.latents
        if latents is None:
            raise RuntimeError("TrainingBatch.latents is required")

        device = latents.device
        dtype = latents.dtype
        batch_size = int(latents.shape[0])
        max_length = self._stage_max_length(state.stage)
        chunk_size = self._stage_chunk_size(state.stage)
        current = int(state.current_length)
        remaining = max_length - current
        if remaining <= 0:
            raise RuntimeError("Streaming sequence has no remaining frames")

        if state.previous_latents is None:
            new_frames = min(chunk_size, remaining)
            overlap = 0
        else:
            new_frames = self._select_new_frame_count(
                state.stage,
                remaining=remaining,
                device=device,
            )
            overlap = min(
                chunk_size - new_frames,
                int(state.previous_latents.shape[1]),
            )

        if new_frames <= 0:
            raise RuntimeError("Streaming new frame count must be positive")
        if current + new_frames > max_length:
            raise RuntimeError("Streaming chunk would exceed max length")

        rollout_block_size = int(self._chunk_size)
        if rollout_block_size <= 0:
            raise ValueError("method.chunk_size must be positive")
        if self._streaming_require_full_blocks and new_frames % rollout_block_size != 0:
            raise ValueError("Streaming new-frame count must be divisible by "
                             "method.chunk_size: "
                             f"new_frames={new_frames}, "
                             f"chunk_size={rollout_block_size}")

        denoising_steps = self._get_denoising_step_list(device)
        num_steps = int(denoising_steps.numel())
        noise_full = torch.randn(
            (
                batch_size,
                new_frames,
                int(latents.shape[2]),
                int(latents.shape[3]),
                int(latents.shape[4]),
            ),
            device=device,
            dtype=dtype,
            generator=self.cuda_generator,
        )

        block_lengths: list[int] = []
        remaining_block_frames = int(new_frames)
        while remaining_block_frames > 0:
            current_block = min(rollout_block_size, remaining_block_frames)
            block_lengths.append(current_block)
            remaining_block_frames -= current_block

        exit_indices = self._sample_exit_indices(
            num_blocks=len(block_lengths),
            num_steps=num_steps,
            device=device,
        )
        block0_exit_idx = int(exit_indices[0])
        denoised_from, denoised_to = self._get_self_forcing_timestep_bounds(
            denoising_steps,
            block0_exit_idx,
        )

        denoised_blocks: list[torch.Tensor] = []
        local_start = 0
        for block_idx, block_frames in enumerate(block_lengths):
            block_start = current + local_start
            noisy_block = noise_full[:, local_start:local_start + block_frames]
            exit_idx = int(exit_indices[block_idx])
            pred_x0_block: torch.Tensor | None = None

            for step_idx, current_timestep in enumerate(denoising_steps):
                timestep_block = current_timestep * torch.ones(
                    (batch_size, block_frames),
                    device=device,
                    dtype=torch.float32,
                )
                exit_flag = step_idx == exit_idx
                enable_grad = (bool(with_grad) and bool(self._enable_gradient_in_rollout) and torch.is_grad_enabled())

                if not exit_flag:
                    with torch.no_grad():
                        pred_noise = self.student.predict_noise_streaming(
                            noisy_block,
                            timestep_block,
                            batch,
                            conditional=True,
                            cache_tag=state.cache_tag,
                            store_kv=False,
                            cur_start_frame=block_start,
                            cfg_uncond=self._cfg_uncond,
                            attn_kind="vsa",
                        )
                        if pred_noise is None:
                            raise RuntimeError("predict_noise_streaming returned None")
                        pred_x0_block = pred_noise_to_pred_video(
                            pred_noise=pred_noise.flatten(0, 1),
                            noise_input_latent=noisy_block.flatten(0, 1),
                            timestep=timestep_block,
                            scheduler=self._sf_scheduler,
                        ).unflatten(0, pred_noise.shape[:2])

                    if step_idx + 1 >= num_steps:
                        break
                    next_timestep = denoising_steps[step_idx + 1]
                    if self._student_sample_type == "sde":
                        noisy_block = self._sf_add_noise(
                            pred_x0_block,
                            torch.randn(
                                pred_x0_block.shape,
                                device=device,
                                dtype=pred_x0_block.dtype,
                                generator=self.cuda_generator,
                            ),
                            next_timestep * torch.ones(
                                (batch_size, block_frames),
                                device=device,
                                dtype=torch.float32,
                            ),
                        )
                    else:
                        next_timestep_block = next_timestep * torch.ones(
                            (batch_size, block_frames),
                            device=device,
                            dtype=torch.float32,
                        )
                        sigma_cur = self._timestep_to_sigma(timestep_block).view(
                            batch_size,
                            block_frames,
                            1,
                            1,
                            1,
                        )
                        sigma_next = self._timestep_to_sigma(next_timestep_block).view(
                            batch_size,
                            block_frames,
                            1,
                            1,
                            1,
                        )
                        eps = (noisy_block - (1 - sigma_cur) * pred_x0_block) / sigma_cur.clamp_min(1e-8)
                        noisy_block = (1 - sigma_next) * pred_x0_block + sigma_next * eps
                    continue

                with torch.set_grad_enabled(enable_grad):
                    pred_noise = self.student.predict_noise_streaming(
                        noisy_block,
                        timestep_block,
                        batch,
                        conditional=True,
                        cache_tag=state.cache_tag,
                        store_kv=False,
                        cur_start_frame=block_start,
                        cfg_uncond=self._cfg_uncond,
                        attn_kind="vsa",
                    )
                    if pred_noise is None:
                        raise RuntimeError("predict_noise_streaming returned None")
                    pred_x0_block = pred_noise_to_pred_video(
                        pred_noise=pred_noise.flatten(0, 1),
                        noise_input_latent=noisy_block.flatten(0, 1),
                        timestep=timestep_block,
                        scheduler=self._sf_scheduler,
                    ).unflatten(0, pred_noise.shape[:2])
                break

            if pred_x0_block is None:
                raise RuntimeError("Streaming rollout produced no block")
            denoised_blocks.append(pred_x0_block)

            with torch.no_grad():
                context_timestep = torch.ones(
                    (batch_size, block_frames),
                    device=device,
                    dtype=torch.float32,
                ) * float(self._context_noise)
                context_latents = self._sf_add_noise(
                    pred_x0_block.detach(),
                    torch.randn(
                        pred_x0_block.shape,
                        device=device,
                        dtype=pred_x0_block.dtype,
                        generator=self.cuda_generator,
                    ),
                    context_timestep,
                )

                _ = self.student.predict_noise_streaming(
                    context_latents,
                    context_timestep,
                    batch,
                    conditional=True,
                    cache_tag=state.cache_tag,
                    store_kv=True,
                    cur_start_frame=block_start,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )

            local_start += block_frames

        if not denoised_blocks:
            raise RuntimeError("Streaming rollout produced no chunk")
        pred_x0_chunk = torch.cat(denoised_blocks, dim=1)
        batch.denoised_timestep_from = denoised_from
        batch.denoised_timestep_to = denoised_to

        if overlap > 0 and state.previous_latents is not None:
            overlap_latents = state.previous_latents[:, -overlap:].detach()
            if self._streaming_reencode_overlap_anchor:
                anchor_source = torch.cat(
                    [state.previous_latents.detach(), pred_x0_chunk],
                    dim=1,
                )
                full_chunk = self._process_first_frame_anchor(
                    anchor_source,
                    target_latents=chunk_size,
                )
            else:
                full_chunk = torch.cat([overlap_latents, pred_x0_chunk], dim=1)
            chunk_mask = torch.zeros(
                (batch_size, chunk_size, 1, 1, 1),
                device=device,
                dtype=torch.bool,
            )
            chunk_mask[:, overlap:] = True
        else:
            full_chunk = pred_x0_chunk
            chunk_mask = torch.ones(
                (batch_size, new_frames, 1, 1, 1),
                device=device,
                dtype=torch.bool,
            )

        state.current_length = current + new_frames
        temporal_compression_ratio = int(self.training_config.pipeline_config.vae_config.arch_config.
                                         temporal_compression_ratio  # type: ignore[union-attr]
                                         )
        chunk_info = _StreamingChunkInfo(
            chunk_start=current - overlap,
            chunk_end=current + new_frames - 1,
            train_start=current,
            train_end=current + new_frames - 1,
            new_frames=new_frames,
            overlap=overlap,
            current_length=state.current_length,
            max_length=max_length,
        )
        score_frame_start = max(0, current - overlap)
        batch.matrixgame_streaming_chunk_info = {
            "frame_start": score_frame_start,
            "frame_end": score_frame_start + int(full_chunk.shape[1]),
            "new_frame_start": int(overlap),
            "new_frames": int(new_frames),
            "gradient_mask": chunk_mask,
            "condition_image_latents": None,
            "condition_image_embeds": None,
            "generator_chunk_start": int(current),
            "generator_chunk_end": int(current + new_frames),
            "generator_action_frame_end": int((current + new_frames - 1) * temporal_compression_ratio + 1),
            "window_label": f"window_{score_frame_start}_{score_frame_start + int(full_chunk.shape[1])}",
        }
        state.previous_latents = full_chunk.detach()[:, -chunk_size:]

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "[StreamingLong] "
                f"stage={state.stage.name} start={current} "
                f"new={new_frames} overlap={overlap} "
                f"blocks={block_lengths} "
                f"current_length={state.current_length}/{max_length}",
                flush=True,
            )

        return full_chunk, chunk_mask, chunk_info

    def _get_self_forcing_timestep_bounds(
        self,
        step_list: torch.Tensor,
        exit_idx: int,
    ) -> tuple[int | None, int | None]:
        scheduler_timesteps = self._sf_scheduler.timesteps.to(
            device=step_list.device,
            dtype=torch.float32,
        )
        exit_idx = min(int(exit_idx), int(step_list.numel()) - 1)
        denoising_timestep = step_list[exit_idx].to(dtype=scheduler_timesteps.dtype)
        denoised_timestep_from = int(self.student.num_train_timesteps) - int(
            torch.argmin(
                (scheduler_timesteps - denoising_timestep).abs(),
                dim=0,
            ).item())
        if exit_idx == int(step_list.numel()) - 1:
            return denoised_timestep_from, 0

        next_denoising_timestep = step_list[exit_idx + 1].to(dtype=scheduler_timesteps.dtype)
        denoised_timestep_to = int(self.student.num_train_timesteps) - int(
            torch.argmin(
                (scheduler_timesteps - next_denoising_timestep).abs(),
                dim=0,
            ).item())
        return denoised_timestep_from, denoised_timestep_to

    def _process_first_frame_anchor(
        self,
        frames: torch.Tensor,
        *,
        target_latents: int,
    ) -> torch.Tensor:
        if frames.shape[1] <= 1:
            return frames

        process_latents = min(int(target_latents), int(frames.shape[1]))
        if process_latents <= 1:
            return frames[:, -process_latents:]

        vae = getattr(self.student, "vae", None)
        if vae is None:
            raise RuntimeError("Streaming first-frame anchor requires student.vae")
        self._require_vae_encoder(vae)

        inject_k = min(int(self._streaming_anchor_inject_k), process_latents)
        if inject_k == 1:
            with torch.no_grad():
                latents_to_decode = frames[:, :-(process_latents - 1), ...]
                decode_latents = self._denormalize_wan_latents(
                    latents_to_decode,
                    vae=vae,
                )
                pixels = vae.decode(decode_latents)
                last_frame_pixel = pixels[:, :, -1:, :, :].float()
                image_latent = self._encode_wan_latent(
                    vae,
                    last_frame_pixel,
                    like=frames,
                )

            remaining = frames[:, -(process_latents - 1):, ...]
            return torch.cat([image_latent, remaining], dim=1)

        out_slice = frames[:, -process_latents:, ...]
        with torch.no_grad():
            first_k = out_slice[:, :inject_k, ...]
            decode_latents = self._denormalize_wan_latents(
                first_k,
                vae=vae,
            )
            pixels = vae.decode(decode_latents).float()
            new_first_k = self._encode_wan_latent(
                vae,
                pixels,
                like=frames,
            )

        remaining = out_slice[:, inject_k:, ...]
        return torch.cat([new_first_k, remaining], dim=1)

    def _denormalize_wan_latents(
        self,
        frames: torch.Tensor,
        *,
        vae: Any,
    ) -> torch.Tensor:
        latents = frames.permute(0, 2, 1, 3, 4).float()
        cfg = getattr(vae, "config", None)
        if (cfg is not None and hasattr(cfg, "latents_mean") and hasattr(cfg, "latents_std")):
            latents_mean = torch.tensor(
                cfg.latents_mean,
                device=latents.device,
                dtype=latents.dtype,
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                cfg.latents_std,
                device=latents.device,
                dtype=latents.dtype,
            ).view(1, -1, 1, 1, 1)
            return latents * latents_std + latents_mean

        if hasattr(vae, "scaling_factor"):
            scaling_factor = self._vae_tensor_attr(
                vae,
                "scaling_factor",
                like=latents,
                default=1.0,
            )
            latents = latents / scaling_factor
        shift_factor = self._vae_tensor_attr(
            vae,
            "shift_factor",
            like=latents,
            default=0.0,
        )
        return latents + shift_factor

    def _normalize_wan_latents(
        self,
        latents: torch.Tensor,
        *,
        vae: Any,
    ) -> torch.Tensor:
        cfg = getattr(vae, "config", None)
        if (cfg is not None and hasattr(cfg, "latents_mean") and hasattr(cfg, "latents_std")):
            latents_mean = torch.tensor(
                cfg.latents_mean,
                device=latents.device,
                dtype=latents.dtype,
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                cfg.latents_std,
                device=latents.device,
                dtype=latents.dtype,
            ).view(1, -1, 1, 1, 1)
            return (latents - latents_mean) / latents_std

        shift_factor = self._vae_tensor_attr(
            vae,
            "shift_factor",
            like=latents,
            default=0.0,
        )
        latents = latents - shift_factor
        if hasattr(vae, "scaling_factor"):
            scaling_factor = self._vae_tensor_attr(
                vae,
                "scaling_factor",
                like=latents,
                default=1.0,
            )
            latents = latents * scaling_factor
        return latents

    def _encode_wan_latent(
        self,
        vae: Any,
        pixels: torch.Tensor,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        encoded = vae.encode(pixels)
        if isinstance(encoded, torch.Tensor):
            raw_latent = encoded
        elif hasattr(encoded, "mode"):
            raw_latent = encoded.mode()
        else:
            latent_dist = getattr(encoded, "latent_dist", None)
            if latent_dist is not None:
                raw_latent = (latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.mean)
            else:
                mean = getattr(encoded, "mean", None)
                if not isinstance(mean, torch.Tensor):
                    raise RuntimeError("Unsupported VAE encode output")
                raw_latent = mean

        raw_latent = raw_latent.to(device=like.device, dtype=torch.float32)
        normalized = self._normalize_wan_latents(
            raw_latent,
            vae=vae,
        )
        return normalized.permute(0, 2, 1, 3, 4).to(dtype=like.dtype)

    def _require_vae_encoder(self, vae: Any) -> None:
        target = getattr(vae, "module", vae)
        encoder = getattr(target, "encoder", None)
        if encoder is None:
            raise RuntimeError("Streaming first-frame anchor requires a VAE "
                               "loaded with encoder weights")
        if (getattr(target, "use_feature_cache", False) and not hasattr(target, "_enc_feat_map")):
            config = getattr(target, "config", None)
            if config is not None:
                config.load_encoder = True
            clear_cache = getattr(target, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
        if (getattr(target, "use_feature_cache", False) and not hasattr(target, "_enc_feat_map")):
            raise RuntimeError("Streaming first-frame anchor requires VAE "
                               "encoder feature-cache buffers")

    def _vae_tensor_attr(
        self,
        vae: Any,
        name: str,
        *,
        like: torch.Tensor,
        default: float,
    ) -> torch.Tensor:
        value = getattr(vae, name, None)
        if value is None and hasattr(vae, "module"):
            value = getattr(vae.module, name, None)
        if value is None:
            value = default
        if isinstance(value, torch.Tensor):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.tensor(value, device=like.device, dtype=like.dtype)

    def _log_train_chunk(
        self,
        *,
        iteration: int,
        stage: DistillStage,
        update_student: bool,
        chunk_info: _StreamingChunkInfo,
    ) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        generator_state = "on" if update_student else "off"
        train_target = "generator+critic" if update_student else "critic"
        print(
            "[StreamingLong] "
            f"step={iteration} stage={stage.name} "
            f"chunk_latents={chunk_info.chunk_start}-{chunk_info.chunk_end} "
            f"train_latents={chunk_info.train_start}-{chunk_info.train_end} "
            f"new={chunk_info.new_frames} overlap={chunk_info.overlap} "
            f"current_length={chunk_info.current_length}/{chunk_info.max_length} "
            f"train={train_target} generator={generator_state} critic=on",
            flush=True,
        )

    def _select_new_frame_count(
        self,
        stage: DistillStage,
        *,
        remaining: int,
        device: torch.device,
    ) -> int:
        chunk_size = self._stage_chunk_size(stage)
        fixed_overlap = stage.streaming_fixed_overlap_latents
        if fixed_overlap is None:
            fixed_overlap = get_optional_int(
                self.method_config,
                "streaming_fixed_overlap_latents",
                where="method.streaming_fixed_overlap_latents",
            )
        if fixed_overlap is not None:
            return min(chunk_size - int(fixed_overlap), remaining)

        min_new = stage.streaming_min_new_frame
        if min_new is None:
            min_new = get_optional_int(
                self.method_config,
                "streaming_min_new_frame",
                where="method.streaming_min_new_frame",
            )
        if min_new is None:
            min_new = chunk_size
        min_new = max(1, int(min_new))

        max_new = min(chunk_size, remaining)
        if max_new <= min_new:
            return max_new

        # Match LongLive's step-by-num_frame_per_block selection. With the
        # official 21/18 setting this chooses 18 until the final short chunk.
        block = int(self._chunk_size)
        choices = list(range(min_new, max_new, block))
        if not choices:
            choices = [max_new]

        if not dist.is_initialized() or dist.get_rank() == 0:
            idx = torch.randint(
                low=0,
                high=len(choices),
                size=(1, ),
                device=device,
                dtype=torch.long,
                generator=self.cuda_generator,
            )
        else:
            idx = torch.empty((1, ), device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return int(choices[int(idx.item())])

    def _stage_max_length(self, stage: DistillStage) -> int:
        return int(stage.streaming_max_length or stage.num_latent_t)

    def _stage_chunk_size(self, stage: DistillStage) -> int:
        chunk_size = stage.streaming_chunk_size
        if chunk_size is None:
            chunk_size = get_optional_int(
                self.method_config,
                "streaming_chunk_size",
                where="method.streaming_chunk_size",
            )
        if chunk_size is None:
            chunk_size = self._chunk_size
        if int(chunk_size) <= 0:
            raise ValueError("streaming_chunk_size must be positive")
        return int(chunk_size)

    def _truncate_batch_latents(self, batch: Any, num_latent_t: int) -> None:
        num_latent_t = int(num_latent_t)
        if batch.latents is not None:
            batch.latents = batch.latents[:, :num_latent_t]
            batch.raw_latent_shape = (
                int(batch.latents.shape[0]),
                int(batch.latents.shape[2]),
                int(batch.latents.shape[1]),
                int(batch.latents.shape[3]),
                int(batch.latents.shape[4]),
            )
        if batch.noise_latents is not None:
            batch.noise_latents = batch.noise_latents[:, :num_latent_t]
        if batch.noisy_model_input is not None:
            batch.noisy_model_input = batch.noisy_model_input[:, :num_latent_t]
        if batch.noise is not None:
            batch.noise = batch.noise[:, :num_latent_t]
        if batch.timesteps is not None and batch.timesteps.ndim == 2:
            batch.timesteps = batch.timesteps[:, :num_latent_t]
        if batch.sigmas is not None and batch.sigmas.ndim >= 1:
            if batch.sigmas.ndim == 1:
                batch.sigmas = batch.sigmas[:num_latent_t]
            else:
                batch.sigmas = batch.sigmas[:, :num_latent_t]
        self._rebuild_attention_metadata(batch)

    def _rebuild_attention_metadata(self, batch: Any) -> None:
        build_metadata = getattr(self.student, "_build_attention_metadata", None)
        if not callable(build_metadata):
            return
        batch.attn_metadata = None
        batch.attn_metadata_vsa = None
        build_metadata(batch)
        batch.attn_metadata_vsa = copy.deepcopy(batch.attn_metadata)
        if batch.attn_metadata is not None:
            batch.attn_metadata.VSA_sparsity = 0.0

    def _critic_flow_matching_loss_for_x0(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
        *,
        chunk_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        device = generator_pred_x0.device
        batch_size = int(generator_pred_x0.shape[0])
        fake_score_timestep = self._sample_score_timestep(
            batch_size,
            device=device,
        )

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
            generator=self.cuda_generator,
        )
        noisy_x0 = self._sf_add_noise(generator_pred_x0, noise, fake_score_timestep)

        pred_noise = self.critic.predict_noise(
            noisy_x0,
            fake_score_timestep,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="dense",
        )
        target = noise - generator_pred_x0
        flow_matching_loss = self._masked_mean(
            (pred_noise - target)**2,
            chunk_mask,
        )

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {"fake_score_latent_vis_dict": batch.fake_score_latent_vis_dict}
        return (
            flow_matching_loss,
            (batch.timesteps, batch.attn_metadata),
            outputs,
        )

    def _dmd_loss_masked(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
        *,
        chunk_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = 1.0
        device = generator_pred_x0.device
        batch_size = int(generator_pred_x0.shape[0])

        with torch.no_grad():
            timestep = self._sample_score_timestep(
                batch_size,
                device=device,
            )
            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
                generator=self.cuda_generator,
            )
            noisy_latents = self._sf_add_noise(generator_pred_x0, noise, timestep)

            faker_x0 = self._predict_x0_with_scheduler(
                self.critic,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            real_cond_x0 = self._predict_x0_with_scheduler(
                self.teacher,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            if self._real_score_cfg:
                real_uncond_x0 = self._predict_x0_with_scheduler(
                    self.teacher,
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=False,
                    attn_kind="dense",
                )
                real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale
            else:
                real_cfg_x0 = real_cond_x0
            denom = torch.abs(generator_pred_x0.float() - real_cfg_x0.float(), ).mean(dim=[1, 2, 3, 4],
                                                                                      keepdim=True).clamp_min(1e-6)
            grad = (faker_x0 - real_cfg_x0) / denom
            grad = torch.nan_to_num(grad)

        loss = 0.5 * (generator_pred_x0.float() - (generator_pred_x0.float() - grad.float()).detach())**2
        return self._masked_mean(loss, chunk_mask)

    def _score_timestep_from_ratio(self, key: str) -> int | None:
        ratio = get_optional_float(
            self.method_config,
            key,
            where=f"method.{key}",
        )
        if ratio is None:
            return None
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"method.{key} must be in [0, 1], got {ratio}")
        return int(round(float(ratio) * int(self.student.num_train_timesteps)))

    def _sample_score_timestep(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        timestep = torch.randint(
            0,
            int(self.student.num_train_timesteps),
            [batch_size],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        timestep = self.student.shift_and_clamp_timestep(timestep)
        min_timestep = self._score_min_timestep
        max_timestep = self._score_max_timestep
        if min_timestep is not None or max_timestep is not None:
            timestep = timestep.clamp(
                min=0 if min_timestep is None else int(min_timestep),
                max=(int(self.student.num_train_timesteps) if max_timestep is None else int(max_timestep)),
            )
        return timestep

    def _masked_mean(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask_f = mask.to(device=values.device, dtype=values.dtype)
        while mask_f.ndim < values.ndim:
            mask_f = mask_f.unsqueeze(-1)
        denom = mask_f.expand_as(values).sum().clamp_min(1.0)
        return (values * mask_f).sum() / denom
