# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing SFT method (TFSFT; algorithm layer)."""

from __future__ import annotations

import copy
import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from fastvideo.models.schedulers.scheduling_diffusion_forcing import (
    DiffusionForcingScheduler, )
from fastvideo.train.methods.base import TrainingMethod, LogScalar
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )


class TeacherForcingSFTMethod(TrainingMethod):
    """Teacher-forcing SFT (TFSFT) for causal students.

    Training uses a concatenated ``[clean | noisy]`` latent sequence plus a
    custom block mask so noisy chunks can attend to a clean prefix while the
    denoising loss is applied only on the noisy half.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("TFSFT requires role 'student'")
        if not self.student._trainable:
            raise ValueError("TFSFT requires student to be trainable")
        if self.training_config.model.precondition_outputs:
            raise ValueError(
                "TFSFT only supports official diffusion-forcing loss; "
                "set training.model.precondition_outputs=false"
            )
        self._attn_kind: Literal["dense", "vsa"] = (self._infer_attn_kind())

        self._chunk_size = self._parse_chunk_size(
            self.method_config.get("chunk_size", None))
        self._tfsft_scheduler = self._build_tfsft_scheduler()
        self._timestep_index_range = self._parse_timestep_index_range()

        # Initialize preprocessors on student.
        self.student.init_preprocessors(self.training_config)

        self._block_mask_cache: dict[tuple[str, int, int, int], Any] = {}
        self._init_optimizers_and_schedulers()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

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
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )

        if training_batch.latents is None:
            raise RuntimeError("prepare_batch() must set TrainingBatch.latents")

        clean_latents = training_batch.latents
        if not torch.is_tensor(clean_latents):
            raise TypeError("TrainingBatch.latents must be a torch.Tensor")
        if clean_latents.ndim != 5:
            raise ValueError("TrainingBatch.latents must be "
                             "[B, T, C, H, W], got "
                             f"shape={tuple(clean_latents.shape)}")

        batch_size, num_latents = (
            int(clean_latents.shape[0]),
            int(clean_latents.shape[1]),
        )

        expected_chunk = getattr(
            self.student.transformer,
            "num_frame_per_block",
            None,
        )
        if (expected_chunk is not None
                and int(expected_chunk) != int(self._chunk_size)):
            raise ValueError("TFSFT chunk_size must match "
                             "transformer.num_frame_per_block for "
                             f"causal training (got {self._chunk_size}, "
                             f"expected {expected_chunk}).")

        timestep_indices = self._sample_t_inhom_indices(
            batch_size=batch_size,
            num_latents=num_latents,
            device=clean_latents.device,
        )
        sp_size = int(self.training_config.distributed.sp_size)
        sp_group = getattr(self.student, "sp_group", None)
        if (sp_size > 1 and sp_group is not None
                and hasattr(sp_group, "broadcast")):
            sp_group.broadcast(timestep_indices, src=0)

        scheduler = self._tfsft_scheduler
        schedule_timesteps = scheduler.timesteps.to(
            device=clean_latents.device,
            dtype=torch.float32,
        )
        t_inhom = schedule_timesteps[timestep_indices]

        noise = torch.randn(
            clean_latents.shape,
            generator=self.cuda_generator,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        noisy_latents = scheduler.add_noise(
            clean_latents.flatten(0, 1),
            noise.flatten(0, 1),
            t_inhom.flatten(),
        ).unflatten(0, clean_latents.shape[:2])
        training_batch.noise = noise

        concat_latents = torch.cat([clean_latents, noisy_latents], dim=1)
        concat_timesteps = torch.cat([torch.zeros_like(t_inhom), t_inhom], dim=1)

        saved_batch_state = self._expand_batch_for_concat(
            training_batch,
            num_concat_latents=int(concat_latents.shape[1]),
        )
        training_batch.timesteps = concat_timesteps
        self._refresh_attention_metadata(training_batch)

        transformer = self.student.transformer
        custom_block_mask = self._get_teacher_forcing_block_mask(
            transformer=transformer,
            latents=concat_latents,
        )
        prev_block_mask = getattr(transformer, "block_mask", None)

        try:
            transformer.block_mask = custom_block_mask
            pred = self.student.predict_noise(
                concat_latents,
                concat_timesteps,
                training_batch,
                conditional=True,
                attn_kind=self._attn_kind,
            )
        finally:
            transformer.block_mask = prev_block_mask
            self._restore_batch_after_concat(training_batch, saved_batch_state)

        pred_noisy = pred[:, num_latents:]
        target = scheduler.training_target(clean_latents, noise, t_inhom)
        per_frame_loss = F.mse_loss(
            pred_noisy.float(),
            target.float(),
            reduction="none",
        ).mean(dim=(2, 3, 4))
        weight = scheduler.training_weight(t_inhom).reshape(
            batch_size,
            num_latents,
        )
        total_loss = (per_frame_loss * weight.float()).mean()

        if self._attn_kind == "vsa":
            attn_metadata = training_batch.attn_metadata_vsa
        else:
            attn_metadata = training_batch.attn_metadata

        loss_map = {"total_loss": total_loss, "tfsft_loss": total_loss}
        outputs: dict[str, Any] = {
            "_fv_backward": (
                concat_timesteps,
                attn_metadata,
            )
        }
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
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
            grad_accum_rounds=grad_accum_rounds,
        )

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

    def _refresh_attention_metadata(self, training_batch: Any) -> None:
        build_fn = getattr(self.student, "_build_attention_metadata", None)
        if not callable(build_fn):
            return

        build_fn(training_batch)
        training_batch.attn_metadata_vsa = copy.deepcopy(
            training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

    def _expand_batch_for_concat(
        self,
        training_batch: Any,
        *,
        num_concat_latents: int,
    ) -> dict[str, Any]:
        saved: dict[str, Any] = {}

        raw_latent_shape = getattr(training_batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and len(raw_latent_shape) == 5:
            saved["raw_latent_shape"] = raw_latent_shape
            batch_size, channels, _num_frames, height, width = raw_latent_shape
            training_batch.raw_latent_shape = (
                batch_size,
                channels,
                num_concat_latents,
                height,
                width,
            )

        temporal_dims = {
            "image_latents": 2,
            "mask_lat_size": 2,
            "viewmats": 1,
            "Ks": 1,
            "action": 1,
            "mouse_cond": 1,
            "keyboard_cond": 1,
        }
        for name, dim in temporal_dims.items():
            value = getattr(training_batch, name, None)
            if isinstance(value, torch.Tensor):
                saved[name] = value
                setattr(training_batch, name, torch.cat([value, value], dim=dim))

        return saved

    def _restore_batch_after_concat(
        self,
        training_batch: Any,
        saved: dict[str, Any],
    ) -> None:
        for name, value in saved.items():
            setattr(training_batch, name, value)

    def _get_teacher_forcing_block_mask(
        self,
        *,
        transformer: Any,
        latents: torch.Tensor,
    ) -> Any:
        num_frames = int(latents.shape[1])
        if num_frames % 2 != 0:
            raise ValueError("TFSFT concat-and-mask requires an even "
                             f"number of frames, got {num_frames}")

        patch_size = getattr(transformer, "patch_size", (1, 1, 1))
        if len(patch_size) != 3:
            raise ValueError("Unexpected transformer.patch_size: "
                             f"{patch_size!r}")

        patch_t, patch_h, patch_w = [int(v) for v in patch_size]
        post_patch_num_frames = num_frames // patch_t
        if post_patch_num_frames * patch_t != num_frames:
            raise ValueError("TFSFT requires num_frames divisible by "
                             f"patch temporal size {patch_t}, got {num_frames}")

        frame_seqlen = (int(latents.shape[-2]) // patch_h) * (
            int(latents.shape[-1]) // patch_w)
        clean_frames = post_patch_num_frames // 2
        if clean_frames <= 0:
            raise ValueError("TFSFT requires at least one clean frame")

        device_key = str(latents.device)
        cache_key = (
            device_key,
            post_patch_num_frames,
            frame_seqlen,
            self._chunk_size,
        )
        cached = self._block_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        total_length = post_patch_num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        total_tokens = total_length + padded_length

        prefix_starts = torch.zeros(total_tokens, device=latents.device, dtype=torch.long)
        prefix_ends = torch.zeros_like(prefix_starts)
        noisy_starts = torch.zeros_like(prefix_starts)
        noisy_ends = torch.zeros_like(prefix_starts)

        for frame_idx in range(post_patch_num_frames):
            token_start = frame_idx * frame_seqlen
            token_end = min(total_length, token_start + frame_seqlen)
            if frame_idx < clean_frames:
                chunk_end = min(clean_frames, ((frame_idx // self._chunk_size) + 1) * self._chunk_size)
                prefix_start = 0
                prefix_end = chunk_end * frame_seqlen
                noisy_start = 0
                noisy_end = 0
            else:
                noisy_frame = frame_idx - clean_frames
                chunk_start = (noisy_frame // self._chunk_size) * self._chunk_size
                chunk_end = min(clean_frames, chunk_start + self._chunk_size)
                prefix_start = 0
                prefix_end = chunk_start * frame_seqlen
                noisy_start = (clean_frames + chunk_start) * frame_seqlen
                noisy_end = (clean_frames + chunk_end) * frame_seqlen

            prefix_starts[token_start:token_end] = prefix_start
            prefix_ends[token_start:token_end] = prefix_end
            noisy_starts[token_start:token_end] = noisy_start
            noisy_ends[token_start:token_end] = noisy_end

        def attention_mask(b, h, q_idx, kv_idx):
            valid = (q_idx < total_length) & (kv_idx < total_length)
            prefix_ok = ((kv_idx >= prefix_starts[q_idx])
                         & (kv_idx < prefix_ends[q_idx]))
            noisy_ok = ((kv_idx >= noisy_starts[q_idx])
                        & (kv_idx < noisy_ends[q_idx]))
            return valid & (prefix_ok | noisy_ok)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_tokens,
            KV_LEN=total_tokens,
            _compile=False,
            device=latents.device,
        )
        self._block_mask_cache[cache_key] = block_mask
        return block_mask

    def _parse_chunk_size(self, raw: Any) -> int:
        if raw in (None, ""):
            return 3
        if isinstance(raw, bool):
            raise ValueError("method_config.chunk_size must be an int, "
                             "got bool")
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("method_config.chunk_size must be an int, "
                             "got float")
        if isinstance(raw, str) and not raw.strip():
            raise ValueError("method_config.chunk_size must be an int, "
                             "got empty string")
        try:
            value = int(raw)
        except (TypeError, ValueError) as e:
            raise ValueError("method_config.chunk_size must be an int, "
                             f"got {type(raw).__name__}") from e
        if value <= 0:
            raise ValueError("method_config.chunk_size must be > 0")
        return value

    def _parse_ratio(
        self,
        raw: Any,
        *,
        where: str,
        default: float,
    ) -> float:
        if raw in (None, ""):
            return float(default)
        if isinstance(raw, bool):
            raise ValueError(f"{where} must be a number/string, got bool")
        if isinstance(raw, int | float):
            return float(raw)
        if isinstance(raw, str) and raw.strip():
            return float(raw)
        raise ValueError(f"{where} must be a number/string, "
                         f"got {type(raw).__name__}")

    def _parse_timestep_index_range(self) -> tuple[int, int]:
        scheduler = self._tfsft_scheduler
        num_steps = int(
            getattr(scheduler, "config", scheduler).num_train_timesteps)

        min_ratio = self._parse_ratio(
            self.method_config.get("min_timestep_ratio", None),
            where="method.min_timestep_ratio",
            default=0.0,
        )
        max_ratio = self._parse_ratio(
            self.method_config.get("max_timestep_ratio", None),
            where="method.max_timestep_ratio",
            default=1.0,
        )

        if not (0.0 <= min_ratio <= 1.0 and 0.0 <= max_ratio <= 1.0):
            raise ValueError("TFSFT timestep ratios must be in [0,1], "
                             f"got min={min_ratio}, max={max_ratio}")
        if max_ratio < min_ratio:
            raise ValueError("method_config.max_timestep_ratio must be "
                             ">= min_timestep_ratio")

        min_index = int(min_ratio * num_steps)
        max_index = int(max_ratio * num_steps)
        min_index = max(0, min(min_index, num_steps - 1))
        max_index = max(0, min(max_index, num_steps - 1))

        if max_index <= min_index:
            max_index = min(num_steps - 1, min_index + 1)

        return min_index, max_index + 1

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 "
                             "for tfsft")

        student_betas = tc.optimizer.betas
        student_sched = str(tc.optimizer.lr_scheduler)
        student_params = [
            p for p in self.student.transformer.parameters() if p.requires_grad
        ]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

    def _sample_t_inhom_indices(
        self,
        *,
        batch_size: int,
        num_latents: int,
        device: torch.device,
    ) -> torch.Tensor:
        chunk_size = self._chunk_size
        num_chunks = ((num_latents + chunk_size - 1) // chunk_size)
        low, high = self._timestep_index_range
        chunk_indices = torch.randint(
            low=low,
            high=high,
            size=(batch_size, num_chunks),
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        expanded = chunk_indices.repeat_interleave(chunk_size, dim=1)
        return expanded[:, :num_latents]

    def _build_tfsft_scheduler(self) -> DiffusionForcingScheduler:
        student_scheduler = getattr(self.student, "noise_scheduler", None)
        if student_scheduler is None:
            raise ValueError("TFSFT requires student.noise_scheduler")
        num_steps = int(
            getattr(
                student_scheduler,
                "config",
                student_scheduler,
            ).num_train_timesteps
        )
        pipeline_config = self.training_config.pipeline_config
        if pipeline_config is None:
            raise ValueError("TFSFT requires training_config.pipeline_config")
        shift = float(
            getattr(
                pipeline_config,
                "flow_shift",
                getattr(self.student, "timestep_shift", 1.0),
            )
        )
        scheduler = DiffusionForcingScheduler(
            num_inference_steps=num_steps,
            num_train_timesteps=num_steps,
            shift=shift,
            sigma_min=0.0,
            extra_one_step=True,
            training=True,
        )
        scheduler.set_timesteps(num_steps, training=True)
        return scheduler
