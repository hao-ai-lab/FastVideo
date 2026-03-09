# SPDX-License-Identifier: Apache-2.0
"""Diffusion-forcing SFT method (DFSFT; algorithm layer)."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import TrainingMethod, LogScalar
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )


class DiffusionForcingSFTMethod(TrainingMethod):
    """Diffusion-forcing SFT (DFSFT): train only ``student``
    with inhomogeneous timesteps.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("DFSFT requires role 'student'")
        if not self.student._trainable:
            raise ValueError("DFSFT requires student to be trainable")
        self._attn_kind: Literal["dense", "vsa"] = (self._parse_attn_kind(self.method_config.get("attn_kind", None)))

        self._chunk_size = self._parse_chunk_size(self.method_config.get("chunk_size", None))
        self._timestep_index_range = (self._parse_timestep_index_range())

        # Initialize preprocessors on student.
        self.student.init_preprocessors(self.training_config)

        self._init_optimizers_and_schedulers()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    # TrainingMethod override: single_train_step
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        del iteration
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            current_vsa_sparsity=current_vsa_sparsity,
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
        if (expected_chunk is not None and int(expected_chunk) != int(self._chunk_size)):
            raise ValueError("DFSFT chunk_size must match "
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
        if (sp_size > 1 and sp_group is not None and hasattr(sp_group, "broadcast")):
            sp_group.broadcast(timestep_indices, src=0)

        scheduler = self.student.noise_scheduler
        if scheduler is None:
            raise ValueError("DFSFT requires student.noise_scheduler")

        schedule_timesteps = scheduler.timesteps.to(device=clean_latents.device, dtype=torch.float32)
        schedule_sigmas = scheduler.sigmas.to(
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        t_inhom = schedule_timesteps[timestep_indices]

        # Override the homogeneous timesteps from prepare_batch
        # so that set_forward_context (in predict_noise and
        # backward) receives the correct per-chunk timesteps.
        training_batch.timesteps = t_inhom

        noise = getattr(training_batch, "noise", None)
        if noise is None:
            noise = torch.randn_like(clean_latents)
        else:
            if not torch.is_tensor(noise):
                raise TypeError("TrainingBatch.noise must be a "
                                "torch.Tensor when set")
            noise = noise.permute(0, 2, 1, 3, 4).to(dtype=clean_latents.dtype)

        noisy_latents = self.student.add_noise(
            clean_latents,
            noise,
            t_inhom.flatten(),
        )

        pred = self.student.predict_noise(
            noisy_latents,
            t_inhom,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )

        if bool(self.training_config.model.precondition_outputs):
            sigmas = schedule_sigmas[timestep_indices]
            sigmas = sigmas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            pred_x0 = noisy_latents - pred * sigmas
            loss = F.mse_loss(pred_x0.float(), clean_latents.float())
        else:
            target = noise - clean_latents
            loss = F.mse_loss(pred.float(), target.float())

        if self._attn_kind == "vsa":
            attn_metadata = training_batch.attn_metadata_vsa
        else:
            attn_metadata = training_batch.attn_metadata

        loss_map = {"total_loss": loss, "dfsft_loss": loss}
        outputs: dict[str, Any] = {
            "_fv_backward": (
                training_batch.timesteps,
                attn_metadata,
            )
        }
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

    # TrainingMethod override: backward
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

    # TrainingMethod override: get_optimizers
    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    # TrainingMethod override: get_lr_schedulers
    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

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

    def _parse_timestep_index_range(self, ) -> tuple[int, int]:
        scheduler = self.student.noise_scheduler
        if scheduler is None:
            raise ValueError("DFSFT requires student.noise_scheduler")
        num_steps = int(getattr(scheduler, "config", scheduler).num_train_timesteps)

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
            raise ValueError("DFSFT timestep ratios must be in [0,1], "
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
                             "for dfsft")

        student_betas = tc.optimizer.betas
        student_sched = str(tc.optimizer.lr_scheduler)
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
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
