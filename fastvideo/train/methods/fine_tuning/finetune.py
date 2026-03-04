# SPDX-License-Identifier: Apache-2.0
"""Supervised finetuning method (algorithm layer)."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import TrainingMethod, LogScalar
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler,
    clip_grad_norm_if_needed,
)
from fastvideo.train.utils.validation import (
    is_validation_enabled,
    parse_validation_dataset_file,
    parse_validation_every_steps,
    parse_validation_guidance_scale,
    parse_validation_num_frames,
    parse_validation_ode_solver,
    parse_validation_output_dir,
    parse_validation_rollout_mode,
    parse_validation_sampler_kind,
    parse_validation_sampling_steps,
)
from fastvideo.train.validators.base import ValidationRequest

if TYPE_CHECKING:
    pass


class FineTuneMethod(TrainingMethod):
    """Supervised finetuning: only ``student`` participates."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
        validator: Any | None = None,
    ) -> None:
        super().__init__(role_models=role_models)

        if "student" not in role_models:
            raise ValueError("FineTuneMethod requires role 'student'")
        self.student = role_models["student"]
        if not getattr(self.student, "_trainable", True):
            raise ValueError("FineTuneMethod requires student to be trainable")

        self.validator = validator
        self.training_config = cfg.training
        self.method_config: dict[str, Any] = dict(cfg.method)
        self.validation_config: dict[str, Any] = dict(getattr(cfg, "validation", {}) or {})
        self._attn_kind: Literal["dense", "vsa"] = (self._parse_attn_kind(self.method_config.get("attn_kind", None)))

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
            current_vsa_sparsity=current_vsa_sparsity,
            latents_source="data",
        )

        if training_batch.latents is None:
            raise RuntimeError("prepare_batch() must set TrainingBatch.latents")
        if training_batch.noisy_model_input is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.noisy_model_input")
        if training_batch.noise is None:
            raise RuntimeError("prepare_batch() must set TrainingBatch.noise")
        if training_batch.sigmas is None:
            raise RuntimeError("prepare_batch() must set TrainingBatch.sigmas")
        if training_batch.timesteps is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.timesteps")

        clean_latents = training_batch.latents
        noisy_latents = training_batch.noisy_model_input.permute(0, 2, 1, 3, 4)
        noise = training_batch.noise.permute(0, 2, 1, 3, 4)
        sigmas = training_batch.sigmas
        timesteps = training_batch.timesteps

        pred = self.student.predict_noise(
            noisy_latents,
            timesteps,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )

        if bool(self.training_config.model.precondition_outputs):
            pred_x0 = noisy_latents - pred * sigmas
            loss = F.mse_loss(pred_x0.float(), clean_latents.float())
        else:
            target = noise - clean_latents
            loss = F.mse_loss(pred.float(), target.float())

        if self._attn_kind == "vsa":
            attn_metadata = training_batch.attn_metadata_vsa
        else:
            attn_metadata = training_batch.attn_metadata

        loss_map = {"total_loss": loss, "finetune_loss": loss}
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
    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    # TrainingMethod override: get_lr_schedulers
    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    # TrainingMethod override: optimizers_schedulers_step
    def optimizers_schedulers_step(self, iteration: int) -> None:
        clip_grad_norm_if_needed(self.student.transformer, self.training_config.optimizer.max_grad_norm)
        super().optimizers_schedulers_step(iteration)

    # Trainer hook: on_train_start
    def on_train_start(self) -> None:
        self.student.on_train_start()

    # Trainer hook: log_validation
    def log_validation(self, iteration: int) -> None:
        validator = getattr(self, "validator", None)
        if validator is None:
            return
        if not is_validation_enabled(self.validation_config):
            return

        every_steps = parse_validation_every_steps(self.validation_config)
        if every_steps <= 0:
            return
        if iteration % every_steps != 0:
            return

        dataset_file = parse_validation_dataset_file(self.validation_config)
        sampling_steps = parse_validation_sampling_steps(self.validation_config)
        guidance_scale = parse_validation_guidance_scale(self.validation_config)
        sampler_kind = parse_validation_sampler_kind(self.validation_config, default="ode")
        rollout_mode = parse_validation_rollout_mode(self.validation_config)
        output_dir = parse_validation_output_dir(self.validation_config)
        num_actions = parse_validation_num_frames(self.validation_config)
        ode_solver = parse_validation_ode_solver(self.validation_config, sampler_kind=sampler_kind)

        request = ValidationRequest(
            sample_handle=self.student,
            dataset_file=dataset_file,
            sampling_steps=sampling_steps,
            sampler_kind=sampler_kind,
            rollout_mode=rollout_mode,
            ode_solver=ode_solver,
            sampling_timesteps=None,
            guidance_scale=guidance_scale,
            num_frames=num_actions,
            output_dir=output_dir,
        )
        validator.log_validation(iteration, request=request)

    # Checkpoint hook: get_rng_generators
    def get_rng_generators(self) -> dict[str, torch.Generator]:
        generators: dict[str, torch.Generator] = {}

        student_gens = self.student.get_rng_generators()
        generators.update(student_gens)

        validator = getattr(self, "validator", None)
        validation_gen = getattr(validator, "validation_random_generator", None)
        if isinstance(validation_gen, torch.Generator):
            generators["validation_cpu"] = validation_gen

        return generators

    def _parse_attn_kind(self, raw: Any) -> Literal["dense", "vsa"]:
        if raw in (None, ""):
            return "dense"
        kind = str(raw).strip().lower()
        if kind not in {"dense", "vsa"}:
            raise ValueError("method_config.attn_kind must be one of "
                             f"{{'dense', 'vsa'}}, got {raw!r}.")
        return cast(Literal["dense", "vsa"], kind)

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config

        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for finetune")

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
