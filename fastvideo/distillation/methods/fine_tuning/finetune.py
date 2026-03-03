# SPDX-License-Identifier: Apache-2.0

"""Supervised finetuning method (algorithm layer).

Config keys used (YAML schema-v2):
- `recipe.method`: must be `"finetune"` for this method.
- `roles`: requires `student` (and `roles.student.trainable=true`).
- `method_config`:
  - `attn_kind` (optional): `dense` or `vsa`
- `training` (selected fields used for optim/schedule):
  - `learning_rate`, `betas`, `lr_scheduler`
  - `weight_decay`, `lr_warmup_steps`, `max_train_steps`, `lr_num_cycles`,
    `lr_power`, `min_lr_ratio`, `max_grad_norm`
- `training.validation.*` (parsed by method; executed via validator):
  - `enabled`, `every_steps`, `dataset_file`, `sampling_steps`
  - optional: `guidance_scale`, `sampler_kind`, `ode_solver`, `rollout_mode`,
    `output_dir`, `num_frames`
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastvideo.distillation.dispatch import register_method
from fastvideo.distillation.methods.base import DistillMethod, LogScalar
from fastvideo.distillation.roles import RoleManager
from fastvideo.distillation.utils.optimizer import (
    build_role_optimizer_and_scheduler,
    clip_grad_norm_if_needed,
)
from fastvideo.distillation.utils.validation import (
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
from fastvideo.distillation.validators.base import ValidationRequest
from fastvideo.distillation.utils.config import (
    DistillRunConfig,
    parse_betas,
)

if TYPE_CHECKING:
    from fastvideo.distillation.models.base import ModelBase


@register_method("finetune")
class FineTuneMethod(DistillMethod):
    """Supervised finetuning as a method: only `student` participates.

    The loss follows the same objective used by the legacy training pipeline:
    - default: flow-matching target `noise - x0`
    - optional (if `training.precondition_outputs=true`): precondition to `x0`
      and regress `x0` directly.
    """

    def __init__(
        self,
        *,
        bundle: RoleManager,
        model: ModelBase,
        method_config: dict[str, Any] | None = None,
        validation_config: dict[str, Any] | None = None,
        validator: Any | None = None,
    ) -> None:
        super().__init__(bundle)
        bundle.require_roles(["student"])
        self.student = bundle.role("student")
        if not self.student.trainable:
            raise ValueError("FineTuneMethod requires roles.student.trainable=true")

        self.model = model
        self.validator = validator
        self.training_args = model.training_args
        self.method_config: dict[str, Any] = dict(method_config or {})
        self.validation_config: dict[str, Any] = dict(validation_config or {})
        self._attn_kind: Literal["dense", "vsa"] = self._parse_attn_kind(
            self.method_config.get("attn_kind", None)
        )

        self._init_optimizers_and_schedulers()

    # DistillMethod override: build
    @classmethod
    def build(
        cls,
        *,
        cfg: DistillRunConfig,
        bundle: RoleManager,
        model: Any,
        validator: Any | None,
    ) -> DistillMethod:
        return cls(
            bundle=bundle,
            model=model,
            method_config=cfg.method_config,
            validation_config=cfg.validation,
            validator=validator,
        )

    # DistillMethod override: single_train_step
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        training_batch = self.model.prepare_batch(
            batch,
            current_vsa_sparsity=current_vsa_sparsity,
            latents_source="data",
        )

        if training_batch.latents is None:
            raise RuntimeError("model.prepare_batch() must set TrainingBatch.latents")
        if training_batch.noisy_model_input is None:
            raise RuntimeError(
                "model.prepare_batch() must set TrainingBatch.noisy_model_input"
            )
        if training_batch.noise is None:
            raise RuntimeError("model.prepare_batch() must set TrainingBatch.noise")
        if training_batch.sigmas is None:
            raise RuntimeError("model.prepare_batch() must set TrainingBatch.sigmas")
        if training_batch.timesteps is None:
            raise RuntimeError("model.prepare_batch() must set TrainingBatch.timesteps")

        clean_latents = training_batch.latents
        noisy_latents = training_batch.noisy_model_input.permute(0, 2, 1, 3, 4)
        noise = training_batch.noise.permute(0, 2, 1, 3, 4)
        sigmas = training_batch.sigmas
        timesteps = training_batch.timesteps

        pred = self.model.predict_noise(
            self.student,
            noisy_latents,
            timesteps,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )

        if bool(getattr(self.training_args, "precondition_outputs", False)):
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
        outputs: dict[str, Any] = {"_fv_backward": (training_batch.timesteps, attn_metadata)}
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

    # DistillMethod override: backward
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
            super().backward(loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return
        self.model.backward(
            loss_map["total_loss"],
            ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    # DistillMethod override: get_optimizers
    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return list(self.student.optimizers.values())

    # DistillMethod override: get_lr_schedulers
    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return list(self.student.lr_schedulers.values())

    # DistillMethod override: optimizers_schedulers_step
    def optimizers_schedulers_step(self, iteration: int) -> None:
        for module in self.student.modules.values():
            clip_grad_norm_if_needed(module, self.training_args)
        super().optimizers_schedulers_step(iteration)

    # DistillTrainer hook: on_train_start
    def on_train_start(self) -> None:
        self.model.on_train_start()

    # DistillTrainer hook: log_validation
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
        ode_solver = parse_validation_ode_solver(
            self.validation_config, sampler_kind=sampler_kind
        )

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
        """Return RNG generators that should be checkpointed for exact resume."""

        generators: dict[str, torch.Generator] = {}

        model = getattr(self, "model", None)
        get_model_generators = getattr(model, "get_rng_generators", None)
        if callable(get_model_generators):
            generators.update(get_model_generators())

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
            raise ValueError(
                "method_config.attn_kind must be one of {'dense', 'vsa'}, "
                f"got {raw!r}."
            )
        return cast(Literal["dense", "vsa"], kind)

    def _init_optimizers_and_schedulers(self) -> None:
        training_args = self.training_args

        student_lr = float(getattr(training_args, "learning_rate", 0.0) or 0.0)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for finetune")

        student_betas = parse_betas(
            getattr(training_args, "betas", None),
            where="training.betas",
        )
        student_sched = str(getattr(training_args, "lr_scheduler", "constant"))
        build_role_optimizer_and_scheduler(
            role="student",
            handle=self.student,
            training_args=self.training_args,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )
