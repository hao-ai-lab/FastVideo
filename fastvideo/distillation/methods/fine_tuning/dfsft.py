# SPDX-License-Identifier: Apache-2.0

"""Diffusion-forcing SFT method (DFSFT; algorithm layer).

Config keys used (YAML schema-v2):
- `recipe.method`: must be `"dfsft"` for this method.
- `roles`: requires `student` (and `roles.student.trainable=true`).
- `method_config`:
  - `attn_kind` (optional): `dense` or `vsa`
  - `chunk_size` (optional; default=3)
  - `min_timestep_ratio` / `max_timestep_ratio` (optional)
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

from fastvideo.distillation.methods.base import DistillMethod, LogScalar
from fastvideo.distillation.dispatch import register_method
from fastvideo.distillation.utils.config import (
    DistillRunConfig,
    parse_betas,
)
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

if TYPE_CHECKING:
    from fastvideo.distillation.models.base import ModelBase


@register_method("dfsft")
class DiffusionForcingSFTMethod(DistillMethod):
    """Diffusion-forcing SFT (DFSFT): train only `student` with inhomogeneous timesteps.

    This is a supervised finetuning objective (flow-matching loss), except that
    we sample *block-wise* (chunk-wise) inhomogeneous timesteps `t_inhom` over
    the latent time dimension to expose the student to noisy-history regimes.
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
            raise ValueError("DFSFT requires roles.student.trainable=true")

        self.model = model
        self.validator = validator
        self.training_args = model.training_args
        self.method_config: dict[str, Any] = dict(method_config or {})
        self.validation_config: dict[str, Any] = dict(validation_config or {})
        self._attn_kind: Literal["dense", "vsa"] = self._parse_attn_kind(
            self.method_config.get("attn_kind", None)
        )

        self._chunk_size = self._parse_chunk_size(self.method_config.get("chunk_size", None))
        self._timestep_index_range = self._parse_timestep_index_range()

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

        clean_latents = training_batch.latents
        if not torch.is_tensor(clean_latents):
            raise TypeError("TrainingBatch.latents must be a torch.Tensor")
        if clean_latents.ndim != 5:
            raise ValueError(
                "TrainingBatch.latents must be [B, T, C, H, W], got "
                f"shape={tuple(clean_latents.shape)}"
            )

        batch_size, num_latents = int(clean_latents.shape[0]), int(clean_latents.shape[1])

        transformer = self.student.require_module("transformer")
        expected_chunk = getattr(transformer, "num_frame_per_block", None)
        if expected_chunk is not None and int(expected_chunk) != int(self._chunk_size):
            raise ValueError(
                "DFSFT chunk_size must match transformer.num_frame_per_block for "
                f"causal training (got {self._chunk_size}, expected {expected_chunk})."
            )

        timestep_indices = self._sample_t_inhom_indices(
            batch_size=batch_size,
            num_latents=num_latents,
            device=clean_latents.device,
        )
        sp_size = int(getattr(self.training_args, "sp_size", 1) or 1)
        sp_group = getattr(self.model, "sp_group", None)
        if sp_size > 1 and sp_group is not None and hasattr(sp_group, "broadcast"):
            sp_group.broadcast(timestep_indices, src=0)

        scheduler = getattr(self.model, "noise_scheduler", None)
        if scheduler is None:
            raise ValueError("DFSFT requires model.noise_scheduler")

        schedule_timesteps = scheduler.timesteps.to(
            device=clean_latents.device, dtype=torch.float32
        )
        schedule_sigmas = scheduler.sigmas.to(
            device=clean_latents.device, dtype=clean_latents.dtype
        )
        t_inhom = schedule_timesteps[timestep_indices]

        noise = getattr(training_batch, "noise", None)
        if noise is None:
            noise = torch.randn_like(clean_latents)
        else:
            if not torch.is_tensor(noise):
                raise TypeError("TrainingBatch.noise must be a torch.Tensor when set")
            noise = noise.permute(0, 2, 1, 3, 4).to(dtype=clean_latents.dtype)

        noisy_latents = self.model.add_noise(
            clean_latents,
            noise,
            t_inhom.flatten(),
        )

        pred = self.model.predict_noise(
            self.student,
            noisy_latents,
            t_inhom,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )

        if bool(getattr(self.training_args, "precondition_outputs", False)):
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

    def _parse_chunk_size(self, raw: Any) -> int:
        if raw in (None, ""):
            return 3
        if isinstance(raw, bool):
            raise ValueError("method_config.chunk_size must be an int, got bool")
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("method_config.chunk_size must be an int, got float")
        if isinstance(raw, str) and not raw.strip():
            raise ValueError("method_config.chunk_size must be an int, got empty string")
        try:
            value = int(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "method_config.chunk_size must be an int, got "
                f"{type(raw).__name__}"
            ) from e
        if value <= 0:
            raise ValueError("method_config.chunk_size must be > 0")
        return value

    def _parse_ratio(self, raw: Any, *, where: str, default: float) -> float:
        if raw in (None, ""):
            return float(default)
        if isinstance(raw, bool):
            raise ValueError(f"{where} must be a number/string, got bool")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw.strip():
            return float(raw)
        raise ValueError(f"{where} must be a number/string, got {type(raw).__name__}")

    def _parse_timestep_index_range(self) -> tuple[int, int]:
        scheduler = getattr(self.model, "noise_scheduler", None)
        if scheduler is None:
            raise ValueError("DFSFT requires model.noise_scheduler")
        num_steps = int(getattr(scheduler, "config", scheduler).num_train_timesteps)

        min_ratio = self._parse_ratio(
            self.method_config.get("min_timestep_ratio", None),
            where="method_config.min_timestep_ratio",
            default=float(getattr(self.training_args, "min_timestep_ratio", 0.0) or 0.0),
        )
        max_ratio = self._parse_ratio(
            self.method_config.get("max_timestep_ratio", None),
            where="method_config.max_timestep_ratio",
            default=float(getattr(self.training_args, "max_timestep_ratio", 1.0) or 1.0),
        )

        if not (0.0 <= min_ratio <= 1.0 and 0.0 <= max_ratio <= 1.0):
            raise ValueError(
                "DFSFT timestep ratios must be in [0,1], got "
                f"min={min_ratio}, max={max_ratio}"
            )
        if max_ratio < min_ratio:
            raise ValueError(
                "method_config.max_timestep_ratio must be >= min_timestep_ratio"
            )

        min_index = int(min_ratio * num_steps)
        max_index = int(max_ratio * num_steps)
        min_index = max(0, min(min_index, num_steps - 1))
        max_index = max(0, min(max_index, num_steps - 1))

        if max_index <= min_index:
            max_index = min(num_steps - 1, min_index + 1)

        # torch.randint expects [low, high), so make high exclusive.
        return min_index, max_index + 1

    def _init_optimizers_and_schedulers(self) -> None:
        student_lr = float(getattr(self.training_args, "learning_rate", 0.0) or 0.0)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for dfsft")

        student_betas = parse_betas(
            getattr(self.training_args, "betas", None),
            where="training.betas",
        )
        student_sched = str(getattr(self.training_args, "lr_scheduler", "constant"))
        build_role_optimizer_and_scheduler(
            role="student",
            handle=self.student,
            training_args=self.training_args,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

    def _sample_t_inhom_indices(self, *, batch_size: int, num_latents: int, device: torch.device) -> torch.Tensor:
        chunk_size = self._chunk_size
        num_chunks = (num_latents + chunk_size - 1) // chunk_size
        low, high = self._timestep_index_range
        chunk_indices = torch.randint(
            low=low,
            high=high,
            size=(batch_size, num_chunks),
            device=device,
            dtype=torch.long,
        )
        expanded = chunk_indices.repeat_interleave(chunk_size, dim=1)
        return expanded[:, :num_latents]
