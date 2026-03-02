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

from typing import Any, Literal, Protocol, cast

import torch
import torch.nn.functional as F

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

from fastvideo.distillation.methods.base import DistillMethod, LogScalar
from fastvideo.distillation.dispatch import register_method
from fastvideo.distillation.roles import RoleHandle, RoleManager
from fastvideo.distillation.utils.config import (
    DistillRunConfig,
    get_optional_int,
    parse_betas,
)
from fastvideo.distillation.validators.base import ValidationRequest


class _DFSFTModel(Protocol):
    """Model contract for diffusion-forcing SFT (DFSFT).

    DFSFT is implemented purely at the method (algorithm) layer and relies only
    on operation-centric primitives exposed by the model plugin.
    """

    training_args: Any
    noise_scheduler: Any

    def on_train_start(self) -> None:
        ...

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> Any:
        ...

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def predict_noise(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        ...

    def backward(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        ...


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
        model: _DFSFTModel,
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
            self._clip_grad_norm(module)
        super().optimizers_schedulers_step(iteration)

    # DistillTrainer hook: on_train_start
    def on_train_start(self) -> None:
        self.model.on_train_start()

    # DistillTrainer hook: log_validation
    def log_validation(self, iteration: int) -> None:
        validator = getattr(self, "validator", None)
        if validator is None:
            return
        if not self._is_validation_enabled():
            return

        every_steps = self._parse_validation_every_steps()
        if every_steps <= 0:
            return
        if iteration % every_steps != 0:
            return

        dataset_file = self._parse_validation_dataset_file()
        sampling_steps = self._parse_validation_sampling_steps()
        guidance_scale = self._parse_validation_guidance_scale()

        sampler_kind_raw = self.validation_config.get("sampler_kind", "ode")
        if not isinstance(sampler_kind_raw, str):
            raise ValueError(
                "training.validation.sampler_kind must be a string when set, got "
                f"{type(sampler_kind_raw).__name__}"
            )
        sampler_kind = sampler_kind_raw.strip().lower()
        if sampler_kind not in {"ode", "sde"}:
            raise ValueError(
                "training.validation.sampler_kind must be one of {ode, sde}, got "
                f"{sampler_kind_raw!r}"
            )
        sampler_kind = cast(Literal["ode", "sde"], sampler_kind)
        ode_solver = self._parse_validation_ode_solver(sampler_kind=sampler_kind)

        rollout_mode_raw = self.validation_config.get("rollout_mode", "parallel")
        if not isinstance(rollout_mode_raw, str):
            raise ValueError(
                "training.validation.rollout_mode must be a string when set, got "
                f"{type(rollout_mode_raw).__name__}"
            )
        rollout_mode = rollout_mode_raw.strip().lower()
        if rollout_mode not in {"parallel", "streaming"}:
            raise ValueError(
                "training.validation.rollout_mode must be one of {parallel, streaming}, "
                f"got {rollout_mode_raw!r}"
            )

        output_dir = self.validation_config.get("output_dir", None)
        if output_dir is not None and not isinstance(output_dir, str):
            raise ValueError(
                "training.validation.output_dir must be a string when set, got "
                f"{type(output_dir).__name__}"
            )

        num_actions = get_optional_int(
            self.validation_config,
            "num_frames",
            where="training.validation.num_frames",
        )
        if num_actions is not None and num_actions <= 0:
            raise ValueError("training.validation.num_frames must be > 0 when set")

        request = ValidationRequest(
            sample_handle=self.student,
            dataset_file=dataset_file,
            sampling_steps=sampling_steps,
            sampler_kind=sampler_kind,
            rollout_mode=cast(Literal["parallel", "streaming"], rollout_mode),
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

    def _build_role_optimizer_and_scheduler(
        self,
        *,
        role: str,
        handle: RoleHandle,
        learning_rate: float,
        betas: tuple[float, float],
        scheduler_name: str,
    ) -> None:
        params: list[torch.nn.Parameter] = []
        for module in handle.modules.values():
            params.extend([p for p in module.parameters() if p.requires_grad])
        if not params:
            raise ValueError(f"Role {role!r} is trainable but has no trainable parameters")

        optimizer = torch.optim.AdamW(
            params,
            lr=float(learning_rate),
            betas=betas,
            weight_decay=float(getattr(self.training_args, "weight_decay", 0.0) or 0.0),
            eps=1e-8,
        )

        scheduler = get_scheduler(
            str(scheduler_name),
            optimizer=optimizer,
            num_warmup_steps=int(getattr(self.training_args, "lr_warmup_steps", 0) or 0),
            num_training_steps=int(getattr(self.training_args, "max_train_steps", 0) or 0),
            num_cycles=int(getattr(self.training_args, "lr_num_cycles", 0) or 0),
            power=float(getattr(self.training_args, "lr_power", 0.0) or 0.0),
            min_lr_ratio=float(getattr(self.training_args, "min_lr_ratio", 0.5) or 0.5),
            last_epoch=-1,
        )

        handle.optimizers = {"main": optimizer}
        handle.lr_schedulers = {"main": scheduler}

    def _init_optimizers_and_schedulers(self) -> None:
        student_lr = float(getattr(self.training_args, "learning_rate", 0.0) or 0.0)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for dfsft")

        student_betas = parse_betas(
            getattr(self.training_args, "betas", None),
            where="training.betas",
        )
        student_sched = str(getattr(self.training_args, "lr_scheduler", "constant"))
        self._build_role_optimizer_and_scheduler(
            role="student",
            handle=self.student,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

    def _is_validation_enabled(self) -> bool:
        cfg = self.validation_config
        if not cfg:
            return False
        enabled = cfg.get("enabled", None)
        if enabled is None:
            return True
        if isinstance(enabled, bool):
            return bool(enabled)
        raise ValueError(
            "training.validation.enabled must be a bool when set, got "
            f"{type(enabled).__name__}"
        )

    def _parse_validation_every_steps(self) -> int:
        raw = self.validation_config.get("every_steps", None)
        if raw is None:
            raise ValueError(
                "training.validation.every_steps must be set when validation is enabled"
            )
        if isinstance(raw, bool):
            raise ValueError("training.validation.every_steps must be an int, got bool")
        if isinstance(raw, int):
            return int(raw)
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        if isinstance(raw, str) and raw.strip():
            return int(raw)
        raise ValueError(
            "training.validation.every_steps must be an int, got "
            f"{type(raw).__name__}"
        )

    def _parse_validation_dataset_file(self) -> str:
        raw = self.validation_config.get("dataset_file", None)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                "training.validation.dataset_file must be set when validation is enabled"
            )
        return raw.strip()

    def _parse_validation_sampling_steps(self) -> list[int]:
        raw = self.validation_config.get("sampling_steps")
        steps: list[int] = []
        if raw is None or raw == "":
            raise ValueError("training.validation.sampling_steps must be set for validation")
        if isinstance(raw, bool):
            raise ValueError("validation sampling_steps must be an int/list/str, got bool")
        if isinstance(raw, int) or (isinstance(raw, float) and raw.is_integer()):
            steps = [int(raw)]
        elif isinstance(raw, str):
            steps = [int(s) for s in raw.split(",") if str(s).strip()]
        elif isinstance(raw, list):
            steps = [int(s) for s in raw]
        else:
            raise ValueError(
                "validation sampling_steps must be an int/list/str, got "
                f"{type(raw).__name__}"
            )
        return [s for s in steps if int(s) > 0]

    def _parse_validation_guidance_scale(self) -> float | None:
        raw = self.validation_config.get("guidance_scale")
        if raw in (None, ""):
            return None
        if isinstance(raw, bool):
            raise ValueError("validation guidance_scale must be a number/string, got bool")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw.strip():
            return float(raw)
        raise ValueError(
            "validation guidance_scale must be a number/string, got "
            f"{type(raw).__name__}"
        )

    def _parse_validation_ode_solver(
        self,
        *,
        sampler_kind: Literal["ode", "sde"],
    ) -> str | None:
        raw = self.validation_config.get("ode_solver", None)
        if raw in (None, ""):
            return None
        if sampler_kind != "ode":
            raise ValueError(
                "training.validation.ode_solver is only valid when "
                "training.validation.sampler_kind='ode'"
            )
        if not isinstance(raw, str):
            raise ValueError(
                "training.validation.ode_solver must be a string when set, got "
                f"{type(raw).__name__}"
            )
        solver = raw.strip().lower()
        if solver in {"unipc", "unipc_multistep", "multistep"}:
            return "unipc"
        if solver in {"euler", "flowmatch", "flowmatch_euler"}:
            return "euler"
        raise ValueError(
            "training.validation.ode_solver must be one of {unipc, euler}, got "
            f"{raw!r}"
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

    def _clip_grad_norm(self, module: torch.nn.Module) -> float:
        max_grad_norm_raw = getattr(self.training_args, "max_grad_norm", None)
        if max_grad_norm_raw is None:
            return 0.0
        try:
            max_grad_norm = float(max_grad_norm_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "training.max_grad_norm must be a number when set, got "
                f"{max_grad_norm_raw!r}"
            ) from e
        if max_grad_norm <= 0.0:
            return 0.0
        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in module.parameters()],
            max_grad_norm,
            foreach=None,
        )
        return float(grad_norm.item()) if grad_norm is not None else 0.0
