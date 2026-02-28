# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

import torch
import torch.nn.functional as F

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

from fastvideo.distillation.roles import RoleHandle, RoleManager
from fastvideo.distillation.methods.base import DistillMethod, LogScalar
from fastvideo.distillation.dispatch import register_method
from fastvideo.distillation.validators.base import ValidationRequest
from fastvideo.distillation.utils.config import DistillRunConfig, parse_betas


class _FineTuneAdapter(Protocol):
    """Adapter contract for :class:`FineTuneMethod`.

    Finetuning is implemented as a method (algorithm layer) on top of the
    model-plugin-provided adapter. The method must remain model-plugin agnostic, so
    it consumes only operation-centric primitives exposed by the adapter.
    """

    training_args: Any

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
        adapter: _FineTuneAdapter,
        method_config: dict[str, Any] | None = None,
        validation_config: dict[str, Any] | None = None,
        validator: Any | None = None,
    ) -> None:
        super().__init__(bundle)
        bundle.require_roles(["student"])
        self.student = bundle.role("student")
        if not self.student.trainable:
            raise ValueError("FineTuneMethod requires roles.student.trainable=true")

        self.adapter = adapter
        self.validator = validator
        self.training_args = adapter.training_args
        self.method_config: dict[str, Any] = dict(method_config or {})
        self.validation_config: dict[str, Any] = dict(validation_config or {})
        self._attn_kind: Literal["dense", "vsa"] = self._parse_attn_kind(
            self.method_config.get("attn_kind", None)
        )

        self._init_optimizers_and_schedulers()

    @classmethod
    def build(
        cls,
        *,
        cfg: DistillRunConfig,
        bundle: RoleManager,
        adapter: Any,
        validator: Any | None,
    ) -> DistillMethod:
        return cls(
            bundle=bundle,
            adapter=adapter,
            method_config=cfg.method_config,
            validation_config=cfg.validation,
            validator=validator,
        )

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

    def _build_role_optimizer_and_scheduler(
        self,
        *,
        role: str,
        handle: RoleHandle,
        learning_rate: float,
        betas: tuple[float, float],
        scheduler_name: str,
    ) -> None:
        modules = handle.modules
        params: list[torch.nn.Parameter] = []
        for module in modules.values():
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
        training_args = self.training_args

        student_lr = float(getattr(training_args, "learning_rate", 0.0) or 0.0)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for finetune")

        student_betas = parse_betas(
            getattr(training_args, "betas", None),
            where="training.betas",
        )
        student_sched = str(getattr(training_args, "lr_scheduler", "constant"))
        self._build_role_optimizer_and_scheduler(
            role="student",
            handle=self.student,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

    def on_train_start(self) -> None:
        self.adapter.on_train_start()

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
            raise ValueError(
                "training.validation.sampling_steps must be set for validation"
            )
        if isinstance(raw, bool):
            raise ValueError(
                "validation sampling_steps must be an int/list/str, got bool"
            )
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
            raise ValueError(
                "validation guidance_scale must be a number/string, got bool"
            )
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

        request = ValidationRequest(
            sample_handle=self.student,
            dataset_file=dataset_file,
            sampling_steps=sampling_steps,
            sampler_kind=sampler_kind,
            rollout_mode=cast(Literal["parallel", "streaming"], rollout_mode),
            ode_solver=ode_solver,
            sampling_timesteps=None,
            guidance_scale=guidance_scale,
            output_dir=output_dir,
        )
        validator.log_validation(iteration, request=request)

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        """Return RNG generators that should be checkpointed for exact resume."""

        generators: dict[str, torch.Generator] = {}

        adapter = getattr(self, "adapter", None)
        get_adapter_generators = getattr(adapter, "get_rng_generators", None)
        if callable(get_adapter_generators):
            generators.update(get_adapter_generators())

        validator = getattr(self, "validator", None)
        validation_gen = getattr(validator, "validation_random_generator", None)
        if isinstance(validation_gen, torch.Generator):
            generators["validation_cpu"] = validation_gen

        return generators

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

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        training_batch = self.adapter.prepare_batch(
            batch,
            current_vsa_sparsity=current_vsa_sparsity,
            latents_source="data",
        )

        if training_batch.latents is None:
            raise RuntimeError("adapter.prepare_batch() must set TrainingBatch.latents")
        if training_batch.noisy_model_input is None:
            raise RuntimeError(
                "adapter.prepare_batch() must set TrainingBatch.noisy_model_input"
            )
        if training_batch.noise is None:
            raise RuntimeError("adapter.prepare_batch() must set TrainingBatch.noise")
        if training_batch.sigmas is None:
            raise RuntimeError("adapter.prepare_batch() must set TrainingBatch.sigmas")
        if training_batch.timesteps is None:
            raise RuntimeError("adapter.prepare_batch() must set TrainingBatch.timesteps")

        clean_latents = training_batch.latents
        noisy_latents = training_batch.noisy_model_input.permute(0, 2, 1, 3, 4)
        noise = training_batch.noise.permute(0, 2, 1, 3, 4)
        sigmas = training_batch.sigmas
        timesteps = training_batch.timesteps

        pred = self.adapter.predict_noise(
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
        outputs: dict[str, Any] = {
            "_fv_backward": (training_batch.timesteps, attn_metadata)
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
            super().backward(loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return
        self.adapter.backward(
            loss_map["total_loss"],
            ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return list(self.student.optimizers.values())

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return list(self.student.lr_schedulers.values())

    def optimizers_schedulers_step(self, iteration: int) -> None:
        for module in self.student.modules.values():
            self._clip_grad_norm(module)
        super().optimizers_schedulers_step(iteration)
