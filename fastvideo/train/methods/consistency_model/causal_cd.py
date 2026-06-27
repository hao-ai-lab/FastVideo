# SPDX-License-Identifier: Apache-2.0
"""Causal Consistency Distillation (Causal-Forcing++ Stage-2b).

ODE-data-free initialization for asymmetric DMD. Ported from
Causal-Forcing ``model/naive_consistency.py``.

For a clean latent and a discrete flow-match schedule of ``discrete_cd_N``
steps, sample an index ``i`` and form ``(t, t_next)``. A frozen *teacher*
takes a single CFG Euler step from ``latent_t`` to ``latent_t_next``. The
trainable *student* predicts ``x0`` at ``t`` and an EMA copy of the student
predicts ``x0`` at ``t_next``; the loss is their MSE. All three forwards run
under clean-history teacher forcing (``clean_x``), so the target is produced
online from ground-truth latents only -- no precomputed ODE pairs.

Roles (all initialized from the same checkpoint, as in the reference):
  * ``student`` -- trainable generator.
  * ``teacher`` -- frozen; the single-step Euler ODE target.
  * ``ema``     -- frozen; EMA of the student, the consistency target.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


class CausalConsistencyDistillationMethod(TrainingMethod):
    """Causal consistency distillation (Causal-Forcing++)."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        for role in ("student", "teacher", "ema"):
            if role not in role_models:
                raise ValueError(f"Causal-CD requires role {role!r} "
                                 "(student trainable; teacher + ema frozen, "
                                 "both initialized from the student's "
                                 "checkpoint)")
        if not self.student._trainable:
            raise ValueError("Causal-CD requires student to be trainable")
        self.teacher = role_models["teacher"]
        self.ema_model = role_models["ema"]

        self._attn_kind = self._infer_attn_kind()
        self._guidance_scale = float(self.method_config.get("guidance_scale", 3.0))
        self._discrete_cd_n = int(self.method_config.get("discrete_cd_N", 48))
        if self._discrete_cd_n < 2:
            raise ValueError("method.discrete_cd_N must be >= 2")
        self._ema_decay = float(self.method_config.get("ema_decay", 0.95))

        self.student.init_preprocessors(self.training_config)
        self._init_optimizers_and_schedulers()

    # ------------------------------------------------------------------

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    # ------------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        clean_latents = training_batch.latents
        if not torch.is_tensor(clean_latents) or clean_latents.ndim != 5:
            raise ValueError("Causal-CD expects [B, T, C, H, W] latents")

        batch_size, num_latents = int(clean_latents.shape[0]), int(clean_latents.shape[1])
        device = clean_latents.device

        cd_timesteps = self._cd_timesteps(device)
        idx = int(torch.randint(0, self._discrete_cd_n - 1, (1, ), generator=self.cuda_generator, device=device).item())
        t = cd_timesteps[idx]
        t_next = cd_timesteps[idx + 1]
        t_pf = t * torch.ones(batch_size, num_latents, device=device)
        t_next_pf = t_next * torch.ones(batch_size, num_latents, device=device)

        noise = torch.randn(
            clean_latents.shape,
            generator=self.cuda_generator,
            device=device,
            dtype=clean_latents.dtype,
        )
        latent_t = self.student.add_noise(clean_latents, noise, t_pf.flatten())

        # Teacher: single CFG Euler step latent_t -> latent_t_next (no grad).
        with torch.no_grad():
            v_cond = self._predict_flow(self.teacher,
                                        latent_t,
                                        t_pf,
                                        training_batch,
                                        conditional=True,
                                        clean_x=clean_latents)
            v_uncond = self._predict_flow(self.teacher,
                                          latent_t,
                                          t_pf,
                                          training_batch,
                                          conditional=False,
                                          clean_x=clean_latents)
            v_pred = v_uncond + self._guidance_scale * (v_cond - v_uncond)
            dt = ((t - t_next) / float(self.student.num_train_timesteps))
            latent_t_next = latent_t - dt * v_pred

        # Student x0 at t (with grad).
        training_batch.timesteps = t_pf
        flow_student = self._predict_flow(self.student,
                                          latent_t,
                                          t_pf,
                                          training_batch,
                                          conditional=True,
                                          clean_x=clean_latents)
        x0_t = self._to_x0(flow_student, latent_t, t_pf)

        # EMA-student x0 at t_next (no grad) -- the consistency target.
        with torch.no_grad():
            flow_ema = self._predict_flow(self.ema_model,
                                          latent_t_next,
                                          t_next_pf,
                                          training_batch,
                                          conditional=True,
                                          clean_x=clean_latents)
            x0_t_next = self._to_x0(flow_ema, latent_t_next, t_next_pf)

        loss = F.mse_loss(x0_t.float(), x0_t_next.float())

        loss_map = {"total_loss": loss, "causal_cd_loss": loss}
        outputs: dict[str, Any] = {"_fv_backward": (t_pf, training_batch.attn_metadata)}
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

    # ------------------------------------------------------------------

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
        self.student.backward(loss_map["total_loss"], ctx, grad_accum_rounds=grad_accum_rounds)

    def optimizers_schedulers_step(self, iteration: int) -> None:
        super().optimizers_schedulers_step(iteration)
        self._update_ema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_flow(
        self,
        model: ModelBase,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        clean_x: torch.Tensor,
    ) -> torch.Tensor:
        """Run a (teacher-forced) flow prediction on ``model``."""
        return model.predict_noise(latents,
                                   timestep,
                                   batch,
                                   conditional=conditional,
                                   cfg_uncond=None,
                                   attn_kind=self._attn_kind,
                                   clean_x=clean_x)

    def _to_x0(
        self,
        flow: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return pred_noise_to_pred_video(
            pred_noise=flow.flatten(0, 1),
            noise_input_latent=latents.flatten(0, 1),
            timestep=timestep,
            scheduler=self.student.noise_scheduler,
        ).unflatten(0, flow.shape[:2])

    def _cd_timesteps(self, device: torch.device) -> torch.Tensor:
        full = self.student.noise_scheduler.timesteps.to(device=device, dtype=torch.float32)
        idx = torch.linspace(0, full.shape[0] - 1, self._discrete_cd_n, device=device).round().long()
        return full[idx]

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self._ema_decay
        for ema_p, p in zip(self.ema_model.transformer.parameters(), self.student.transformer.parameters(),
                            strict=True):
            ema_p.mul_(decay).add_(p.detach().to(ema_p.dtype), alpha=1.0 - decay)

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 for causal-cd")
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )
