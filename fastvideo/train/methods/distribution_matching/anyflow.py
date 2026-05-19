# SPDX-License-Identifier: Apache-2.0
"""AnyFlow on-policy distillation method.

Stage 2 of the AnyFlow two-stage recipe. Continues from a pretrained
flow-map student and refines it via distribution-matching distillation
(DMD2) where the student is rolled out for ``student_sample_steps``
Euler-flow steps from pure noise. One randomly-chosen step in the
rollout is gradient-enabled (and broadcast across ranks so every worker
agrees on which step to gradient-enable); the rest run under
``torch.no_grad``.

Inherits ``DMD2Method`` for the alternating student / critic update
machinery and the existing DMD VSD-with-fake-score loss. Overrides
``_student_rollout`` to drive the multi-step Euler-flow rollout with
``r = t_next`` (mean-velocity sampling — matches the AnyFlow paper's
``WanAnyFlowPipeline.training_rollout`` with ``use_mean_velocity=True``).

Reference: ``pipeline_wan_anyflow.py::training_rollout`` in
NVlabs/AnyFlow at commit ``549236a``.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.distributed as dist

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
)


class AnyFlowMethod(DMD2Method):
    """AnyFlow on-policy distillation (multi-step rollout)."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config

        student_sample_steps = get_optional_int(
            mcfg, "student_sample_steps", where="method.student_sample_steps")
        if student_sample_steps is None:
            student_sample_steps = 4
        if int(student_sample_steps) <= 0:
            raise ValueError(
                "method.student_sample_steps must be positive, "
                f"got {student_sample_steps}")
        self._student_sample_steps = int(student_sample_steps)

        use_mean_velocity_raw = mcfg.get("use_mean_velocity", True)
        if not isinstance(use_mean_velocity_raw, bool):
            raise ValueError(
                "method.use_mean_velocity must be a bool, "
                f"got {type(use_mean_velocity_raw).__name__}")
        self._use_mean_velocity = bool(use_mean_velocity_raw)

        # Optional pinned rollout schedule (descending, absolute t-units).
        # Falls back to dmd_denoising_steps when absent.
        raw_t_list = mcfg.get("t_list_override", None)
        if raw_t_list is None:
            self._t_list_override: list[float] | None = None
        else:
            if not isinstance(raw_t_list, list) or not raw_t_list:
                raise ValueError(
                    "method.t_list_override must be a non-empty list of "
                    f"floats when set, got {raw_t_list!r}")
            t_list = [float(x) for x in raw_t_list]
            for i in range(len(t_list) - 1):
                if t_list[i] < t_list[i + 1]:
                    raise ValueError(
                        "method.t_list_override must be descending, "
                        f"got {t_list!r}")
            self._t_list_override = t_list

        # Scoring conditioning: AnyFlow scores against r=0 for the DMD branch.
        score_r_raw = mcfg.get("dmd_score_r_value", 0.0)
        try:
            self._dmd_score_r = float(score_r_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "method.dmd_score_r_value must be numeric, "
                f"got {score_r_raw!r}") from exc

        # Optional teacher guidance scale for the DMD loss (carry over from
        # DMD2Method's behavior; default 1.0).
        guidance = get_optional_float(
            mcfg, "real_score_guidance_scale",
            where="method.real_score_guidance_scale")
        self._real_score_guidance = float(guidance) if guidance is not None else 1.0

    # ------------------------------------------------------------------
    # Rollout schedule

    def _get_rollout_schedule(self, *, device: torch.device) -> torch.Tensor:
        """Build the descending timestep schedule used by the on-policy
        rollout. Length is ``num_steps + 1`` so ``num_steps`` Euler steps
        consume the full range.

        Order of precedence:
          1. ``method.t_list_override`` — used verbatim (absolute units).
          2. ``method.dmd_denoising_steps`` (inherited from DMD2) appended
             with a final 0 boundary if the last entry isn't already 0.
        """
        if self._t_list_override is not None:
            return torch.tensor(
                self._t_list_override, device=device, dtype=torch.float32)

        steps = self._get_denoising_step_list(device).to(dtype=torch.float32)
        if float(steps[-1].item()) != 0.0:
            zero = torch.zeros(1, device=device, dtype=torch.float32)
            steps = torch.cat([steps, zero], dim=0)
        return steps

    def _broadcast_grad_step_index(
        self,
        num_steps: int,
        *,
        device: torch.device,
    ) -> int:
        """Pick the rollout step that gets gradient enabled. In distributed
        runs the choice is broadcast from rank 0 so every worker agrees."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if dist.is_initialized() and dist.get_rank() != 0:
            idx_tensor = torch.empty(
                (1,), dtype=torch.long, device=device)
        else:
            idx_tensor = torch.randint(
                0, num_steps, (1,),
                device=device, dtype=torch.long,
                generator=self.cuda_generator)
        if dist.is_initialized():
            dist.broadcast(idx_tensor, src=0)
        return int(idx_tensor.item())

    # ------------------------------------------------------------------
    # Rollout

    def _student_rollout(
        self,
        batch: Any,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        """Multi-step Euler-flow rollout from pure noise.

        Returns the predicted clean latent ``x_0`` after the chosen
        gradient step (or the final ``x`` after the last step if
        ``with_grad`` is False — used by the critic path).
        """
        latents = batch.latents
        if latents is None or latents.ndim != 5:
            raise RuntimeError(
                "AnyFlow on-policy rollout requires TrainingBatch.latents "
                "of shape [B, T, C, H, W] for shape templating")
        device = latents.device
        dtype = latents.dtype

        schedule = self._get_rollout_schedule(device=device)
        num_entries = int(schedule.numel())
        num_steps = num_entries - 1
        if num_steps <= 0:
            raise RuntimeError(
                "rollout schedule must have at least two entries "
                f"(got {num_entries})")
        if num_steps > self._student_sample_steps:
            # Trim to the configured cap, keeping the last (=0) boundary.
            schedule = torch.cat(
                [schedule[: self._student_sample_steps], schedule[-1:]],
                dim=0)
            num_steps = self._student_sample_steps

        grad_step = self._broadcast_grad_step_index(
            num_steps, device=device) if with_grad else -1

        attn_kind: Literal["dense", "vsa"] = "vsa"
        n_train = float(self.student.num_train_timesteps)

        x = torch.randn(
            latents.shape, device=device, dtype=dtype,
            generator=self.cuda_generator)
        last_pred_x0: torch.Tensor | None = None
        batch_size = int(latents.shape[0])

        for i in range(num_steps):
            t_cur = schedule[i].expand(batch_size)
            t_next = schedule[i + 1].expand(batch_size)
            r = t_next if self._use_mean_velocity else t_cur

            enable_grad = bool(with_grad) and (i == grad_step)
            with torch.set_grad_enabled(enable_grad):
                v = self.student.predict_velocity_with_r(
                    x, t_cur, r, batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind=attn_kind,
                )
            # Euler step in absolute units: x ← x - ((t_cur - t_next) / N) * v.
            view = [-1] + [1] * (x.ndim - 1)
            dt = ((t_cur - t_next) / n_train).view(*view)
            x = x - dt * v

            if enable_grad:
                # We treat the rollout output (post-step) as a predicted
                # clean latent — AnyFlow's last Euler step lands at t=0.
                last_pred_x0 = x

        if last_pred_x0 is None:
            # No gradient step taken (with_grad=False path).
            last_pred_x0 = x

        if hasattr(batch, "dmd_latent_vis_dict"):
            batch.dmd_latent_vis_dict["generator_timestep"] = (
                schedule[-1].detach().clone())
        return last_pred_x0
