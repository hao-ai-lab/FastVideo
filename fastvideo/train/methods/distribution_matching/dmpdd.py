# SPDX-License-Identifier: Apache-2.0
"""Distribution Matching with Parallel Decoding Distillation (DMPDD).

Port of FastGen's ``DMPDDModel``
(``fastgen/methods/distribution_matching/dmpdd.py``): a carry-based method. One trajectory
on the highest-NFE PDD grid (``pdd_steps / block_size_min`` steps) is carried
across iterations exactly like backward-simulation DMD2, so every iteration
takes a single PDD forward at an on-policy state, with gradients only on student updates.

That one materialized prediction serves three purposes:

1. its full remaining tail is integrated to a clean estimate, the PDD analog
   of DMD2's one-step x0 prediction, which feeds the DMD2 objective. The tail
   deliberately bypasses ``block_size_max``: heads beyond the PDD loss window
   train through the DMD loss alone;
2. one head inside the ``block_size_max`` window is supervised with the PDD
   teacher objective (student phase only);
3. its current coarse block is integrated under ``no_grad`` to carry the
   trajectory into a later iteration.

With ``method.rollout_mode: data_latent`` the student state is instead real
data noised to a sampled aligned block start (standalone data-dependent PDD);
no trajectory is carried and purposes 1 and 2 are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.distribution_matching.packed_dmd import PackedDMDMixin
from fastvideo.train.methods.trajectory_matching.pdd import PDDMethodMixin
from fastvideo.train.methods.trajectory_matching.pdd_core import advance_block
from fastvideo.train.models.base import ModelBase

logger = init_logger(__name__)


class DMPDDMethod(PDDMethodMixin, PackedDMDMixin):
    """DMD2 whose carried trajectory and generator are a PDD student."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        self._reject_dmd2_ladder_knobs()
        self._init_pdd(
            carried=self._rollout_mode == "simulate",
            update_interval=self._generator_update_interval(),
        )

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        """Match FastGen DMPDD: clip the student, never the fake score."""
        if self._should_update_student(iteration):
            return {"student": self.student.transformer}
        return {}

    def get_grad_sanitize_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        """Sanitize both active roles although only the student is clipped."""
        if self._should_update_student(iteration):
            return {"student": self.student.transformer}
        return {"critic": self.critic.transformer}

    # -- training -------------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        """DMD2 alternation with a PDD-block carry instead of a renoise carry."""
        if self._rollout_mode == "data_latent":
            training_batch, state, step_idx, fine_grid = self._pdd_data_latent_inputs(batch)
            loss_map, outputs, metrics, _ = self._pdd_dmd_step(state, step_idx, fine_grid, training_batch, iteration)
            metrics["pdd_block_start"] = float(step_idx[0])
            return loss_map, outputs, metrics

        walk = self._pdd_resume_walk(batch)
        loss_map, outputs, metrics, pred = self._pdd_dmd_step(
            walk.state,
            walk.step_idx,
            walk.fine_grid,
            walk.training_batch,
            iteration,
        )
        next_state: torch.Tensor | None = None
        if not walk.is_last:
            # Carry: advance the current coarse block from the same heads.
            with torch.no_grad():
                next_state = advance_block(
                    walk.state,
                    pred.detach(),
                    walk.step_idx,
                    walk.next_idx,
                    walk.fine_grid,
                    self._pdd_blocks(walk.training_batch),
                    precision_decoding=self.pdd_config.precision_decoding,
                )
        self._pdd_commit_walk(walk, next_state)
        metrics.update(walk.metrics())
        return loss_map, outputs, metrics

    def _pdd_dmd_step(
        self,
        state: torch.Tensor,
        step_idx: torch.Tensor,
        fine_grid: torch.Tensor,
        training_batch: Any,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar], torch.Tensor]:
        """One PDD forward: clean tail estimate for DMD2 plus the PDD head loss.

        Returns the DMD2 step outputs and the materialized heads (detached by
        the caller when it carries the trajectory).
        """
        device = state.device
        blocks = self._pdd_blocks(training_batch)
        update_student = self._should_update_student(iteration)
        full_end = torch.full_like(step_idx, self.pdd_steps)
        precision_decoding = self.pdd_config.precision_decoding

        with torch.set_grad_enabled(update_student):
            pred = self._pdd_heads(state, step_idx, fine_grid, training_batch)
            student_ctx: tuple[Any, Any] | None = (training_batch.timesteps, training_batch.attn_metadata_vsa)
            # The clean tail estimate is a prediction, not a sampler advance, so
            # it deliberately integrates past block_size_max to the grid's end.
            generator_pred_x0 = advance_block(
                state,
                pred,
                step_idx,
                full_end,
                fine_grid,
                blocks,
                precision_decoding=precision_decoding,
            )

        generator_loss = torch.zeros((), device=device, dtype=torch.float32)
        fake_score_loss = torch.zeros_like(generator_loss)
        critic_ctx = None
        critic_outputs: dict[str, Any] = {}
        generator_metrics: dict[str, LogScalar] = {}
        critic_metrics: dict[str, LogScalar] = {}
        aux_losses: dict[str, torch.Tensor] = {}
        if update_student:
            self._dmd_real_uncond_forwards = self._dmd_fake_uncond_forwards = 0
            generator_loss, generator_metrics = self._dmd_loss(generator_pred_x0, training_batch)
            # Evidence (only where guidance is configured) that each unconditional score forward ran.
            if float(self.method_config.get("real_score_guidance_scale", 1.0)) != 1.0:
                generator_metrics["dmd_real_uncond_forwards"] = float(self._dmd_real_uncond_forwards)
            if self._fake_score_guidance_scale != 1.0:
                generator_metrics["dmd_fake_uncond_forwards"] = float(self._dmd_fake_uncond_forwards)
            if self.pdd_config.pdd_loss_weight > 0:
                self._pdd_teacher_uncond_forwards = 0
                head = self._compute_pdd_loss_from_prediction(state, step_idx, pred, fine_grid, training_batch)
                generator_metrics.update({
                    f"pdd_loss_{name}": value.detach()
                    for name, value in head.per_modality.items()
                })
                if self.pdd_config.teacher_guidance_scale != 1.0:
                    generator_metrics["pdd_teacher_uncond_forwards"] = float(self._pdd_teacher_uncond_forwards)
                aux_losses = {"dmd_loss": generator_loss, "pdd_loss": head.total}
                generator_loss = generator_loss + self.pdd_config.pdd_loss_weight * head.total
            training_batch.dmd_latent_vis_dict["generator_pred_video"] = generator_pred_x0.detach()
        else:
            student_ctx = None
            (
                fake_score_loss,
                critic_ctx,
                critic_outputs,
                critic_metrics,
            ) = self._critic_flow_matching_loss(training_batch, generator_pred_x0=generator_pred_x0)
        training_batch.dmd_latent_vis_dict["generator_timestep"] = self.student.pdd_timestep(
            fine_grid[step_idx]).float()

        loss_map = {
            "total_loss": generator_loss + fake_score_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
            **aux_losses,
        }
        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            **generator_metrics,
            **critic_metrics,
        }
        self.latent_vis = {
            **(training_batch.fake_score_latent_vis_dict or {}),
            **(training_batch.dmd_latent_vis_dict or {}),
            "_fv_latent_layout":
            getattr(training_batch, "minimax_h3_dmd_layout", None),
        }
        return loss_map, outputs, metrics, pred


__all__ = ["DMPDDMethod"]
