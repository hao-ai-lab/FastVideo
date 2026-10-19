# SPDX-License-Identifier: Apache-2.0
"""Carried, single-head DMD with alternating critic/student updates.

Each accumulation slot keeps one detached trajectory and its conditioning.
After a one-time uniform pre-walk, one student forward advances each slot by
one trained rung. Carries are intentionally transient across checkpoint resumes.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.distribution_matching.packed_dmd import PackedDMDMixin
from fastvideo.train.methods.trajectory_carry import (
    TrajectoryCarry,
    parse_carry_slots,
    snapshot_raw_batch,
    stagger_offset,
    stagger_rank_and_groups,
    validate_carry_coverage,
)
from fastvideo.train.models.base import ModelBase


class CarriedDMDMethod(PackedDMDMixin):
    """Single-head DMD; the critic and student never update on the same step.

    With interval five, steps 1--4 fit the critic to detached student samples
    and step 5 updates only the student against the teacher/critic score gap.
    Both active roles participate in the configured gradient clipping callback.
    The existing public DMD2 method retains its independent cadence.
    """

    def __init__(self, *, cfg: Any, role_models: dict[str, ModelBase]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if self._rollout_mode != "simulate":
            raise ValueError("CarriedDMDMethod requires method.rollout_mode=simulate")
        if self.method_config.get("rollout_carry", True) is not True:
            raise ValueError("CarriedDMDMethod requires method.rollout_carry=true")
        if self.method_config.get("rollout_data_forcing", False) is not False:
            raise ValueError("CarriedDMDMethod does not support data-forced trajectories")
        if self.method_config.get("warp_denoising_step", False) is not False:
            raise ValueError("CarriedDMDMethod requires explicit unwarped DMD rungs")
        self._rollout_sample_type = str(self.method_config.get("rollout_sample_type", "ode")).strip().lower()
        if self._rollout_sample_type not in ("ode", "sde"):
            raise ValueError("method.rollout_sample_type must be ode or sde")
        if self._rollout_sample_type == "ode" and not (callable(getattr(self.student, "extract_eps_for_batch", None))
                                                       or callable(getattr(self.student, "extract_eps", None))):
            raise ValueError("ODE carry requires the model's extract_eps hook")
        raw = self.method_config.get("dmd_denoising_steps")
        if (not isinstance(raw, list) or not raw
                or any(type(step) is not int or not 0 < step <= self.student.num_train_timesteps for step in raw)
                or any(left <= right for left, right in zip(raw, raw[1:], strict=False))):
            raise ValueError("method.dmd_denoising_steps must be strictly decreasing positive integer rungs")
        self._carry = TrajectoryCarry(parse_carry_slots(self.method_config, self.training_config.loop))
        _, groups = stagger_rank_and_groups(self.training_config)
        validate_carry_coverage(streams=groups * self._carry.num_slots,
                                num_steps=len(raw),
                                interval=self._generator_update_interval(),
                                owner="CarriedDMDMethod")

    def get_grad_clip_targets(self, iteration: int) -> dict[str, torch.nn.Module]:
        role = "student" if self._should_update_student(iteration) else "critic"
        model = self.student if role == "student" else self.critic
        return {role: model.transformer}

    def inference_checkpoint_json_overrides(self, role: str = "student") -> dict[str, dict[str, Any]]:
        """Export the trained ladder and effective clocks with the student.

        H3's public inference API counts sigma-grid points, including terminal
        zero: an eight-forward student therefore requests nine grid points.
        Only the distilled student has a deployment contract for this method.
        """
        if role != "student":
            raise ValueError("CarriedDMDMethod supports inference export only for role 'student'")
        steps = list(self.method_config["dmd_denoising_steps"])
        video_shift = float(self.student.video_scheduler_shift)
        audio_shift = float(self.student.audio_scheduler_shift)
        return {
            "fastvideo_inference.json": {
                "schema_version": "fasth3-inference-contract-v1",
                "num_inference_steps": len(steps) + 1,
                "transformer_forwards": len(steps),
                "dmd_denoising_steps": steps,
                "video_scheduler_shift": video_shift,
                "audio_scheduler_shift": audio_shift,
            },
            "scheduler/scheduler_config.json": {
                "shift": video_shift
            },
            "audio_scheduler/scheduler_config.json": {
                "shift": audio_shift
            },
        }

    @staticmethod
    def _extract_eps_for_batch(
        model: ModelBase,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        hook = getattr(model, "extract_eps_for_batch", None)
        if hook is not None and batch is not None:
            return hook(noisy_latents, clean_latents, timestep, batch)
        extractor = getattr(model, "extract_eps", None)
        if not callable(extractor):
            raise TypeError(f"{type(model).__name__} does not implement extract_eps")
        return extractor(noisy_latents, clean_latents, timestep)

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        """One backward-simulation call: one generation forward, carried state.

        A multistep student is only ever correct on its own sampling
        trajectory, and walking the full grid every call costs
        ``len(dmd_denoising_steps)`` forwards. Instead the walk is spread
        over consecutive calls: each call pays for exactly one student
        forward at the carried rung, both phases (student and critic)
        consume it — the critic is fit on the same simulated states the
        student trains on — and both advance the trajectory. Each
        grad-accum round owns an independent slot, selected round-robin
        because the trainer does not pass the round index.

        An empty slot (first ever use, cleared after a finished trajectory,
        or after a resume — the carry is transient and never checkpointed)
        starts a fresh trajectory from noise and adopts the incoming loader
        batch's conditioning; mid-walk calls ignore the fresh loader batch
        and rebuild the training batch from the carried raw batch, since a
        trajectory keeps the prompt it set out with.
        """
        slot = self._carry.next_slot()

        carried = self._carry.slots[slot]
        raw_batch = (snapshot_raw_batch(batch, self.student.device) if carried is None else carried["raw_batch"])

        training_batch = self.student.prepare_batch(
            raw_batch,
            generator=self.cuda_generator,
            latents_source="zeros",
        )
        latents = training_batch.latents
        device = latents.device
        step_list = self._get_denoising_step_list(device)

        if carried is None:
            rung = 0
            state = torch.randn(
                latents.shape,
                device=device,
                dtype=latents.dtype,
                generator=self.cuda_generator,
            )
            # Stagger only the first-ever fill of each slot; later fresh
            # starts begin at rung 0 with no pre-walk and stay out of phase
            # naturally.
            if not self._carry.seeded[slot]:
                self._carry.seeded[slot] = True
                state, rung = self._staggered_start(
                    state,
                    training_batch,
                    step_list,
                    slot,
                )
        else:
            rung = int(carried["rung"])
            state = carried["state"]

        timestep = step_list[rung] * torch.ones(
            1,
            device=device,
            dtype=torch.long,
        )

        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros((), device=device, dtype=torch.float32)
        fake_score_loss = torch.zeros_like(generator_loss)
        student_ctx = None
        critic_ctx = None
        critic_outputs: dict[str, Any] = {}
        generator_metrics: dict[str, LogScalar] = {}
        critic_metrics: dict[str, LogScalar] = {}
        if update_student:
            generator_pred_x0 = self.student.predict_x0(
                state,
                timestep,
                training_batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="vsa",
            )
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss, generator_metrics = self._dmd_loss(generator_pred_x0, training_batch)
            training_batch.dmd_latent_vis_dict["generator_pred_video"] = generator_pred_x0.detach()
        else:
            with torch.no_grad():
                generator_pred_x0 = self.student.predict_x0(
                    state,
                    timestep,
                    training_batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )
            (
                fake_score_loss,
                critic_ctx,
                critic_outputs,
                critic_metrics,
            ) = self._critic_flow_matching_loss(
                training_batch,
                generator_pred_x0=generator_pred_x0,
            )
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep.float().detach()

        # Advance after the loss path: both phases generated, so both hand
        # the trajectory on.
        self._advance_carry(
            slot,
            state,
            generator_pred_x0,
            timestep,
            rung,
            step_list,
            training_batch,
            raw_batch,
        )

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }
        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            "rollout_step": float(rung),
            **generator_metrics,
            **critic_metrics,
        }
        # Rank-local latent snapshots for LatentVisCallback.
        self.latent_vis = {
            **(training_batch.fake_score_latent_vis_dict or {}),
            **(training_batch.dmd_latent_vis_dict or {}),
            "_fv_latent_layout":
            getattr(training_batch, "minimax_h3_dmd_layout", None),
        }
        return loss_map, outputs, metrics

    def _staggered_start(
        self,
        state: torch.Tensor,
        training_batch: Any,
        step_list: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, int]:
        """Pre-walk a fresh trajectory and snapshot it at this stream's rung.

        First-ever fill of a slot only. Each (rank, slot) stream starts at
        ``(stagger_rank * slots + slot) % len(grid)``, spreading the
        streams evenly across the grid so student updates (which revisit
        rungs modulo ``gcd(len(grid), generator_update_interval)``) see
        every rung. Every rank walks the whole grid under ``no_grad``
        regardless of its offset so the FSDP forwards issue a uniform
        collective count — a rank-dependent count would desynchronize the
        all-gathers and hang; only the kept snapshot differs per rank.
        """
        rank, _ = stagger_rank_and_groups(self.training_config)
        grid_len = len(step_list)
        offset = stagger_offset(stagger_rank=rank, num_slots=self._carry.num_slots, slot=slot, num_steps=grid_len)
        device = state.device
        snapshot = state
        with torch.no_grad():
            for rung in range(grid_len - 1):
                timestep = step_list[rung] * torch.ones(
                    1,
                    device=device,
                    dtype=torch.long,
                )
                pred_x0 = self.student.predict_x0(
                    state,
                    timestep,
                    training_batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )
                state = self._renoise(
                    state,
                    pred_x0.detach(),
                    timestep,
                    rung + 1,
                    step_list,
                    training_batch,
                )
                if rung + 1 == offset:
                    snapshot = state
        return snapshot, offset

    def _renoise(
        self,
        state: torch.Tensor,
        pred_x0: torch.Tensor,
        timestep: torch.Tensor,
        next_rung: int,
        step_list: torch.Tensor,
        batch: Any | None = None,
    ) -> torch.Tensor:
        """Re-noise an x0 prediction made at ``timestep`` onto the next rung.

        ``sde`` draws fresh noise — the existing full-rollout hop.
        ``ode`` reuses the noise the current state implies per modality
        (``eps_m = (x_t - alpha_m(t) x0) / sigma_m(t)`` with each
        modality's shifted sigma, via the adapter's ``extract_eps``), the
        deterministic step the FastGen H3 recipe uses.
        """
        device = state.device
        next_timestep = step_list[next_rung] * torch.ones(
            1,
            device=device,
            dtype=torch.long,
        )
        if self._rollout_sample_type == "ode":
            eps = self._extract_eps_for_batch(self.student, state, pred_x0, timestep, batch)
        else:
            eps = torch.randn(
                state.shape,
                device=device,
                dtype=pred_x0.dtype,
                generator=self.cuda_generator,
            )
        return self._add_noise_for_batch(self.student, pred_x0, eps, next_timestep, batch)

    def _advance_carry(
        self,
        slot: int,
        state: torch.Tensor,
        generator_pred_x0: torch.Tensor,
        timestep: torch.Tensor,
        rung: int,
        step_list: torch.Tensor,
        training_batch: Any,
        raw_batch: dict[str, Any],
    ) -> None:
        """Hand the one paid-for step to the slot, or clear a finished walk.

        The advanced state is detached and produced under ``no_grad``: it
        feeds a later call, not a gradient path. Walking past the last rung
        ends the trajectory (the terminal clean sample is never trained
        on), so the slot empties and the next call starts fresh at rung 0.
        """
        if rung + 1 >= len(step_list):
            self._carry.clear(slot)
            return
        with torch.no_grad():
            next_state = self._renoise(
                state,
                generator_pred_x0.detach(),
                timestep,
                rung + 1,
                step_list,
                training_batch,
            )
        self._carry.set(slot, state=next_state, rung=rung + 1, raw_batch=raw_batch)
