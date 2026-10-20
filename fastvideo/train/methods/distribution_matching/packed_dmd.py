# SPDX-License-Identifier: Apache-2.0
"""Packed distribution-matching objectives for parallel-decoding students.

These hooks normalize video and audio independently over real latent scalars,
carry batch-local backward contexts, and alternate critic/student optimization.
They are isolated from DMD2's existing public loss, sampling, and update contract.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Literal

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.methods.trajectory_matching.pdd_core import (
    latent_loss_region,
    latent_loss_valid_mask,
    validate_latent_loss_regions,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
    parse_betas,
    parse_cfg_uncond,
    parse_rollout_mode,
    uses_negative_prompt_conditioning,
    validate_text_only_rollout,
)
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


class PackedDMDMixin(DMD2Method):
    """DMPDD score losses and alternating role updates, shared with its PDD walk."""

    def __init__(self, *, cfg: Any, role_models: dict[str, ModelBase]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        self._score_timestep_shift = self._parse_score_timestep_shift()
        self._score_timestep_warp_max = self._parse_score_timestep_warp_max()
        self._score_timestep_continuous = self._parse_score_timestep_continuous()
        self._fake_score_loss_space = self._parse_fake_score_loss_space()
        self._fake_score_guidance_scale = self._parse_fake_score_guidance_scale()
        self._fake_score_cfg_dropout_rate = self._parse_fake_score_cfg_dropout_rate()
        self._dmd_real_uncond_forwards = 0
        self._dmd_fake_uncond_forwards = 0

    @abstractmethod
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        """Supply a packed generator walk with one active role per update.

        The legacy DMD2 step intentionally remains unchanged and must not be
        combined with these alternating-role losses and backward contexts.
        """
        raise NotImplementedError

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        backward_ctx = outputs.get("_fv_backward")
        if not isinstance(backward_ctx, dict):
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            return

        update_student = bool(backward_ctx.get("update_student", False))
        if update_student:
            student_ctx = backward_ctx.get("student_ctx")
            if student_ctx is None:
                raise RuntimeError("Missing student backward context")
            backward_loss = loss_map["generator_loss"]
            loss_scale = float(backward_ctx.get("sequence_packing_loss_scale", 1.0))
            if loss_scale != 1.0:
                backward_loss = backward_loss * loss_scale
            self.student.backward(
                backward_loss,
                student_ctx,
                grad_accum_rounds=grad_accum_rounds,
            )
            return

        critic_ctx = backward_ctx.get("critic_ctx")
        if critic_ctx is None:
            raise RuntimeError("Missing critic backward context")
        backward_loss = loss_map["fake_score_loss"]
        loss_scale = float(backward_ctx.get("sequence_packing_loss_scale", 1.0))
        if loss_scale != 1.0:
            backward_loss = backward_loss * loss_scale
        self.critic.backward(
            backward_loss,
            critic_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        if self._should_update_student(iteration):
            return [self._student_optimizer]
        return [self._critic_optimizer]

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        if self._should_update_student(iteration):
            return [self._student_lr_scheduler]
        return [self._critic_lr_scheduler]

    def _parse_rollout_mode(self, ) -> Literal["simulate", "data_latent"]:
        """Parse how DMD2 obtains the latent point used for rollout.

        ``simulate`` starts from fresh noise and lets the student create an
        artificial latent trajectory, so it can run with text-only data.
        ``data_latent`` starts from preprocessed VAE latents and perturbs them
        at a sampled denoising timestep.
        """
        return "simulate" if parse_rollout_mode(self.method_config) == "simulate" else "data_latent"

    def _validate_preprocessed_data_type(self) -> None:
        data_type = str(getattr(
            self.training_config.data,
            "preprocessed_data_type",
            "t2v",
        )).strip().lower()
        validate_text_only_rollout(data_type, self._rollout_mode)

    def _uses_negative_prompt_conditioning(self) -> bool:
        return uses_negative_prompt_conditioning(self._cfg_uncond)

    def _configure_student_negative_conditioning(self) -> None:
        setter = getattr(
            self.student,
            "set_requires_negative_conditioning",
            None,
        )
        if setter is not None:
            setter(self._uses_negative_prompt_conditioning())

    def _parse_cfg_uncond(self, ) -> dict[str, Any] | None:
        return parse_cfg_uncond(self.method_config)

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config

        # Student optimizer/scheduler.
        student_lr = float(tc.optimizer.learning_rate)
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

        # Critic optimizer/scheduler — must be set in
        # method config.
        critic_lr_raw = get_optional_float(
            self.method_config,
            "fake_score_learning_rate",
            where="method.fake_score_learning_rate",
        )
        if critic_lr_raw is None or not math.isfinite(critic_lr_raw) or critic_lr_raw <= 0.0:
            raise ValueError("method.fake_score_learning_rate must "
                             "be set to a positive value")
        critic_lr = float(critic_lr_raw)

        critic_betas_raw = self.method_config.get("fake_score_betas", None)
        if critic_betas_raw is None:
            raise ValueError("method.fake_score_betas must be set "
                             "(e.g. [0.0, 0.999])")
        critic_betas = parse_betas(
            critic_betas_raw,
            where="method.fake_score_betas",
        )

        critic_sched_raw = self.method_config.get("fake_score_lr_scheduler", None)
        if critic_sched_raw is None:
            raise ValueError("method.fake_score_lr_scheduler must "
                             "be set (e.g. 'constant')")
        critic_sched = str(critic_sched_raw)
        critic_weight_decay_raw = get_optional_float(
            self.method_config,
            "fake_score_weight_decay",
            where="method.fake_score_weight_decay",
        )
        critic_weight_decay = (float(tc.optimizer.weight_decay)
                               if critic_weight_decay_raw is None else float(critic_weight_decay_raw))
        if not math.isfinite(critic_weight_decay) or critic_weight_decay < 0.0:
            raise ValueError("method.fake_score_weight_decay must be non-negative")
        critic_params = [p for p in self.critic.transformer.parameters() if p.requires_grad]
        (
            self._critic_optimizer,
            self._critic_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=critic_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=critic_lr,
            betas=critic_betas,
            scheduler_name=critic_sched,
            weight_decay=critic_weight_decay,
        )

    @staticmethod
    def _add_noise_for_batch(
        model: ModelBase,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        """Call the batch-aware hook while retaining lightweight test doubles."""
        hook = getattr(model, "add_noise_for_batch", None)
        if hook is not None and batch is not None:
            return hook(clean_latents, noise, timestep, batch)
        return model.add_noise(clean_latents, noise, timestep)

    def _modality_slices(
        self,
        batch: Any,
        *,
        valid: torch.Tensor | None = None,
    ) -> tuple[tuple[str, slice], ...] | None:
        """Return real scalar-column slices for independent modality reductions.

        Validate against the scalar latent layout, not joint transformer rows:
        the two coordinates have different video/audio patch widths.
        """
        batch_getter = getattr(self.student, "modality_slices_for_batch", None)
        if batch_getter is not None:
            slices = tuple(batch_getter(batch))
        else:
            getter = getattr(self.student, "modality_slices", None)
            if getter is None:
                return None
            slices = tuple(getter())
        valid = latent_loss_valid_mask(batch) if valid is None else valid
        validate_latent_loss_regions(valid, tuple(region for _, region in slices))
        return slices or None

    def _packed_document_regions(
        self,
        batch: Any,
        *,
        valid: torch.Tensor | None = None,
    ) -> tuple[tuple[int, str, slice], ...] | None:
        """Return real scalar-column regions only for a cross-sample pack.

        Explicitly check that document regions exclude invalid latent columns;
        capacity padding must not become another document or loss denominator.
        """
        num_documents = int(getattr(batch, "sequence_packing_num_documents", 1) or 1)
        if num_documents <= 1:
            return None
        getter = getattr(self.student, "document_modality_slices_for_batch", None)
        if not callable(getter):
            raise RuntimeError("A sequence-packed batch requires document-aware latent loss regions")
        regions = tuple(getter(batch))
        document_ids = {int(document_id) for document_id, _, _ in regions}
        if document_ids != set(range(num_documents)):
            raise ValueError("Packed DMD loss regions do not cover every document exactly")
        valid = latent_loss_valid_mask(batch) if valid is None else valid
        validate_latent_loss_regions(valid, tuple(region for _, _, region in regions))
        return regions

    def _score_timestep_for_batch(self, device: torch.device, batch: Any) -> torch.Tensor:
        num_documents = int(getattr(batch, "sequence_packing_num_documents", 1) or 1)
        if num_documents <= 1:
            # Preserve test-double compatibility and the exact legacy RNG call.
            return self._sample_score_timestep(device)
        return self._sample_score_timestep(device, count=num_documents)

    def _modality_weight(self, name: str) -> float:
        raw = self.method_config.get("modality_loss_weights", None)
        if not isinstance(raw, dict):
            return 1.0
        value = raw.get(name, 1.0)
        return 1.0 if value is None else float(value)

    def apply_configured_lrs(self) -> None:
        """Force student/critic LRs back to the configured values (post-resume)."""
        student_lr = float(self.training_config.optimizer.learning_rate)
        critic_lr = float(self.method_config.get("fake_score_learning_rate"))
        for optimizer, scheduler, lr in (
            (self._student_optimizer, self._student_lr_scheduler, student_lr),
            (self._critic_optimizer, self._critic_lr_scheduler, critic_lr),
        ):
            for group in optimizer.param_groups:
                group["lr"] = lr
                if "initial_lr" in group:
                    group["initial_lr"] = lr
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [lr] * len(scheduler.base_lrs)

    def _parse_score_timestep_shift(self) -> float:
        """Resolve the rational warp used by score-time sampling.

        Legacy integer sampling draws uniformly in the warped coordinate and
        inverts it. Continuous FastGen parity draws the pre-warp coordinate
        directly and applies this inverse warp before the model adapter adds
        its modality-specific clock.
        """
        shift = get_optional_float(
            self.method_config,
            "score_timestep_shift",
            where="method.score_timestep_shift",
        )
        shift = 1.0 if shift is None else float(shift)
        if shift <= 0.0:
            raise ValueError("method.score_timestep_shift must be > 0, "
                             f"got {shift}")
        return shift

    def _parse_score_timestep_warp_max(self) -> float:
        """Resolve the endpoint used by the continuous rational time warp."""
        warp_max = get_optional_float(
            self.method_config,
            "score_timestep_warp_max",
            where="method.score_timestep_warp_max",
        )
        warp_max = 1.0 if warp_max is None else float(warp_max)
        if not 0.0 < warp_max <= 1.0:
            raise ValueError("method.score_timestep_warp_max must satisfy "
                             f"0 < max <= 1, got {warp_max}")
        max_ratio = self._score_max_timestep / float(self.student.num_train_timesteps)
        if max_ratio > warp_max:
            raise ValueError("method.max_timestep_ratio must not exceed "
                             "method.score_timestep_warp_max, got "
                             f"{max_ratio} > {warp_max}")
        return warp_max

    def _parse_score_timestep_continuous(self) -> bool:
        """Select FastGen-style continuous score times instead of integer bins."""
        raw = self.method_config.get("score_timestep_continuous", False)
        if not isinstance(raw, bool):
            raise ValueError("method.score_timestep_continuous must be a bool, "
                             f"got {type(raw).__name__}")
        return raw

    def _parse_fake_score_guidance_scale(self) -> float:
        """Guidance applied to the critic's (fake) score inside the DMD gradient.

        ``s == 1`` (default) uses the conditional critic prediction alone, exactly as before.
        Otherwise the fake score is ``uncond + s * (cond - uncond)``, built with the same
        ``method.cfg_uncond`` construction as the teacher's ``real_score_guidance_scale``.
        """
        scale = get_optional_float(
            self.method_config,
            "fake_score_guidance_scale",
            where="method.fake_score_guidance_scale",
        )
        scale = 1.0 if scale is None else float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"method.fake_score_guidance_scale must be positive, got {scale}")
        if scale != 1.0 and self._cfg_uncond is None:
            raise ValueError("method.fake_score_guidance_scale != 1 requires method.cfg_uncond for the "
                             "critic's unconditional forward")
        return scale

    def _parse_fake_score_cfg_dropout_rate(self) -> float:
        """Probability that a critic flow-matching update trains the unconditional critic.

        A guided fake score needs a critic that has learned an unconditional prediction of the
        student distribution; dropout (default 0: never) supplies it with the same ``cfg_uncond``.
        """
        rate = get_optional_float(
            self.method_config,
            "fake_score_cfg_dropout_rate",
            where="method.fake_score_cfg_dropout_rate",
        )
        rate = 0.0 if rate is None else float(rate)
        if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
            raise ValueError(f"method.fake_score_cfg_dropout_rate must be in [0, 1), got {rate}")
        if rate > 0.0 and self._cfg_uncond is None:
            raise ValueError("method.fake_score_cfg_dropout_rate > 0 requires method.cfg_uncond for the "
                             "critic's unconditional training forward")
        return rate

    def _critic_condition_dropped(self) -> bool:
        """Draw one critic condition-dropout decision, identical on every rank of an SP group.

        The draw comes from the method's SP-group-seeded generator, and SP rank 0's value is
        broadcast (as for the shared noise amount), so all ranks sharding one document agree.
        """
        generator = self.cuda_generator
        draw = torch.rand((1, ), generator=generator, device=generator.device, dtype=torch.float32)
        sp_group = getattr(self.student, "sp_group", None)
        if int(self.training_config.distributed.sp_size or 1) > 1 and sp_group is not None:
            sp_group.broadcast(draw, src=0)
        return bool(draw.item() < self._fake_score_cfg_dropout_rate)

    def _parse_fake_score_loss_space(self) -> dict[str, str]:
        """Resolve the critic regression space, globally or per modality.

        ``velocity`` is plain velocity MSE. A global ``x0`` setting calls the
        critic's x0 prediction directly, matching FastGen without estimating
        sigma from rounded latents. Legacy mixed mappings retain the original
        single-forward sigma-squared conversion for their x0 modalities.
        """
        raw = self.method_config.get("fake_score_loss_space", None)
        if raw is None:
            return {"__default__": "velocity"}
        if isinstance(raw, str):
            mapping = {"__default__": raw}
        elif isinstance(raw, dict):
            mapping = {str(k).strip().lower(): str(v) for k, v in raw.items()}
            mapping.setdefault("__default__", "velocity")
        else:
            raise ValueError("method.fake_score_loss_space must be a string "
                             "or a {modality: space} mapping, got "
                             f"{type(raw).__name__}")
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            space = str(value).strip().lower()
            if space not in ("velocity", "x0"):
                raise ValueError("method.fake_score_loss_space values must be "
                                 f"one of {{velocity, x0}}, got {value!r} "
                                 f"for {key!r}")
            normalized[key] = space
        return normalized

    def _fake_score_space_for(self, modality_name: str) -> str:
        return self._fake_score_loss_space.get(
            modality_name,
            self._fake_score_loss_space["__default__"],
        )

    def _sample_score_timestep(
        self,
        device: torch.device,
        *,
        count: int = 1,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError("Score timestep sample count must be positive")
        shift = self._score_timestep_shift
        num_timesteps = float(self.student.num_train_timesteps)
        t_lo = self._score_min_timestep / num_timesteps
        t_hi = self._score_max_timestep / num_timesteps

        if getattr(self, "_score_timestep_continuous", False):
            # FastGen draws the pre-warp coordinate continuously in float64,
            # then applies the rational shift. This method stores the inverse
            # shift because model adapters (H3: video 12, audio 3) apply their
            # own modality clocks afterwards. Bounds therefore belong to U,
            # not to the inverse-warped base time.
            u = torch.rand(
                [count],
                device=device,
                dtype=torch.float64,
                generator=self.cuda_generator,
            ) * (t_hi - t_lo) + t_lo
            inverse_shift = 1.0 / shift
            warp_max = getattr(self, "_score_timestep_warp_max", 1.0)
            t = (u * inverse_shift * warp_max / (u * (inverse_shift - 1.0) + warp_max))
            timestep = t * num_timesteps
            timestep = self.student.shift_and_clamp_timestep(timestep)
            return timestep.clamp(0.0, warp_max * num_timesteps)

        if shift == 1.0:
            # Draw inside the bounds directly; drawing over the full range
            # and clamping piles probability atoms onto both endpoints.
            timestep = torch.randint(
                self._score_min_timestep,
                self._score_max_timestep + 1,
                [count],
                device=device,
                dtype=torch.long,
                generator=self.cuda_generator,
            )
        else:
            sigma_lo = shift * t_lo / (1.0 + (shift - 1.0) * t_lo)
            sigma_hi = shift * t_hi / (1.0 + (shift - 1.0) * t_hi)
            u = torch.rand(
                [count],
                device=device,
                dtype=torch.float32,
                generator=self.cuda_generator,
            ) * (sigma_hi - sigma_lo) + sigma_lo
            t = u / (shift - (shift - 1.0) * u)
            timestep = (t * num_timesteps).round().to(torch.long)
        timestep = self.student.shift_and_clamp_timestep(timestep)
        return timestep.clamp(
            self._score_min_timestep,
            self._score_max_timestep,
        )

    def _critic_flow_matching_loss(
        self,
        batch: Any,
        *,
        generator_pred_x0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, dict[str, Any], dict[str, LogScalar]]:
        if generator_pred_x0 is None:
            with torch.no_grad():
                generator_pred_x0 = self._student_rollout(batch, with_grad=False)

        device = generator_pred_x0.device
        valid = latent_loss_valid_mask(batch)
        document_regions = self._packed_document_regions(batch, valid=valid)
        fake_score_timestep = self._score_timestep_for_batch(device, batch)

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
            generator=self.cuda_generator,
        )
        noisy_x0 = self._add_noise_for_batch(
            self.student,
            generator_pred_x0,
            noise,
            fake_score_timestep,
            batch,
        )
        # Opt-in condition dropout trains the critic's unconditional prediction (needed by a
        # guided fake score). Rate 0 draws nothing, so existing recipes are unchanged.
        critic_conditional = True
        if self._fake_score_cfg_dropout_rate > 0.0:
            critic_conditional = not self._critic_condition_dropped()

        slices: tuple[tuple[str, slice], ...] = ()
        if document_regions is None:
            resolved_slices = self._modality_slices(batch, valid=valid)
            emit_modality_metrics = resolved_slices is not None
            slices = resolved_slices or (("packed", slice(None)), )
            all_x0 = all(self._fake_score_space_for(name) == "x0" for name, _ in slices)
        else:
            emit_modality_metrics = True
            all_x0 = all(self._fake_score_space_for(name) == "x0" for _, name, _ in document_regions)

        pred_x0: torch.Tensor | None = None
        pred_noise: torch.Tensor | None = None
        target: torch.Tensor | None = None
        if all_x0:
            # Match FastGen's fake_score_pred_type=x0 objective directly.
            # Reweighting raw velocity MSE by sigma^2 is algebraically equal
            # before rounding, but estimating sigma from BF16 x_t introduces a
            # low-noise bias (especially for H3 audio).
            pred_x0 = self.critic.predict_x0(
                noisy_x0,
                fake_score_timestep,
                batch,
                conditional=critic_conditional,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            if not isinstance(pred_x0, torch.Tensor):
                raise TypeError("DMD2 critic predict_x0 must return one packed tensor")
        else:
            # Retain the single-forward legacy path for mixed per-modality
            # velocity/x0 configurations. Strict H3 parity uses global x0 and
            # therefore always takes the direct branch above.
            pred_noise = self.critic.predict_noise(
                noisy_x0,
                fake_score_timestep,
                batch,
                conditional=critic_conditional,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            if not isinstance(pred_noise, torch.Tensor):
                raise TypeError("DMD2 critic predict_noise must return one packed tensor")
            target = noise - generator_pred_x0

        def region(tensor: torch.Tensor, modality: slice) -> torch.Tensor:
            return latent_loss_region(tensor, modality, valid).float()

        flow_matching_loss = torch.zeros((), device=device, dtype=torch.float32)
        metrics: dict[str, LogScalar] = {}
        if self._fake_score_cfg_dropout_rate > 0.0:
            # Evidence of the critic's unconditional training forward (step average = drop fraction).
            metrics["critic_cfg_dropout"] = 0.0 if critic_conditional else 1.0
        if document_regions is None:
            # Keep the single-document reduction and operation order exact.
            for name, modality in slices:
                if all_x0:
                    assert pred_x0 is not None
                    loss_m = torch.mean((region(pred_x0, modality) - region(generator_pred_x0, modality))**2)
                else:
                    assert pred_noise is not None and target is not None
                    loss_m = torch.mean((region(pred_noise, modality) - region(target, modality))**2)
                if not all_x0 and self._fake_score_space_for(name) == "x0":
                    # For affine rectified flow, x0 MSE is sigma_m(t)^2 times
                    # velocity MSE. This compatibility path is retained only for
                    # legacy mixed-space recipes.
                    assert target is not None
                    with torch.no_grad():
                        num = torch.mean((region(noisy_x0, modality) - region(generator_pred_x0, modality))**2)
                        den = torch.mean(region(target, modality)**2)
                        sigma_sq = num / den
                    loss_m = sigma_sq * loss_m
                flow_matching_loss = flow_matching_loss + self._modality_weight(name) * loss_m
                if emit_modality_metrics:
                    metrics[f"fake_score_loss_{name}"] = loss_m.detach()
        else:
            num_documents = int(batch.sequence_packing_num_documents)
            document_losses = [torch.zeros((), device=device, dtype=torch.float32) for _ in range(num_documents)]
            modality_losses: dict[str, list[torch.Tensor]] = {}
            for document_id, name, modality in document_regions:
                if all_x0:
                    assert pred_x0 is not None
                    loss_m = torch.mean((region(pred_x0, modality) - region(generator_pred_x0, modality))**2)
                else:
                    assert pred_noise is not None and target is not None
                    loss_m = torch.mean((region(pred_noise, modality) - region(target, modality))**2)
                if not all_x0 and self._fake_score_space_for(name) == "x0":
                    assert target is not None
                    with torch.no_grad():
                        num = torch.mean((region(noisy_x0, modality) - region(generator_pred_x0, modality))**2)
                        den = torch.mean(region(target, modality)**2)
                        sigma_sq = num / den
                    loss_m = sigma_sq * loss_m
                document_losses[document_id] = document_losses[document_id] + self._modality_weight(name) * loss_m
                modality_losses.setdefault(name, []).append(loss_m.detach())
            flow_matching_loss = torch.stack(document_losses).mean()
            metrics.update({
                f"fake_score_loss_{name}": torch.stack(values).mean()
                for name, values in modality_losses.items()
            })

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {"fake_score_latent_vis_dict": (batch.fake_score_latent_vis_dict)}
        return (
            flow_matching_loss,
            (batch.timesteps, batch.attn_metadata),
            outputs,
            metrics,
        )

    def _dmd_loss(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, LogScalar]]:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = 1.0
        device = generator_pred_x0.device
        valid = latent_loss_valid_mask(batch)
        document_regions = self._packed_document_regions(batch, valid=valid)

        with torch.no_grad():
            timestep = self._score_timestep_for_batch(device, batch)

            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
                generator=self.cuda_generator,
            )
            noisy_latents = self._add_noise_for_batch(
                self.student,
                generator_pred_x0,
                noise,
                timestep,
                batch,
            )

            faker_x0 = self.critic.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            if self._fake_score_guidance_scale != 1.0:
                # Mirror the real score: guide the critic with the same unconditional construction.
                self._dmd_fake_uncond_forwards += 1
                faker_uncond_x0 = self.critic.predict_x0(
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=False,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="dense",
                )
                faker_x0 = faker_uncond_x0 + (faker_x0 - faker_uncond_x0) * self._fake_score_guidance_scale
            real_cond_x0 = self.teacher.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            if float(guidance_scale) == 1.0:
                # Scale 1 is the conditional prediction and needs no
                # unconditional forward.
                real_cfg_x0 = real_cond_x0
            else:
                self._dmd_real_uncond_forwards += 1
                real_uncond_x0 = self.teacher.predict_x0(
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=False,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="dense",
                )
                real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale

            # LatentVisCallback decodes these estimates on rank 0.
            batch.dmd_latent_vis_dict.update({
                "real_score_pred_video": real_cfg_x0.detach(),
                "faker_score_pred_video": faker_x0.detach(),
                "dmd_timestep": timestep.detach(),
            })

        slices: tuple[tuple[str, slice], ...] = ()
        if document_regions is None:
            resolved_slices = self._modality_slices(batch, valid=valid)
            emit_modality_metrics = resolved_slices is not None
            slices = resolved_slices or (("packed", slice(None)), )
        else:
            emit_modality_metrics = True
        loss = torch.zeros((), device=device, dtype=torch.float32)
        metrics: dict[str, LogScalar] = {}
        if document_regions is None:
            # Keep the single-document reduction and operation order exact.
            for name, modality in slices:
                gen_m = latent_loss_region(generator_pred_x0, modality, valid).float()
                with torch.no_grad():
                    # Keep the VSD weight stable for low-precision or degenerate
                    # teacher residuals. Invalid scalars never enter this mean.
                    real_m = latent_loss_region(real_cfg_x0, modality, valid).float()
                    fake_m = latent_loss_region(faker_x0, modality, valid).float()
                    denom = (gen_m - real_m).abs().mean() + 1e-6
                    grad = torch.nan_to_num((fake_m - real_m) / denom)
                loss_m = 0.5 * F.mse_loss(gen_m, (gen_m - grad).detach())
                loss = loss + self._modality_weight(name) * loss_m
                if emit_modality_metrics:
                    metrics[f"generator_loss_{name}"] = loss_m.detach()
        else:
            num_documents = int(batch.sequence_packing_num_documents)
            document_losses = [torch.zeros((), device=device, dtype=torch.float32) for _ in range(num_documents)]
            modality_losses: dict[str, list[torch.Tensor]] = {}
            for document_id, name, modality in document_regions:
                gen_m = latent_loss_region(generator_pred_x0, modality, valid).float()
                with torch.no_grad():
                    real_m = latent_loss_region(real_cfg_x0, modality, valid).float()
                    fake_m = latent_loss_region(faker_x0, modality, valid).float()
                    denom = (gen_m - real_m).abs().mean() + 1e-6
                    grad = torch.nan_to_num((fake_m - real_m) / denom)
                loss_m = 0.5 * F.mse_loss(gen_m, (gen_m - grad).detach())
                document_losses[document_id] = document_losses[document_id] + self._modality_weight(name) * loss_m
                modality_losses.setdefault(name, []).append(loss_m.detach())
            loss = torch.stack(document_losses).mean()
            metrics.update({
                f"generator_loss_{name}": torch.stack(values).mean()
                for name, values in modality_losses.items()
            })
        return loss, metrics

    def _generator_update_interval(self) -> int:
        interval = get_optional_int(
            self.method_config,
            "generator_update_interval",
            where="method.generator_update_interval",
        )
        if interval is None:
            interval = 5
        # Roles alternate exclusively: the student updates on every Nth
        # iteration and the critic on all others. DMD2's interval 1 (both roles
        # every step) would therefore never train the critic here.
        if interval < 2:
            raise ValueError("method.generator_update_interval must be at least 2: DMPDD updates only the critic "
                             f"between student updates, got {interval}")
        return interval

    def _should_update_student(
        self,
        iteration: int,
    ) -> bool:
        return iteration % self._generator_update_interval() == 0
