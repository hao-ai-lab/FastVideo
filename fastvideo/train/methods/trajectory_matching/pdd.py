# SPDX-License-Identifier: Apache-2.0
"""Parallel Decoding Distillation (PDD) training method.

Port of FastGen's ``PDDModelMixin`` / ``PDDModel``
(``fastgen/methods/trajectory_matching/pdd.py``). A PDD-widened student
predicts every fine-grid head in one forward; one sampled head inside the
``block_size_max`` window is supervised by the teacher's mean sigma velocity
over that head's interval, and the same prediction advances the student's
own trajectory when the method carries one.

Two student-input regimes exist, selected with ``method.rollout_mode``:

- ``data_latent`` noises real packed latents to a sampled block start aligned
  to ``block_size_min`` (FastGen ``backward_simulation=False``);
- ``simulate`` carries one on-policy trajectory per gradient-accumulation
  slot along the highest-NFE partition of the fine grid, exactly one student
  forward per call (FastGen backward simulation with ``CarryCallback``,
  using this repository's shared trajectory-carry cadence).

The shared PDD machinery lives in :class:`PDDMethodMixin` for reuse by ``DMPDDMethod``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from fastvideo.dataset.sequence_capacity import sequence_capacity_metrics
from fastvideo.layers.pdd import pdd_student_step_indices, validate_finite_nonzero
from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.trajectory_carry import (
    TrajectoryCarry,
    parse_carry_slots,
    snapshot_raw_batch,
    stagger_offset,
    stagger_rank_and_groups,
    validate_carry_coverage,
)
from fastvideo.train.methods.trajectory_matching.pdd_core import (
    RK_STAGES,
    PDDHeadLoss,
    PDDModalityBlock,
    advance_block,
    advance_fused,
    aligned_block_starts,
    carried_step_count,
    head_losses,
    latent_loss_valid_mask,
    require_pdd_student,
    sample_supervision_index,
    teacher_block_target,
    x0_from_local_head,
    x0_from_points,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    parse_cfg_uncond,
    parse_rollout_mode,
    parse_torch_dtype,
    require_choice,
    require_non_negative_float,
    require_positive_int,
    uses_negative_prompt_conditioning,
    validate_text_only_rollout,
)
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

logger = init_logger(__name__)

# DMD2 knobs that describe its integer denoising ladder and renoise carry;
# the PDD hybrids sample on the student's fine grid instead.
_DMD2_LADDER_KNOBS = (
    "rollout_sample_type",
    "rollout_data_forcing",
    "allow_mixed_rollout_regimes",
    "dmd_denoising_steps",
    "warp_denoising_step",
)


@dataclass(frozen=True)
class PDDConfig:
    """PDD knobs shared by the standalone method and the DMD2 hybrids.

    Controls the teacher target, local head window, and carried fine grid.
    """

    # Training-start stride (also the carried grid resolution) and the widest
    # supervised / fused sampling block.
    block_size_min: int
    block_size_max: int
    # Teacher target integrator on each fine interval.
    type_rk: str
    # Precision of the trajectory integration; model forwards keep bf16.
    precision_decoding: torch.dtype
    # Classifier-free guidance scale for the teacher target (1.0: one forward).
    teacher_guidance_scale: float
    # Explicit fine-grid node maps ``{num_steps: [indices]}``; balanced otherwise.
    student_grid_indices: dict[int, list[int]] | None
    # Weight of the auxiliary PDD term in distribution matching.
    pdd_loss_weight: float


def parse_pdd_config(method_config: Mapping[str, Any], *, where: str = "method") -> PDDConfig:
    config = dict(method_config)
    block_size_min = require_positive_int(config, "block_size_min", default=32, where=f"{where}.block_size_min")
    block_size_max = require_positive_int(config, "block_size_max", default=128, where=f"{where}.block_size_max")
    if block_size_min > block_size_max:
        raise ValueError(f"{where}.block_size_min={block_size_min} must not exceed "
                         f"{where}.block_size_max={block_size_max}.")
    type_rk = require_choice(config, "type_rk", set(RK_STAGES), default="euler", where=f"{where}.type_rk")
    precision_decoding = parse_torch_dtype(
        config,
        "precision_decoding",
        default="fp32",
        where=f"{where}.precision_decoding",
    )

    guidance = get_optional_float(config, "teacher_guidance_scale", where=f"{where}.teacher_guidance_scale")
    guidance = 1.0 if guidance is None else float(guidance)
    if guidance <= 0.0:
        raise ValueError(f"{where}.teacher_guidance_scale must be positive, got {guidance}")

    indices_raw = config.get("student_grid_indices")
    grid_indices: dict[int, list[int]] | None = None
    if indices_raw is not None:
        if not isinstance(indices_raw, Mapping):
            raise ValueError(f"{where}.student_grid_indices must map a step count to fine-grid indices")
        grid_indices = {int(key): [int(index) for index in value] for key, value in indices_raw.items()}

    return PDDConfig(
        block_size_min=block_size_min,
        block_size_max=block_size_max,
        type_rk=type_rk,
        precision_decoding=precision_decoding,
        teacher_guidance_scale=guidance,
        student_grid_indices=grid_indices,
        pdd_loss_weight=float(
            require_non_negative_float(config, "pdd_loss_weight", default=1.0, where=f"{where}.pdd_loss_weight")),
    )


@dataclass
class PDDWalk:
    """One carried stream's position for the current ``single_train_step`` call."""

    slot: int
    rung: int
    num_steps: int
    state: torch.Tensor
    fine_grid: torch.Tensor
    indices: torch.Tensor
    training_batch: Any
    raw_batch: dict[str, Any]

    @property
    def step_idx(self) -> torch.Tensor:
        return self.indices[self.rung].expand(self.state.shape[0])

    @property
    def next_idx(self) -> torch.Tensor:
        return self.indices[self.rung + 1].expand(self.state.shape[0])

    @property
    def is_last(self) -> bool:
        return self.rung + 1 >= self.num_steps

    def metrics(self) -> dict[str, LogScalar]:
        return {"rollout_step": float(self.rung), "pdd_block_start": float(self.indices[self.rung])}


class PDDMethodMixin:
    """PDD training primitives shared by the standalone and hybrid methods.

    Host classes provide ``student``, ``teacher``, ``cuda_generator``,
    ``training_config``, ``method_config`` and ``_cfg_uncond`` (the
    ``TrainingMethod`` / ``DMD2Method`` attributes) and call :meth:`_init_pdd`
    once their role models are bound.
    """

    student: Any
    teacher: Any
    cuda_generator: torch.Generator | None
    training_config: Any
    method_config: dict[str, Any]
    validation_config: dict[str, Any]
    _cfg_uncond: dict[str, Any] | None
    pdd_config: PDDConfig
    pdd_steps: int
    _pdd_carry: TrajectoryCarry | None

    # Distillation learning rates move the widened heads and norm gains by
    # less than half a bf16 ulp; keep sharded FP32 masters.
    requires_fp32_master_weights = True

    def _init_pdd(self, *, carried: bool, update_interval: int = 1) -> None:
        """Bind the PDD contract; with ``carried`` also build the trajectory carry.

        ``update_interval`` is how often (in calls) a carried stream receives
        a student update; the standalone method updates on every call.
        """
        require_pdd_student(self.student)
        self.pdd_config = parse_pdd_config(self.method_config)
        self.pdd_steps = int(self.student.pdd_steps)
        if self.pdd_config.block_size_max > self.pdd_steps:
            raise ValueError(f"method.block_size_max={self.pdd_config.block_size_max} exceeds the student's "
                             f"pdd_steps={self.pdd_steps}")
        if self.pdd_config.teacher_guidance_scale != 1.0 and self._cfg_uncond is None:
            raise ValueError("method.teacher_guidance_scale != 1 requires method.cfg_uncond for the "
                             "teacher's unconditional forward")
        # Model forwards happen at the model boundary dtype (bf16 for H3);
        # trajectory arithmetic happens in ``precision_decoding``.
        self._pdd_model_dtype: torch.dtype = self.student.pdd_model_dtype
        self._pdd_weights_validated = False
        self._validate_pdd_validation_steps()
        self._pdd_carry = None
        if carried:
            self._pdd_carry = TrajectoryCarry(parse_carry_slots(self.method_config, self.training_config.loop))
            self._validate_pdd_carried_grid(update_interval)

    def _validate_pdd_validation_steps(self) -> None:
        """Validation NFEs must partition the grid into blocks no wider than the trained window.

        FastGen's sampler rejects a runtime block wider than ``block_size_max``
        at sampling time; the pipeline sampler has no such guard, so check
        the configured ``callbacks.validation.sampling_steps`` up front. The
        pipeline samples the balanced partition, so that is what is checked;
        ``student_grid_indices`` only shape the training walks.
        """
        if not self.validation_config:
            return
        # ValidationCallback samples 40 steps when the recipe names none.
        steps = self.validation_config.get("sampling_steps") or [40]
        configured = self.pdd_config.student_grid_indices or {}
        for num_steps in steps:
            if int(num_steps) in configured:
                logger.warning(
                    "student_grid_indices[%s] shapes training only; validation and exports sample the balanced "
                    "%s-block partition", int(num_steps), int(num_steps))
            indices = pdd_student_step_indices(self.pdd_steps, int(num_steps), device=torch.device("cpu"))
            widest = int((indices[1:] - indices[:-1]).max())
            if widest > self.pdd_config.block_size_max:
                raise ValueError(f"callbacks.validation.sampling_steps={list(steps)}: {int(num_steps)} fused blocks of "
                                 f"the {self.pdd_steps}-interval grid need a {widest}-wide block, wider than the "
                                 f"trained method.block_size_max={self.pdd_config.block_size_max}")

    def _validate_pdd_carried_grid(self, update_interval: int) -> None:
        """The carried highest-NFE grid must fit the trained block sizes and cover every rung."""
        assert self._pdd_carry is not None
        indices = self._pdd_carried_indices(torch.device("cpu"))
        max_block_size = int((indices[1:] - indices[:-1]).max())
        if max_block_size > self.pdd_config.block_size_max:
            raise ValueError(f"Carried PDD block {max_block_size} exceeds the trained "
                             f"block_size_max={self.pdd_config.block_size_max}.")
        _, stagger_groups = stagger_rank_and_groups(self.training_config)
        validate_carry_coverage(
            streams=stagger_groups * self._pdd_carry.num_slots,
            num_steps=int(indices.numel()) - 1,
            interval=update_interval,
            owner=f"Carried {type(self).__name__}",
        )

    def _reject_dmd2_ladder_knobs(self) -> tuple[bool, int, Literal["ode", "sde"]]:
        """Reject integer-ladder settings for methods using the PDD fine grid.

        The hybrids sample on the student's PDD fine grid, so DMD2's integer
        denoising ladder and renoise carry must stay unconfigured.
        """
        owner = type(self).__name__
        if self.method_config.get("rollout_carry"):
            raise ValueError(f"{owner} walks the student's PDD fine grid itself; method.rollout_carry must be false")
        for key in _DMD2_LADDER_KNOBS:
            if key in self.method_config:
                raise ValueError(f"{owner} samples on the student's PDD fine grid; remove method.{key} and "
                                 "configure block_size_min/block_size_max/student_grid_indices instead")
        return False, 0, "ode"

    # -- schedule ---------------------------------------------------------------

    def _pdd_fine_grid(self, device: torch.device) -> torch.Tensor:
        return self.student.pdd_fine_grid(device)

    def _pdd_blocks(self, batch: Any) -> tuple[PDDModalityBlock, ...]:
        blocks = tuple(self.student.pdd_modality_schedules(batch))
        if not self._pdd_weights_validated:
            # The schedules are fixed per model: check their coefficients once.
            fine_grid = self._pdd_fine_grid(torch.device("cpu"))
            for block in blocks:
                validate_finite_nonzero(block.schedule.integration_weights(fine_grid),
                                        f"PDD {block.name} integration weight")
            self._pdd_weights_validated = True
        return blocks

    def _pdd_carried_indices(self, device: torch.device) -> torch.Tensor:
        """Node indices of the highest-NFE partition the carried trajectory walks."""
        return pdd_student_step_indices(
            self.pdd_steps,
            carried_step_count(self.pdd_steps, self.pdd_config.block_size_min),
            configured_indices=self.pdd_config.student_grid_indices,
            device=device,
        )

    def inference_checkpoint_contract(self, role: str = "student") -> dict[str, Any] | None:
        """The sampling contract a deployable export of *role* carries, or None.

        A student export samples the balanced partition of its highest-NFE
        grid (``pdd_steps / block_size_min`` fused blocks), as validation does
        for the same step count; ``student_grid_indices`` shape training only.
        """
        contract = getattr(self.student, "inference_contract", None)
        if role != "student" or not callable(contract):
            return None
        num_blocks = carried_step_count(self.pdd_steps, self.pdd_config.block_size_min)
        indices = pdd_student_step_indices(self.pdd_steps, num_blocks, device=torch.device("cpu"))
        return contract(pdd_step_indices=[int(index) for index in indices.tolist()])

    # -- teacher ----------------------------------------------------------------

    def _pdd_teacher_predictor(self, batch: Any) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the guided teacher's native ``eps - x0`` in decoding precision.

        The teacher forward happens at the model boundary dtype and every
        forward (conditional and, under guidance, unconditional) runs on every
        rank so FSDP collectives never diverge.
        """
        cfg = self.pdd_config

        def predict(state: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            timestep = self.student.pdd_timestep(base)
            model_input = state.to(self._pdd_model_dtype)
            conditional = self.teacher.predict_noise(
                model_input,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            ).to(cfg.precision_decoding)
            if cfg.teacher_guidance_scale == 1.0:
                return conditional
            # Counted so a guided PDD target can prove its unconditional teacher forwards ran.
            self._pdd_teacher_uncond_forwards = getattr(self, "_pdd_teacher_uncond_forwards", 0) + 1
            unconditional = self.teacher.predict_noise(
                model_input,
                timestep,
                batch,
                conditional=False,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            ).to(cfg.precision_decoding)
            return unconditional + (conditional - unconditional) * cfg.teacher_guidance_scale

        return predict

    # -- student ----------------------------------------------------------------

    def _pdd_heads(
        self,
        state: torch.Tensor,
        step_idx: torch.Tensor,
        fine_grid: torch.Tensor,
        batch: Any,
        *,
        head_window: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """One student forward materializing the PDD heads at ``step_idx``.

        ``head_window=(start, end)`` keeps only those heads, so a rollout
        segment retains just the block it integrates.
        """
        return self.student.predict_pdd_heads(
            state.to(self._pdd_model_dtype),
            self.student.pdd_timestep(fine_grid[step_idx]),
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="vsa",
            head_window=head_window,
        )

    def _compute_pdd_loss_from_prediction(
        self,
        x_n: torch.Tensor,
        step_idx: torch.Tensor,
        pred: torch.Tensor,
        fine_grid: torch.Tensor,
        batch: Any,
        *,
        block_end_idx: torch.Tensor | None = None,
        head_offset: int = 0,
    ) -> PDDHeadLoss:
        """Supervise one head from an already-materialized PDD prediction.

        Reusing the prediction matters for in-iteration rollouts: one student
        forward both advances its segment and contributes its PDD objective.
        ``block_end_idx`` is exclusive and confines the sampled head to that
        segment; ``head_offset`` is the grid index of ``pred``'s first head.
        """
        cfg = self.pdd_config
        k = sample_supervision_index(
            step_idx,
            pdd_steps=self.pdd_steps,
            block_size_max=cfg.block_size_max,
            block_end_idx=block_end_idx,
            generator=self.cuda_generator,
        )
        batch_idx = torch.arange(x_n.shape[0], device=x_n.device)
        p_k = pred[batch_idx, k - head_offset]
        blocks = self._pdd_blocks(batch)
        with torch.no_grad():
            x_k = advance_block(
                x_n,
                pred,
                step_idx,
                k,
                fine_grid,
                blocks,
                allow_empty=True,
                precision_decoding=cfg.precision_decoding,
                head_offset=head_offset,
            )
            p_teacher = teacher_block_target(
                x_k,
                k,
                fine_grid,
                blocks,
                solver=cfg.type_rk,
                predict_teacher=self._pdd_teacher_predictor(batch),
            )
        losses = head_losses(p_k, p_teacher, blocks, valid=latent_loss_valid_mask(batch))
        return PDDHeadLoss(total=sum(losses.values()), per_modality=losses, x_k=x_k, k=k, p_k=p_k)

    # -- sampling ---------------------------------------------------------------

    def _pdd_advance_student_block(
        self,
        state: torch.Tensor,
        start: int,
        end: int,
        fine_grid: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        """Advance one explicit fused PDD block (the distilled sampler's step)."""
        cfg = self.pdd_config
        timestep = self.student.pdd_timestep(fine_grid[start].expand(state.shape[0]))
        mean_velocity = self.student.predict_pdd_fused(
            state.to(self._pdd_model_dtype),
            timestep,
            start,
            end,
            fine_grid,
            batch,
            precision_decoding=cfg.precision_decoding,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="vsa",
        )
        return advance_fused(
            state,
            mean_velocity,
            start,
            end,
            fine_grid,
            self._pdd_blocks(batch),
            precision_decoding=cfg.precision_decoding,
        )

    def _pdd_fresh_state(self, batch: Any, fine_grid: torch.Tensor) -> torch.Tensor:
        """Fresh noise at the grid's first node: ``sigma_m(u_0) * eps`` per modality."""
        latents = batch.latents
        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=self.cuda_generator,
        )
        timestep = self.student.pdd_timestep(fine_grid[:1].expand(latents.shape[0]))
        return self.student.add_noise_for_batch(torch.zeros_like(latents), noise, timestep, batch)

    def _pdd_staggered_start(
        self,
        state: torch.Tensor,
        batch: Any,
        fine_grid: torch.Tensor,
        indices: torch.Tensor,
        *,
        slot: int,
        num_slots: int,
    ) -> tuple[torch.Tensor, int]:
        """Pre-walk a fresh trajectory and keep this stream's rung.

        Every rank walks the whole carried grid under ``no_grad`` so the
        fused forwards issue a uniform collective count; only the retained
        snapshot differs per (rank, slot) stream. When every rank shares
        offset 0 (a single stagger group) there is nothing to walk.
        """
        stagger_rank, stagger_groups = stagger_rank_and_groups(self.training_config)
        num_steps = int(indices.numel()) - 1
        offset = stagger_offset(stagger_rank=stagger_rank, num_slots=num_slots, slot=slot, num_steps=num_steps)
        if stagger_groups == 1 and offset == 0:
            return state, 0
        snapshot = state
        transformer = self.student.transformer
        was_training = transformer.training
        try:
            transformer.eval()
            with torch.no_grad():
                for rung in range(num_steps - 1):
                    state = self._pdd_advance_student_block(
                        state,
                        int(indices[rung]),
                        int(indices[rung + 1]),
                        fine_grid,
                        batch,
                    )
                    if rung + 1 == offset:
                        snapshot = state
        finally:
            transformer.train(was_training)
        return snapshot, offset

    # -- student inputs -----------------------------------------------------------

    def _pdd_data_latent_inputs(self, batch: dict[str, Any]) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Real latents noised to an aligned block start (FastGen data-dependent PDD).

        Returns ``(training_batch, x_n, step_idx, fine_grid)``.
        """
        training_batch = self.student.prepare_batch(batch, generator=self.cuda_generator, latents_source="data")
        real = training_batch.latents
        device = real.device
        fine_grid = self._pdd_fine_grid(device)
        step_idx = aligned_block_starts(
            real.shape[0],
            pdd_steps=self.pdd_steps,
            block_size_min=self.pdd_config.block_size_min,
            device=device,
            generator=self.cuda_generator,
        )
        noise = torch.randn(real.shape, device=device, dtype=real.dtype, generator=self.cuda_generator)
        x_n = self.student.add_noise_for_batch(real, noise, self.student.pdd_timestep(fine_grid[step_idx]),
                                               training_batch)
        return training_batch, x_n, step_idx, fine_grid

    def _pdd_resume_walk(self, batch: dict[str, Any]) -> PDDWalk:
        """Select this call's carried stream and materialize its state.

        An empty slot starts a fresh trajectory from noise (staggered on its
        first-ever fill) and adopts the incoming batch's conditioning; a
        carried slot ignores the fresh batch and rebuilds the training batch
        from its snapshot, since a trajectory keeps the prompt it set out with.
        """
        assert self._pdd_carry is not None
        carry = self._pdd_carry
        slot = carry.next_slot()
        carried = carry.slots[slot]
        raw_batch = snapshot_raw_batch(batch, self.student.device) if carried is None else carried["raw_batch"]
        training_batch = self.student.prepare_batch(raw_batch, generator=self.cuda_generator, latents_source="zeros")
        device = training_batch.latents.device
        fine_grid = self._pdd_fine_grid(device)
        indices = self._pdd_carried_indices(device)

        if carried is None:
            rung = 0
            state = self._pdd_fresh_state(training_batch, fine_grid)
            if not carry.seeded[slot]:
                carry.seeded[slot] = True
                state, rung = self._pdd_staggered_start(
                    state,
                    training_batch,
                    fine_grid,
                    indices,
                    slot=slot,
                    num_slots=carry.num_slots,
                )
        else:
            rung = int(carried["rung"])
            state = carried["state"]
        return PDDWalk(
            slot=slot,
            rung=rung,
            num_steps=int(indices.numel()) - 1,
            state=state,
            fine_grid=fine_grid,
            indices=indices,
            training_batch=training_batch,
            raw_batch=raw_batch,
        )

    def _pdd_commit_walk(self, walk: PDDWalk, next_state: torch.Tensor | None) -> None:
        """Carry the stream to its next rung, or release it after the last one."""
        assert self._pdd_carry is not None
        if walk.is_last:
            self._pdd_carry.clear(walk.slot)
            return
        if next_state is None:
            raise ValueError("A carried PDD stream needs its next state before the last rung")
        self._pdd_carry.set(walk.slot, state=next_state, rung=walk.rung + 1, raw_batch=walk.raw_batch)


class PDDMethod(PDDMethodMixin, TrainingMethod):
    """Standalone Parallel Decoding Distillation (student + frozen teacher)."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "student" not in role_models:
            raise ValueError("PDDMethod requires role 'student'")
        if "teacher" not in role_models:
            raise ValueError("PDDMethod requires role 'teacher'")
        self.teacher = role_models["teacher"]
        if not self.student._trainable:
            raise ValueError("PDDMethod requires student to be trainable")
        if self.teacher._trainable:
            raise ValueError("PDDMethod requires teacher to be non-trainable")

        self._cfg_uncond = parse_cfg_uncond(self.method_config)
        self._rollout_mode = parse_rollout_mode(self.method_config)
        validate_text_only_rollout(
            getattr(self.training_config.data, "preprocessed_data_type", "t2v"),
            self._rollout_mode,
        )
        setter = getattr(self.student, "set_requires_negative_conditioning", None)
        if setter is not None:
            setter(uses_negative_prompt_conditioning(self._cfg_uncond))
        self._init_pdd(carried=self._rollout_mode == "simulate")

        self.student.init_preprocessors(self.training_config)
        self._init_optimizer_and_scheduler()
        self.latent_vis: dict[str, Any] = {}

    # -- optimizer ------------------------------------------------------------------

    def _init_optimizer_and_scheduler(self) -> None:
        tc = self.training_config
        params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=float(tc.optimizer.learning_rate),
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        return [self._student_lr_scheduler]

    def apply_configured_lrs(self) -> None:
        lr = float(self.training_config.optimizer.learning_rate)
        for group in self._student_optimizer.param_groups:
            group["lr"] = lr
            if "initial_lr" in group:
                group["initial_lr"] = lr
        if hasattr(self._student_lr_scheduler, "base_lrs"):
            self._student_lr_scheduler.base_lrs = [lr] * len(self._student_lr_scheduler.base_lrs)

    # -- training ---------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        if self._rollout_mode == "data_latent":
            training_batch, x_n, step_idx, fine_grid = self._pdd_data_latent_inputs(batch)
            loss_map, outputs, _ = self._pdd_train_step(x_n, step_idx, fine_grid, training_batch, next_idx=None)
            return loss_map, outputs, {
                "pdd_block_start": float(step_idx[0]),
                **sequence_capacity_metrics(training_batch),
            }

        # One backward-simulation call: one student forward at the carried
        # rung; the materialized heads supply the PDD loss and their coarse
        # block is integrated under ``no_grad`` to carry the trajectory on.
        walk = self._pdd_resume_walk(batch)
        loss_map, outputs, x_next = self._pdd_train_step(
            walk.state,
            walk.step_idx,
            walk.fine_grid,
            walk.training_batch,
            next_idx=walk.next_idx,
        )
        self._pdd_commit_walk(walk, x_next)
        return loss_map, outputs, {**walk.metrics(), **sequence_capacity_metrics(walk.training_batch)}

    def _pdd_train_step(
        self,
        x_n: torch.Tensor,
        step_idx: torch.Tensor,
        fine_grid: torch.Tensor,
        batch: Any,
        *,
        next_idx: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor | None]:
        """Materialize the heads once: loss map, backward outputs, and the optional carry state.

        The backward context (the student forward's timesteps and VSA
        metadata) is captured right after the student forward, before the
        teacher forwards rewrite the batch's timestep fields.
        """
        pred = self._pdd_heads(x_n, step_idx, fine_grid, batch)
        student_ctx = (batch.timesteps, batch.attn_metadata_vsa)
        head = self._compute_pdd_loss_from_prediction(x_n, step_idx, pred, fine_grid, batch)
        blocks = self._pdd_blocks(batch)
        x_next: torch.Tensor | None = None
        if next_idx is None:
            x0_preview = x0_from_local_head(head.x_k, head.k, fine_grid, blocks, head.p_k.detach())
        else:
            with torch.no_grad():
                x_next = advance_block(
                    x_n,
                    pred,
                    step_idx,
                    next_idx,
                    fine_grid,
                    blocks,
                    precision_decoding=self.pdd_config.precision_decoding,
                )
            x0_preview = x0_from_points(x_n.detach(), x_next, step_idx, next_idx, fine_grid, blocks)

        loss_map = {
            "total_loss": head.total,
            "pdd_loss": head.total,
            **{
                f"pdd_loss_{name}": value
                for name, value in head.per_modality.items()
            },
        }
        outputs: dict[str, Any] = {"_fv_backward": {"student_ctx": student_ctx}}
        self.latent_vis = {
            "generator_pred_video": x0_preview.detach(),
            "_fv_latent_layout": getattr(batch, "minimax_h3_dmd_layout", None),
        }
        return loss_map, outputs, x_next

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        self.student.backward(
            loss_map["total_loss"],
            outputs["_fv_backward"]["student_ctx"],
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )


__all__ = ["PDDConfig", "PDDMethod", "PDDMethodMixin", "PDDWalk", "parse_pdd_config"]
