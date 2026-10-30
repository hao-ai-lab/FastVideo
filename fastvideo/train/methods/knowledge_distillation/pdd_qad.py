# SPDX-License-Identifier: Apache-2.0
"""Per-forward quantization-aware distillation (QAD) of a FastH3 OmniRef PDD student.

Student = the teacher's own weights + NVFP4 fake-quantized block linears
(Stage A calibrated static FFN scales, unit attention/gate scales) + the
deployed sparse-FP4 attention numerics. Loss = per-forward regression of the
denoiser output (x0) to the frozen bf16 teacher, at identical inputs on the
eight-forward PDD ladder; no DMD, no adversarial term.

Each optimizer micro-step takes one OmniRef row: the teacher walks its PDD
trajectory (teacher forcing), or with probability ``rollout_fraction`` the
student walks its own (DAgger-style exposure-bias correction); at every
forward both models see the same state and the student's x0 error is
backpropagated at once, inside that forward's scope.

Guardrails: step-0 checks (quantization disabled -> bit-identical to the
teacher; quantized error vs the Stage A PTQ number), per-(rung, modality)
loss normalization by the step-0 PTQ error, held-out evaluation every
``eval_every`` steps (teacher-forced x0 rel-L2 per forward, student-rollout
endpoint, latent keyframe fidelity), best-checkpoint requests, and a stop
request after ``patience`` consecutive evaluations worse than step 0 (teacher
alignment counts full evaluations only). The step-0 baseline and the guard
state are checkpointed (``method.qad_baseline``), so a resumed run keeps them.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import BlockReconstruction
from fastvideo.train.methods.knowledge_distillation.pdd_qad_step0 import (load_step0, restore_step0, save_step0,
                                                                          step0_fingerprint_inputs)
from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import (MODALITIES, RungStats, keyframe_latents,
                                                                            rel_l2, step_packed, summarize,
                                                                            target_slices, x0_pair)
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

logger = init_logger(__name__)

# Teacher alignment: the student's own sample (same seed) and its keyframes vs the teacher's; full evaluations only.
ALIGNMENT_KEYS = ("endpoint/video", "endpoint/audio", "keyframe/vs_teacher/first", "keyframe/vs_teacher/last")
BASELINE_FORMAT = 1


class _BaselineState:
    """DCP-stateful step-0 baseline and guard state of an ``OmniRefPDDQADMethod`` (one JSON string)."""

    def __init__(self, method: OmniRefPDDQADMethod) -> None:
        self.method = method

    def state_dict(self) -> dict[str, Any]:
        return {"json": json.dumps(self.method._baseline_state(), sort_keys=True)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.method._load_baseline_state(json.loads(state_dict["json"]))


def _unit_hash(*parts: Any) -> float:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2.0**64


class OmniRefPDDQADMethod(TrainingMethod):
    """Teacher-forced (+ optional student-rollout) per-forward x0 regression; see module docstring."""

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "teacher" not in role_models:
            raise ValueError("OmniRefPDDQADMethod needs a frozen 'teacher' role")
        self.teacher = role_models["teacher"]
        if getattr(self.teacher, "_trainable", False) or self.teacher.qad_linears:
            raise ValueError("the teacher must be frozen and unquantized")
        mc = self.method_config
        self.weights = {"video": float(mc.get("video_weight", 1.0)), "audio": float(mc.get("audio_weight", 1.0))}
        self.rollout_fraction = float(mc.get("rollout_fraction", 0.0))
        self.rollout_start_step = int(mc.get("rollout_start_step", 0))
        # Student forward+backward on this many of the 8 forwards per row (the teacher walks all of them).
        self.rungs_per_row = int(mc.get("rungs_per_row", 0))
        # "output": per-forward x0 regression. "recon": layer-wise reconstruction only.
        # "output+recon": x0 regression + lambda * reconstruction.
        self.objective = str(mc.get("objective", "output"))
        if self.objective not in ("output", "recon", "output+recon"):
            raise ValueError(f"objective must be output, recon or output+recon, got {self.objective!r}")
        recon_lambda = float(mc.get("recon_lambda", 1.0))
        self.output_weight = {"output": 1.0, "recon": 0.0, "output+recon": 1.0 / (1.0 + recon_lambda)}[self.objective]
        self.recon_weight = {
            "output": 0.0,
            "recon": 1.0,
            "output+recon": recon_lambda / (1.0 + recon_lambda)
        }[self.objective]
        if self.objective == "recon" and float(mc.get("rollout_fraction", 0.0)) > 0:
            raise ValueError("reconstruction trains on teacher states only; set rollout_fraction: 0")
        # Best checkpoint by this eval key (lower is better): "score" (teacher-forced x0 rel-L2) or
        # "endpoint/video" (the student's own sample vs the teacher's: teacher alignment).
        self.select_metric = str(mc.get("select_metric", "score"))
        # Save each eval's final latents (student's own sample, teacher's) for side-by-side decoding.
        self.eval_save_dir = mc.get("eval_save_dir")
        # Full evaluation (with the student's own 8-step sample) every this many steps; teacher-forced only otherwise.
        self.full_eval_every = int(mc.get("full_eval_every", 0)) or int(mc.get("eval_every", 50))
        # Start the student from another run's tagged weights (e.g. Stage 1 best) instead of the teacher's.
        self.init_student_dcp = mc.get("init_student_dcp")
        self._eval_iteration = 0
        self.eval_every = int(mc.get("eval_every", 50))
        self.patience = int(mc.get("patience", 3))
        self.fidelity_tolerance = float(mc.get("fidelity_tolerance", 0.05))
        self.ptq_reference_json = mc.get("ptq_reference_json")
        self.ptq_tolerance = float(mc.get("ptq_tolerance", 0.05))
        self.require_bitwise_init = bool(mc.get("require_bitwise_init", True))
        self.max_eval_rows = int(mc.get("max_eval_rows", 0))
        self.save_best = bool(mc.get("save_best", True))
        # Reconstruction per row group (target video, target audio, conditioning): over the whole packed sequence
        # audio is ~1.4% of rows and text tokens dominate the norm, so an ungrouped ratio barely trains audio.
        self.recon_group_weights = dict(
            mc.get("recon_group_weights") or {
                "video": 1.0,
                "audio": 1.0,
                "condition": 0.25
            })
        # "step0": divide each forward's reconstruction loss by its step-0 value (as the x0 terms are), so
        # recon_lambda means what it says; raw reconstruction gradients are ~1e3-1e4x smaller than x0's.
        self.recon_normalize = str(mc.get("recon_normalize", "none"))
        if self.recon_normalize not in ("none", "step0"):
            raise ValueError(f"recon_normalize must be none or step0, got {self.recon_normalize!r}")
        self.recon_norm_rows = int(mc.get("recon_norm_rows", 4))
        self.recon_norm: list[float] | None = None
        # Full evaluations roll out (8 more student forwards) on at most this many rows; others are teacher-forced.
        self.max_rollout_rows = int(mc.get("max_rollout_rows", 0))
        # A checkpoint is "best" only if these full-eval metrics stay <= step0 * (1 + tolerance). With
        # noise_floor_eval the tolerance is 2 sigma of a seed+1 re-evaluation at step 0 (>= min_eligibility_tolerance);
        # otherwise fidelity_tolerance.
        self.eligibility_keys = tuple(
            mc.get("eligibility_keys")
            or ("endpoint/audio", "keyframe/vs_teacher/first", "keyframe/vs_teacher/last", "x0_rel_l2/audio/mean"))
        self.noise_floor_eval = bool(mc.get("noise_floor_eval", False))
        self.min_eligibility_tolerance = float(mc.get("min_eligibility_tolerance", 0.02))
        self.tolerances: dict[str, float] = {}
        # Stage A's per-rung reference metric for the step-0 PTQ match (v_rel is sign-independent).
        self.ptq_reference_metric = str(mc.get("ptq_reference_metric", "v_rel_l2"))
        self._recon_criteria: dict[str, Any] = {}
        # Reuse (or write) this run's step-0 evaluation, keyed by a fingerprint of everything it depends on.
        self.step0_cache = mc.get("step0_cache")
        self._method_config_snapshot = mc
        self.student.init_preprocessors(cfg.training)
        tc = self.training_config
        params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        self.optimizer, self.lr_scheduler = build_optimizer_and_scheduler(params=params,
                                                                          optimizer_config=tc.optimizer,
                                                                          loop_config=tc.loop,
                                                                          learning_rate=float(
                                                                              tc.optimizer.learning_rate),
                                                                          betas=tuple(tc.optimizer.betas),
                                                                          scheduler_name=str(tc.optimizer.lr_scheduler))
        self.normalizers: dict[str, list[float]] | None = None
        self.step0: dict[str, float] = {}
        self.best_score = math.inf
        # Consecutive evaluations with a worse teacher-forced score (every eval) and with worse teacher alignment
        # (full evals only); ``bad_evals`` is the larger.
        self.bad_score_evals = 0
        self.bad_alignment_evals = 0
        self.bad_evals = 0
        self.stop_requested = False
        self._checkpoint_request: str | None = None
        if self.init_student_dcp:
            import torch.distributed.checkpoint as dcp

            # The same DCP wrapper the checkpoint manager saves tagged/regular student weights with.
            states = {
                key: value
                for key, value in self.checkpoint_state().items() if key == "roles.student.transformer"
            }
            if not states:
                raise ValueError("init_student_dcp needs a trainable 'student' role with a transformer")
            dcp.load(states, checkpoint_id=str(Path(self.init_student_dcp) / "dcp"))
            logger.info("QAD student initialized from %s", self.init_student_dcp)
        world = get_world_group()
        self.sp_size = int(tc.distributed.sp_size or 1)
        self.dp_rank, self.dp_size = world.rank // self.sp_size, world.world_size // self.sp_size
        self.is_sp_leader = world.rank % self.sp_size == 0
        self.sampling_seed = int(tc.data.seed or 0)

    # ------------------------------------------------------------------ optimizer plumbing
    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        return [self.optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        return [self.lr_scheduler]

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self.optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self.lr_scheduler}

    def backward(self,
                 loss_map: dict[str, torch.Tensor],
                 outputs: dict[str, Any],
                 *,
                 grad_accum_rounds: int = 1) -> None:
        """No-op: each forward's loss was backpropagated inside its own rung scope."""

    def consume_checkpoint_request(self) -> str | None:
        request, self._checkpoint_request = self._checkpoint_request, None
        return request

    # ------------------------------------------------------------------ resume
    def checkpoint_state(self) -> dict[str, Any]:
        """Roles, optimizer and scheduler, plus the step-0 baseline and guard state.

        Without the baseline a resumed run would re-measure step 0 on its trained weights (wrong loss scale,
        stop history and best-checkpoint reference); a checkpoint without ``method.qad_baseline`` cannot be
        resumed (DCP reports the missing key).
        """
        return {**super().checkpoint_state(), "method.qad_baseline": _BaselineState(self)}

    def _baseline_state(self) -> dict[str, Any]:
        provisional = {}
        if getattr(self.student, "provisional_rows", 0) > 0:
            provisional = {
                name: m.input_amax
                for name, m in self.student.qad_linears.items() if m.act_scale == "static" and m.input_amax is not None
            }
        return {
            "format": BASELINE_FORMAT,
            "normalizers": self.normalizers,
            "step0": self.step0,
            "recon_norm": self.recon_norm,
            "tolerances": self.tolerances,
            "best_score": self.best_score,
            "bad_score_evals": self.bad_score_evals,
            "bad_alignment_evals": self.bad_alignment_evals,
            "provisional_amax": provisional,
        }

    def _load_baseline_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != BASELINE_FORMAT:
            raise ValueError(f"unsupported QAD baseline state format {state.get('format')!r}")
        if state["normalizers"] is None:
            raise ValueError("the checkpoint has no step-0 baseline (saved before the step-0 evaluation)")
        self.normalizers = {m: [float(v) for v in values] for m, values in state["normalizers"].items()}
        self.step0 = {k: float(v) for k, v in state["step0"].items()}
        self.recon_norm = None if state["recon_norm"] is None else [float(v) for v in state["recon_norm"]]
        self.tolerances = {k: float(v) for k, v in state["tolerances"].items()}
        self.best_score = float(state["best_score"])
        self.bad_score_evals = int(state["bad_score_evals"])
        self.bad_alignment_evals = int(state["bad_alignment_evals"])
        self.bad_evals = max(self.bad_score_evals, self.bad_alignment_evals)
        for name, amax in state.get("provisional_amax", {}).items():
            method = self.student.qad_linears.get(name)
            if method is None:
                raise KeyError(f"checkpoint has a provisional scale for {name}, which is not a QAD linear here")
            if method.input_amax is None:
                method.set_input_amax(float(amax))
        if get_world_group().rank == 0:
            logger.info("QAD step-0 baseline restored from the checkpoint (score %.5f, bad evals %d/%d)",
                        self.step0.get("score", float("nan")), self.bad_score_evals, self.bad_alignment_evals)

    # ------------------------------------------------------------------ one training row
    def _row(self, spec: Any) -> Any:
        return self.student.load_row(spec)

    def single_train_step(self, batch: dict[str, Any], iteration: int) -> tuple[dict, dict, dict[str, LogScalar]]:
        if self.normalizers is None:
            raise RuntimeError("step-0 evaluation must run before training (it sets the loss normalizers)")
        spec = batch["row"]
        prepared = self._row(spec)
        rollout = self._use_rollout(spec.id, iteration)
        schedulers = self.student.schedulers()
        timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
        accum = max(1, int(self.training_config.loop.gradient_accumulation_steps or 1))
        num_rungs = self.student.num_rungs
        trained = self._trained_rungs(spec.id, iteration, num_rungs)
        video, audio = prepared.video, prepared.audio
        total = torch.zeros((), device=self.student.device)
        metrics: dict[str, LogScalar] = {
            "train/rollout": float(rollout),
            "train/tokens": float(prepared.layout.sequence_length)
        }
        started = time.perf_counter()
        for rung in range(num_rungs):
            recon = None
            with torch.no_grad(), self.teacher.rung_scope(prepared, rung, timesteps) as rt:
                if rung in trained and self.recon_weight > 0:
                    # Student blocks train inside the teacher forward, on the teacher's sub-layer inputs.
                    norm = self.recon_norm[rung] if self.recon_norm else 1.0
                    scale = self.recon_weight / (norm * 2 * len(self.student.transformer.transformer_blocks) *
                                                 len(trained) * accum)
                    with BlockReconstruction(self.teacher.transformer, self.student.transformer, scale,
                                             self._recon_criterion(prepared)) as recon:
                        teacher_out = self.teacher.forward_rung(prepared, video, audio, rt)
                else:
                    teacher_out = self.teacher.forward_rung(prepared, video, audio, rt)
            if recon is not None:
                recon_loss = recon.loss()
                if not torch.isfinite(recon_loss):
                    raise FloatingPointError(f"non-finite reconstruction loss at step {iteration}, rung {rung}")
                norm = self.recon_norm[rung] if self.recon_norm else 1.0
                total += self.recon_weight * recon_loss / (norm * len(trained))
                metrics[f"train/recon_vs_step0/rung{rung}"] = recon_loss / norm
                for key, value in recon.metrics("train/recon").items():
                    metrics[f"{key}/rung{rung}"] = value
            if rung not in trained or self.output_weight == 0:
                if rollout:  # the student's own trajectory still needs its (no-grad) forward here
                    with torch.no_grad(), self.student.rung_scope(prepared, rung, timesteps) as rt:
                        student_out = self.student.forward_rung(prepared, video, audio, rt)
                video, audio = step_packed(schedulers, prepared, rung, video, audio,
                                           *(student_out if rollout else teacher_out))
                continue
            with self.student.rung_scope(prepared, rung, timesteps) as rt:
                student_out = self.student.forward_rung(prepared, video, audio, rt)
                pair = x0_pair(prepared, schedulers, rung, video, audio, {
                    "student": student_out,
                    "teacher": teacher_out
                })
                loss = self._rung_loss(pair, rung, metrics)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite QAD loss at step {iteration}, row {spec.id}, rung {rung}")
                (self.output_weight * loss / (len(trained) * accum)).backward()
            total += self.output_weight * loss.detach() / len(trained)
            step_out = tuple(v.detach() for v in student_out) if rollout else teacher_out
            video, audio = step_packed(schedulers, prepared, rung, video, audio, *step_out)
        metrics["train/row_seconds"] = time.perf_counter() - started
        if get_world_group().rank == 0:
            logger.info("QAD step %d row %s (%s, %d tokens, rollout=%s): loss %.5f in %.1fs", iteration, spec.id,
                        spec.case, prepared.layout.sequence_length, rollout, float(total), metrics["train/row_seconds"])
        return {"total_loss": total}, {}, metrics

    def _recon_criterion(self, prepared: Any) -> Any:
        from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import (GroupedRelError, local_row_groups,
                                                                                  packed_row_groups)
        key = prepared.spec.id
        if key not in self._recon_criteria:
            self._recon_criteria = {}  # one row at a time
            groups = packed_row_groups(prepared.layout, int(prepared.layout.sequence_length))
            self._recon_criteria[key] = GroupedRelError(local_row_groups(groups, self.student.device),
                                                        self.recon_group_weights)
        return self._recon_criteria[key]

    @torch.no_grad()
    def _calibrate_recon(self, rows: list[tuple[Any, bool]]) -> dict[str, LogScalar]:
        """Step-0 reconstruction error per forward (measure-only), the Stage-2 normalizer."""
        num_rungs = self.student.num_rungs
        sums = torch.zeros(num_rungs, dtype=torch.float64, device=self.student.device)
        count = torch.zeros((), dtype=torch.float64, device=self.student.device)
        per_rank = -(-self.recon_norm_rows // self.dp_size)
        for spec, counted in rows[:per_rank]:
            prepared = self._row(spec)
            schedulers = self.student.schedulers()
            timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
            video, audio = prepared.video, prepared.audio
            for rung in range(num_rungs):
                with self.teacher.rung_scope(prepared, rung,
                                             timesteps) as rt, BlockReconstruction(self.teacher.transformer,
                                                                                   self.student.transformer,
                                                                                   0.0,
                                                                                   self._recon_criterion(prepared),
                                                                                   measure_only=True) as recon:
                    out = self.teacher.forward_rung(prepared, video, audio, rt)
                if counted:
                    sums[rung] += recon.loss().double()
                video, audio = step_packed(schedulers, prepared, rung, video, audio, *out)
            count += float(counted)
        if not self.is_sp_leader:
            sums.zero_()
            count.zero_()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(count)
        self.recon_norm = [max(float(v), 1e-12) for v in (sums / count.clamp_min(1.0))]
        return {f"init/recon_step0/rung{rung}": value for rung, value in enumerate(self.recon_norm)}

    def _noise_floor(self, rows: list[tuple[Any, bool]], step0: dict[str, float]) -> dict[str, LogScalar]:
        """Re-evaluate step 0 at seed+1; per-metric tolerance = max(2 sigma, minimum) with sigma relative."""
        import dataclasses
        shifted = [(dataclasses.replace(spec, seed=spec.seed + 1), counted) for spec, counted in rows]
        other = self._evaluate(shifted, rollout=True, save_tag="seed1")
        out: dict[str, LogScalar] = {}
        for key in set(self.eligibility_keys) | {"endpoint/video", "score"}:
            if key in step0 and key in other:
                mean = 0.5 * (step0[key] + other[key])
                sigma = abs(step0[key] - other[key]) / (math.sqrt(2.0) * max(mean, 1e-12))
                self.tolerances[key] = max(2.0 * sigma, self.min_eligibility_tolerance)
                out[f"init/noise_sigma/{key}"] = sigma
                out[f"init/seed1/{key}"] = other[key]
        return out

    def _tolerance(self, key: str) -> float:
        return self.tolerances.get(key, self.fidelity_tolerance)

    def _sampling_key(self, row_id: str) -> str:
        """Keep forward/backward choices in lockstep across the FSDP shard group.

        FSDP spans data-parallel groups, whose rows differ. A row-dependent choice can make one group gather
        teacher block weights while another gathers student block weights, corrupting both forwards. Single-group
        runs retain their existing row-dependent sampling.
        """
        return row_id if self.dp_size == 1 else f"fsdp-shared:{self.sampling_seed}"

    def _use_rollout(self, row_id: str, iteration: int) -> bool:
        return (iteration >= self.rollout_start_step
                and _unit_hash(self._sampling_key(row_id), iteration) < self.rollout_fraction)

    def _trained_rungs(self, row_id: str, iteration: int, num_rungs: int) -> set[int]:
        """All forwards, or a deterministic subset shared by every FSDP rank."""
        if self.rungs_per_row <= 0 or self.rungs_per_row >= num_rungs:
            return set(range(num_rungs))
        order = sorted(range(num_rungs),
                       key=lambda rung: _unit_hash(self._sampling_key(row_id), iteration, "rung", rung))
        return set(order[:self.rungs_per_row])

    def _rung_loss(self, pair: dict[str, dict[str, torch.Tensor]], rung: int, metrics: dict[str,
                                                                                            LogScalar]) -> torch.Tensor:
        assert self.normalizers is not None
        loss = torch.zeros((), device=self.student.device)
        for modality in MODALITIES:
            student, teacher = pair[modality]["student"], pair[modality]["teacher"]
            mse = (student - teacher).pow(2).mean()
            normalized = mse / self.normalizers[modality][rung]
            loss = loss + self.weights[modality] * normalized
            metrics[f"train/rel_mse_vs_ptq/{modality}/rung{rung}"] = normalized.detach()
        return loss / sum(self.weights.values())

    # ------------------------------------------------------------------ evaluation and guardrails
    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        # Smoke runs without Stage A's table: calibrate the FFN scales before anything uses them, whether step 0
        # is measured, restored from the cache, or skipped by a resume. A no-op once every scale is set.
        calibration = self._provisional_calibration()
        first = self.normalizers is None
        if not first and (self.eval_every <= 0 or iteration % self.eval_every):
            return dict(calibration)
        rows = self._rank_rows()
        metrics: dict[str, LogScalar] = dict(calibration)
        started = time.perf_counter()
        self._eval_iteration = iteration
        cached = self._load_step0() if first else None
        if cached is not None:
            summary = self._restore_step0(cached)
            metrics.update(cached)
        else:
            if first and iteration == 0 and not self.init_student_dcp:
                # Fresh start only: a resumed run (e.g. Stage 2 from a Stage-1 checkpoint) is no longer the teacher.
                metrics.update(self._initial_checks(rows))
            if first and self._needs_recon_norm():
                metrics.update(self._calibrate_recon(rows))
            full = first or (iteration % self.full_eval_every == 0)
            summary = self._evaluate(rows, rollout=full)
            metrics.update({f"eval/{k}": v for k, v in summary.items()})
            if first and self.noise_floor_eval:
                metrics.update(self._noise_floor(rows, summary))
            if first and self.step0_cache and get_world_group().rank == 0:
                save_step0(self.step0_cache, self._step0_inputs(), {k: float(v) for k, v in metrics.items()})
        if first:
            self.normalizers = {
                m: [max(summary[f"x0_mse/{m}/rung{r}"], 1e-20) for r in range(self.student.num_rungs)]
                for m in MODALITIES
            }
            self.step0 = dict(summary)
            self.best_score = summary.get(self.select_metric, summary["score"])
        else:
            metrics.update(self._guardrails(iteration, summary))
        metrics["eval/seconds"] = time.perf_counter() - started
        metrics["eval/score_vs_ptq"] = summary["score"] / max(self.step0.get("score", summary["score"]), 1e-30)
        if get_world_group().rank == 0:
            logger.info("QAD eval @%d: score %.5f (PTQ %.5f) %s", iteration, summary["score"],
                        self.step0.get("score", float("nan")),
                        json.dumps({
                            k: round(float(v), 5)
                            for k, v in metrics.items()
                        }))
        return metrics

    def _needs_recon_norm(self) -> bool:
        return self.recon_normalize == "step0" and self.recon_weight > 0

    def _step0_inputs(self) -> dict[str, Any]:
        return step0_fingerprint_inputs(self._method_config_snapshot,
                                        self.student.data_config,
                                        self.student.qad_config,
                                        self.student.amax_sha256,
                                        len(self.student.eval_rows),
                                        student_init_from=self.student._init_from,
                                        teacher_init_from=self.teacher._init_from)

    def _load_step0(self) -> dict[str, float] | None:
        if not self.step0_cache:
            return None
        cached = load_step0(self.step0_cache, self._step0_inputs())
        if cached is not None and get_world_group().rank == 0:
            logger.info("QAD step 0 restored from %s (no step-0 evaluation)", self.step0_cache)
        return cached

    def _restore_step0(self, cached: dict[str, float]) -> dict[str, float]:
        summary, recon_norm, tolerances = restore_step0(cached, self.student.num_rungs, self._needs_recon_norm(),
                                                        self.noise_floor_eval, self.min_eligibility_tolerance)
        if recon_norm is not None:
            self.recon_norm = recon_norm
        self.tolerances.update(tolerances)
        return summary

    def _rank_rows(self) -> list[tuple[Any, bool]]:
        """This data-parallel group's eval rows, padded so every group runs the same number of forwards.

        FSDP all-gathers span every rank, so groups must stay in lockstep; padded
        repeats run but are not counted (``(row, counted)`` pairs).
        """
        rows = self.student.eval_rows[:self.max_eval_rows] if self.max_eval_rows else self.student.eval_rows
        if not rows:
            return []
        per_rank = -(-len(rows) // self.dp_size)
        return [(rows[(i * self.dp_size + self.dp_rank) % len(rows)], i * self.dp_size + self.dp_rank < len(rows))
                for i in range(per_rank)]

    def _eligible(self, summary: dict[str, float]) -> bool:
        """Full-eval audio/keyframe metrics within step0 * (1 + tolerance); a teacher-forced-only eval cannot tell."""
        if "endpoint/video" not in summary:
            return False
        return all(summary[key] <= self.step0[key] * (1.0 + self._tolerance(key)) for key in self.eligibility_keys
                   if key in self.step0 and key in summary)

    def _guardrails(self, iteration: int, summary: dict[str, float]) -> dict[str, LogScalar]:
        worse_score = summary["score"] > self.step0["score"]
        self.bad_score_evals = self.bad_score_evals + 1 if worse_score else 0
        # Teacher alignment is measured by full evaluations only; a teacher-forced-only one leaves its count as is.
        measured = [key for key in ALIGNMENT_KEYS if key in summary and key in self.step0]
        if measured:
            misaligned = any(summary[key] > self.step0[key] * (1.0 + self._tolerance(key)) for key in measured)
            self.bad_alignment_evals = self.bad_alignment_evals + 1 if misaligned else 0
        self.bad_evals = max(self.bad_score_evals, self.bad_alignment_evals)
        selected = summary.get(self.select_metric)
        eligible = self._eligible(summary)
        if selected is not None and selected < self.best_score and eligible:
            self.best_score = selected
            self._checkpoint_request = "best" if self.save_best else None
        if self.bad_evals >= self.patience:
            self.stop_requested = True
            logger.warning("QAD stop: %d consecutive evaluations worse than the PTQ start (step %d)", self.bad_evals,
                           iteration)
        return {
            "eval/eligible": float(eligible),
            "eval/bad_evals": float(self.bad_evals),
            "eval/bad_score_evals": float(self.bad_score_evals),
            "eval/bad_alignment_evals": float(self.bad_alignment_evals),
            "eval/best_score": self.best_score,
            "eval/stop_requested": float(self.stop_requested)
        }

    @torch.no_grad()
    def _evaluate(self,
                  rows: list[tuple[Any, bool]],
                  *,
                  rollout: bool,
                  attention: bool = True,
                  linears: bool = True,
                  save_tag: str = "") -> dict[str, float]:
        num_rungs = self.student.num_rungs
        stats = RungStats(num_rungs)
        counted_stats = stats
        # Same rollout count on every data-parallel group (lockstep), covering the first max_rollout_rows rows.
        rollout_rows = -(-self.max_rollout_rows // self.dp_size) if self.max_rollout_rows else len(rows)
        with self.student.quantization(linears=linears, attention=attention):
            for position, (spec, counted) in enumerate(rows):
                stats = counted_stats if counted else RungStats(num_rungs)
                prepared = self._row(spec)
                schedulers = self.student.schedulers()
                timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
                video, audio = prepared.video, prepared.audio
                for rung in range(num_rungs):
                    with self.teacher.rung_scope(prepared, rung, timesteps) as rt:
                        teacher_out = self.teacher.forward_rung(prepared, video, audio, rt)
                    with self.student.rung_scope(prepared, rung, timesteps) as rt:
                        student_out = self.student.forward_rung(prepared, video, audio, rt)
                    stats.add_rung(
                        rung,
                        x0_pair(prepared, schedulers, rung, video, audio, {
                            "student": student_out,
                            "teacher": teacher_out
                        }))
                    video, audio = step_packed(schedulers, prepared, rung, video, audio, *teacher_out)
                if rollout and position < rollout_rows:
                    self._rollout_metrics(stats,
                                          prepared,
                                          schedulers,
                                          timesteps,
                                          video,
                                          audio,
                                          save=counted,
                                          save_tag=save_tag)
        means = counted_stats.reduce(contribute=self.is_sp_leader)
        summary = summarize(means, num_rungs, self.weights)
        for name in ("first", "last"):
            student, teacher = summary.get(f"keyframe/student/{name}"), summary.get(f"keyframe/teacher/{name}")
            if student is not None and teacher is not None:
                summary[f"keyframe/drop/{name}"] = student / max(teacher, 1e-12) - 1.0
        return summary

    def _rollout_metrics(self,
                         stats: RungStats,
                         prepared: Any,
                         schedulers: tuple[Any, Any],
                         timesteps: Any,
                         teacher_video: torch.Tensor,
                         teacher_audio: torch.Tensor,
                         save: bool = False,
                         save_tag: str = "") -> None:
        """Student's own 8-forward sample vs the teacher's: endpoint rel-L2 and latent keyframe fidelity."""
        video, audio = prepared.video, prepared.audio
        for rung in range(self.student.num_rungs):
            with self.student.rung_scope(prepared, rung, timesteps) as rt:
                out = self.student.forward_rung(prepared, video, audio, rt)
            video, audio = step_packed(schedulers, prepared, rung, video, audio, *out)
        cut = target_slices(prepared)
        if save and self.eval_save_dir and self.is_sp_leader:
            step_dir = f"step{self._eval_iteration:05d}" + (f"_{save_tag}" if save_tag else "")
            path = Path(self.eval_save_dir) / step_dir / f"{prepared.spec.id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "student_video": video[cut["video"]].float().cpu(),
                    "teacher_video": teacher_video[cut["video"]].float().cpu(),
                    # Audio for decoded stats (spectral centroid, >8 kHz share, dynamic range, log-mel distance).
                    "student_audio": audio[cut["audio"]].float().cpu(),
                    "teacher_audio": teacher_audio[cut["audio"]].float().cpu(),
                    "latent_shape": prepared.latent_shape,
                    "case": prepared.spec.case,
                    "seed": prepared.spec.seed
                },
                path)
        for modality, student, teacher in (("video", video, teacher_video), ("audio", audio, teacher_audio)):
            s, t = student[cut[modality]].float(), teacher[cut[modality]].float()
            stats.add_scalar(f"endpoint.{modality}", (s - t).norm() / t.norm().clamp_min(1e-12))
        teacher_frames = keyframe_latents(prepared, teacher_video)
        for name, (frame, reference) in keyframe_latents(prepared, video).items():
            stats.add_scalar(f"keyframe.vs_teacher.{name}", rel_l2(frame, teacher_frames[name][0]))
            if reference is not None:
                stats.add_scalar(f"keyframe.student.{name}", rel_l2(frame, reference))
                stats.add_scalar(f"keyframe.teacher.{name}", rel_l2(teacher_frames[name][0], reference))

    def _initial_checks(self, rows: list[tuple[Any, bool]]) -> dict[str, LogScalar]:
        """Bit-identity with quantization off, and the PTQ error with bf16 attention (Stage A's protocol)."""
        out: dict[str, LogScalar] = {}
        diff = torch.zeros((), device=self.student.device, dtype=torch.float64)
        if rows:
            prepared = self._row(rows[0][0])
            schedulers = self.student.schedulers()
            timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
            with torch.no_grad(), self.student.quantization(linears=False, attention=False):
                for rung in (0, self.student.num_rungs - 1):
                    outs = []
                    for model in (self.teacher, self.student):
                        with model.rung_scope(prepared, rung, timesteps) as rt:
                            outs.append(model.forward_rung(prepared, prepared.video, prepared.audio, rt))
                    for teacher_t, student_t in zip(*outs, strict=True):
                        diff = torch.maximum(diff, (teacher_t.double() - student_t.double()).abs().max())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(diff, op=torch.distributed.ReduceOp.MAX)
        out["init/noquant_vs_teacher_max_abs"] = float(diff)
        sha = getattr(self.student, "amax_sha256", "")
        if sha:  # first 32 bits of the pinned calibration table's sha256, for W&B
            out["init/amax_sha256_u32"] = float(int(sha[:8], 16))
        if self.require_bitwise_init and float(diff) != 0.0:
            raise RuntimeError(f"student with quantization disabled differs from the teacher (max |diff| "
                               f"{float(diff):.3e}); refusing to train from a student that is not the teacher")
        weights_only = self._evaluate(rows, rollout=False, attention=False)
        out.update({
            f"init/ptq_bf16_attn/{k}": v
            for k, v in weights_only.items() if k.startswith(("x0_rel", "v_rel", "score"))
        })
        out.update(self._compare_ptq_reference(weights_only))
        if getattr(self.student, "compression", None) is not None:
            # Training-free token compression alone (bf16 linears): how much of the step-0 error it contributes.
            compress_only = self._evaluate(rows, rollout=False, linears=False)
            out.update({
                f"init/compress_only/{k}": v
                for k, v in compress_only.items() if k.startswith(("x0_rel", "v_rel", "score"))
            })
        return out

    @torch.no_grad()
    def _provisional_calibration(self) -> dict[str, LogScalar]:
        """Smoke runs without Stage A's table: max-calibrate the FFN inputs on dense teacher trajectories."""
        from fastvideo.layers.quantization.nvfp4_qad import calibrating, nvfp4_qad_export_scales
        pending = [m for m in self.student.qad_linears.values() if m.act_scale == "static" and m.input_amax is None]
        if not pending:
            return {}
        count = int(getattr(self.student, "provisional_rows", 0))
        rows = self.student.train_rows[:count]
        per_rank = -(-len(rows) // self.dp_size)
        mine = [rows[(i * self.dp_size + self.dp_rank) % len(rows)] for i in range(per_rank)]
        with self.student.quantization(linears=False, attention=False), calibrating(self.student.qad_linears):
            for spec in mine:
                prepared = self._row(spec)
                schedulers = self.student.schedulers()
                timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
                video, audio = prepared.video, prepared.audio
                for rung in range(self.student.num_rungs):
                    with self.student.rung_scope(prepared, rung, timesteps) as rt:
                        out = self.student.forward_rung(prepared, video, audio, rt)
                    video, audio = step_packed(schedulers, prepared, rung, video, audio, *out)
        table = nvfp4_qad_export_scales(self.student.qad_linears)
        if get_world_group().rank == 0:
            path = Path(self.training_config.checkpoint.output_dir) / "provisional_amax.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(table, indent=1, sort_keys=True))
            logger.warning("PROVISIONAL FFN scales from %d rows written to %s; use Stage A's table for real runs",
                           len(rows), path)
        return {"init/provisional_calibration_rows": float(len(rows))}

    def _compare_ptq_reference(self, weights_only: dict[str, float]) -> dict[str, LogScalar]:
        """Compare step-0 weights-only x0 rel-L2 per rung to Stage A's (``{"video": [...], "audio": [...]}``)."""
        if not self.ptq_reference_json:
            return {}
        reference = json.loads(Path(self.ptq_reference_json).read_text())
        deviations = []
        for modality in MODALITIES:
            for rung, value in enumerate(reference.get(modality, [])):
                ours = weights_only.get(f"{self.ptq_reference_metric}/{modality}/rung{rung}")
                if ours is not None and value:
                    deviations.append(abs(ours / float(value) - 1.0))
        worst = max(deviations, default=float("nan"))
        if deviations and worst > self.ptq_tolerance:
            raise RuntimeError(f"step-0 PTQ error deviates {worst:.1%} from Stage A's ({self.ptq_reference_json}); "
                               "the QAD student does not start at the calibrated PTQ point")
        return {"init/ptq_vs_stage_a_max_rel_dev": worst}


__all__ = ["OmniRefPDDQADMethod"]
