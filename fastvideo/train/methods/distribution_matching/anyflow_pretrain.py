# SPDX-License-Identifier: Apache-2.0
"""AnyFlow pretrain (flow-map central-difference) training method.

Stage 1 of the AnyFlow two-stage recipe. Trains a single student network
``u_θ(x_t, t, r)`` to predict the average velocity from time ``t`` back
to time ``r`` via the central-difference target

    target = (eps - x_0) - ((t - r) / N) * dF/dt

where ``N = num_train_timesteps`` and ``dF/dt`` is estimated from the
student's own forward at ``(t ± δ, r)`` (with one-sided fallback near the
schedule endpoints).

Per-batch ``(t, r)`` sampling follows the AnyFlow paper:

- ``diffusion_ratio`` fraction: ``r = t`` (recovers plain flow matching).
- ``consistency_ratio`` fraction: ``r = 0`` (consistency to clean data).
- Remaining fraction: ``(t, r) = (max, min)`` of two independent uniform
  draws (full reconstruction range).

Reference: ``trainer_wan_anyflow_pretrain.py`` in NVlabs/AnyFlow at
commit ``549236a``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
)
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


def _sample_pair_timesteps(
    *,
    batch_size: int,
    diffusion_ratio: float,
    consistency_ratio: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``(t, r)`` per the AnyFlow paper.

    Two uniform draws ``u1, u2 ∈ [0, 1]`` per sample, then
    ``t = max(u1, u2)`` and ``r = min(u1, u2)``. After the base sample,
    the first ``diffusion_ratio * B`` entries get ``r = t`` (diffusion
    branch, plain flow matching), and the next ``consistency_ratio * B``
    entries get ``r = 0`` (consistency branch).

    Returns
    -------
    t, r, is_diffusion, is_consistency
        Each tensor has shape ``(batch_size,)``. ``t`` and ``r`` are in
        ``[0, 1]`` (i.e. *not* yet shifted and *not* yet in absolute
        train-timestep units). ``is_diffusion`` and ``is_consistency``
        are bool masks that partition a subset of the batch — entries
        outside both masks are the "free" reconstruction fraction.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if diffusion_ratio < 0.0 or consistency_ratio < 0.0:
        raise ValueError(
            "diffusion_ratio and consistency_ratio must be non-negative")
    if diffusion_ratio + consistency_ratio > 1.0:
        raise ValueError(
            "diffusion_ratio + consistency_ratio must be <= 1, "
            f"got {diffusion_ratio} + {consistency_ratio}")

    u1 = torch.rand(batch_size, device=device, generator=generator)
    u2 = torch.rand(batch_size, device=device, generator=generator)
    t = torch.maximum(u1, u2)
    r = torch.minimum(u1, u2)

    n_diff = int(diffusion_ratio * batch_size)
    n_cons = int(consistency_ratio * batch_size)
    is_diffusion = torch.zeros(batch_size, dtype=torch.bool, device=device)
    is_consistency = torch.zeros(batch_size, dtype=torch.bool, device=device)
    is_diffusion[:n_diff] = True
    is_consistency[n_diff:n_diff + n_cons] = True

    # Override per the AnyFlow paper:
    # - diffusion entries: r = t (plain flow matching)
    # - consistency entries: r = 0 (consistency to clean data)
    r = torch.where(is_diffusion, t, r)
    r = torch.where(is_consistency, torch.zeros_like(r), r)
    return t, r, is_diffusion, is_consistency


@torch.no_grad()
def _central_difference_dF_dt(
    *,
    student: Any,
    batch: Any,
    noisy: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    delta: float,
    num_train_timesteps: float,
    attn_kind: str = "dense",
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Estimate ``dF/dt`` for the AnyFlow central-difference target.

    Computes a symmetric finite difference of the velocity prediction in
    *absolute train-timestep units*:

        dF/dt ≈ [u_θ(x_{t+δ}, t+δ, r) - u_θ(x_{t-δ}, t-δ, r)] / (2 * δ * guidance)

    The sample is also moved along the flow trajectory by the same
    finite step (``v_pred * (δ / N)``) to match AnyFlow's reference
    formulation in ``trainer_wan_anyflow_pretrain.py::compute_central_difference``.
    Wrapped in ``torch.no_grad`` so the two extra forwards never enter
    the backward graph.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if guidance_scale <= 0.0:
        raise ValueError(
            f"guidance_scale must be positive, got {guidance_scale}")

    v_pred = noise - latents  # ground-truth flow velocity
    delta_x = delta / float(num_train_timesteps)

    t_plus = t + delta
    noisy_plus = noisy + v_pred * delta_x
    f_plus = student.predict_velocity_with_r(
        noisy_plus, t_plus, r, batch,
        conditional=True, attn_kind=attn_kind)

    t_minus = t - delta
    noisy_minus = noisy - v_pred * delta_x
    f_minus = student.predict_velocity_with_r(
        noisy_minus, t_minus, r, batch,
        conditional=True, attn_kind=attn_kind)

    return (f_plus - f_minus) / (2.0 * delta * guidance_scale)


class AnyFlowPretrainMethod(TrainingMethod):
    """AnyFlow flow-map pretrain method.

    Single-student training; no teacher or critic. The student must
    implement ``predict_velocity_with_r(noisy, t, r, batch, ...)`` —
    typically a ``WanModel`` with ``r_embedder=True`` in its arch config.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "student" not in role_models:
            raise ValueError("AnyFlowPretrainMethod requires role 'student'")
        if not self.student._trainable:
            raise ValueError(
                "AnyFlowPretrainMethod requires student to be trainable")

        mcfg = self.method_config
        self._diffusion_ratio = float(
            get_optional_float(
                mcfg, "diffusion_ratio", where="method.diffusion_ratio")
            or 0.5)
        self._consistency_ratio = float(
            get_optional_float(
                mcfg, "consistency_ratio", where="method.consistency_ratio")
            or 0.25)
        if self._diffusion_ratio + self._consistency_ratio > 1.0:
            raise ValueError(
                "method.diffusion_ratio + method.consistency_ratio must "
                f"be <= 1, got {self._diffusion_ratio} + "
                f"{self._consistency_ratio}")

        # δ: finite-difference step in absolute train-timestep units.
        epsilon = get_optional_int(mcfg, "epsilon", where="method.epsilon")
        self._fd_epsilon = float(epsilon) if epsilon is not None else 5.0

        # Loss weighting scheme (uniform / gaussian / beta08).
        raw_weight_type = mcfg.get("weight_type", "beta08")
        if not isinstance(raw_weight_type, str):
            raise ValueError(
                "method.weight_type must be a string, got "
                f"{type(raw_weight_type).__name__}")
        weight_type = raw_weight_type.strip().lower()
        if weight_type not in {"uniform", "gaussian", "beta08"}:
            raise ValueError(
                "method.weight_type must be one of "
                "{uniform, gaussian, beta08}, "
                f"got {raw_weight_type!r}")
        self._weight_type = weight_type

        # Guidance fused into the training target (default 1.0 = unused).
        fg = get_optional_float(
            mcfg, "fuse_guidance_scale", where="method.fuse_guidance_scale")
        self._fuse_guidance_scale = float(fg) if fg is not None else 1.0
        if self._fuse_guidance_scale <= 0.0:
            raise ValueError(
                "method.fuse_guidance_scale must be positive, "
                f"got {self._fuse_guidance_scale}")

        # Flow-map scheduler — uses pipeline_config.flow_shift if present
        # and falls back to method.shift (and finally 1.0).
        shift = float(
            getattr(self.training_config.pipeline_config, "flow_shift", 0.0)
            or 0.0)
        if shift <= 0.0:
            shift_override = get_optional_float(
                mcfg, "shift", where="method.shift")
            shift = float(shift_override) if shift_override is not None else 1.0
        self._shift = shift

        # Lazy-imported to avoid circular imports on package load.
        from fastvideo.models.schedulers.scheduling_flow_map_euler_discrete import (
            FlowMapEulerDiscreteScheduler, )
        self._flow_map_scheduler = FlowMapEulerDiscreteScheduler(
            num_train_timesteps=int(self.student.num_train_timesteps),
            shift=self._shift,
        )

        self.student.init_preprocessors(self.training_config)
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(
        self,
        iteration: int,
    ) -> Sequence[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> Sequence[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        del iteration  # AnyFlow pretrain has no iteration-dependent dispatch.

        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        latents = training_batch.latents  # [B, T, C, H, W] (post-permute in prepare_batch).
        if latents is None or latents.ndim != 5:
            raise RuntimeError(
                "AnyFlow pretrain expects TrainingBatch.latents of shape "
                "[B, T, C, H, W] after prepare_batch; got "
                f"{None if latents is None else tuple(latents.shape)}")
        device = latents.device
        dtype = latents.dtype
        batch_size = int(latents.shape[0])

        # AnyFlow (t, r) sampling — overrides the timestep drawn by
        # WanModel._sample_timesteps inside prepare_batch.
        t_norm, r_norm, is_diffusion, is_consistency = _sample_pair_timesteps(
            batch_size=batch_size,
            diffusion_ratio=self._diffusion_ratio,
            consistency_ratio=self._consistency_ratio,
            device=device,
            generator=self.cuda_generator,
        )

        sched = self._flow_map_scheduler
        n_train = float(self.student.num_train_timesteps)
        t = (sched.apply_shift(t_norm) * n_train).to(device=device, dtype=dtype)
        r = (sched.apply_shift(r_norm) * n_train).to(device=device, dtype=dtype)

        # Fresh noise drawn from the method's RNG; ignore the noise that
        # prepare_batch attached (it pairs with the discarded timestep).
        noise = torch.randn(
            latents.shape,
            device=device,
            dtype=dtype,
            generator=self.cuda_generator,
        )
        noisy = sched.add_noise(latents, noise, t)

        # Keep training_batch coherent with the new (t, noisy): downstream
        # forward_context uses these fields.
        training_batch.timesteps = t
        training_batch.noise = noise
        training_batch.noisy_model_input = noisy.permute(0, 2, 1, 3, 4)

        # Student velocity prediction at (t, r).
        noise_pred = self.student.predict_velocity_with_r(
            noisy, t, r, training_batch,
            conditional=True,
            attn_kind="dense",
        )

        # Optional guidance distillation — fuse CFG into the training target so
        # the resulting checkpoint can be sampled at guidance_scale=1.0.
        if self._fuse_guidance_scale != 1.0:
            with torch.no_grad():
                noise_pred_uncond = self.student.predict_velocity_with_r(
                    noisy, t, r, training_batch,
                    conditional=False,
                    attn_kind="dense",
                )
            g = float(self._fuse_guidance_scale)
            noise_pred = (noise_pred - (1.0 - g) * noise_pred_uncond) / g

        dF_dt = _central_difference_dF_dt(
            student=self.student,
            batch=training_batch,
            noisy=noisy,
            latents=latents,
            noise=noise,
            t=t,
            r=r,
            delta=self._fd_epsilon,
            num_train_timesteps=n_train,
            attn_kind="dense",
            guidance_scale=self._fuse_guidance_scale,
        )

        # AnyFlow target: target = (eps - x_0) - (t - r) * dF/dt
        # dF/dt is in (velocity per absolute t-unit); (t - r) is in absolute units;
        # the product cancels back to velocity units, matching noise_pred.
        view = [batch_size] + [1] * (latents.ndim - 1)
        target = (noise - latents) - (t - r).view(*view) * dF_dt

        # Per-sample squared error, then per-timestep weight, then scale-balance
        # so the non-diffusion branches stay on the same magnitude as the
        # diffusion branch (matches AnyFlow's stop-grad rescaling).
        per_sample = torch.mean(
            ((noise_pred.float() - target.float()) ** 2).reshape(
                batch_size, -1),
            dim=-1,
        )
        weight = sched.get_train_weight(t, weight_type=self._weight_type)
        per_sample = per_sample * weight

        with torch.no_grad():
            diff_mask = is_diffusion
            diff_mean = per_sample[diff_mask].mean() if diff_mask.any() else per_sample.mean()
            non_diff_mask = ~diff_mask
            if non_diff_mask.any():
                scale = diff_mean / (per_sample[non_diff_mask] + 1e-5)
            else:
                scale = torch.tensor(1.0, device=device, dtype=per_sample.dtype)
        non_diff_idx = torch.nonzero(non_diff_mask, as_tuple=False).flatten()
        if non_diff_idx.numel() > 0:
            per_sample = per_sample.clone()
            per_sample[non_diff_idx] = per_sample[non_diff_idx] * scale

        total_loss = per_sample.mean()

        loss_map = {"total_loss": total_loss}
        metrics: dict[str, LogScalar] = {
            "diffusion_fraction": float(is_diffusion.float().mean()),
            "consistency_fraction": float(is_consistency.float().mean()),
            "scale_weight_mean": float(scale.mean()) if isinstance(
                scale, torch.Tensor) else float(scale),
        }
        outputs = {
            "student_ctx": (
                training_batch.timesteps,
                training_batch.attn_metadata,
            ),
        }
        return loss_map, outputs, metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        """Route the loss backward through the student's forward_context
        so attn metadata stays attached during gradient computation."""
        student_ctx = outputs.get("student_ctx")
        if student_ctx is None:
            super().backward(
                loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return
        self.student.backward(
            loss_map["total_loss"],
            student_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    def _init_optimizer_and_scheduler(self) -> None:
        tc = self.training_config
        params = [
            p for p in self.student.transformer.parameters() if p.requires_grad
        ]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=float(tc.optimizer.learning_rate),
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )
