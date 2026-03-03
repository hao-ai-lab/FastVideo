# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing distillation method (algorithm layer).

Self-Forcing is a distribution-matching variant that changes *how the student
sample (gen_data) is obtained*: instead of a single-step rollout, we perform a
multi-step simulate rollout and only keep gradients at a stochastic "exit"
denoising step per temporal block (chunk).

This implementation follows the structure of:
- FastGen: `fastgen/methods/distribution_matching/self_forcing.py`
- FastVideo legacy: `fastvideo/training/self_forcing_distillation_pipeline.py`

Config keys used (YAML schema-v2):
- `recipe.method`: must be `"self_forcing"` for this method.
- `roles`: requires `student`, `teacher`, `critic` (trainable critic).
- `method_config`:
  - `rollout_mode`: must be `"simulate"` for self-forcing.
  - `dmd_denoising_steps` (list[int]): the candidate denoising steps.
  - `cfg_uncond` (optional): same as DMD2.
  - self-forcing rollout knobs:
    - `chunk_size` (int, optional; default=3)
    - `student_sample_type` (`"sde"` or `"ode"`, optional; default="sde")
    - `same_step_across_blocks` (bool, optional; default=false)
    - `last_step_only` (bool, optional; default=false)
    - `context_noise` (float, optional; default=0.0)
    - `enable_gradient_in_rollout` (bool, optional; default=true)
    - `start_gradient_frame` (int, optional; default=0)
- `training` (selected fields used for optim/schedule): same as DMD2.
- `training.validation.*`: parsed by base DMD2 method; executed via validator.

Notes / limitations:
- This method uses `SelfForcingFlowMatchScheduler` for simulate-rollout
  (matching legacy behavior). Teacher/critic losses also interpret noise under
  the same scheduler for consistency.
- Context/no-grad KV cache updates are model-family specific in legacy. The new
  framework keeps method/model decoupled, so this method implements a cache-free
  rollout using the model plugin's `predict_noise` primitive.
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch
import torch.distributed as dist

from fastvideo.distillation.dispatch import register_method
from fastvideo.distillation.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.distillation.utils.config import get_optional_float, get_optional_int
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from fastvideo.models.utils import pred_noise_to_pred_video

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch


def _require_bool(raw: Any, *, where: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"Expected bool at {where}, got {type(raw).__name__}")


def _require_str(raw: Any, *, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Expected non-empty string at {where}")
    return raw


@register_method("self_forcing")
class SelfForcingMethod(DMD2Method):
    """Self-Forcing DMD2 (distribution matching) method.

    Inherits DMD2 losses/update policy, but replaces the student rollout
    procedure with a self-forcing rollout.
    """

    def __init__(
        self,
        *,
        bundle: Any,
        model: Any,
        method_config: dict[str, Any] | None = None,
        validation_config: dict[str, Any] | None = None,
        validator: Any | None = None,
    ) -> None:
        super().__init__(
            bundle=bundle,
            model=model,
            method_config=method_config,
            validation_config=validation_config,
            validator=validator,
        )

        if self._rollout_mode != "simulate":
            raise ValueError("SelfForcingMethod only supports method_config.rollout_mode='simulate'")

        cfg = self.method_config

        chunk_size = get_optional_int(
            cfg,
            "chunk_size",
            where="method_config.chunk_size",
        )
        if chunk_size is None:
            chunk_size = 3
        if chunk_size <= 0:
            raise ValueError(
                "method_config.chunk_size must be a positive integer, got "
                f"{chunk_size}"
            )
        self._chunk_size = int(chunk_size)

        sample_type_raw = cfg.get("student_sample_type", "sde")
        sample_type = _require_str(sample_type_raw, where="method_config.student_sample_type")
        sample_type = sample_type.strip().lower()
        if sample_type not in {"sde", "ode"}:
            raise ValueError(
                "method_config.student_sample_type must be one of {sde, ode}, got "
                f"{sample_type_raw!r}"
            )
        self._student_sample_type: Literal["sde", "ode"] = sample_type  # type: ignore[assignment]

        same_step_raw = cfg.get("same_step_across_blocks", False)
        if same_step_raw is None:
            same_step_raw = False
        self._same_step_across_blocks = _require_bool(
            same_step_raw, where="method_config.same_step_across_blocks"
        )

        last_step_raw = cfg.get("last_step_only", False)
        if last_step_raw is None:
            last_step_raw = False
        self._last_step_only = _require_bool(
            last_step_raw, where="method_config.last_step_only"
        )

        context_noise = get_optional_float(
            cfg,
            "context_noise",
            where="method_config.context_noise",
        )
        if context_noise is None:
            context_noise = 0.0
        if context_noise < 0.0:
            raise ValueError(
                "method_config.context_noise must be >= 0, got "
                f"{context_noise}"
            )
        self._context_noise = float(context_noise)

        enable_grad_raw = cfg.get("enable_gradient_in_rollout", True)
        if enable_grad_raw is None:
            enable_grad_raw = True
        self._enable_gradient_in_rollout = _require_bool(
            enable_grad_raw, where="method_config.enable_gradient_in_rollout"
        )

        start_grad_frame = get_optional_int(
            cfg,
            "start_gradient_frame",
            where="method_config.start_gradient_frame",
        )
        if start_grad_frame is None:
            start_grad_frame = 0
        if start_grad_frame < 0:
            raise ValueError(
                "method_config.start_gradient_frame must be >= 0, got "
                f"{start_grad_frame}"
            )
        self._start_gradient_frame = int(start_grad_frame)

        # Legacy Self-Forcing uses a dedicated scheduler (different sigma_min).
        shift = float(getattr(self.training_args.pipeline_config, "flow_shift", 0.0) or 0.0)
        self._sf_scheduler = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=int(self.model.num_train_timesteps),
            shift=shift,
            sigma_min=0.0,
            extra_one_step=True,
            training=True,
        )

        # Cache for warped denoising step list (device specific).
        self._sf_denoising_step_list: torch.Tensor | None = None

    def _get_denoising_step_list(self, device: torch.device) -> torch.Tensor:
        if (
            self._sf_denoising_step_list is not None
            and self._sf_denoising_step_list.device == device
        ):
            return self._sf_denoising_step_list

        raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method_config.dmd_denoising_steps must be set for self_forcing")
        steps = torch.tensor([int(s) for s in raw], dtype=torch.long, device=device)

        warp = self.method_config.get("warp_denoising_step", None)
        if warp is None:
            warp = getattr(self.training_args, "warp_denoising_step", False)
        if bool(warp):
            timesteps = torch.cat(
                (
                    self._sf_scheduler.timesteps.to("cpu"),
                    torch.tensor([0], dtype=torch.float32),
                )
            ).to(device)
            steps = timesteps[int(self.model.num_train_timesteps) - steps]

        self._sf_denoising_step_list = steps
        return steps

    def _predict_x0_with_scheduler(
        self,
        handle: Any,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        attn_kind: Literal["dense", "vsa"],
    ) -> torch.Tensor:
        pred_noise = self.model.predict_noise(
            handle,
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=self._cfg_uncond,
            attn_kind=attn_kind,
        )
        pred_x0 = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latents.flatten(0, 1),
            timestep=timestep,
            scheduler=self._sf_scheduler,
        ).unflatten(0, pred_noise.shape[:2])
        return pred_x0

    def _sf_add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, t = clean_latents.shape[:2]
        noisy = self._sf_scheduler.add_noise(
            clean_latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep,
        ).unflatten(0, (b, t))
        return noisy

    def _timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        sigmas = self._sf_scheduler.sigmas.to(device=timestep.device, dtype=torch.float32)
        timesteps = self._sf_scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        t = timestep.to(device=timestep.device, dtype=torch.float32)
        if t.ndim == 2:
            t = t.flatten(0, 1)
        elif t.ndim == 1 and t.numel() == 1:
            t = t.expand(1)
        elif t.ndim != 1:
            raise ValueError(f"Invalid timestep shape: {tuple(timestep.shape)}")
        idx = torch.argmin((timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(), dim=1)
        return sigmas[idx]

    def _sample_exit_indices(
        self,
        *,
        num_blocks: int,
        num_steps: int,
        device: torch.device,
    ) -> list[int]:
        if num_blocks <= 0:
            return []
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        shape = (1,) if self._same_step_across_blocks else (num_blocks,)

        if not dist.is_initialized() or dist.get_rank() == 0:
            if self._last_step_only:
                indices = torch.full(
                    shape,
                    num_steps - 1,
                    dtype=torch.long,
                    device=device,
                )
            else:
                indices = torch.randint(
                    low=0,
                    high=num_steps,
                    size=shape,
                    device=device,
                )
        else:
            indices = torch.empty(shape, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)

        if self._same_step_across_blocks:
            return [int(indices.item()) for _ in range(num_blocks)]
        return [int(i) for i in indices.tolist()]

    def _student_rollout(self, batch: Any, *, with_grad: bool) -> torch.Tensor:
        latents = batch.latents
        if latents is None:
            raise RuntimeError("TrainingBatch.latents is required for self-forcing rollout")
        if latents.ndim != 5:
            raise ValueError(
                "TrainingBatch.latents must be [B, T, C, H, W], got "
                f"shape={tuple(latents.shape)}"
            )

        device = latents.device
        dtype = latents.dtype
        batch_size = int(latents.shape[0])
        num_frames = int(latents.shape[1])

        denoising_steps = self._get_denoising_step_list(device)
        num_steps = int(denoising_steps.numel())

        noise_full = torch.randn_like(latents, device=device, dtype=dtype)

        chunk = int(self._chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")

        remaining = num_frames % chunk
        num_blocks = num_frames // chunk
        if num_blocks == 0:
            num_blocks = 1
            remaining = num_frames

        exit_indices = self._sample_exit_indices(
            num_blocks=num_blocks,
            num_steps=num_steps,
            device=device,
        )

        context_latents: list[torch.Tensor] = []
        context_timesteps: list[torch.Tensor] = []
        denoised_blocks: list[torch.Tensor] = []

        for block_idx in range(num_blocks):
            if block_idx == 0:
                start = 0
                end = remaining + chunk if remaining else chunk
            else:
                start = remaining + block_idx * chunk
                end = remaining + (block_idx + 1) * chunk
            start = int(start)
            end = int(min(end, num_frames))
            if start >= end:
                break

            noisy_block = noise_full[:, start:end]
            exit_idx = int(exit_indices[block_idx])

            context_len = start
            future_latents = noise_full[:, end:]
            future_len = int(future_latents.shape[1])

            for step_idx, current_timestep in enumerate(denoising_steps):
                exit_flag = step_idx == exit_idx

                timestep_block = current_timestep * torch.ones(
                    (batch_size, end - start),
                    device=device,
                    dtype=torch.float32,
                )

                if context_len > 0:
                    context_lat = torch.cat(context_latents, dim=1)
                    context_t = torch.cat(context_timesteps, dim=1)
                    if int(context_lat.shape[1]) != context_len:
                        raise RuntimeError("context_latents length mismatch")
                else:
                    context_lat = None
                    context_t = None

                if context_lat is None:
                    model_latents = torch.cat([noisy_block, future_latents], dim=1)
                    timestep_full = torch.cat(
                        [
                            timestep_block,
                            current_timestep
                            * torch.ones(
                                (batch_size, future_len),
                                device=device,
                                dtype=torch.float32,
                            ),
                        ],
                        dim=1,
                    )
                else:
                    model_latents = torch.cat([context_lat, noisy_block, future_latents], dim=1)
                    timestep_full = torch.cat(
                        [
                            context_t,
                            timestep_block,
                            current_timestep
                            * torch.ones(
                                (batch_size, future_len),
                                device=device,
                                dtype=torch.float32,
                            ),
                        ],
                        dim=1,
                    )

                enable_grad = (
                    bool(with_grad)
                    and bool(self._enable_gradient_in_rollout)
                    and torch.is_grad_enabled()
                    and start >= int(self._start_gradient_frame)
                )

                if not exit_flag:
                    with torch.no_grad():
                        pred_x0_full = self._predict_x0_with_scheduler(
                            self.student,
                            model_latents,
                            timestep_full,
                            batch,
                            conditional=True,
                            attn_kind="vsa",
                        )
                    pred_x0_chunk = pred_x0_full[:, context_len:context_len + (end - start)]
                    if step_idx + 1 >= num_steps:
                        break
                    next_timestep = denoising_steps[step_idx + 1]
                    if self._student_sample_type == "sde":
                        noisy_block = self._sf_add_noise(
                            pred_x0_chunk,
                            torch.randn_like(pred_x0_chunk),
                            next_timestep
                            * torch.ones(
                                (batch_size, end - start),
                                device=device,
                                dtype=torch.float32,
                            ),
                        )
                    else:
                        sigma_cur = self._timestep_to_sigma(timestep_block).view(
                            batch_size, end - start, 1, 1, 1
                        )
                        sigma_next = self._timestep_to_sigma(
                            next_timestep
                            * torch.ones(
                                (batch_size, end - start),
                                device=device,
                                dtype=torch.float32,
                            )
                        ).view(batch_size, end - start, 1, 1, 1)
                        eps = (noisy_block - (1 - sigma_cur) * pred_x0_chunk) / sigma_cur.clamp_min(
                            1e-8
                        )
                        noisy_block = (1 - sigma_next) * pred_x0_chunk + sigma_next * eps
                    continue

                with torch.set_grad_enabled(enable_grad):
                    pred_x0_full = self._predict_x0_with_scheduler(
                        self.student,
                        model_latents,
                        timestep_full,
                        batch,
                        conditional=True,
                        attn_kind="vsa",
                    )
                pred_x0_chunk = pred_x0_full[:, context_len:context_len + (end - start)]
                break

            denoised_blocks.append(pred_x0_chunk)

            # Update context frames (cache-free): store context-noised frames + timesteps.
            if self._context_noise > 0.0:
                context_timestep = torch.ones(
                    (batch_size, end - start),
                    device=device,
                    dtype=torch.float32,
                ) * float(self._context_noise)
                with torch.no_grad():
                    context_latents.append(
                        self._sf_add_noise(
                            pred_x0_chunk.detach(),
                            torch.randn_like(pred_x0_chunk),
                            context_timestep,
                        )
                    )
                context_timesteps.append(context_timestep)
            else:
                context_latents.append(pred_x0_chunk.detach())
                context_timesteps.append(
                    torch.zeros(
                        (batch_size, end - start),
                        device=device,
                        dtype=torch.float32,
                    )
                )

        if not denoised_blocks:
            raise RuntimeError("Self-forcing rollout produced no blocks")

        return torch.cat(denoised_blocks, dim=1)

    def _critic_flow_matching_loss(
        self, batch: Any
    ) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        with torch.no_grad():
            generator_pred_x0 = self._student_rollout(batch, with_grad=False)

        device = generator_pred_x0.device
        fake_score_timestep = torch.randint(
            0,
            int(self.model.num_train_timesteps),
            [1],
            device=device,
            dtype=torch.long,
        )
        fake_score_timestep = self.model.shift_and_clamp_timestep(fake_score_timestep)

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
        )
        noisy_x0 = self._sf_add_noise(generator_pred_x0, noise, fake_score_timestep)

        pred_noise = self.model.predict_noise(
            self.critic,
            noisy_x0,
            fake_score_timestep,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="dense",
        )
        target = noise - generator_pred_x0
        flow_matching_loss = torch.mean((pred_noise - target) ** 2)

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {"fake_score_latent_vis_dict": batch.fake_score_latent_vis_dict}
        return flow_matching_loss, (batch.timesteps, batch.attn_metadata), outputs

    def _dmd_loss(self, generator_pred_x0: torch.Tensor, batch: Any) -> torch.Tensor:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method_config.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = float(getattr(self.training_args, "real_score_guidance_scale", 1.0))
        device = generator_pred_x0.device

        with torch.no_grad():
            timestep = torch.randint(
                0,
                int(self.model.num_train_timesteps),
                [1],
                device=device,
                dtype=torch.long,
            )
            timestep = self.model.shift_and_clamp_timestep(timestep)

            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
            )
            noisy_latents = self._sf_add_noise(generator_pred_x0, noise, timestep)

            faker_x0 = self._predict_x0_with_scheduler(
                self.critic,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            real_cond_x0 = self._predict_x0_with_scheduler(
                self.teacher,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            real_uncond_x0 = self._predict_x0_with_scheduler(
                self.teacher,
                noisy_latents,
                timestep,
                batch,
                conditional=False,
                attn_kind="dense",
            )
            real_cfg_x0 = real_cond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale

            denom = torch.abs(generator_pred_x0 - real_cfg_x0).mean()
            grad = (faker_x0 - real_cfg_x0) / denom
            grad = torch.nan_to_num(grad)

        loss = 0.5 * torch.mean(
            (generator_pred_x0.float() - (generator_pred_x0.float() - grad.float()).detach())
            ** 2
        )
        return loss
