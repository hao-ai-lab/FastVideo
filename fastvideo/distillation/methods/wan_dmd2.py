# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Any

import torch

from fastvideo.distributed import get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
)
from fastvideo.utils import set_random_seed

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.adapters.wan import WanPipelineAdapter


class WanDMD2Method(DistillMethod):
    def __init__(
        self,
        *,
        bundle: ModelBundle,
        adapter: WanPipelineAdapter,
    ) -> None:
        super().__init__(bundle)
        self.adapter = adapter
        self.training_args = adapter.pipeline.training_args
        self.world_group = get_world_group()

    def on_train_start(self) -> None:
        seed = self.training_args.seed
        if seed is None:
            raise ValueError("training_args.seed must be set for distillation")

        pipeline = self.adapter.pipeline
        if pipeline.sp_world_size > 1:
            sp_group_seed = seed + (pipeline.global_rank // pipeline.sp_world_size)
            set_random_seed(sp_group_seed)
        else:
            set_random_seed(seed + pipeline.global_rank)

        pipeline.noise_random_generator = torch.Generator(
            device="cpu").manual_seed(seed)
        pipeline.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(seed)
        if pipeline.device.type == "cuda":
            pipeline.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(seed)
        else:
            pipeline.noise_gen_cuda = torch.Generator(
                device=pipeline.device).manual_seed(seed)

        self.adapter.ensure_negative_conditioning()

    def log_validation(self, iteration: int) -> None:
        pipeline = self.adapter.pipeline
        training_args = pipeline.training_args
        if not getattr(training_args, "log_validation", False):
            return

        if getattr(pipeline, "validation_pipeline", None) is None:
            pipeline.initialize_validation_pipeline(training_args)

        old_inference_mode = training_args.inference_mode
        old_dit_cpu_offload = training_args.dit_cpu_offload
        try:
            pipeline._log_validation(
                pipeline.transformer,
                training_args,
                iteration,
            )
        finally:
            training_args.inference_mode = old_inference_mode
            training_args.dit_cpu_offload = old_dit_cpu_offload

    def _should_update_student(self, iteration: int) -> bool:
        interval = int(self.training_args.generator_update_interval or 1)
        if interval <= 0:
            return True
        return iteration % interval == 0

    def _clip_grad_norm(self, module: torch.nn.Module) -> float:
        max_grad_norm = self.training_args.max_grad_norm
        if not max_grad_norm:
            return 0.0
        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in module.parameters()],
            float(max_grad_norm),
            foreach=None,
        )
        return float(grad_norm.item()) if grad_norm is not None else 0.0

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        pipeline = self.adapter.pipeline
        pipeline.current_trainstep = iteration

        training_batch = self.adapter.prepare_batch(
            batch,
            current_vsa_sparsity=current_vsa_sparsity,
        )

        update_student = self._should_update_student(iteration)
        device = pipeline.device
        device_type = device.type

        generator_loss = torch.zeros(
            (),
            device=device,
            dtype=training_batch.latents.dtype,
        )
        batch_gen = None
        student_backward_ctx = None
        if update_student:
            batch_gen = copy.deepcopy(training_batch)
            with torch.autocast(device_type, dtype=batch_gen.latents.dtype):
                with set_forward_context(
                    current_timestep=batch_gen.timesteps,
                    attn_metadata=batch_gen.attn_metadata_vsa,
                ):
                    if self.training_args.simulate_generator_forward:
                        generator_pred_video = (
                            pipeline._generator_multi_step_simulation_forward(
                                batch_gen))
                    else:
                        generator_pred_video = pipeline._generator_forward(batch_gen)

                with set_forward_context(
                    current_timestep=batch_gen.timesteps,
                    attn_metadata=batch_gen.attn_metadata,
                ):
                    generator_loss = pipeline._dmd_forward(
                        generator_pred_video=generator_pred_video,
                        training_batch=batch_gen,
                    )
            student_backward_ctx = (batch_gen.timesteps, batch_gen.attn_metadata_vsa)

        batch_fake = copy.deepcopy(training_batch)
        with torch.autocast(device_type, dtype=batch_fake.latents.dtype):
            _, fake_score_loss = pipeline.faker_score_forward(batch_fake)

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }
        outputs = {}
        if update_student and batch_gen is not None:
            outputs["dmd_latent_vis_dict"] = batch_gen.dmd_latent_vis_dict
        outputs["fake_score_latent_vis_dict"] = batch_fake.fake_score_latent_vis_dict
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_backward_ctx,
            "critic_ctx": (batch_fake.timesteps, batch_fake.attn_metadata),
        }
        return loss_map, outputs

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
            super().backward(loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return

        update_student = bool(backward_ctx.get("update_student", False))
        if update_student:
            student_ctx = backward_ctx.get("student_ctx")
            if student_ctx is None:
                raise RuntimeError("Missing student backward context")
            timesteps, attn_metadata = student_ctx
            with set_forward_context(
                current_timestep=timesteps,
                attn_metadata=attn_metadata,
            ):
                (loss_map["generator_loss"] / grad_accum_rounds).backward()

        timesteps, attn_metadata = backward_ctx["critic_ctx"]
        with set_forward_context(
            current_timestep=timesteps,
            attn_metadata=attn_metadata,
        ):
            (loss_map["fake_score_loss"] / grad_accum_rounds).backward()

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        optimizers: list[torch.optim.Optimizer] = []
        optimizers.extend(self.bundle.role("critic").optimizers.values())
        if self._should_update_student(iteration):
            optimizers.extend(self.bundle.role("student").optimizers.values())
        return optimizers

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        schedulers: list[Any] = []
        schedulers.extend(self.bundle.role("critic").lr_schedulers.values())
        if self._should_update_student(iteration):
            schedulers.extend(self.bundle.role("student").lr_schedulers.values())
        return schedulers

    def optimizers_schedulers_step(self, iteration: int) -> None:
        if self._should_update_student(iteration):
            for module in self.bundle.role("student").modules.values():
                self._clip_grad_norm(module)
        for module in self.bundle.role("critic").modules.values():
            self._clip_grad_norm(module)

        super().optimizers_schedulers_step(iteration)
