# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import cast

import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import pyarrow_schema_ode_trajectory
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline import WanCausalDMDPipeline
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases)
from fastvideo.distributed import get_local_torch_device
from fastvideo.forward_context import set_forward_context

logger = init_logger(__name__)


class ODEInitTrainingPipeline(TrainingPipeline):
    """
    Training pipeline for ODE-init using precomputed denoising trajectories.

    Supervision: predict the next latent in the stored trajectory by
    - feeding current latent at timestep t into the transformer to predict noise
    - stepping the scheduler with the predicted noise
    - minimizing MSE to the stored next latent at timestep t_next
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Match the preprocess/generation scheduler for consistent stepping
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_ode_trajectory

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=self.timestep_shift)

        self.dmd_denoising_steps = [1000, 750, 500, 250]
        self.dmd_denoising_steps = torch.tensor(self.dmd_denoising_steps, dtype=torch.long, device=get_local_torch_device())
        logger.info("Initialized ODE-init training pipeline with %s denoising steps", len(self.dmd_denoising_steps))
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps
        # self.min_timestep = int(self.training_args.min_timestep_ratio *
        #                         self.num_train_timestep)
        # self.max_timestep = int(self.training_args.max_timestep_ratio *
        #                         self.num_train_timestep)
        # self.real_score_guidance_scale = self.training_args.real_score_guidance_scale

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        # Warm start validation with current transformer
        self.validation_pipeline = WanCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)


    def _get_next_batch(self, training_batch):  # type: ignore[override]
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        # Required fields from parquet (ODE trajectory schema)
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        infos = batch['info_list']

        # Trajectory tensors may include a leading singleton batch dim per row
        trajectory_latents = batch['trajectory_latents']
        if trajectory_latents.dim() == 7:
            # [B, 1, S, C, T, H, W] -> [B, S, C, T, H, W]
            trajectory_latents = trajectory_latents[:, 0]
        elif trajectory_latents.dim() == 6:
            # already [B, S, C, T, H, W]
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_latents dim: {trajectory_latents.dim()}")

        trajectory_timesteps = batch['trajectory_timesteps']
        if trajectory_timesteps.dim() == 3:
            # [B, 1, S] -> [B, S]
            trajectory_timesteps = trajectory_timesteps[:, 0]
        elif trajectory_timesteps.dim() == 2:
            # [B, S]
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_timesteps dim: {trajectory_timesteps.dim()}")

        # Move to device
        device = get_local_torch_device()
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            device, dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=torch.bfloat16)
        training_batch.infos = infos

        return training_batch, trajectory_latents.to(device, dtype=torch.bfloat16), trajectory_timesteps.to(device)

    def _step_predict_next_latent(self, traj_latents: torch.Tensor,
                                   traj_timesteps: torch.Tensor,
                                   encoder_hidden_states: torch.Tensor,
                                   encoder_attention_mask: torch.Tensor
                                   ) -> torch.Tensor:
        device = get_local_torch_device()

        # Shapes: traj_latents [B, S, C, T, H, W], traj_timesteps [B, S]
        B, S = traj_latents.shape[0], traj_latents.shape[1]

        # Sample a DMD timestep and find the closest stored timestep per sample
        dmd_idx = torch.randint(0, len(self.dmd_denoising_steps), (1,), device=device)
        logger.info(f"dmd_idx: {dmd_idx}")
        dmd_timestep = int(self.dmd_denoising_steps[dmd_idx].item())

        # Compute per-sample nearest indices in the stored trajectory
        distances = (traj_timesteps.to(torch.long) - dmd_timestep).abs()  # [B, S]
        nearest_idx = torch.argmin(distances, dim=1)  # [B]

        batch_indices = torch.arange(B, device=device)
        noisy_input = traj_latents[batch_indices, nearest_idx]  # [B, C, T, H, W]
        target_latent = traj_latents[batch_indices, -1]  # [B, C, T, H, W]
        t = traj_timesteps[batch_indices, nearest_idx]  # [B]

        # Scale model input as in inference for consistency with stored trajectories
        # scaled_latents = self.modules["scheduler"].scale_model_input(current_latents, t)

        # Prepare inputs for transformer
        input_kwargs = {
            "hidden_states": noisy_input,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": t.to(device, dtype=torch.bfloat16),
            "encoder_attention_mask": encoder_attention_mask,
            "return_dict": False,
        }
        # Predict noise and step the scheduler to obtain next latent
        with set_forward_context(current_timestep=t,
                            attn_metadata=None,
                            forward_batch=None):
            noise_pred = self.transformer(**input_kwargs)
        if isinstance(noise_pred, (tuple, list)):
            noise_pred = noise_pred[0]
        # next_latent_pred = self.modules["scheduler"].step(
        #     noise_pred, t, current_latents, return_dict=False)[0]
        return noise_pred, target_latent, t
    
    def train_one_step(self, training_batch):  # type: ignore[override]
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        args = cast(TrainingArgs, self.training_args)

        for _ in range(args.gradient_accumulation_steps):
            training_batch, traj_latents, traj_timesteps = self._get_next_batch(training_batch)
            text_embeds = training_batch.encoder_hidden_states
            text_attention_mask = training_batch.encoder_attention_mask
            assert traj_latents.shape[0] == 1

            # Shapes: traj_latents [B, S, C, T, H, W], traj_timesteps [B, S]
            B, S = traj_latents.shape[0], traj_latents.shape[1]
            if S < 2:
                raise ValueError("Trajectory must contain at least 2 steps")

            # Sample per-sample current step i in [0, S-2]


            # idx = torch.randint(low=0, high=S - 1, size=(B, ),
            #                     device=traj_latents.device)

            # Gather current latents and next latents
            # batch_indices = torch.arange(B, device=traj_latents.device)
            # current_latents = traj_latents[batch_indices, idx]  # [B, C, T,H,W]
            # current_latent = traj_timesteps[:, -1, :, :, :, :]
            # target_latents = traj_latents[:, -1, :, :, :, :]

            # Corresponding timesteps t (long) -> cast per sample
            # t = traj_timesteps[:, -1, :, :, :, :]
            # if t.dtype != torch.long:
            #     t = t.long()

            # Forward to predict next latent by stepping scheduler with predicted noise
            noise_pred, target_latent, t = self._step_predict_next_latent(traj_latents,
                                                                traj_timesteps,
                                                              text_embeds,
                                                              text_attention_mask)

            mask = t != 0                                                            

            # Compute loss
            loss = F.mse_loss(noise_pred[mask], target_latent[mask], reduction="mean")
            loss = loss / args.gradient_accumulation_steps

            with set_forward_context(current_timestep=t,
                            attn_metadata=None,
                            forward_batch=None):
                loss.backward()
            avg_loss = loss.detach().clone()
            training_batch.total_loss += avg_loss.item()

        # Clip grad and step optimizers
        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in self.transformer.parameters() if p.requires_grad],
            args.max_grad_norm if args.max_grad_norm is not None else 0.0)

        self.optimizer.step()
        self.lr_scheduler.step()

        if grad_norm is None:
            grad_value = 0.0
        else:
            try:
                if isinstance(grad_norm, torch.Tensor):
                    grad_value = float(grad_norm.detach().float().item())
                else:
                    grad_value = float(grad_norm)
            except Exception:
                grad_value = 0.0
        training_batch.grad_norm = grad_value
        return training_batch


def main(args) -> None:
    logger.info("Starting ODE-init training pipeline...")
    pipeline = ODEInitTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("ODE-init training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)


