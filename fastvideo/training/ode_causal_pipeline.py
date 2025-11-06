# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_text_only)
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline)
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases)

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
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.modules["scheduler"].set_timesteps(num_inference_steps=1000,
                                                training=True)

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_ode_trajectory_text_only

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        assert self.timestep_shift == 5.0, "flow_shift must be 5.0"
        self.noise_scheduler = SelfForcingFlowMatchScheduler(
            shift=self.timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.noise_scheduler.set_timesteps(num_inference_steps=1000,
                                           training=True)

        logger.info("dmd_denoising_steps: %s",
                    self.training_args.pipeline_config.dmd_denoising_steps)
        self.dmd_denoising_steps = torch.tensor([1000, 750, 500, 250],
                                                dtype=torch.long,
                                                device=get_local_torch_device())
        if training_args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.noise_scheduler.timesteps.cpu(),
                                   torch.tensor([0],
                                                dtype=torch.float32))).cuda()
            logger.info("timesteps: %s", timesteps)
            self.dmd_denoising_steps = timesteps[1000 -
                                                 self.dmd_denoising_steps]
            logger.info("warped self.dmd_denoising_steps: %s",
                        self.dmd_denoising_steps)
        else:
            raise ValueError("warp_denoising_step must be true")

        self.dmd_denoising_steps = self.dmd_denoising_steps.to(
            get_local_torch_device())

        logger.info("denoising_step_list: %s", self.dmd_denoising_steps)

        logger.info(
            "Initialized ODE-init training pipeline with %s denoising steps",
            len(self.dmd_denoising_steps))
        # Cache for nearest trajectory index per DMD step (computed lazily on first batch)
        self._cached_closest_idx_per_dmd = None
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps
        self.manual_idx = 0

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

    def _get_next_batch(
            self,
            training_batch) -> tuple[TrainingBatch, torch.Tensor, torch.Tensor]:
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
                f"Unexpected trajectory_latents dim: {trajectory_latents.dim()}"
            )

        trajectory_timesteps = batch['trajectory_timesteps']
        if trajectory_timesteps.dim() == 3:
            # [B, 1, S] -> [B, S]
            trajectory_timesteps = trajectory_timesteps[:, 0]
        elif trajectory_timesteps.dim() == 2:
            # [B, S]
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_timesteps dim: {trajectory_timesteps.dim()}"
            )
        # [B, S, C, T, H, W] -> [B, S, T, C, H, W] to match self-forcing
        trajectory_latents = trajectory_latents.permute(0, 1, 3, 2, 4, 5)

        # Move to device
        device = get_local_torch_device()
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            device, dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=torch.bfloat16)
        training_batch.infos = infos

        return training_batch, trajectory_latents.to(
            device, dtype=torch.bfloat16), trajectory_timesteps.to(device)

        ## TEMP Used for loading the sf .pt files directly
        """
        self.manual_idx = self.manual_idx % 155
        path = f"/mnt/weka/home/hao.zhang/wl/Self-Forcing/ode_pt_vidprom_1000/{self.manual_idx:05d}.pt"
        logger.info("path: %s", path)
        self.manual_idx += 1
        # path = "/mnt/weka/home/hao.zhang/wl/Self-Forcing/ode_single_full/00000.pt"
        b = torch.load(path)
        training_batch.encoder_hidden_states = b["text_embedding"][0].unsqueeze(
            0).to(device, dtype=torch.bfloat16)
        trajectory_latents = b["ode_latent"].to(device, dtype=torch.bfloat16)
        logger.info("trajectory_latents: %s", trajectory_latents.shape)
        logger.info("encoder_hidden_states: %s",
                    training_batch.encoder_hidden_states.shape)
        assert trajectory_latents.shape[1] <= 10, "trajectory_latents.shape[1] must be <= 10"
        return training_batch, trajectory_latents.to(
            device, dtype=torch.bfloat16), trajectory_timesteps.to(device)
        """

    def _get_timestep(self,
                      min_timestep: int,
                      max_timestep: int,
                      batch_size: int,
                      num_frame: int,
                      num_frame_per_block: int,
                      uniform_timestep: bool = False) -> torch.Tensor:
        if uniform_timestep:
            timestep = torch.randint(min_timestep,
                                     max_timestep, [batch_size, 1],
                                     device=self.device,
                                     dtype=torch.long).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(min_timestep,
                                     max_timestep, [batch_size, num_frame],
                                     device=self.device,
                                     dtype=torch.long)
            # logger.info(f"individual timestep: {timestep}")
            # make the noise level the same within every block
            timestep = timestep.reshape(timestep.shape[0], -1,
                                        num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def _step_predict_next_latent(
        self, traj_latents: torch.Tensor, traj_timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str,
                                                              torch.Tensor]]:
        latent_vis_dict: dict[str, torch.Tensor] = {}
        device = get_local_torch_device()
        target_latent = traj_latents[:, -1]

        # Shapes: traj_latents [B, S, C, T, H, W], traj_timesteps [B, S]
        B, S, num_frames, num_channels, height, width = traj_latents.shape

        # Lazily cache nearest trajectory index per DMD step based on the (fixed) S timesteps
        if self._cached_closest_idx_per_dmd is None:
            self._cached_closest_idx_per_dmd = torch.tensor(
                [0, 12, 24, 36], dtype=torch.long).cpu()
            # [0, 1, 2, 3], dtype=torch.long).cpu()
            logger.info("self._cached_closest_idx_per_dmd: %s",
                        self._cached_closest_idx_per_dmd)
            logger.info(
                "corresponding timesteps: %s", self.noise_scheduler.timesteps[
                    self._cached_closest_idx_per_dmd])

        # Select the K indexes from traj_latents using self._cached_closest_idx_per_dmd
        # traj_latents: [B, S, C, T, H, W], self._cached_closest_idx_per_dmd: [K]
        # Output: [B, K, C, T, H, W]
        assert self._cached_closest_idx_per_dmd is not None
        relevant_traj_latents = torch.index_select(
            traj_latents,
            dim=1,
            index=self._cached_closest_idx_per_dmd.to(traj_latents.device))
        logger.info("relevant_traj_latents: %s", relevant_traj_latents.shape)
        # assert relevant_traj_latents.shape[0] == 1

        indexes = self._get_timestep(  # [B, num_frames]
            0,
            len(self.dmd_denoising_steps),
            B,
            num_frames,
            3,
            uniform_timestep=False)
        logger.info("indexes: %s", indexes.shape)
        logger.info("indexes: %s", indexes)
        # noisy_input = relevant_traj_latents[indexes]
        noisy_input = torch.gather(
            relevant_traj_latents,
            dim=1,
            index=indexes.reshape(B, 1, num_frames, 1, 1,
                                  1).expand(-1, -1, -1, num_channels, height,
                                            width).to(self.device)).squeeze(1)
        timestep = self.dmd_denoising_steps[indexes]
        logger.info("selected timestep for rank %s: %s",
                    self.global_rank,
                    timestep,
                    local_main_process_only=False)

        # Prepare inputs for transformer
        latent_vis_dict["noisy_input"] = noisy_input.permute(
            0, 2, 1, 3, 4).detach().clone().cpu()
        latent_vis_dict["x0"] = target_latent.permute(0, 2, 1, 3,
                                                      4).detach().clone().cpu()

        input_kwargs = {
            "hidden_states": noisy_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep.to(device, dtype=torch.bfloat16),
            "return_dict": False,
        }
        # Predict noise and step the scheduler to obtain next latent
        with set_forward_context(current_timestep=timestep,
                                 attn_metadata=None,
                                 forward_batch=None):
            noise_pred = self.transformer(**input_kwargs).permute(0, 2, 1, 3, 4)

        from fastvideo.models.utils import pred_noise_to_pred_video
        pred_video = pred_noise_to_pred_video(
            pred_noise=noise_pred.flatten(0, 1),
            noise_input_latent=noisy_input.flatten(0, 1),
            timestep=timestep.to(dtype=torch.bfloat16).flatten(0, 1),
            scheduler=self.modules["scheduler"]).unflatten(
                0, noise_pred.shape[:2])
        latent_vis_dict["pred_video"] = pred_video.permute(
            0, 2, 1, 3, 4).detach().clone().cpu()

        return pred_video, target_latent, timestep, latent_vis_dict

    def train_one_step(self, training_batch):  # type: ignore[override]
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        args = cast(TrainingArgs, self.training_args)

        # Using cached nearest index per DMD step; computation happens in _step_predict_next_latent

        for _ in range(args.gradient_accumulation_steps):
            training_batch, traj_latents, traj_timesteps = self._get_next_batch(
                training_batch)
            text_embeds = training_batch.encoder_hidden_states
            text_attention_mask = training_batch.encoder_attention_mask
            assert traj_latents.shape[0] == 1

            # Shapes: traj_latents [B, S, C, T, H, W], traj_timesteps [B, S]
            _, S = traj_latents.shape[0], traj_latents.shape[1]
            if S < 2:
                raise ValueError("Trajectory must contain at least 2 steps")

            # Forward to predict next latent by stepping scheduler with predicted noise
            noise_pred, target_latent, t, latent_vis_dict = self._step_predict_next_latent(
                traj_latents, traj_timesteps, text_embeds, text_attention_mask)

            training_batch.latent_vis_dict.update(latent_vis_dict)

            mask = t != 0

            # Compute loss
            loss = F.mse_loss(noise_pred[mask],
                              target_latent[mask],
                              reduction="mean")
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

    def visualize_intermediate_latents(self, training_batch: TrainingBatch,
                                       training_args: TrainingArgs, step: int):
        tracker_loss_dict: dict[str, Any] = {}
        latents_vis_dict = training_batch.latent_vis_dict
        latent_log_keys = ['noisy_input', 'x0', 'pred_video']
        for latent_key in latent_log_keys:
            assert latent_key in latents_vis_dict and latents_vis_dict[
                latent_key] is not None
            latent = latents_vis_dict[latent_key]
            pixel_latent = self.validation_pipeline.decoding_stage.decode(
                latent, training_args)

            video = pixel_latent.cpu().float()
            video = video.permute(0, 2, 1, 3, 4)
            video = (video * 255).numpy().astype(np.uint8)
            video_artifact = self.tracker.video(
                video, fps=16, format="mp4")  # change to 16 for Wan2.1
            if video_artifact is not None:
                tracker_loss_dict[latent_key] = video_artifact
            # Clean up references
            del video, pixel_latent, latent

        if self.global_rank == 0 and tracker_loss_dict:
            self.tracker.log_artifacts(tracker_loss_dict, step)


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
