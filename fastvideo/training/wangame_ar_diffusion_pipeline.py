# SPDX-License-Identifier: Apache-2.0

import sys
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_wangame
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.hyworld.pose import process_custom_actions
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.pipelines.basic.wan.wangame_causal_dmd_pipeline import (
    WanGameCausalDMDPipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases)
from fastvideo.utils import shallow_asdict

logger = init_logger(__name__)


class WanGameARDiffusionPipeline(TrainingPipeline):

    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.modules["scheduler"].set_timesteps(num_inference_steps=1000,
                                                training=True)

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_wangame

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        self.num_frame_per_block = getattr(training_args, 'num_frame_per_block', 3)
        self.timestep_shift = training_args.pipeline_config.flow_shift
        self.ar_noise_scheduler = SelfForcingFlowMatchScheduler(
            shift=self.timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.ar_noise_scheduler.set_timesteps(num_inference_steps=1000,
                                              training=True)

        logger.info("AR Diffusion pipeline initialized with "
                    "num_frame_per_block=%d, timestep_shift=%.1f",
                    self.num_frame_per_block, self.timestep_shift)

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True

        validation_scheduler = SelfForcingFlowMatchScheduler(
            shift=args_copy.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)
        validation_scheduler.set_timesteps(num_inference_steps=1000,
                                           training=True)

        num_val_steps = int(
            training_args.validation_sampling_steps.split(",")[0])
        step_size = 1000 // num_val_steps
        args_copy.pipeline_config.dmd_denoising_steps = list(
            range(1000, 0, -step_size))
        args_copy.pipeline_config.warp_denoising_step = True
        training_args.pipeline_config.dmd_denoising_steps = (
            args_copy.pipeline_config.dmd_denoising_steps)
        training_args.pipeline_config.warp_denoising_step = True

        logger.info("Validation: %d-step causal denoising, "
                    "dmd_denoising_steps has %d entries",
                    num_val_steps,
                    len(args_copy.pipeline_config.dmd_denoising_steps))

        self.validation_pipeline = WanGameCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "vae": self.get_module("vae"),
                "scheduler": validation_scheduler,
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        """
        Sample per-block timesteps.
        """
        device = get_local_torch_device()
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, 1],
                device=device, dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, num_frame],
                device=device, dtype=torch.long
            )
            # Make the noise level the same within every block
            timestep = timestep.reshape(
                timestep.shape[0], -1, num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_dataset.sampler.set_epoch(self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        clip_features = batch['clip_feature']
        image_latents = batch['first_frame_latent']
        image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = batch['pil_image']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.preprocessed_image = pil_image.to(
            get_local_torch_device())
        training_batch.image_embeds = clip_features.to(get_local_torch_device())
        training_batch.image_latents = image_latents.to(
            get_local_torch_device())
        training_batch.infos = infos

        # Action conditioning
        if 'mouse_cond' in batch and batch['mouse_cond'].numel() > 0:
            training_batch.mouse_cond = batch['mouse_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.mouse_cond = None

        if 'keyboard_cond' in batch and batch['keyboard_cond'].numel() > 0:
            training_batch.keyboard_cond = batch['keyboard_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.keyboard_cond = None

        # Validate action temporal dimensions match video num_frames
        expected_num_frames = (self.training_args.num_latent_t - 1) * 4 + 1
        if training_batch.keyboard_cond is not None:
            assert training_batch.keyboard_cond.shape[1] >= expected_num_frames, (
                f"keyboard_cond has {training_batch.keyboard_cond.shape[1]} frames "
                f"but need at least {expected_num_frames}")
            training_batch.keyboard_cond = training_batch.keyboard_cond[:, :expected_num_frames]
        if training_batch.mouse_cond is not None:
            assert training_batch.mouse_cond.shape[1] >= expected_num_frames, (
                f"mouse_cond has {training_batch.mouse_cond.shape[1]} frames "
                f"but need at least {expected_num_frames}")
            training_batch.mouse_cond = training_batch.mouse_cond[:, :expected_num_frames]

        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""
        assert self.training_args is not None
        latents = training_batch.latents  # [B, C, T, H, W]
        batch_size = latents.shape[0]
        num_latent_t = latents.shape[2]

        # Reshape latents to [B, T, C, H, W] for per-frame operations
        latents_btchw = latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        # Sample per-block independent timestep indices: [B, T]
        timestep_indices = self._get_timestep(
            min_timestep=0,
            max_timestep=self.ar_noise_scheduler.num_train_timesteps,
            batch_size=batch_size,
            num_frame=num_latent_t,
            num_frame_per_block=self.num_frame_per_block,
            uniform_timestep=False)

        # Convert indices to actual timestep values: [B, T]
        self.ar_noise_scheduler.timesteps = self.ar_noise_scheduler.timesteps.to(
            get_local_torch_device())
        timesteps = self.ar_noise_scheduler.timesteps[timestep_indices]

        # Generate noise: [B, T, C, H, W]
        noise = torch.randn_like(latents_btchw)

        # Add noise per-frame: noisy = (1-σ) * clean + σ * noise
        noisy_latents = self.ar_noise_scheduler.add_noise(
            latents_btchw.flatten(0, 1),  # [B*T, C, H, W]
            noise.flatten(0, 1),          # [B*T, C, H, W]
            timesteps.flatten(0, 1)       # [B*T]
        ).unflatten(0, (batch_size, num_latent_t))  # [B, T, C, H, W]

        # Convert back to [B, C, T, H, W] for transformer input
        noisy_model_input = noisy_latents.permute(0, 2, 1, 3, 4)

        # I2V concatenation: [mask(1ch), image_latent(16ch)] → 17+16=33 ch total
        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        temporal_compression_ratio = self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (num_latent_t - 1) * temporal_compression_ratio + 1
        _, num_channels, _, latent_height, latent_width = image_latents.shape
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, 1:] = 0

        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=temporal_compression_ratio)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]],
                                  dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1,
                                           temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(
            image_latents.device).to(dtype=torch.bfloat16)

        noisy_model_input = torch.cat(
            [noisy_model_input, mask_lat_size, image_latents], dim=1)

        # Compute flow-matching training target: target = noise - clean
        # Shape: [B, T, C, H, W]
        training_target = self.ar_noise_scheduler.training_target(
            latents_btchw.flatten(0, 1),
            noise.flatten(0, 1),
            timesteps.flatten(0, 1)
        ).unflatten(0, (batch_size, num_latent_t))

        # Store everything on training_batch
        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps  # [B, T] per-frame timesteps
        training_batch.noise = noise.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        training_batch.raw_latent_shape = latents.shape
        # Store extra data for the custom loss function
        training_batch._ar_training_target = training_target  # [B, T, C, H, W]

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Build transformer kwargs with action conditioning and per-frame timesteps."""
        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)

        # Process actions for each batch sample
        batch_size = training_batch.noisy_model_input.shape[0]
        keyboard_cond = training_batch.keyboard_cond
        mouse_cond = training_batch.mouse_cond

        if keyboard_cond is not None and mouse_cond is not None:
            viewmats_list, intrinsics_list, action_labels_list = [], [], []
            for b in range(batch_size):
                v, i, a = process_custom_actions(keyboard_cond[b],
                                                 mouse_cond[b])
                viewmats_list.append(v)
                intrinsics_list.append(i)
                action_labels_list.append(a)
            viewmats = torch.stack(viewmats_list,
                                   dim=0).to(get_local_torch_device(),
                                             dtype=torch.bfloat16)
            intrinsics = torch.stack(intrinsics_list,
                                     dim=0).to(get_local_torch_device(),
                                               dtype=torch.bfloat16)
            action_labels = torch.stack(action_labels_list,
                                        dim=0).to(get_local_torch_device(),
                                                  dtype=torch.bfloat16)
        else:
            viewmats = None
            intrinsics = None
            action_labels = None

        # Per-frame timesteps: [B, T]
        timesteps = training_batch.timesteps
        assert timesteps.ndim == 2, (
            f"Expected per-frame timesteps [B, T], got shape {timesteps.shape}")

        training_batch.input_kwargs = {
            "hidden_states": training_batch.noisy_model_input,
            "encoder_hidden_states": None,  # No text conditioning for WanGame
            "timestep": timesteps.to(get_local_torch_device(),
                                     dtype=torch.bfloat16),
            "encoder_hidden_states_image": image_embeds,
            "viewmats": viewmats,
            "Ks": intrinsics,
            "action": action_labels,
            "num_frame_per_block": self.num_frame_per_block,
            "return_dict": False,
        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Run transformer forward pass and compute flow-matching loss.
        """
        input_kwargs = training_batch.input_kwargs

        # Forward with causal attention via set_forward_context
        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=None,
                                 forward_batch=None):
            # model_pred: [B, C, T, H, W] (flow prediction)
            model_pred = self.transformer(**input_kwargs)

            # model_pred is [B, C, T, H, W], convert to [B, T, C, H, W]
            model_pred_btchw = model_pred.permute(0, 2, 1, 3, 4)

            # Training target: [B, T, C, H, W]
            training_target = training_batch._ar_training_target.to(
                model_pred_btchw.device, dtype=model_pred_btchw.dtype)

            batch_size, num_frame = model_pred_btchw.shape[:2]

            # Per-frame MSE loss with training weight
            # loss shape before weight: [B, T]
            loss = F.mse_loss(
                model_pred_btchw.float(),
                training_target.float(),
                reduction='none'
            ).mean(dim=(2, 3, 4))  # Average over C, H, W → [B, T]

            # Apply per-timestep training weight from scheduler
            timesteps = training_batch.timesteps  # [B, T]
            weights = self.ar_noise_scheduler.training_weight(
                timesteps.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
            loss = (loss * weights).mean()

            loss = loss / self.training_args.gradient_accumulation_steps
            loss.backward()

        avg_loss = loss.detach().clone()
        training_batch.total_loss += avg_loss.item()

        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Override to use custom AR diffusion training logic."""
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        args = cast(TrainingArgs, self.training_args)

        for _ in range(args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            # Prepare noisy inputs with per-block timesteps + I2V concat
            training_batch = self._prepare_dit_inputs(training_batch)

            # Build transformer input kwargs (action conditioning etc.)
            training_batch = self._build_input_kwargs(training_batch)

            # Forward + loss
            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        # Clip grad and step
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
        training_batch.raw_latent_shape = training_batch.latents.shape
        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch.get(
            'image_path') or validation_batch.get('video_path')
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )
        if "image" in validation_batch and validation_batch["image"] is not None:
            batch.pil_image = validation_batch["image"]

        if "keyboard_cond" in validation_batch and validation_batch[
                "keyboard_cond"] is not None:
            keyboard_cond = validation_batch["keyboard_cond"]
            keyboard_cond = torch.tensor(keyboard_cond, dtype=torch.bfloat16)
            keyboard_cond = keyboard_cond.unsqueeze(0)
            batch.keyboard_cond = keyboard_cond

        if "mouse_cond" in validation_batch and validation_batch[
                "mouse_cond"] is not None:
            mouse_cond = validation_batch["mouse_cond"]
            mouse_cond = torch.tensor(mouse_cond, dtype=torch.bfloat16)
            mouse_cond = mouse_cond.unsqueeze(0)
            batch.mouse_cond = mouse_cond

        return batch

    def _post_process_validation_frames(
            self, frames: list[np.ndarray],
            batch: ForwardBatch) -> list[np.ndarray]:
        """Apply action overlay to validation frames."""
        keyboard_cond = getattr(batch, 'keyboard_cond', None)
        mouse_cond = getattr(batch, 'mouse_cond', None)

        if keyboard_cond is None and mouse_cond is None:
            return frames

        from fastvideo.models.dits.matrixgame.utils import (draw_keys_on_frame,
                                                            draw_mouse_on_frame)

        if keyboard_cond is not None:
            keyboard_cond = keyboard_cond.squeeze(
                0).cpu().float().numpy()
        if mouse_cond is not None:
            mouse_cond = mouse_cond.squeeze(0).cpu().float().numpy()

        key_names = ["W", "S", "A", "D", "left", "right"]

        processed_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = np.ascontiguousarray(frame.copy())

            if keyboard_cond is not None and frame_idx < len(keyboard_cond):
                keys = {
                    key_names[i]: bool(keyboard_cond[frame_idx, i])
                    for i in range(min(len(key_names), keyboard_cond.shape[1]))
                }
                draw_keys_on_frame(frame, keys, mode='universal')

            if mouse_cond is not None and frame_idx < len(mouse_cond):
                pitch = float(mouse_cond[frame_idx, 0])
                yaw = float(mouse_cond[frame_idx, 1])
                draw_mouse_on_frame(frame, pitch, yaw)

            processed_frames.append(frame)

        return processed_frames


def main(args) -> None:
    logger.info("Starting WanGame AR diffusion training pipeline...")

    pipeline = WanGameARDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("WanGame AR diffusion training pipeline done")


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
