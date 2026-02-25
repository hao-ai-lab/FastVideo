# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_wangame
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.hyworld.pose import process_custom_actions
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.basic.wan.wangame_i2v_pipeline import (
    WanGameActionImageToVideoPipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.training_utils import shift_timestep
from fastvideo.utils import is_vsa_available, shallow_asdict

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class WanGameDistillationPipeline(DistillationPipeline):
    """
    DMD distillation pipeline for WanGame.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize WanGame-specific scheduler."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """May be used in future refactors."""
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_wangame

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        device = get_local_torch_device()
        dtype = torch.bfloat16

        clip_feature = batch['clip_feature']
        first_frame_latent = batch['first_frame_latent']
        keyboard_cond = batch.get('keyboard_cond', None)
        mouse_cond = batch.get('mouse_cond', None)
        infos = batch['info_list']

        if self.training_args.simulate_generator_forward:
            # When simulating, we don't need real VAE latents — just use zeros
            batch_size = clip_feature.shape[0]
            vae_config = self.training_args.pipeline_config.vae_config.arch_config
            num_channels = vae_config.z_dim
            spatial_compression_ratio = vae_config.spatial_compression_ratio

            latent_height = self.training_args.num_height // spatial_compression_ratio
            latent_width = self.training_args.num_width // spatial_compression_ratio

            latents = torch.zeros(
                batch_size,
                num_channels,
                self.training_args.num_latent_t,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        else:
            if 'vae_latent' not in batch:
                raise ValueError(
                    "vae_latent not found in batch and simulate_generator_forward is False. "
                    "Either preprocess data with VAE latents or set --simulate_generator_forward."
                )
            latents = batch['vae_latent']
            latents = latents[:, :, :self.training_args.num_latent_t]
            latents = latents.to(device, dtype=dtype)

        training_batch.latents = latents.to(device, dtype=dtype)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.image_embeds = clip_feature.to(device, dtype=dtype)
        training_batch.image_latents = first_frame_latent.to(device, dtype=dtype)

        # Action conditioning
        if keyboard_cond is not None and keyboard_cond.numel() > 0:
            training_batch.keyboard_cond = keyboard_cond.to(device, dtype=dtype)
        else:
            training_batch.keyboard_cond = None
        if mouse_cond is not None and mouse_cond.numel() > 0:
            training_batch.mouse_cond = mouse_cond.to(device, dtype=dtype)
        else:
            training_batch.mouse_cond = None

        training_batch.infos = infos
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""
        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        training_batch.conditional_dict = {
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
        }
        training_batch.unconditional_dict = None

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        # Build mask + image_latent -> cond_concat (20 channels)
        temporal_compression_ratio = self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (self.training_args.num_latent_t -
                      1) * temporal_compression_ratio + 1
        batch_size, num_channels, _, latent_height, latent_width = image_latents.shape
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

        # cond_concat = [mask(4), image_latent(16)] = 20 channels
        image_latents = torch.cat([mask_lat_size, image_latents], dim=1)

        if self.sp_world_size > 1:
            total_frames = image_latents.shape[2]
            # Split cond latents to local SP shard only when tensor is still global.
            if total_frames == self.training_args.num_latent_t:
                if total_frames % self.sp_world_size != 0:
                    raise ValueError(
                        "image_latents temporal dim is not divisible by SP world size: "
                        f"frames={total_frames}, sp_world_size={self.sp_world_size}"
                    )
                image_latents = rearrange(image_latents,
                                          "b c (n t) h w -> b c n t h w",
                                          n=self.sp_world_size).contiguous()
                image_latents = image_latents[:, :, self.rank_in_sp_group, :, :,
                                              :]

        training_batch.image_latents = image_latents

        return training_batch

    def _build_distill_input_kwargs(
            self, noise_input: torch.Tensor, timestep: torch.Tensor,
            text_dict: dict[str, torch.Tensor] | None,
            training_batch: TrainingBatch) -> TrainingBatch:
        """Build model input with WanGame
        """
        # Image embeds (CLIP features) for cross-attention conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)

        # already prepared in _prepare_dit_inputs
        image_latents = training_batch.image_latents

        # Process action conditioning
        keyboard_cond = training_batch.keyboard_cond
        mouse_cond = training_batch.mouse_cond

        if keyboard_cond is not None and mouse_cond is not None:
            viewmats_list = []
            intrinsics_list = []
            action_labels_list = []
            for b in range(noise_input.shape[0]):
                viewmats, intrinsics, action_labels = process_custom_actions(
                    keyboard_cond[b], mouse_cond[b])
                viewmats_list.append(viewmats)
                intrinsics_list.append(intrinsics)
                action_labels_list.append(action_labels)

            viewmats = torch.stack(viewmats_list, dim=0).to(
                device=get_local_torch_device(), dtype=torch.bfloat16)
            intrinsics = torch.stack(intrinsics_list, dim=0).to(
                device=get_local_torch_device(), dtype=torch.bfloat16)
            action_labels = torch.stack(action_labels_list, dim=0).to(
                device=get_local_torch_device(), dtype=torch.bfloat16)
        else:
            viewmats = None
            intrinsics = None
            action_labels = None

        # I2V concatenation: [noise_input(16ch), image_latents(20ch)] -> 36ch
        noisy_model_input = torch.cat(
            [noise_input, image_latents.permute(0, 2, 1, 3, 4)], dim=2)

        training_batch.input_kwargs = {
            "hidden_states": noisy_model_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": None,
            "timestep": timestep,
            "encoder_hidden_states_image": image_embeds,
            "viewmats": viewmats,
            "Ks": intrinsics,
            "action": action_labels,
            "return_dict": False,
        }
        training_batch.noise_latents = noise_input

        return training_batch

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch: TrainingBatch) -> torch.Tensor:
        """Compute DMD loss for WanGame."""
        original_latent = generator_pred_video
        with torch.no_grad():
            timestep = torch.randint(0,
                                     self.num_train_timestep, [1],
                                     device=self.device,
                                     dtype=torch.long)

            timestep = shift_timestep(timestep, self.timestep_shift,
                                      self.num_train_timestep)

            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            noise = torch.randn(self.video_latent_shape,
                                device=self.device,
                                dtype=generator_pred_video.dtype)

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep).detach().unflatten(0, (1, generator_pred_video.shape[1]))

            # Build input kwargs for critic/teacher
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)

            # fake_score_transformer forward
            current_fake_score_transformer = self._get_fake_score_transformer(
                timestep)
            fake_score_pred_noise = current_fake_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score_transformer forward
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            real_score_pred_noise = current_real_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            real_score_pred_video = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise.shape[:2])

            # No CFG for WanGame - use real_score_pred_video directly
            grad = (faker_score_pred_video - real_score_pred_video) / torch.abs(
                original_latent - real_score_pred_video).mean()
            grad = torch.nan_to_num(grad)

        dmd_loss = 0.5 * F.mse_loss(
            original_latent.float(),
            (original_latent.float() - grad.float()).detach())

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video":
            original_latent.detach(),
            "real_score_pred_video":
            real_score_pred_video.detach(),
            "faker_score_pred_video":
            faker_score_pred_video.detach(),
            "dmd_timestep":
            timestep.detach(),
        })

        return dmd_loss

    def faker_score_forward(
            self, training_batch: TrainingBatch
    ) -> tuple[TrainingBatch, torch.Tensor]:
        """Forward pass for critic training with WanGame action conditioning."""
        with torch.no_grad(), set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata_vsa):
            if self.training_args.simulate_generator_forward:
                generator_pred_video = self._generator_multi_step_simulation_forward(
                    training_batch)
            else:
                generator_pred_video = self._generator_forward(training_batch)

        fake_score_timestep = torch.randint(0,
                                            self.num_train_timestep, [1],
                                            device=self.device,
                                            dtype=torch.long)

        fake_score_timestep = shift_timestep(fake_score_timestep,
                                             self.timestep_shift,
                                             self.num_train_timestep)

        fake_score_timestep = fake_score_timestep.clamp(self.min_timestep,
                                                        self.max_timestep)

        fake_score_noise = torch.randn(self.video_latent_shape,
                                       device=self.device,
                                       dtype=generator_pred_video.dtype)

        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1),
            fake_score_noise.flatten(0, 1), fake_score_timestep).unflatten(
                0, (1, generator_pred_video.shape[1]))

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            training_batch = self._build_distill_input_kwargs(
                noisy_generator_pred_video, fake_score_timestep,
                training_batch.conditional_dict, training_batch)

            current_fake_score_transformer = self._get_fake_score_transformer(
                fake_score_timestep)
            fake_score_pred_noise = current_fake_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generator_pred_video
        flow_matching_loss = torch.mean((fake_score_pred_noise - target)**2)

        training_batch.fake_score_latent_vis_dict = {
            "training_batch_fakerscore_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video": generator_pred_video,
            "fake_score_timestep": fake_score_timestep,
        }

        return training_batch, flow_matching_loss

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True

        validation_pipeline = WanGameActionImageToVideoPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "vae": self.get_module("vae"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline

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
            if isinstance(keyboard_cond, torch.Tensor):
                keyboard_cond = keyboard_cond.detach().clone().to(dtype=torch.bfloat16)
            else:
                keyboard_cond = torch.tensor(keyboard_cond, dtype=torch.bfloat16)
            keyboard_cond = keyboard_cond.unsqueeze(0)
            batch.keyboard_cond = keyboard_cond[:num_frames]

        if "mouse_cond" in validation_batch and validation_batch[
                "mouse_cond"] is not None:
            mouse_cond = validation_batch["mouse_cond"]
            if isinstance(mouse_cond, torch.Tensor):
                mouse_cond = mouse_cond.detach().clone().to(dtype=torch.bfloat16)
            else:
                mouse_cond = torch.tensor(mouse_cond, dtype=torch.bfloat16)
            mouse_cond = mouse_cond.unsqueeze(0)
            batch.mouse_cond = mouse_cond[:num_frames]

        return batch

    def _post_process_validation_frames(
            self, frames: list[np.ndarray],
            batch: ForwardBatch) -> list[np.ndarray]:
        """Apply action overlay to validation frames for WanGame.
        
        Draws keyboard (WASD) and mouse (pitch/yaw) indicators on the video frames.
        """
        # Check if action data is available
        keyboard_cond = getattr(batch, 'keyboard_cond', None)
        mouse_cond = getattr(batch, 'mouse_cond', None)

        if keyboard_cond is None and mouse_cond is None:
            return frames

        # Import overlay functions
        from fastvideo.models.dits.matrixgame.utils import (draw_keys_on_frame,
                                                            draw_mouse_on_frame)

        # Convert tensors to numpy if needed (bfloat16 -> float32 -> numpy)
        if keyboard_cond is not None:
            keyboard_cond = keyboard_cond.squeeze(
                0).cpu().float().numpy()  # (T, 6)
        if mouse_cond is not None:
            mouse_cond = mouse_cond.squeeze(0).cpu().float().numpy()  # (T, 2)

        # WanGame convention: keyboard [W, S, A, D, left, right], mouse [Pitch, Yaw]
        key_names = ["W", "S", "A", "D", "left", "right"]

        processed_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = np.ascontiguousarray(frame.copy())

            # Draw keyboard overlay
            if keyboard_cond is not None and frame_idx < len(keyboard_cond):
                keys = {
                    key_names[i]: bool(keyboard_cond[frame_idx, i])
                    for i in range(min(len(key_names), keyboard_cond.shape[1]))
                }
                draw_keys_on_frame(frame, keys, mode='universal')

            # Draw mouse overlay
            if mouse_cond is not None and frame_idx < len(mouse_cond):
                pitch = float(mouse_cond[frame_idx, 0])
                yaw = float(mouse_cond[frame_idx, 1])
                draw_mouse_on_frame(frame, pitch, yaw)

            processed_frames.append(frame)

        return processed_frames


def main(args) -> None:
    logger.info("Starting WanGame DMD distillation pipeline...")

    pipeline = WanGameDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    pipeline.train()
    logger.info("WanGame DMD distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
