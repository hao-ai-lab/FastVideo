# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_wangame
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.wan.wangame_i2v_pipeline import WanGameActionImageToVideoPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.utils import is_vsa_available, shallow_asdict

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanGameTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for WanGame-2.1-Fun-1.3B-InP.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_wangame

    def set_trainable(self) -> None:
        """
        Override to only train newly added action-related parameters:
        - condition_embedder.action_embedder: embeds action into timestep
        - blocks.*.to_out_prope: projects PRoPE attention output
        
        This freezes the base model (q/k/v projections, FFN, etc.) while
        allowing the action-conditioning path to be trained.
        """
        train_action_only = getattr(self.fastvideo_args, "train_action_only", False)
        
        if not train_action_only:
            # Default behavior: train all parameters
            super().set_trainable()
            return
        
        # Freeze all transformer parameters first
        transformer = self.get_module("transformer")
        transformer.train()
        transformer.requires_grad_(False)
        
        # Define which parameter name patterns to train
        action_param_patterns = [
            "condition_embedder.action_embedder",  # Action embedding MLP
            "to_out_prope",  # PRoPE output projections in each block
        ]
        
        # Enable gradients for action-related parameters only
        trainable_count = 0
        frozen_count = 0
        for name, param in transformer.named_parameters():
            should_train = any(pattern in name for pattern in action_param_patterns)
            if should_train:
                param.requires_grad_(True)
                trainable_count += 1
                logger.info(f"Trainable: {name} ({param.numel()} params)")
            else:
                frozen_count += 1
        
        logger.info(f"Action-only training: {trainable_count} trainable param groups, "
                    f"{frozen_count} frozen param groups")

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        # args_copy.pipeline_config.vae_config.load_encoder = False
        # validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
        self.validation_pipeline = WanGameActionImageToVideoPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            dit_cpu_offload=False)

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        # encoder_hidden_states = batch['text_embedding']
        # encoder_attention_mask = batch['text_attention_mask']
        clip_features = batch['clip_feature']
        image_latents = batch['first_frame_latent']
        image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = batch['pil_image']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        # MatrixGame doesn't use text encoder
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

        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""

        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

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

        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, mask_lat_size, image_latents],
            dim=1)

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:

        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)
        encoder_hidden_states_image = image_embeds

        from fastvideo.models.dits.hyworld.pose import process_custom_actions
        
        # Process actions for each batch sample
        batch_size = training_batch.noisy_model_input.shape[0]
        viewmats_list, intrinsics_list, action_labels_list = [], [], []
        for b in range(batch_size):
            v, i, a = process_custom_actions(
                training_batch.keyboard_cond[b], training_batch.mouse_cond[b])
            viewmats_list.append(v)
            intrinsics_list.append(i)
            action_labels_list.append(a)
        viewmats = torch.stack(viewmats_list, dim=0).to(get_local_torch_device(), dtype=torch.bfloat16)
        intrinsics = torch.stack(intrinsics_list, dim=0).to(get_local_torch_device(), dtype=torch.bfloat16)
        action_labels = torch.stack(action_labels_list, dim=0).to(get_local_torch_device(), dtype=torch.bfloat16)

        # NOTE: noisy_model_input already contains concatenated image_latents from _prepare_dit_inputs
        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,  # None for MatrixGame
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            # "encoder_attention_mask":
            # training_batch.encoder_attention_mask,
            "encoder_hidden_states_image":
            encoder_hidden_states_image,
            # Action conditioning
            "viewmats": viewmats,
            "Ks": intrinsics,
            "action": action_labels,
            "return_dict":
            False,
        }
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

    def _post_process_validation_frames(self, frames: list[np.ndarray],
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
        from fastvideo.models.dits.matrixgame.utils import (
            draw_keys_on_frame, draw_mouse_on_frame)
        
        # Convert tensors to numpy if needed (bfloat16 -> float32 -> numpy)
        if keyboard_cond is not None:
            keyboard_cond = keyboard_cond.squeeze(0).cpu().float().numpy()  # (T, 6)
        if mouse_cond is not None:
            mouse_cond = mouse_cond.squeeze(0).cpu().float().numpy()  # (T, 2)
        
        # MatrixGame convention: keyboard [W, S, A, D, left, right], mouse [Pitch, Yaw]
        key_names = ["W", "S", "A", "D", "left", "right"]
        
        processed_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = np.ascontiguousarray(frame.copy())
            
            # Draw keyboard overlay
            if keyboard_cond is not None and frame_idx < len(keyboard_cond):
                keys = {key_names[i]: bool(keyboard_cond[frame_idx, i]) 
                        for i in range(min(len(key_names), keyboard_cond.shape[1]))}
                draw_keys_on_frame(frame, keys, mode='universal')
            
            # Draw mouse overlay
            if mouse_cond is not None and frame_idx < len(mouse_cond):
                pitch = float(mouse_cond[frame_idx, 0])
                yaw = float(mouse_cond[frame_idx, 1])
                draw_mouse_on_frame(frame, pitch, yaw)
            
            processed_frames.append(frame)
        
        return processed_frames


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanGameTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


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