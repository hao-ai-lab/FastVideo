# SPDX-License-Identifier: Apache-2.0
import copy
import gc
import json
import os
import time
from abc import abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.validation_dataset import ValidationDataset
from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import (ComposedPipelineBase, ForwardBatch,
                                 TrainingBatch)
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    EMA_FSDP, clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler, load_distillation_checkpoint, save_distillation_checkpoint,
    shift_timestep)
from fastvideo.utils import (is_vsa_available, maybe_download_model,
                             set_random_seed, verify_model_config_and_directory)

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class DistillationPipeline(TrainingPipeline):
    """
    A distillation pipeline for training a 3 step model.
    Inherits from TrainingPipeline to reuse training infrastructure.
    """
    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
    ]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0
    init_steps: int
    current_trainstep: int
    video_latent_shape: tuple[int, ...]
    video_latent_shape_sp: tuple[int, ...]
    train_fake_score_transformer_2: bool = False

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the distillation training pipeline with multiple models."""
        logger.info("Initializing distillation pipeline...")

        super().initialize_training_pipeline(training_args)

        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=self.timestep_shift)

        if self.training_args.boundary_ratio is not None:
            self.boundary_timestep = self.training_args.boundary_ratio * self.noise_scheduler.num_train_timesteps
        else:
            self.boundary_timestep = None

        if training_args.real_score_model_path:
            logger.info("Loading real score transformer from: %s",
                        training_args.real_score_model_path)
            training_args.override_transformer_cls_name = "WanTransformer3DModel"
            self.real_score_transformer = self.load_module_from_path(
                training_args.real_score_model_path, "transformer",
                training_args)
            try:
                self.real_score_transformer_2 = self.load_module_from_path(
                    training_args.real_score_model_path, "transformer_2",
                    training_args)
                logger.info("Loaded real score transformer_2 for MoE support")
            except Exception:
                logger.info(
                    "real score transformer_2 not found, using single transformer"
                )
                self.real_score_transformer_2 = None
        else:
            self.real_score_transformer = self.get_module(
                "real_score_transformer")
            self.real_score_transformer_2 = self.get_module(
                "real_score_transformer_2")

        if training_args.fake_score_model_path:
            logger.info("Loading fake score transformer from: %s",
                        training_args.fake_score_model_path)
            training_args.override_transformer_cls_name = "WanTransformer3DModel"
            self.fake_score_transformer = self.load_module_from_path(
                training_args.fake_score_model_path, "transformer",
                training_args)
            try:
                self.fake_score_transformer_2 = self.load_module_from_path(
                    training_args.fake_score_model_path, "transformer_2",
                    training_args)
                logger.info("Loaded fake score transformer_2 for MoE support")
            except Exception:
                logger.info(
                    "fake score transformer_2 not found, using single transformer"
                )
                self.fake_score_transformer_2 = None
        else:
            self.fake_score_transformer = self.get_module(
                "fake_score_transformer")
            self.fake_score_transformer_2 = self.get_module(
                "fake_score_transformer_2")

        self.real_score_transformer.requires_grad_(False)
        self.real_score_transformer.eval()
        if self.real_score_transformer_2 is not None:
            self.real_score_transformer_2.requires_grad_(False)
            self.real_score_transformer_2.eval()

        # Set training modes for fake score transformers (trainable)
        self.fake_score_transformer.requires_grad_(True)
        self.fake_score_transformer.train()
        if self.fake_score_transformer_2 is not None:
            self.fake_score_transformer_2.requires_grad_(True)
            self.fake_score_transformer_2.train()

        if training_args.enable_gradient_checkpointing_type is not None:
            self.fake_score_transformer = apply_activation_checkpointing(
                self.fake_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            if self.fake_score_transformer_2 is not None:
                self.fake_score_transformer_2 = apply_activation_checkpointing(
                    self.fake_score_transformer_2,
                    checkpointing_type=training_args.
                    enable_gradient_checkpointing_type)

            self.real_score_transformer = apply_activation_checkpointing(
                self.real_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            if self.real_score_transformer_2 is not None:
                self.real_score_transformer_2 = apply_activation_checkpointing(
                    self.real_score_transformer_2,
                    checkpointing_type=training_args.
                    enable_gradient_checkpointing_type)

        # Initialize optimizers
        fake_score_params = list(
            filter(lambda p: p.requires_grad,
                   self.fake_score_transformer.parameters()))

        # Use separate learning rate for fake_score_transformer if specified
        fake_score_lr = training_args.fake_score_learning_rate
        if fake_score_lr == 0.0:
            fake_score_lr = training_args.learning_rate

        betas_str = training_args.fake_score_betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))

        self.fake_score_optimizer = torch.optim.AdamW(
            fake_score_params,
            lr=fake_score_lr,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.fake_score_lr_scheduler = get_scheduler(
            training_args.fake_score_lr_scheduler,
            optimizer=self.fake_score_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        if self.fake_score_transformer_2 is not None:
            fake_score_params_2 = list(
                filter(lambda p: p.requires_grad,
                       self.fake_score_transformer_2.parameters()))
            self.fake_score_optimizer_2 = torch.optim.AdamW(
                fake_score_params_2,
                lr=fake_score_lr,
                betas=betas,
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )
            self.fake_score_lr_scheduler_2 = get_scheduler(
                training_args.fake_score_lr_scheduler,
                optimizer=self.fake_score_optimizer_2,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )

        logger.info(
            "Distillation optimizers initialized: generator and fake_score")

        self.generator_update_interval = self.training_args.generator_update_interval
        logger.info(
            "Distillation pipeline initialized with generator_update_interval=%s",
            self.generator_update_interval)

        self.denoising_step_list = torch.tensor(
            self.training_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device())

        if training_args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.noise_scheduler.timesteps.cpu(),
                                   torch.tensor([0],
                                                dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 -
                                                 self.denoising_step_list]
            logger.info("Warping denoising_step_list")

        self.denoising_step_list = self.denoising_step_list.to(
            get_local_torch_device())
        logger.info("Distillation generator model to %s denoising steps: %s",
                    len(self.denoising_step_list), self.denoising_step_list)
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps

        self.min_timestep = int(self.training_args.min_timestep_ratio *
                                self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio *
                                self.num_train_timestep)

        self.real_score_guidance_scale = self.training_args.real_score_guidance_scale

        self.generator_ema: EMA_FSDP | None = None
        self.generator_ema_2: EMA_FSDP | None = None
        if (self.training_args.ema_decay
                is not None) and (self.training_args.ema_decay > 0.0):
            self.generator_ema = EMA_FSDP(self.transformer,
                                          decay=self.training_args.ema_decay)
            logger.info("Initialized generator EMA with decay=%s",
                        self.training_args.ema_decay)

            # Initialize EMA for transformer_2 if it exists
            if self.transformer_2 is not None:
                self.generator_ema_2 = EMA_FSDP(
                    self.transformer_2, decay=self.training_args.ema_decay)
                logger.info("Initialized generator EMA_2 with decay=%s",
                            self.training_args.ema_decay)
        else:
            logger.info("Generator EMA disabled (ema_decay <= 0.0)")

    def load_module_from_path(self, model_path: str, module_type: str,
                              training_args: "TrainingArgs"):
        """
        Load a module from a specific path using the same loading logic as the pipeline.
        
        Args:
            model_path: Path to the model
            module_type: Type of module to load (e.g., "transformer")
            training_args: Training arguments
            
        Returns:
            The loaded module
        """
        logger.info("Loading %s from custom path: %s", module_type, model_path)
        # Set flag to prevent custom weight loading for teacher/critic models
        training_args._loading_teacher_critic_model = True

        try:
            from fastvideo.models.loader.component_loader import (
                PipelineComponentLoader)

            # Download the model if it's a Hugging Face model ID
            local_model_path = maybe_download_model(model_path)
            logger.info("Model downloaded/found at: %s", local_model_path)
            config = verify_model_config_and_directory(local_model_path)

            if module_type not in config:
                if hasattr(self, '_extra_config_module_map'
                           ) and module_type in self._extra_config_module_map:
                    extra_module = self._extra_config_module_map[module_type]
                    if extra_module in config:
                        module_type = extra_module
                        logger.info("Using %s for %s", extra_module,
                                    module_type)
                    else:
                        raise ValueError(
                            f"Module {module_type} not found in config at {local_model_path}"
                        )
                else:
                    raise ValueError(
                        f"Module {module_type} not found in config at {local_model_path}"
                    )

            module_info = config[module_type]
            if module_info is None:
                raise ValueError(
                    f"Module {module_type} has null value in config at {local_model_path}"
                )

            transformers_or_diffusers, architecture = module_info
            component_path = os.path.join(local_model_path, module_type)
            module = PipelineComponentLoader.load_module(
                module_name=module_type,
                component_model_path=component_path,
                transformers_or_diffusers=transformers_or_diffusers,
                fastvideo_args=training_args,
            )

            logger.info("Successfully loaded %s from %s", module_type,
                        component_path)
            return module
        finally:
            # Always clean up the flag
            if hasattr(training_args, '_loading_teacher_critic_model'):
                delattr(training_args, '_loading_teacher_critic_model')

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method")

    def _prepare_distillation(self,
                              training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        self.transformer.requires_grad_(True)
        self.transformer.train()
        if self.transformer_2 is not None:
            self.transformer_2.requires_grad_(True)
            self.transformer_2.train()
        self.fake_score_transformer.requires_grad_(True)
        self.fake_score_transformer.train()
        if self.fake_score_transformer_2 is not None:
            self.fake_score_transformer_2.requires_grad_(True)
            self.fake_score_transformer_2.train()

        return training_batch

    def apply_ema_to_model(self, model):
        """Apply EMA weights to the model for validation or inference."""
        if model is self.transformer and self.generator_ema is not None:
            with self.generator_ema.apply_to_model(model):
                return model
        elif model is self.transformer_2 and self.generator_ema_2 is not None:
            with self.generator_ema_2.apply_to_model(model):
                return model
        return model

    def get_ema_model_copy(self) -> torch.nn.Module | None:
        """Get a copy of the model with EMA weights applied."""
        if self.generator_ema is not None:
            ema_model = copy.deepcopy(self.transformer)
            self.generator_ema.copy_to_unwrapped(ema_model)
            return ema_model
        return None

    def get_ema_2_model_copy(self) -> torch.nn.Module | None:
        """Get a copy of the transformer_2 model with EMA weights applied."""
        if self.generator_ema_2 is not None and self.transformer_2 is not None:
            ema_2_model = copy.deepcopy(self.transformer_2)
            self.generator_ema_2.copy_to_unwrapped(ema_2_model)
            return ema_2_model
        return None

    def is_ema_ready(self, current_step: int | None = None):
        """Check if EMA is ready for use (after ema_start_step)."""
        if current_step is None:
            current_step = getattr(self, 'current_trainstep', 0)
        return (self.generator_ema is not None
                and current_step >= self.training_args.ema_start_step)

    def save_ema_weights(self, output_dir: str, step: int):
        """Save EMA weights separately for inference purposes."""
        if self.generator_ema is None and self.generator_ema_2 is None:
            logger.warning("Cannot save EMA weights: No EMA initialized")
            return

        if not self.is_ema_ready():
            logger.warning(
                "Cannot save EMA weights: EMA not ready yet (step < ema_start_step)"
            )
            return

        try:
            # Save main transformer EMA
            if self.generator_ema is not None:
                ema_model = self.get_ema_model_copy()
                if ema_model is None:
                    logger.warning("Failed to create EMA model copy")
                else:
                    ema_save_dir = os.path.join(output_dir,
                                                f"ema_checkpoint-{step}")
                    os.makedirs(ema_save_dir, exist_ok=True)

                    # save as diffusers format
                    from safetensors.torch import save_file

                    from fastvideo.training.training_utils import (
                        custom_to_hf_state_dict, gather_state_dict_on_cpu_rank0)
                    cpu_state = gather_state_dict_on_cpu_rank0(ema_model,
                                                               device=None)

                    if self.global_rank == 0:
                        weight_path = os.path.join(
                            ema_save_dir, "diffusion_pytorch_model.safetensors")
                        diffusers_state_dict = custom_to_hf_state_dict(
                            cpu_state, ema_model.reverse_param_names_mapping)
                        save_file(diffusers_state_dict, weight_path)

                        config_dict = ema_model.hf_config
                        if "dtype" in config_dict:
                            del config_dict["dtype"]
                        config_path = os.path.join(ema_save_dir, "config.json")
                        with open(config_path, "w") as f:
                            json.dump(config_dict, f, indent=4)

                        logger.info("EMA weights saved to %s", weight_path)

                    del ema_model

            # Save transformer_2 EMA
            if self.generator_ema_2 is not None:
                ema_2_model = self.get_ema_2_model_copy()
                if ema_2_model is None:
                    logger.warning("Failed to create EMA_2 model copy")
                else:
                    ema_2_save_dir = os.path.join(output_dir,
                                                  f"ema_2_checkpoint-{step}")
                    os.makedirs(ema_2_save_dir, exist_ok=True)

                    # save as diffusers format
                    from safetensors.torch import save_file

                    from fastvideo.training.training_utils import (
                        custom_to_hf_state_dict, gather_state_dict_on_cpu_rank0)
                    cpu_state_2 = gather_state_dict_on_cpu_rank0(ema_2_model,
                                                                 device=None)

                    if self.global_rank == 0:
                        weight_path_2 = os.path.join(
                            ema_2_save_dir,
                            "diffusion_pytorch_model.safetensors")
                        diffusers_state_dict_2 = custom_to_hf_state_dict(
                            cpu_state_2,
                            ema_2_model.reverse_param_names_mapping)
                        save_file(diffusers_state_dict_2, weight_path_2)

                        config_dict_2 = ema_2_model.hf_config
                        if "dtype" in config_dict_2:
                            del config_dict_2["dtype"]
                        config_path_2 = os.path.join(ema_2_save_dir,
                                                     "config.json")
                        with open(config_path_2, "w") as f:
                            json.dump(config_dict_2, f, indent=4)

                        logger.info("EMA_2 weights saved to %s", weight_path_2)

                    del ema_2_model

        except Exception as e:
            logger.error("Failed to save EMA weights: %s", str(e))

    def get_ema_stats(self) -> dict[str, Any]:
        """Get EMA statistics for monitoring."""
        ema_enabled = self.generator_ema is not None
        ema_2_enabled = self.generator_ema_2 is not None

        if not ema_enabled and not ema_2_enabled:
            return {
                "ema_enabled": False,
                "ema_2_enabled": False,
                "ema_decay": None,
                "ema_start_step": self.training_args.ema_start_step,
                "ema_ready": False,
                "ema_2_ready": False,
                "ema_step": self.current_trainstep,
            }

        return {
            "ema_enabled": ema_enabled,
            "ema_2_enabled": ema_2_enabled,
            "ema_decay": self.training_args.ema_decay,
            "ema_start_step": self.training_args.ema_start_step,
            "ema_ready": self.is_ema_ready() if ema_enabled else False,
            "ema_2_ready": self.is_ema_ready() if ema_2_enabled else False,
            "ema_step": self.current_trainstep,
        }

    def reset_ema(self):
        """Reset EMA to current model weights."""
        if self.generator_ema is not None:
            logger.info("Resetting EMA to current model weights")
            self.generator_ema.update(self.transformer)
            # Force update to current weights by setting decay to 0 temporarily
            original_decay = self.generator_ema.decay
            self.generator_ema.decay = 0.0
            self.generator_ema.update(self.transformer)
            self.generator_ema.decay = original_decay
            logger.info("EMA reset completed")
        else:
            logger.warning("Cannot reset EMA: EMA not initialized")

        if self.generator_ema_2 is not None:
            logger.info("Resetting EMA_2 to current model weights")
            self.generator_ema_2.update(self.transformer_2)
            # Force update to current weights by setting decay to 0 temporarily
            original_decay_2 = self.generator_ema_2.decay
            self.generator_ema_2.decay = 0.0
            self.generator_ema_2.update(self.transformer_2)
            self.generator_ema_2.decay = original_decay_2
            logger.info("EMA_2 reset completed")

    def _get_real_score_transformer(self, timestep: torch.Tensor):
        """
        Get the appropriate real score transformer based on timestep and boundary logic.
        """
        if self.real_score_transformer_2 is not None and self.boundary_timestep is not None:
            if timestep.item() < self.boundary_timestep:
                return self.real_score_transformer_2  # Low noise expert
            else:
                return self.real_score_transformer  # High noise expert
        else:
            return self.real_score_transformer

    def _get_fake_score_transformer(self, timestep: torch.Tensor):
        """
        Get the appropriate fake score transformer based on timestep and boundary logic.
        """
        if self.fake_score_transformer_2 is not None and self.boundary_timestep is not None:
            if timestep.item() < self.boundary_timestep:
                self.train_fake_score_transformer_2 = True
                return self.fake_score_transformer_2  # Low noise expert
            else:
                self.train_fake_score_transformer_2 = False
                return self.fake_score_transformer  # High noise expert
        else:
            self.train_fake_score_transformer_2 = False
            return self.fake_score_transformer

    def _build_distill_input_kwargs(
            self, noise_input: torch.Tensor, timestep: torch.Tensor,
            text_dict: dict[str, torch.Tensor] | None,
            training_batch: TrainingBatch) -> TrainingBatch:
        if text_dict is None:
            raise ValueError(
                "text_dict cannot be None for distillation pipeline")

        training_batch.input_kwargs = {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep,
            "return_dict": False,
        }

        return training_batch

    def _generator_forward(self, training_batch: TrainingBatch) -> torch.Tensor:

        latents = training_batch.latents
        dtype = latents.dtype
        index = torch.randint(0,
                              len(self.denoising_step_list), [1],
                              device=self.device,
                              dtype=torch.long)
        timestep = self.denoising_step_list[index]
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep

        noise = torch.randn(self.video_latent_shape,
                            device=self.device,
                            dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise,
                              "b (n t) c h w -> b n t c h w",
                              n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]
        noisy_latent = self.noise_scheduler.add_noise(latents.flatten(0, 1),
                                                      noise.flatten(0, 1),
                                                      timestep).unflatten(
                                                          0,
                                                          (1, latents.shape[1]))

        training_batch = self._build_distill_input_kwargs(
            noisy_latent, timestep, training_batch.conditional_dict,
            training_batch)

        pred_noise = self.transformer(**training_batch.input_kwargs).permute(
            0, 2, 1, 3, 4)
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latent.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler).unflatten(0, pred_noise.shape[:2])

        return pred_video

    def _generator_multi_step_simulation_forward(
            self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student transformer matching inference procedure."""
        latents = training_batch.latents
        dtype = latents.dtype

        # Step 1: Randomly sample a target timestep index from denoising_step_list
        target_timestep_idx = torch.randint(0,
                                            len(self.denoising_step_list), [1],
                                            device=self.device,
                                            dtype=torch.long)
        target_timestep_idx_int = target_timestep_idx.item()
        target_timestep = self.denoising_step_list[target_timestep_idx]

        # Step 2: Simulate the multi-step inference process up to the target timestep
        # Start from pure noise like in inference
        current_noise_latents = torch.randn(self.video_latent_shape,
                                            device=self.device,
                                            dtype=dtype)
        if self.sp_world_size > 1:
            current_noise_latents = rearrange(
                current_noise_latents,
                "b (n t) c h w -> b n t c h w",
                n=self.sp_world_size).contiguous()
            current_noise_latents = current_noise_latents[:, self.
                                                          rank_in_sp_group, :, :, :, :]
        current_noise_latents_copy = current_noise_latents.clone()

        # Only run intermediate steps if target_timestep_idx > 0
        max_target_idx = len(self.denoising_step_list) - 1
        noise_latents = []
        noise_latent_index = target_timestep_idx_int - 1
        if max_target_idx > 0:
            # Run student model for all steps before the target timestep
            with torch.no_grad():
                for step_idx in range(max_target_idx):
                    current_timestep = self.denoising_step_list[step_idx]
                    current_timestep_tensor = current_timestep * torch.ones(
                        1, device=self.device, dtype=torch.long)
                    # Run student model to get flow prediction
                    training_batch_temp = self._build_distill_input_kwargs(
                        current_noise_latents, current_timestep_tensor,
                        training_batch.conditional_dict, training_batch)
                    pred_flow = self.transformer(
                        **training_batch_temp.input_kwargs).permute(
                            0, 2, 1, 3, 4)
                    pred_clean = pred_noise_to_pred_video(
                        pred_noise=pred_flow.flatten(0, 1),
                        noise_input_latent=current_noise_latents.flatten(0, 1),
                        timestep=current_timestep_tensor,
                        scheduler=self.noise_scheduler).unflatten(
                            0, pred_flow.shape[:2])

                    # Add noise for the next timestep
                    next_timestep = self.denoising_step_list[step_idx + 1]
                    next_timestep_tensor = next_timestep * torch.ones(
                        1, device=self.device, dtype=torch.long)
                    noise = torch.randn(self.video_latent_shape,
                                        device=self.device,
                                        dtype=pred_clean.dtype)
                    if self.sp_world_size > 1:
                        noise = rearrange(noise,
                                          "b (n t) c h w -> b n t c h w",
                                          n=self.sp_world_size).contiguous()
                        noise = noise[:, self.rank_in_sp_group, :, :, :, :]
                    current_noise_latents = self.noise_scheduler.add_noise(
                        pred_clean.flatten(0, 1), noise.flatten(0, 1),
                        next_timestep_tensor).unflatten(0, pred_clean.shape[:2])
                    latent_copy = current_noise_latents.clone()
                    noise_latents.append(latent_copy)

        # Step 3: Use the simulated noisy input for the final training step
        # For timestep index 0, this is pure noise
        # For timestep index k > 0, this is the result after k denoising steps + noise at target level
        if noise_latent_index >= 0:
            assert noise_latent_index < len(
                self.denoising_step_list
            ) - 1, "noise_latent_index is out of bounds"
            noisy_input = noise_latents[noise_latent_index]
        else:
            noisy_input = current_noise_latents_copy

        # Step 4: Final student prediction (this is what we train on)
        training_batch = self._build_distill_input_kwargs(
            noisy_input, target_timestep, training_batch.conditional_dict,
            training_batch)
        pred_noise = self.transformer(**training_batch.input_kwargs).permute(
            0, 2, 1, 3, 4)
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_input.flatten(0, 1),
            timestep=target_timestep,
            scheduler=self.noise_scheduler).unflatten(0, pred_noise.shape[:2])
        training_batch.dmd_latent_vis_dict[
            "generator_timestep"] = target_timestep.float().detach()
        return pred_video

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch: TrainingBatch) -> torch.Tensor:
        """Compute DMD (Diffusion Model Distillation) loss."""
        original_latent = generator_pred_video
        with torch.no_grad():
            timestep = torch.randint(0,
                                     self.num_train_timestep, [1],
                                     device=self.device,
                                     dtype=torch.long)
            world_group = get_world_group()
            if world_group.world_size > 1:
                world_group.broadcast(timestep, src=0)

            timestep = shift_timestep(
                timestep,
                self.timestep_shift,  # type: ignore
                self.num_train_timestep)

            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            noise = torch.randn(self.video_latent_shape,
                                device=self.device,
                                dtype=generator_pred_video.dtype)
            if self.sp_world_size > 1:
                noise = rearrange(noise,
                                  "b (n t) c h w -> b n t c h w",
                                  n=self.sp_world_size).contiguous()
                noise = noise[:, self.rank_in_sp_group, :, :, :, :]

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep).detach().unflatten(0,
                                             (1, generator_pred_video.shape[1]))

            # fake_score_transformer forward
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
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

            # real_score_transformer cond forward
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            real_score_pred_noise_cond = current_real_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            pred_real_video_cond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_cond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_cond.shape[:2])

            # real_score_transformer uncond forward
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.unconditional_dict,
                training_batch)
            # Use same transformer as conditional forward for consistency
            real_score_pred_noise_uncond = current_real_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            pred_real_video_uncond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_uncond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_uncond.shape[:2])

            real_score_pred_video = pred_real_video_cond + (
                pred_real_video_cond -
                pred_real_video_uncond) * self.real_score_guidance_scale

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
        world_group = get_world_group()
        if world_group.world_size > 1:
            world_group.broadcast(fake_score_timestep, src=0)

        fake_score_timestep = shift_timestep(
            fake_score_timestep,
            self.timestep_shift,  # type: ignore
            self.num_train_timestep)

        fake_score_timestep = fake_score_timestep.clamp(self.min_timestep,
                                                        self.max_timestep)

        fake_score_noise = torch.randn(self.video_latent_shape,
                                       device=self.device,
                                       dtype=generator_pred_video.dtype)
        if self.sp_world_size > 1:
            fake_score_noise = rearrange(fake_score_noise,
                                         "b (n t) c h w -> b n t c h w",
                                         n=self.sp_world_size).contiguous()
            fake_score_noise = fake_score_noise[:, self.
                                                rank_in_sp_group, :, :, :, :]

        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1), fake_score_noise.flatten(0, 1),
            fake_score_timestep).unflatten(0,
                                           (1, generator_pred_video.shape[1]))

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

    def _clip_model_grad_norm_(self, training_batch: TrainingBatch,
                               transformer) -> TrainingBatch:

        max_grad_norm = self.training_args.max_grad_norm

        if max_grad_norm is not None:
            model_parts = [transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float('nan') or grad_norm is not float(
                'inf')
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        super()._prepare_dit_inputs(training_batch)
        conditional_dict = {
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "encoder_attention_mask": training_batch.encoder_attention_mask,
        }
        if getattr(self, "negative_prompt_embeds", None) is not None:
            unconditional_dict = {
                "encoder_hidden_states": self.negative_prompt_embeds,
                "encoder_attention_mask": self.negative_prompt_attention_mask,
            }
            training_batch.unconditional_dict = unconditional_dict

        training_batch.dmd_latent_vis_dict = {}
        training_batch.fake_score_latent_vis_dict = {}

        training_batch.conditional_dict = conditional_dict
        training_batch.raw_latent_shape = training_batch.latents.shape
        training_batch.latents = training_batch.latents.permute(0, 2, 1, 3, 4)
        self.video_latent_shape = training_batch.latents.shape

        if self.sp_world_size > 1:
            training_batch.latents = rearrange(
                training_batch.latents,
                "b (n t) c h w -> b n t c h w",
                n=self.sp_world_size).contiguous()
            training_batch.latents = training_batch.latents[:, self.
                                                            rank_in_sp_group, :, :, :, :]

        self.video_latent_shape_sp = training_batch.latents.shape

        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        gradient_accumulation_steps = getattr(self.training_args,
                                              'gradient_accumulation_steps', 1)
        batches = []
        # Collect N batches for gradient accumulation
        for _ in range(gradient_accumulation_steps):
            batch = self._prepare_distillation(training_batch)
            batch = self._get_next_batch(batch)
            batch = self._normalize_dit_input(batch)
            batch = self._prepare_dit_inputs(batch)
            batch = self._build_attention_metadata(batch)
            batch.attn_metadata_vsa = copy.deepcopy(batch.attn_metadata)
            if batch.attn_metadata is not None:
                batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore
            batches.append(batch)

        self.optimizer.zero_grad()
        total_dmd_loss = 0.0
        dmd_latent_vis_dict = {}
        fake_score_latent_vis_dict = {}
        if (self.current_trainstep % self.generator_update_interval == 0):
            for batch in batches:
                batch_gen = copy.deepcopy(batch)

                with set_forward_context(
                        current_timestep=batch_gen.timesteps,
                        attn_metadata=batch_gen.attn_metadata_vsa):
                    if self.training_args.simulate_generator_forward:
                        generator_pred_video = self._generator_multi_step_simulation_forward(
                            batch_gen)
                    else:
                        generator_pred_video = self._generator_forward(
                            batch_gen)

                with set_forward_context(current_timestep=batch_gen.timesteps,
                                         attn_metadata=batch_gen.attn_metadata):
                    dmd_loss = self._dmd_forward(
                        generator_pred_video=generator_pred_video,
                        training_batch=batch_gen)

                with set_forward_context(
                        current_timestep=batch_gen.timesteps,
                        attn_metadata=batch_gen.attn_metadata_vsa):
                    (dmd_loss / gradient_accumulation_steps).backward()
                total_dmd_loss += dmd_loss.detach().item()

            # Only clip gradients for the model that is currently training
            self._clip_model_grad_norm_(batch_gen, self.transformer)
            for param in self.transformer.parameters():
                # check if the gradient is not None and not zero
                assert param.grad is not None and param.grad.abs().sum() > 0
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.generator_ema is not None:
                self.generator_ema.update(self.transformer)
            if self.generator_ema_2 is not None:
                self.generator_ema_2.update(self.transformer_2)

            avg_dmd_loss = torch.tensor(total_dmd_loss /
                                        gradient_accumulation_steps,
                                        device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_dmd_loss,
                                   op=torch.distributed.ReduceOp.AVG)
            training_batch.generator_loss = avg_dmd_loss.item()
            dmd_latent_vis_dict = batch_gen.dmd_latent_vis_dict
        else:
            training_batch.generator_loss = 0.0

        self.fake_score_optimizer.zero_grad()
        if self.fake_score_transformer_2 is not None:
            self.fake_score_optimizer_2.zero_grad()
        total_fake_score_loss = 0.0
        for batch in batches:
            batch_fake = copy.deepcopy(batch)
            batch_fake, fake_score_loss = self.faker_score_forward(batch_fake)
            with set_forward_context(current_timestep=batch_fake.timesteps,
                                     attn_metadata=batch_fake.attn_metadata):
                (fake_score_loss / gradient_accumulation_steps).backward()
            total_fake_score_loss += fake_score_loss.detach().item()
            fake_score_latent_vis_dict.update(
                batch_fake.fake_score_latent_vis_dict)
        if self.train_fake_score_transformer_2 and self.fake_score_transformer_2 is not None:
            self._clip_model_grad_norm_(batch_fake,
                                        self.fake_score_transformer_2)
        else:
            self._clip_model_grad_norm_(batch_fake, self.fake_score_transformer)

        # Check gradients for fake score transformer
        for param in self.fake_score_transformer.parameters():
            if param.requires_grad:
                assert param.grad is not None and param.grad.abs().sum() > 0

        # Check gradients for fake score transformer_2 if available
        if self.train_fake_score_transformer_2 and self.fake_score_transformer_2 is not None:
            for param in self.fake_score_transformer_2.parameters():
                if param.requires_grad:
                    assert param.grad is not None and param.grad.abs().sum() > 0

        if self.train_fake_score_transformer_2 and self.fake_score_transformer_2 is not None:
            self.fake_score_optimizer_2.step()
            self.fake_score_lr_scheduler_2.step()
        else:
            self.fake_score_optimizer.step()
            self.fake_score_lr_scheduler.step()

        # Step the appropriate scheduler
        self.lr_scheduler.step()

        self.fake_score_optimizer.zero_grad(set_to_none=True)
        if self.fake_score_transformer_2 is not None:
            self.fake_score_optimizer_2.zero_grad(set_to_none=True)
        avg_fake_score_loss = torch.tensor(total_fake_score_loss /
                                           gradient_accumulation_steps,
                                           device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_fake_score_loss,
                               op=torch.distributed.ReduceOp.AVG)
        training_batch.fake_score_loss = avg_fake_score_loss.item()
        training_batch.dmd_latent_vis_dict = dmd_latent_vis_dict
        training_batch.fake_score_latent_vis_dict = batch_fake.fake_score_latent_vis_dict

        training_batch.total_loss = training_batch.generator_loss + training_batch.fake_score_loss
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint with distillation models."""

        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)

        resumed_step = load_distillation_checkpoint(
            self.transformer,
            self.fake_score_transformer,
            self.global_rank,
            self.training_args.resume_from_checkpoint,
            self.optimizer,
            self.fake_score_optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.fake_score_lr_scheduler,
            self.noise_random_generator,
            self.generator_ema,
            # MoE support
            generator_transformer_2=getattr(self, 'transformer_2', None),
            real_score_transformer_2=getattr(self, 'real_score_transformer_2',
                                             None),
            fake_score_transformer_2=getattr(self, 'fake_score_transformer_2',
                                             None),
            generator_optimizer_2=getattr(self, 'optimizer_2', None),
            fake_score_optimizer_2=getattr(self, 'fake_score_optimizer_2',
                                           None),
            generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
            fake_score_scheduler_2=getattr(self, 'fake_score_lr_scheduler_2',
                                           None),
            generator_ema_2=getattr(self, 'generator_ema_2', None))

        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = -1

    def _log_training_info(self) -> None:
        """Log distillation-specific training information."""
        # First call parent class method to get basic training info
        super()._log_training_info()

        # Then add distillation-specific information
        logger.info("Distillation-specific settings:")
        logger.info("  Generator update ratio: %s",
                    self.generator_update_interval)
        assert isinstance(self.training_args, TrainingArgs)
        logger.info("  Max gradient norm: %s", self.training_args.max_grad_norm)

        logger.info(
            "  Real score transformer (high noise expert) parameters: %s B",
            sum(p.numel()
                for p in self.real_score_transformer.parameters()) / 1e9)

        if self.real_score_transformer_2 is not None:
            logger.info(
                "  Real score transformer_2 (low noise expert) parameters: %s B",
                sum(p.numel()
                    for p in self.real_score_transformer_2.parameters()) / 1e9)
            logger.info("  Real score MoE enabled with boundary_timestep: %s",
                        self.boundary_timestep)

        logger.info(
            "  Fake score transformer (high noise expert) parameters: %s B",
            sum(p.numel()
                for p in self.fake_score_transformer.parameters()) / 1e9)

        if self.fake_score_transformer_2 is not None:
            logger.info(
                "  Fake score transformer_2 (low noise expert) parameters: %s B",
                sum(p.numel()
                    for p in self.fake_score_transformer_2.parameters()) / 1e9)
            logger.info("  Fake score MoE enabled with boundary_timestep: %s",
                        self.boundary_timestep)

        if self.generator_ema is not None:
            logger.info("  Generator EMA enabled with decay: %s",
                        self.training_args.ema_decay)
            logger.info("  Generator EMA start step: %s",
                        self.training_args.ema_start_step)
        else:
            logger.info("  Generator EMA disabled")

        if self.generator_ema is not None:
            logger.info("  Generator EMA enabled with decay: %s",
                        self.training_args.ema_decay)
            logger.info("  Generator EMA start step: %s",
                        self.training_args.ema_start_step)
        else:
            logger.info("  Generator EMA disabled")

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        training_args.inference_mode = True
        training_args.dit_cpu_offload = True
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Set deterministic seed for validation

        logger.info("Using validation seed: %s", self.seed)

        # Prepare validation prompts
        logger.info('rank: %s: fastvideo_args.validation_dataset_file: %s',
                    self.global_rank,
                    training_args.validation_dataset_file,
                    local_main_process_only=False)
        validation_dataset = ValidationDataset(
            training_args.validation_dataset_file)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=None,
                                           num_workers=0)

        # Set both transformers to eval mode
        transformer.eval()
        if hasattr(self, 'transformer_2') and self.transformer_2 is not None:
            self.transformer_2.eval()

        # Optionally use EMA model for validation if available and ready
        use_ema_for_validation = (self.training_args.use_ema
                                  and self.is_ema_ready(global_step))
        ema_context = None
        ema_2_context = None

        if use_ema_for_validation:
            logger.info("Using EMA model for validation")
            # Use self.transformer for consistency (the passed transformer should be self.transformer anyway)
            validation_transformer = self.transformer
            if self.generator_ema is not None:
                ema_context = self.generator_ema.apply_to_model(
                    validation_transformer)

            # Handle transformer_2 EMA if available
            if hasattr(
                    self, 'transformer_2'
            ) and self.transformer_2 is not None and self.generator_ema_2 is not None:
                ema_2_context = self.generator_ema_2.apply_to_model(
                    self.transformer_2)
                logger.info("Using EMA_2 model for transformer_2 validation")
        else:
            # Use self.transformer for consistency, but the passed transformer should be the same
            validation_transformer = self.transformer

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]
        # Log validation results for this step
        world_group = get_world_group()
        num_sp_groups = world_group.world_size // self.sp_group.world_size
        # Process each validation prompt for each validation step
        for num_inference_steps in validation_steps:
            logger.info("rank: %s: num_inference_steps: %s",
                        self.global_rank,
                        num_inference_steps,
                        local_main_process_only=False)
            step_videos: list[np.ndarray] = []
            step_captions: list[str] = []

            # Helper function to run validation with optional EMA contexts
            def run_validation_with_ema(
                    steps: int) -> tuple[list[np.ndarray], list[str]]:
                videos: list[np.ndarray] = []
                captions: list[str] = []
                for validation_batch in validation_dataloader:
                    batch = self._prepare_validation_batch(
                        sampling_param, training_args, validation_batch, steps)

                    negative_prompt = batch.negative_prompt
                    batch_negative = ForwardBatch(
                        data_type="video",
                        prompt=negative_prompt,
                        prompt_embeds=[],
                        prompt_attention_mask=[],
                    )
                    result_batch = self.validation_pipeline.prompt_encoding_stage(  # type: ignore
                        batch_negative, training_args)
                    self.negative_prompt_embeds, self.negative_prompt_attention_mask = result_batch.prompt_embeds[
                        0], result_batch.prompt_attention_mask[0]

                    logger.info(
                        "rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                        self.global_rank,
                        self.rank_in_sp_group,
                        batch.prompt,
                        local_main_process_only=False)

                    assert batch.prompt is not None and isinstance(
                        batch.prompt, str)
                    captions.append(batch.prompt)

                    # Run validation inference
                    with torch.no_grad():
                        output_batch = self.validation_pipeline.forward(
                            batch, training_args)
                    samples = output_batch.output
                    if self.rank_in_sp_group != 0:
                        continue

                    # Process outputs
                    video = rearrange(samples, "b c t h w -> t b c h w")
                    frames = []
                    for x in video:
                        x = torchvision.utils.make_grid(x, nrow=6)
                        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                        frames.append((x * 255).numpy().astype(np.uint8))
                    videos.append(frames)

                return videos, captions

            # Apply EMA contexts if available (nested context managers)
            if ema_context is not None and ema_2_context is not None:
                with ema_context, ema_2_context:
                    step_videos, step_captions = run_validation_with_ema(
                        num_inference_steps)
            elif ema_context is not None:
                with ema_context:
                    step_videos, step_captions = run_validation_with_ema(
                        num_inference_steps)
            elif ema_2_context is not None:
                with ema_2_context:
                    step_videos, step_captions = run_validation_with_ema(
                        num_inference_steps)
            else:
                step_videos, step_captions = run_validation_with_ema(
                    num_inference_steps)

            # Log validation results for this step
            world_group = get_world_group()
            num_sp_groups = world_group.world_size // self.sp_group.world_size

            # Only sp_group leaders (rank_in_sp_group == 0) need to send their
            # results to global rank 0
            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    # Global rank 0 collects results from all sp_group leaders
                    all_videos = step_videos  # Start with own results
                    all_captions = step_captions

                    # Receive from other sp_group leaders
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size  # Global rank of other sp_group leaders
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    video_filenames = []
                    for i, (video, caption) in enumerate(
                            zip(all_videos, all_captions, strict=True)):
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
                            f"validation_step_{global_step}_inference_steps_{num_inference_steps}_video_{i}.mp4"
                        )
                        imageio.mimsave(filename, video, fps=sampling_param.fps)
                        video_filenames.append(filename)

                    artifacts = []
                    for filename, caption in zip(video_filenames,
                                                 all_captions,
                                                 strict=True):
                        video_artifact = self.tracker.video(filename,
                                                            caption=caption)
                        if video_artifact is not None:
                            artifacts.append(video_artifact)
                    if artifacts:
                        logs = {
                            f"validation_videos_{num_inference_steps}_steps":
                            artifacts
                        }
                        self.tracker.log_artifacts(logs, global_step)
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        # Re-enable gradients for training - set both transformers back to train mode
        transformer.train()
        if hasattr(self, 'transformer_2') and self.transformer_2 is not None:
            self.transformer_2.train()
        gc.collect()

    def visualize_intermediate_latents(self, training_batch: TrainingBatch,
                                       training_args: TrainingArgs, step: int):
        """Add visualization data to tracker logging and save frames to disk."""
        tracker_loss_dict: dict[str, Any] = {}
        dmd_latents_vis_dict = training_batch.dmd_latent_vis_dict
        fake_score_latents_vis_dict = training_batch.fake_score_latent_vis_dict
        fake_score_log_keys = ['generator_pred_video']
        dmd_log_keys = ['faker_score_pred_video', 'real_score_pred_video']

        for latent_key in fake_score_log_keys:
            latents = fake_score_latents_vis_dict[latent_key]
            latents = latents.permute(0, 2, 1, 3, 4)

            if isinstance(self.vae.scaling_factor, torch.Tensor):
                latents = latents / self.vae.scaling_factor.to(
                    latents.device, latents.dtype)
            else:
                latents = latents / self.vae.scaling_factor

            # Apply shifting if needed
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    latents += self.vae.shift_factor.to(latents.device,
                                                        latents.dtype)
                else:
                    latents += self.vae.shift_factor
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video = self.vae.decode(latents)
                video = (video / 2 + 0.5).clamp(0, 1)
                video = video.cpu().float()
                video = video.permute(0, 2, 1, 3, 4)
                video = (video * 255).numpy().astype(np.uint8)
                video_artifact = self.tracker.video(
                    video, fps=24, format="mp4")  # change to 16 for Wan2.1
                if video_artifact is not None:
                    tracker_loss_dict[latent_key] = video_artifact
                # Clean up references
                del video, latents

        # Process DMD training data if available - use decode_stage instead of self.vae.decode
        if 'generator_pred_video' in dmd_latents_vis_dict:
            for latent_key in dmd_log_keys:
                latents = dmd_latents_vis_dict[latent_key]
                latents = latents.permute(0, 2, 1, 3, 4)
                # decoded_latent = decode_stage(ForwardBatch(data_type="video", latents=latents), training_args)
                if isinstance(self.vae.scaling_factor, torch.Tensor):
                    latents = latents / self.vae.scaling_factor.to(
                        latents.device, latents.dtype)
                else:
                    latents = latents / self.vae.scaling_factor

                # Apply shifting if needed
                if (hasattr(self.vae, "shift_factor")
                        and self.vae.shift_factor is not None):
                    if isinstance(self.vae.shift_factor, torch.Tensor):
                        latents += self.vae.shift_factor.to(
                            latents.device, latents.dtype)
                    else:
                        latents += self.vae.shift_factor
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video = self.vae.decode(latents)
                video = (video / 2 + 0.5).clamp(0, 1)
                video = video.cpu().float()
                video = video.permute(0, 2, 1, 3, 4)
                video = (video * 255).numpy().astype(np.uint8)
                video_artifact = self.tracker.video(
                    video, fps=24, format="mp4")  # change to 16 for Wan2.1
                if video_artifact is not None:
                    tracker_loss_dict[latent_key] = video_artifact
                # Clean up references
                del video, latents

        # Log to tracker
        if self.global_rank == 0 and tracker_loss_dict:
            self.tracker.log_artifacts(tracker_loss_dict, step)

    def train(self) -> None:
        """Main training loop with distillation-specific logging."""
        assert self.training_args.seed is not None, "seed must be set"
        seed = self.training_args.seed

        # Set the same seed within each SP group to ensure reproducibility
        if self.sp_world_size > 1:
            # Use the same seed for all processes within the same SP group
            sp_group_seed = seed + (self.global_rank // self.sp_world_size)
            set_random_seed(sp_group_seed)
            logger.info("Rank %s: Using SP group seed %s", self.global_rank,
                        sp_group_seed)
        else:
            set_random_seed(seed + self.global_rank)

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", seed)

        # Initialize current_trainstep for EMA ready checks
        #TODO: check if needed
        self.current_trainstep = self.init_steps

        # Resume from checkpoint if specified (this will restore random states)
        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()
            logger.info("Resumed from checkpoint, random states restored")
        else:
            logger.info("Starting training from scratch")

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.transformer, self.training_args,
                             self.init_steps)

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )

        use_vsa = vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN"
        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
            if use_vsa:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                if vsa_decay_interval_steps > 1:
                    current_decay_times = min(step // vsa_decay_interval_steps,
                                              vsa_sparsity // vsa_decay_rate)
                    current_vsa_sparsity = current_decay_times * vsa_decay_rate
                else:
                    current_vsa_sparsity = vsa_sparsity
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            self.current_trainstep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity

            if (step >= self.training_args.ema_start_step) and \
                    (self.generator_ema is None) and (self.training_args.ema_decay > 0):
                self.generator_ema = EMA_FSDP(
                    self.transformer, decay=self.training_args.ema_decay)
                logger.info("Created generator EMA at step %s with decay=%s",
                            step, self.training_args.ema_decay)

                # Create EMA for transformer_2 if it exists
                if self.transformer_2 is not None and self.generator_ema_2 is None:
                    self.generator_ema_2 = EMA_FSDP(
                        self.transformer_2, decay=self.training_args.ema_decay)
                    logger.info(
                        "Created generator EMA_2 at step %s with decay=%s",
                        step, self.training_args.ema_decay)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                training_batch = self.train_one_step(training_batch)

            total_loss = training_batch.total_loss
            generator_loss = training_batch.generator_loss
            fake_score_loss = training_batch.fake_score_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "total_loss":
                f"{total_loss:.4f}",
                "generator_loss":
                f"{generator_loss:.4f}",
                "fake_score_loss":
                f"{fake_score_loss:.4f}",
                "step_time":
                f"{step_time:.2f}s",
                "grad_norm":
                grad_norm,
                "ema":
                "✓" if (self.generator_ema is not None and self.is_ema_ready())
                else "✗",
                "ema2":
                "✓" if (self.generator_ema_2 is not None
                        and self.is_ema_ready()) else "✗",
            })
            progress_bar.update(1)

            if self.global_rank == 0:
                # Prepare logging data
                log_data = {
                    "train_total_loss":
                    total_loss,
                    "train_fake_score_loss":
                    fake_score_loss,
                    "learning_rate":
                    self.lr_scheduler.get_last_lr()[0],
                    "fake_score_learning_rate":
                    self.fake_score_lr_scheduler.get_last_lr()[0],
                    "step_time":
                    step_time,
                    "avg_step_time":
                    avg_step_time,
                    "grad_norm":
                    grad_norm,
                }
                # Only log generator loss when generator is actually trained
                if (step % self.generator_update_interval == 0):
                    log_data["train_generator_loss"] = generator_loss
                if use_vsa:
                    log_data["VSA_train_sparsity"] = current_vsa_sparsity

                if self.generator_ema is not None or self.generator_ema_2 is not None:
                    log_data["ema_enabled"] = self.generator_ema is not None
                    log_data["ema_2_enabled"] = self.generator_ema_2 is not None
                    log_data["ema_decay"] = self.training_args.ema_decay
                else:
                    log_data["ema_enabled"] = False
                    log_data["ema_2_enabled"] = False

                ema_stats = self.get_ema_stats()
                log_data.update(ema_stats)

                if training_batch.dmd_latent_vis_dict:
                    dmd_additional_logs = {
                        "generator_timestep":
                        training_batch.
                        dmd_latent_vis_dict["generator_timestep"].item(),
                        "dmd_timestep":
                        training_batch.dmd_latent_vis_dict["dmd_timestep"].item(
                        ),
                    }
                    log_data.update(dmd_additional_logs)

                faker_score_additional_logs = {
                    "fake_score_timestep":
                    training_batch.
                    fake_score_latent_vis_dict["fake_score_timestep"].item(),
                }
                log_data.update(faker_score_additional_logs)

                self.tracker.log(log_data, step)

            # Save training state checkpoint (for resuming training)
            if (self.training_args.training_state_checkpointing_steps > 0
                    and step %
                    self.training_args.training_state_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save training state checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer,
                    self.fake_score_transformer,
                    self.global_rank,
                    self.training_args.output_dir,
                    step,
                    self.optimizer,
                    self.fake_score_optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                    self.fake_score_lr_scheduler,
                    self.noise_random_generator,
                    self.generator_ema,
                    # MoE support
                    generator_transformer_2=getattr(self, 'transformer_2',
                                                    None),
                    real_score_transformer_2=getattr(
                        self, 'real_score_transformer_2', None),
                    fake_score_transformer_2=getattr(
                        self, 'fake_score_transformer_2', None),
                    generator_optimizer_2=getattr(self, 'optimizer_2', None),
                    fake_score_optimizer_2=getattr(self,
                                                   'fake_score_optimizer_2',
                                                   None),
                    generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
                    fake_score_scheduler_2=getattr(self,
                                                   'fake_score_lr_scheduler_2',
                                                   None),
                    generator_ema_2=getattr(self, 'generator_ema_2', None))

                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()

            # Save weight-only checkpoint
            if (self.training_args.weight_only_checkpointing_steps > 0
                    and step %
                    self.training_args.weight_only_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save weight-only checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer,
                    self.fake_score_transformer,
                    self.global_rank,
                    self.training_args.output_dir,
                    f"{step}_weight_only",
                    only_save_generator_weight=True,
                    generator_ema=self.generator_ema,
                    # MoE support
                    generator_transformer_2=getattr(self, 'transformer_2',
                                                    None),
                    real_score_transformer_2=getattr(
                        self, 'real_score_transformer_2', None),
                    fake_score_transformer_2=getattr(
                        self, 'fake_score_transformer_2', None),
                    generator_optimizer_2=getattr(self, 'optimizer_2', None),
                    fake_score_optimizer_2=getattr(self,
                                                   'fake_score_optimizer_2',
                                                   None),
                    generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
                    fake_score_scheduler_2=getattr(self,
                                                   'fake_score_lr_scheduler_2',
                                                   None),
                    generator_ema_2=getattr(self, 'generator_ema_2', None))

                if self.training_args.use_ema and self.is_ema_ready():
                    self.save_ema_weights(self.training_args.output_dir, step)

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                if self.training_args.log_visualization:
                    self.visualize_intermediate_latents(training_batch,
                                                        self.training_args,
                                                        step)
                self._log_validation(self.transformer, self.training_args, step)

        self.tracker.finish()

        # Save final training state checkpoint
        print("rank", self.global_rank,
              "save final training state checkpoint at step",
              self.training_args.max_train_steps)
        save_distillation_checkpoint(
            self.transformer,
            self.fake_score_transformer,
            self.global_rank,
            self.training_args.output_dir,
            self.training_args.max_train_steps,
            self.optimizer,
            self.fake_score_optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.fake_score_lr_scheduler,
            self.noise_random_generator,
            self.generator_ema,
            # MoE support
            generator_transformer_2=getattr(self, 'transformer_2', None),
            real_score_transformer_2=getattr(self, 'real_score_transformer_2',
                                             None),
            fake_score_transformer_2=getattr(self, 'fake_score_transformer_2',
                                             None),
            generator_optimizer_2=getattr(self, 'optimizer_2', None),
            fake_score_optimizer_2=getattr(self, 'fake_score_optimizer_2',
                                           None),
            generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
            fake_score_scheduler_2=getattr(self, 'fake_score_lr_scheduler_2',
                                           None),
            generator_ema_2=getattr(self, 'generator_ema_2', None))

        if self.training_args.use_ema and self.is_ema_ready():
            self.save_ema_weights(self.training_args.output_dir,
                                  self.training_args.max_train_steps)

        if envs.FASTVIDEO_TORCH_PROFILER_DIR:
            logger.info("Stopping profiler...")
            self.profiler_controller.stop()
            logger.info("Profiler stopped.")

        if get_sp_group():
            cleanup_dist_env_and_memory()
