# SPDX-License-Identifier: Apache-2.0
# HYWorld Training Pipeline with Camera Pose and Memory Support
# Adapted from HY-WorldPlay trainer

import math
import os
import time
from collections import deque
from copy import deepcopy

import imageio
import numpy as np
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from fastvideo.dataset.hyworld_camera_dataset import (
    HYWorldCameraDataset,
    build_hyworld_camera_dataloader,
)
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    get_local_torch_device,
    get_sp_group,
    get_world_group,
)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch, TrainingBatch
from fastvideo.configs.sample import SamplingParam
from fastvideo.utils import shallow_asdict
from fastvideo.platforms import current_platform
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling,
    count_trainable,
    get_scheduler,
    get_sigmas,
    load_checkpoint,
    save_checkpoint,
)
from fastvideo.training.muon import get_muon_optimizer
from fastvideo.utils import is_vsa_available, set_random_seed
import wandb

vsa_available = is_vsa_available()

logger = init_logger(__name__)


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    """Merge two tensors based on a mask."""
    assert tensor_1.shape == tensor_2.shape
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp


def _unwrap_module(m):
    """
    Handle common wrappers (FSDP/DDP/torch.compile) without importing their types.

    NOTE: Do NOT unwrap a `.vae` attribute here. Some VAE wrappers may expose a `.vae`
    field that can be `None` (or not the actual decode implementation). Unwrapping it
    can incorrectly turn a valid module into `None`.
    """
    for attr in ("module", "_orig_mod"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m


def _resolve_vae_scaling_and_shift(vae):
    """
    Best-effort resolve (scaling_factor, shift_factor) from a variety of VAE implementations.
    Returns:
        (scaling_factor: float, shift_factor: float|None)
    """
    vae_u = _unwrap_module(vae)
    # Try common locations
    scaling = getattr(vae_u, "scaling_factor", None)
    cfg = getattr(vae_u, "config", None)
    if scaling is None and cfg is not None:
        scaling = getattr(cfg, "scaling_factor", None)
    # Some trainers keep scaling in nested config objects
    if scaling is None:
        cfg2 = getattr(vae_u, "vae_config", None)
        if cfg2 is not None:
            scaling = getattr(cfg2, "scaling_factor", None)
    # Some configs may have scaling_factor=0 as default; treat that as unknown
    try:
        if scaling is not None:
            scaling = float(scaling)
            if scaling == 0.0:
                scaling = None
    except Exception:
        scaling = None

    shift = None
    if cfg is not None:
        shift = getattr(cfg, "shift_factor", None)
        try:
            if shift is not None:
                shift = float(shift)
        except Exception:
            shift = None

    # Be strict: visualization must decode correctly.
    if scaling is None:
        raise RuntimeError(
            f"VAE scaling_factor not found (vae={type(vae_u).__name__}). "
            "Expected `vae.scaling_factor` or `vae.config.scaling_factor` to exist."
        )
    return scaling, shift


def _vae_decode_video_frames(vae, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode VAE latents into video frames for visualization.

    Args:
        vae: VAE module (supports .decode and has .scaling_factor or .config.scaling_factor)
        latents: (B,C,T,H,W) in *scaled latent space* (i.e. encoder output multiplied by scaling_factor).

    Returns:
        frames: (B,3,F,H,W) float in [0,1]
    """
    vae_u = _unwrap_module(vae)
    scaling, shift_factor = _resolve_vae_scaling_and_shift(vae_u)
    # Match HY-WorldPlay pipeline behavior when possible: divide by scaling_factor, optionally add shift_factor.
    z = latents / scaling
    if shift_factor is not None:
        z = z + shift_factor

    # Decode on the VAE's device to avoid device mismatch (CPU/GPU offload variants).
    try:
        vae_device = next(vae_u.parameters()).device
    except StopIteration:
        vae_device = z.device
    z = z.to(device=vae_device)

    dec = vae_u.decode(z)
    # Some VAE impls return tuple/list; some return tensor directly.
    if isinstance(dec, (tuple, list)):
        dec = dec[0]
    # diffusers-style DecoderOutput
    if hasattr(dec, "sample"):
        dec = dec.sample
    frames = (dec / 2 + 0.5).clamp(0, 1)
    return frames


class HYWorldTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for HY-World with camera pose and memory support.
    Supports:
    - Autoregressive training with action control
    - Memory training (window-based with FOV-based frame selection)
    - I2V (image-to-video) and T2V (text-to-video) multitask
    - ProPE (Projective Positional Encoding) for camera-aware attention
    """

    _required_config_modules = ["scheduler", "transformer"]

    # HYWorld-specific attributes
    train_dataset: HYWorldCameraDataset
    action: bool = True
    causal: bool = True
    train_time_shift: float = 1.0

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize the HYWorld training pipeline."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler()

    def create_training_stages(self, training_args: TrainingArgs):
        """May be used in future refactors."""
        pass

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the training pipeline with HYWorld-specific setup."""
        logger.info("Initializing HYWorld training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        self.seed = training_args.seed

        # HYWorld-specific settings
        self.action = getattr(training_args, 'action', True)
        self.causal = getattr(training_args, 'causal', True)
        self.train_time_shift = getattr(training_args, 'train_time_shift', 1.0)

        # Set random seeds
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()

        # Apply activation checkpointing if enabled
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        # Setup trainable parameters
        self.set_trainable()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters()))

        # Use Muon optimizer (more memory-efficient than AdamW)
        # Muon uses momentum only (no second moment), reducing optimizer state memory by ~50%
        # - Muon is used for parameters with ndim >= 2 (matrices)
        # - AdamW is used internally for parameters with ndim < 2 (biases, norms)
        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            adamw_betas=(0.95, 0.95),
            adamw_eps=1e-8,
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        # Build HYWorld camera dataloader
        json_path = getattr(training_args, 'json_path', training_args.data_path)
        window_frames = getattr(training_args, 'window_frames', 24)
        i2v_rate = getattr(training_args, 'i2v_rate', 0.5)
        neg_prompt_path = getattr(training_args, 'neg_prompt_path', None)
        neg_byt5_path = getattr(training_args, 'neg_byt5_path', None)

        self.train_dataset, self.train_dataloader = build_hyworld_camera_dataloader(
            json_path=json_path,
            causal=self.causal,
            window_frames=window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            cfg_rate=training_args.training_cfg_rate,
            i2v_rate=i2v_rate,
            neg_prompt_path=neg_prompt_path,
            neg_byt5_path=neg_byt5_path,
        )

        self.noise_scheduler = self.modules["scheduler"]

        # Calculate training steps
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                          self.num_update_steps_per_epoch)
        self.current_epoch = 0

        # Initialize tracker
        from fastvideo.training.trackers import (Trackers,
                                                 initialize_trackers)
        from dataclasses import asdict

        trackers = list(training_args.trackers)
        if not trackers and training_args.tracker_project_name:
            trackers.append(Trackers.WANDB.value)
        if self.global_rank != 0:
            trackers = []

        tracker_log_dir = training_args.output_dir or os.getcwd()
        if trackers:
            tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

        tracker_config = asdict(training_args) if trackers else None
        tracker_run_name = training_args.wandb_run_name or None
        project = training_args.tracker_project_name or "fastvideo-hyworld"

        self.tracker = initialize_trackers(
            trackers,
            experiment_name=project,
            config=tracker_config,
            log_dir=tracker_log_dir,
            run_name=tracker_run_name,
        )

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize the validation pipeline for HYWorld."""
        logger.info("Initializing HYWorld validation pipeline...")
        from fastvideo.pipelines.basic.hyworld.hyworld_pipeline import HYWorldPipeline

        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        args_copy.dit_cpu_offload = True

        self.validation_pipeline = HYWorldPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            dit_cpu_offload=True,
        )
        self._train_vis_vae = None

    def _log_validation(self, transformer, training_args: TrainingArgs,
                        global_step: int) -> None:
        """Generate validation videos. Uses training data if no validation_dataset_file is set."""
        training_args.inference_mode = True
        training_args.dit_cpu_offload = True
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            logger.warning("Validation pipeline is not initialized, skipping validation")
            return

        logger.info("Starting validation at step %s", global_step)

        # Create sampling parameters
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Use training dataloader if no validation_dataset_file is set
        if not training_args.validation_dataset_file:
            logger.info("No validation_dataset_file set, using training dataloader for validation")
            validation_dataloader = [next(iter(self.train_dataloader))]
        else:
            from fastvideo.training.training_pipeline import ValidationDataset
            from torch.utils.data import DataLoader
            validation_dataset = ValidationDataset(training_args.validation_dataset_file)
            validation_dataloader = DataLoader(validation_dataset, batch_size=None, num_workers=0)

        self.transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]

        world_group = get_world_group()
        num_sp_groups = world_group.world_size // self.sp_group.world_size

        for num_inference_steps in validation_steps:
            logger.info("rank: %s: num_inference_steps: %s",
                        self.global_rank, num_inference_steps)
            step_videos: list[np.ndarray] = []
            step_captions: list[str] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_batch(
                    sampling_param, training_args, validation_batch, num_inference_steps)

                logger.info("rank: %s: batch.prompt: %s", self.global_rank, batch.prompt)
                step_captions.append(batch.prompt or "")


                output_batch = self.validation_pipeline.forward(batch, training_args)
                samples = output_batch.output

                if self.rank_in_sp_group != 0:
                    continue

                # Process outputs
                import torchvision
                video = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                step_videos.append(frames)

            # Only sp_group leaders need to send results to global rank 0
            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    all_videos = step_videos
                    all_captions = step_captions

                    # Receive from other sp_group leaders
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    # Save and log videos
                    vis_dir = os.path.join(training_args.output_dir, "validation_videos")
                    os.makedirs(vis_dir, exist_ok=True)

                    for i, (video, caption) in enumerate(zip(all_videos, all_captions)):
                        filename = os.path.join(
                            vis_dir,
                            f"validation_step_{global_step}_infer_{num_inference_steps}_video_{i}.mp4")
                        imageio.mimsave(filename, video, fps=sampling_param.fps)

                        self.tracker.log(
                            {f"validation_video_{num_inference_steps}steps": wandb.Video(filename, caption=caption)},
                            step=global_step,
                        )
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        self.transformer.train()

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict,
                                  num_inference_steps: int) -> ForwardBatch:
        """Prepare a ForwardBatch for validation with HYWorld-specific inputs."""
        # Helper to get first element if value is a list/tuple
        def _get_first(val):
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return val[0]
            return val

        # Get prompt from validation batch
        prompt = _get_first(validation_batch.get('prompt', ''))

        sampling_param.prompt = prompt
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        # Note: HYWorld training data doesn't have image_path, it has image_cond (VAE latent)
        # Don't set image_path for HYWorld - we'll use image_cond directly
        sampling_param.image_path = None
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        # Calculate latent sizes
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t - 1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8
        ]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        # Add HYWorld-specific inputs from validation batch (take first sample if batched)
        # Note: denoising stage expects 'viewmats', 'Ks', 'action' (not w2c/intrinsic)
        # Use float32 for camera matrices since they may be converted to numpy for frame selection
        if 'w2c' in validation_batch:
            w2c = validation_batch['w2c']
            if w2c.dim() > 3:  # Batched: [B, T, 4, 4] -> [T, 4, 4]
                w2c = w2c[0]
            if w2c.dim() == 3:  # Add batch dim: [T, 4, 4] -> [1, T, 4, 4]
                w2c = w2c.unsqueeze(0)
            batch.viewmats = w2c.to(get_local_torch_device(), dtype=torch.float32)
        if 'intrinsic' in validation_batch:
            intrinsic = validation_batch['intrinsic']
            if intrinsic.dim() > 3:  # Batched
                intrinsic = intrinsic[0]
            if intrinsic.dim() == 3:  # Add batch dim
                intrinsic = intrinsic.unsqueeze(0)
            batch.Ks = intrinsic.to(get_local_torch_device(), dtype=torch.float32)
        if 'action' in validation_batch:
            action = validation_batch['action']
            if action.dim() > 1:  # Batched
                action = action[0]
            if action.dim() == 1:  # Add batch dim
                action = action.unsqueeze(0)
            batch.action = action.to(get_local_torch_device(), dtype=torch.float32)

        # HYWorld: use pre-computed embeddings from training data
        # Pass these directly so encoding stages use them instead of encoding from scratch
        
        # image_cond → image_latent (VAE-encoded first frame, used by denoising stage)
        # Note: image_cond from training data has shape [B, C, 1, H, W] (single frame)
        # We need to expand it to match the full temporal dimension like training does
        if 'image_cond' in validation_batch and validation_batch['image_cond'] is not None:
            image_cond = validation_batch['image_cond']
            if image_cond.dim() > 4:  # Batched: [B, C, T, H, W] -> [C, T, H, W]
                image_cond = image_cond[0]
            # Ensure batch dimension
            if image_cond.dim() == 4:
                image_cond = image_cond.unsqueeze(0)
            
            # Expand temporal dimension to match latent frames (like _prepare_cond_latents does)
            # image_cond is [1, C, 1, H, W], expand to [1, C, T, H, W]
            latent_temporal = training_args.num_latent_t
            if image_cond.shape[2] == 1 and latent_temporal > 1:
                # Repeat along temporal dimension and zero out all except first frame
                image_cond = image_cond.repeat(1, 1, latent_temporal, 1, 1)
                image_cond[:, :, 1:, :, :] = 0.0
            
            # Add mask channel (1 for first frame, 0 for rest) like HYWorldImageEncodingStage does
            mask = torch.zeros(1, 1, latent_temporal, image_cond.shape[3], image_cond.shape[4], 
                              device=image_cond.device, dtype=image_cond.dtype)
            mask[:, :, 0, :, :] = 1.0
            image_latent = torch.cat([image_cond, mask], dim=1)
            
            batch.image_latent = image_latent.to(get_local_torch_device(), dtype=torch.bfloat16)

        # vision_states → image_embeds (CLIP/SigLIP features, used by denoising stage)
        if 'vision_states' in validation_batch and validation_batch['vision_states'] is not None:
            vision_states = validation_batch['vision_states']
            if vision_states.dim() > 3:  # Batched: [B, num_tokens, dim] -> [num_tokens, dim]
                vision_states = vision_states[0]
            # Ensure batch dimension and wrap in list (matching HYWorldImageEncodingStage output)
            if vision_states.dim() == 2:
                vision_states = vision_states.unsqueeze(0)
            batch.image_embeds = [vision_states.to(get_local_torch_device(), dtype=torch.bfloat16)]

        # Pre-computed text embeddings
        # Need shape [1, seq_len, dim] with batch dimension for pipeline
        if 'prompt_embed' in validation_batch and validation_batch['prompt_embed'] is not None:
            prompt_embed = validation_batch['prompt_embed']
            if prompt_embed.dim() > 2:  # Batched: [B, seq_len, dim] -> take first sample
                prompt_embed = prompt_embed[0:1]  # Keep batch dim: [1, seq_len, dim]
            elif prompt_embed.dim() == 2:  # Add batch dim if missing
                prompt_embed = prompt_embed.unsqueeze(0)
            batch.prompt_embeds = [prompt_embed.to(get_local_torch_device(), dtype=torch.bfloat16)]

        if 'byt5_text_states' in validation_batch and validation_batch['byt5_text_states'] is not None:
            byt5_states = validation_batch['byt5_text_states']
            if byt5_states.dim() > 2:  # Batched
                byt5_states = byt5_states[0:1]  # Keep batch dim
            elif byt5_states.dim() == 2:
                byt5_states = byt5_states.unsqueeze(0)
            # Add as second element in prompt_embeds list (HYWorld uses dual text encoders)
            if hasattr(batch, 'prompt_embeds') and batch.prompt_embeds:
                batch.prompt_embeds.append(byt5_states.to(get_local_torch_device(), dtype=torch.bfloat16))

        # Attention masks
        if 'prompt_mask' in validation_batch and validation_batch['prompt_mask'] is not None:
            prompt_mask = validation_batch['prompt_mask']
            if prompt_mask.dim() > 1:
                prompt_mask = prompt_mask[0:1]
            batch.prompt_attention_mask = [prompt_mask.to(get_local_torch_device(), dtype=torch.bfloat16)]

        if 'byt5_text_mask' in validation_batch and validation_batch['byt5_text_mask'] is not None:
            byt5_mask = validation_batch['byt5_text_mask']
            if byt5_mask.dim() > 1:
                byt5_mask = byt5_mask[0:1]
            if hasattr(batch, 'prompt_attention_mask') and batch.prompt_attention_mask:
                batch.prompt_attention_mask.append(byt5_mask.to(get_local_torch_device(), dtype=torch.bfloat16))

        # Set default negative embeddings (zeros) for classifier-free guidance
        if hasattr(batch, 'prompt_embeds') and batch.prompt_embeds:
            batch.negative_prompt_embeds = [torch.zeros_like(e) for e in batch.prompt_embeds]
            batch.negative_attention_mask = batch.prompt_attention_mask

        # HYWorld AR model uses chunk_latent_frames=4
        batch.chunk_latent_frames = 4
        return batch

    def _get_train_vis_vae(self):
        """
        Lazily load VAE decoder for train-time visualization.
        
        Training pipeline does not load the VAE by default (`_required_config_modules` 
        excludes it). For train-time video visualization, we lazily load a VAE decoder 
        from the pretrained model only when needed.
        
        Uses FastVideo's VAE loader to handle HYWorld's 3D conv VAE format correctly.
        """
        if hasattr(self, "_train_vis_vae") and self._train_vis_vae is not None:
            return self._train_vis_vae

        model_root = getattr(self.training_args, "pretrained_model_name_or_path", None) or \
                     getattr(self.training_args, "model_path", None)
        if not model_root:
            raise RuntimeError(
                "Cannot load visualization VAE: model_path is empty.")

        vae_dir = os.path.join(str(model_root), "vae")
        if not os.path.isdir(vae_dir):
            raise RuntimeError(
                f"Cannot load visualization VAE: missing vae directory: {vae_dir}"
            )

        import glob
        import json
        from safetensors.torch import load_file as safetensors_load_file
        from fastvideo.models.registry import ModelRegistry

        # Load VAE config to get class name
        config_path = os.path.join(vae_dir, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(
                f"Cannot load visualization VAE: missing config.json in {vae_dir}"
            )

        with open(config_path) as f:
            config = json.load(f)

        class_name = config.get("_class_name")
        if class_name is None:
            raise RuntimeError(
                f"VAE config.json does not contain _class_name: {config_path}")

        # Use FastVideo's model registry to get the correct VAE class
        vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        # Prepare config for VAE initialization
        # Filter out private fields (e.g., _class_name, _diffusers_version) that aren't part of arch_config
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        vae_config = self.training_args.pipeline_config.vae_config
        vae_config.update_model_arch(config)

        device = get_local_torch_device()
        vis_vae = vae_cls(vae_config).to(device)

        # Load weights from safetensors
        safetensors_list = glob.glob(os.path.join(vae_dir, "*.safetensors"))
        if not safetensors_list:
            raise RuntimeError(f"No safetensors files found in {vae_dir}")

        loaded = {}
        for sf_file in safetensors_list:
            loaded.update(safetensors_load_file(sf_file))

        vis_vae.load_state_dict(loaded, strict=False)
        vis_vae = vis_vae.eval()

        for p in vis_vae.parameters():
            p.requires_grad_(False)

        self._train_vis_vae = vis_vae
        logger.info(
            "Loaded visualization VAE (%s) for train video logging from %s",
            class_name, vae_dir)
        return self._train_vis_vae

    def _should_log_train_videos(self, step: int) -> bool:
        every = int(
            getattr(self.training_args, "train_video_log_steps", 0) or 0)
        return every > 0 and (step % every) == 0

    @torch.no_grad()
    def _log_train_videos_to_wandb(self, training_batch: TrainingBatch,
                                   step: int) -> None:
        """
        Log gt/noisy/pred decoded videos to wandb.

        Fast path for multi-GPU debugging (no inter-rank collectives):
        - Each rank loads its own VAE decoder (cached after first use).
        - At each log event we select `vis_rank = log_idx % world_size` (round-robin).
          Only `vis_rank` decodes & writes the MP4 (from its own local batch).
        - Rank0 logs the MP4 to WandB exactly once, by polling the filesystem for the expected output file.
        """
        if not self._should_log_train_videos(step):
            return
        if self.training_args.sp_size > 1:
            # SP shards tokens/latents; reconstructing full video for logging is non-trivial.
            logger.warning(
                "Skipping train video logging because sp_size > 1 (sp_size=%s).",
                self.training_args.sp_size,
            )
            return

        if training_batch.latents is None or training_batch.noisy_model_input is None or training_batch.noise is None:
            return
        if training_batch.model_pred is None:
            return

        # -----------------------------
        # Choose visualization rank (round-robin).
        # -----------------------------
        world_size = int(getattr(self, "world_size", 1) or 1)
        rank = int(getattr(self, "global_rank", 0) or 0)
        every = int(
            getattr(self.training_args, "train_video_log_steps", 0) or 0)
        log_idx = (int(step) // max(1, every)) if every > 0 else int(step)
        vis_rank = int(log_idx % max(1, world_size))

        max_samples = int(
            getattr(self.training_args, "train_video_log_max_samples", 1) or 1)
        max_samples = max(1, max_samples)
        fps = int(getattr(self.training_args, "train_video_log_fps", 25) or 25)

        # Keep training output dir tidy: write VAE-decoded previews under a dedicated subfolder.
        vis_dir = os.path.join(self.training_args.output_dir,
                               "vae_during_training")
        os.makedirs(vis_dir, exist_ok=True)
        triptych_path = os.path.join(
            vis_dir, f"train_step_{step:06d}_rank{vis_rank}_gt_noisy_x0hat.mp4")

        # Load VAE on every rank (cached) so that when it's a rank's turn it can decode immediately.
        # NOTE: training pipeline does not necessarily load VAE; use a dedicated VAE for visualization.
        _ = self.get_module("vae", None) or self._get_train_vis_vae()

        # -----------------------------
        # Decode only on vis_rank, using that rank's local batch.
        # -----------------------------
        if rank == vis_rank:
            device = get_local_torch_device()
            vae = self.get_module("vae", None) or self._get_train_vis_vae()

            # Build x0_hat latents for visualization (does NOT affect training).
            if self.training_args.precondition_outputs:
                x0_hat_latents = training_batch.model_pred
            else:
                x0_hat_latents = training_batch.noise - training_batch.model_pred

            gt_latents = training_batch.latents[:max_samples].to(device)
            noisy_latents = training_batch.noisy_model_input[:max_samples].to(
                device)
            x0_hat_latents = x0_hat_latents[:max_samples].to(device)

            # Local scalar stats for caption/overlay (written to a sidecar JSON for rank0 to read).
            t_mean = float("nan")
            s_mean = float("nan")
            if training_batch.timesteps is not None:
                t_mean = float(training_batch.timesteps.detach().float().mean().
                               cpu().item())
            if training_batch.sigmas is not None:
                s_mean = float(
                    training_batch.sigmas.detach().float().mean().cpu().item())

            with torch.autocast(device_type="cuda",
                                dtype=torch.float16,
                                enabled=True):
                gt_frames = _vae_decode_video_frames(vae, gt_latents)
                noisy_frames = _vae_decode_video_frames(vae, noisy_latents)
                x0_hat_frames = _vae_decode_video_frames(vae, x0_hat_latents)

            def _to_uint8_video(frames01: torch.Tensor) -> np.ndarray:
                # (B,3,T,H,W) -> (T,H,W,3) uint8 (only first sample)
                x = frames01[0].detach().cpu()
                x = (x * 255.0).clamp(0, 255).to(torch.uint8)
                return x.permute(1, 2, 3, 0).numpy()

            gt_vid = _to_uint8_video(gt_frames)
            noisy_vid = _to_uint8_video(noisy_frames)
            x0_hat_vid = _to_uint8_video(x0_hat_frames)

            # Add overlay for clarity (compact for low-res videos).
            overlay_lines = [f"step={step}"]
            # Mark in-window vs out-window(memory) sampling.
            try:
                is_out_window = int(
                    getattr(training_batch, "select_window_out_flag", 0)
                    or 0) == 1
            except Exception:
                is_out_window = False
            overlay_lines.append(
                "mode=out-window" if is_out_window else "mode=in-window")
            # Helpful for interpreting out-window videos (they are re-packed sequences, not continuous time).
            try:
                latent_T = int(gt_latents.shape[2])
                overlay_lines.append(f"latent_T={latent_T}")
            except Exception:
                pass
            if is_out_window:
                # current_frame_idx is a latent start index in the ORIGINAL sequence (before repacking).
                cfi = getattr(training_batch, "current_frame_idx", None)
                sel = getattr(training_batch, "selected_history_frame_id", None)
                try:
                    if cfi is not None:
                        cfi_int = int(cfi)
                        overlay_lines.append(
                            f"cur_latent={cfi_int}  cur_chunk={cfi_int // 4}")
                    if sel:
                        # Show which history chunks were included (exclude the current chunk indices).
                        cur_start = int(cfi) if cfi is not None else None
                        cur_set = set(
                            range(cur_start, cur_start +
                                  4)) if cur_start is not None else set()
                        hist_chunks = sorted(
                            {int(i) // 4
                             for i in sel if int(i) not in cur_set})
                        # Compact formatting; keep HUD short.
                        if len(hist_chunks) <= 12:
                            overlay_lines.append("ctx_chunks=" + ",".join(
                                str(x) for x in hist_chunks))
                        else:
                            overlay_lines.append("ctx_chunks=" + ",".join(
                                str(x) for x in hist_chunks[:10]) + ",...")
                except Exception:
                    # Never crash training due to HUD formatting.
                    pass
            if not math.isnan(t_mean) and not math.isnan(s_mean):
                overlay_lines.append(f"t={t_mean:.0f}  σ={s_mean:.3f}")
            elif not math.isnan(t_mean):
                overlay_lines.append(f"t={t_mean:.0f}")
            elif not math.isnan(s_mean):
                overlay_lines.append(f"σ={s_mean:.3f}")
            overlay = "\n".join(overlay_lines)

            def _draw_overlay(frames: np.ndarray) -> np.ndarray:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                out = frames.copy()
                for i in range(out.shape[0]):
                    im = Image.fromarray(out[i])
                    draw = ImageDraw.Draw(im)
                    hud_h = 86
                    draw.rectangle([(0, 0), (im.size[0], hud_h)],
                                   fill=(0, 0, 0))
                    draw.multiline_text((4, 2),
                                        overlay,
                                        fill=(255, 255, 255),
                                        font=font,
                                        spacing=0)
                    out[i] = np.asarray(im)
                return out

            gt_vid = _draw_overlay(gt_vid)
            noisy_vid = _draw_overlay(noisy_vid)
            x0_hat_vid = _draw_overlay(x0_hat_vid)

            # Combine into one side-by-side (GT | noisy | x0_hat) so wandb shows a single video panel.
            T = min(gt_vid.shape[0], noisy_vid.shape[0], x0_hat_vid.shape[0])
            gt_vid = gt_vid[:T]
            noisy_vid = noisy_vid[:T]
            x0_hat_vid = x0_hat_vid[:T]
            triptych = np.concatenate([gt_vid, noisy_vid, x0_hat_vid], axis=2)
            imageio.mimsave(triptych_path, triptych, fps=fps, format="mp4")

        if self.global_rank == 0:
            # Rank0 logs exactly once to WandB (avoid duplicate panels).
            # Avoid collectives by polling filesystem for the expected MP4.
            timeout_s = float(os.environ.get("TRAIN_VIDEO_LOG_TIMEOUT_S",
                                             "120"))
            t0 = time.time()
            while (not os.path.exists(triptych_path)) and (time.time() - t0
                                                           < timeout_s):
                time.sleep(0.2)

            if not os.path.exists(triptych_path):
                logger.warning(
                    "Train video preview not found within timeout (step=%s, vis_rank=%s, path=%s). Skipping wandb video log.",
                    step,
                    vis_rank,
                    triptych_path,
                )
                return

            caption = f"step={step}  vis_rank={vis_rank}"
            wandb.log(
                {
                    "train_video_gt_noisy_x0hat":
                    wandb.Video(triptych_path, caption=caption),
                },
                step=step,
            )

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Get the next batch from the dataloader with HYWorld-specific data."""
        batch = next(self.train_loader_iter, None)
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        latents = batch["latent"]
        prompt_embed = batch["prompt_embed"]

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.prompt_embed = prompt_embed.to(get_local_torch_device(),
                                                      dtype=torch.bfloat16)

        if self.action:
            # HYWorld-specific: camera pose and action data
            training_batch.w2c = batch['w2c'].to(get_local_torch_device(),
                                                 dtype=torch.bfloat16)
            training_batch.intrinsic = batch['intrinsic'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.action = batch['action'].to(get_local_torch_device(),
                                                       dtype=torch.bfloat16)
            training_batch.video_path = batch['video_path'][0]
            training_batch.image_cond = batch['image_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.vision_states = batch['vision_states'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.prompt_mask = batch['prompt_mask'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.byt5_text_states = batch['byt5_text_states'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.byt5_text_mask = batch['byt5_text_mask'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.select_window_out_flag = batch[
                'select_window_out_flag'][0]
            selected_history_frame_id = batch.get("selected_history_frame_id")
            current_frame_idx = batch.get("current_frame_idx")
            temporal_context_size = batch.get("temporal_context_size")
            try:
                training_batch.selected_history_frame_id = selected_history_frame_id[
                    0]
            except Exception:
                training_batch.selected_history_frame_id = None
            try:
                training_batch.current_frame_idx = current_frame_idx[0]
            except Exception:
                training_batch.current_frame_idx = None
            try:
                training_batch.temporal_context_size = temporal_context_size[0]
            except Exception:
                training_batch.temporal_context_size = None
            training_batch.i2v_mask = batch['i2v_mask'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.video_path = batch.get('video_path', [''])[0]

        return training_batch

    def timestep_transform(self, t, shift=1.0, num_timesteps=1000.0):
        """Transform timesteps for AR training."""
        t = t / num_timesteps
        t = shift * t / (1 + (shift - 1) * t)
        t = t * num_timesteps
        return t

    def _prepare_ar_dit_inputs(self,
                               training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare AR-style DIT inputs with chunk-based timesteps."""
        latents = training_batch.latents
        batch_size = latents.shape[0]
        latent_t = latents.shape[2]
        latent_h = latents.shape[3]
        latent_w = latents.shape[4]

        noise = torch.randn(latents.shape,
                            generator=self.noise_gen_cuda,
                            device=latents.device,
                            dtype=latents.dtype)

        # Chunk-based timestep sampling for AR training
        chunk_latent_num = 4
        first_chunk_num = 4
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=batch_size *
            ((latent_t - first_chunk_num) // chunk_latent_num + 1),
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        )
        u = u.reshape(batch_size, -1)

        if first_chunk_num == 1:
            first_u = u[:, :first_chunk_num]
            rest_u = u[:, first_chunk_num:]
            rest_u = rest_u.unsqueeze(-1).repeat_interleave(chunk_latent_num,
                                                            dim=-1).reshape(
                                                                batch_size, -1)
            u = torch.cat([first_u, rest_u], dim=1).reshape(-1)
        else:
            u = u.unsqueeze(-1).repeat_interleave(
                chunk_latent_num, dim=-1).reshape(batch_size, -1).reshape(-1)

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        indices = (
            self.noise_scheduler.config.num_train_timesteps -
            self.timestep_transform(indices, self.train_time_shift)).long()
        # Clamp indices to valid range [0, num_train_timesteps - 1]
        indices = indices.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)

        # For memory training, modify timesteps for outside window
        if hasattr(training_batch, 'select_window_out_flag'
                   ) and training_batch.select_window_out_flag == 1:
            for i in range(0, indices.shape[0] - 4, 4):
                rand_val = torch.randint(500, 985, (1, ), device=latents.device)
                indices[i:i + 4] = rand_val

        timesteps = self.noise_scheduler.timesteps[indices].to(
            device=self.device)

        if self.training_args.sp_size > 1:
            sp_group = get_sp_group()
            sp_group.broadcast(timesteps, src=0)

        sigmas = get_sigmas(
            self.noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        sigmas = rearrange(sigmas, '(B D) C T H W -> B C (D T) H W', D=latent_t)
        noisy_model_input = (1.0 -
                             sigmas) * training_batch.latents + sigmas * noise

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = training_batch.latents.shape

        return training_batch

    def _prepare_cond_latents(self, task_type, cond_latents, latents,
                              multitask_mask):
        """Prepare conditional latents and mask for multitask training."""
        latents_concat = None

        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1],
                                         latents.shape[2], latents.shape[3],
                                         latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2],
                                 latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2],
                               latents.shape[3], latents.shape[4])
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(),
                                           mask_ones.cpu(),
                                           mask=multitask_mask.cpu(),
                                           dim=2).to(device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)
        return cond_latents

    def get_task_mask(self, task_type, latent_target_length):
        """Get task mask for I2V or T2V."""
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported!")
        return mask

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Build input kwargs for HYWorld transformer."""
        extra_kwargs = {
            "byt5_text_states": training_batch.byt5_text_states,
            "byt5_text_mask": training_batch.byt5_text_mask,
        }

        multitask_mask = self.get_task_mask(
            "i2v", training_batch.noisy_model_input.shape[2]).to(self.device)
        cond_latents = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, training_batch.noisy_model_input,
            multitask_mask)

        latents_concat = torch.concat(
            [training_batch.noisy_model_input, cond_latents], dim=1)

        # Build input kwargs for HYWorld transformer
        training_batch.input_kwargs = {
            "hidden_states":
            latents_concat,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "timestep_txt":
            torch.tensor(0).unsqueeze(0).to(get_local_torch_device(),
                                            dtype=torch.bfloat16),
            "encoder_hidden_states": [
                training_batch.prompt_embed,
                training_batch.byt5_text_states,
            ],
            "encoder_attention_mask": [
                training_batch.prompt_mask,
                training_batch.byt5_text_mask,
            ],
            "encoder_hidden_states_image": [training_batch.vision_states],
            "viewmats":
            training_batch.w2c,
            "Ks":
            training_batch.intrinsic,
            "action":
            training_batch.action.reshape(-1),
        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        """Forward pass through transformer and compute loss."""
        input_kwargs = training_batch.input_kwargs

        with set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=True):
                model_pred = self.transformer(**input_kwargs)

            if self.training_args.precondition_outputs:
                assert training_batch.sigmas is not None
                model_pred = training_batch.noisy_model_input - model_pred * training_batch.sigmas

            # Stash model output for optional visualization (detach to avoid autograd retention)
            if self._should_log_train_videos(training_batch.current_timestep):
                training_batch.model_pred = model_pred.detach()

            assert training_batch.latents is not None
            assert training_batch.noise is not None

            target = (training_batch.latents
                      if self.training_args.precondition_outputs else
                      training_batch.noise - training_batch.latents)

            i2v_mask = training_batch.i2v_mask
            if hasattr(
                    training_batch, 'select_window_out_flag'
            ) and training_batch.select_window_out_flag == 1 and self.causal:
                i2v_mask[:, :, :-4, ...] = 0  # Only compute loss for last chunk

            assert model_pred.shape == target.shape, \
                f"model_pred.shape: {model_pred.shape}, target.shape: {target.shape}"

            diff = (model_pred.float() * i2v_mask -
                    target.float() * i2v_mask)**2
            loss = diff.sum() / max(
                i2v_mask.sum(),
                1) / self.training_args.gradient_accumulation_steps

            loss.backward()
            avg_loss = loss.detach().clone()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.MAX)
        training_batch.total_loss += avg_loss.item()

        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Clip gradients with max norm."""
        max_grad_norm = self.training_args.max_grad_norm

        if max_grad_norm is not None:
            model_parts = [self.transformer]
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

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Execute one training step."""
        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)
            training_batch = self._prepare_ar_dit_inputs(training_batch)
            training_batch = self._build_input_kwargs(training_batch)
            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        training_batch = self._clip_grad_norm(training_batch)
        grad_norm = torch.tensor(training_batch.grad_norm).to(
            get_local_torch_device())
        dist.all_reduce(grad_norm, op=dist.ReduceOp.MAX)
        training_batch.grad_norm = grad_norm.item()

        if self.global_rank == 0 and training_batch.grad_norm >= 10.0:
            logger.warning(
                "High grad norm: rank=%d, grad_norm=%.2f, timestep=%s, video=%s",
                self.global_rank, training_batch.grad_norm,
                training_batch.current_timestep, training_batch.video_path)

        # NOTE:
        # `clip_grad_norm_...` returns the *pre-clipping* total norm. Even when max_grad_norm is small
        # (e.g. 1.0), this value can stay >10 and would cause us to skip optimizer.step() forever.
        # Rely on clipping for stability; only skip the step for non-finite norms.
        if not torch.isfinite(torch.tensor(training_batch.grad_norm)):
            if self.global_rank == 0:
                logger.warning(
                    "Non-finite grad_norm=%s at step=%s, skipping optimizer step.",
                    training_batch.grad_norm,
                    training_batch.current_timestep,
                )
        else:
            self.optimizer.step()
            self.lr_scheduler.step()

        return training_batch

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare for training iteration."""
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        return training_batch

    def train(self) -> None:
        """Main training loop."""
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info('rank: %s: start training', self.global_rank)

        if not self.post_init_called:
            self.post_init()

        num_trainable_params = count_trainable(self.transformer)
        logger.info("Starting training with %s B trainable parameters",
                    round(num_trainable_params / 1e9, 3))

        # Set random seeds
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(
            device=current_platform.device_name).manual_seed(self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)

        logger.info("Initialized random seeds with seed: %s", self.seed)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()

        # Training loop
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )

        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            # Update dataset max frames (progressive training)
            self.train_dataset.update_max_frames(step)

            start_time = time.perf_counter()

            # VSA sparsity scheduling
            if vsa_available:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                current_decay_times = min(step // vsa_decay_interval_steps,
                                          vsa_sparsity // vsa_decay_rate)
                current_vsa_sparsity = current_decay_times * vsa_decay_rate
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity
            training_batch.attn_metadata = None

            training_batch = self.train_one_step(training_batch)

            loss = training_batch.total_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)

            # Log metrics
            if self.global_rank == 0:
                self.tracker.log(
                    {
                        "train_loss": loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "vsa_sparsity": current_vsa_sparsity,
                    },
                    step=step,
                )

            # Optional: log decoded train-time videos (gt/noisy/pred).
            # IMPORTANT: must run on ALL ranks so round-robin `vis_rank` can decode/save from its local batch.
            # Rank0 will be the only one that logs to WandB; other ranks only write mp4 when selected.
            self._log_train_videos_to_wandb(training_batch, step)

            # Save checkpoints and run full validation
            if step % self.training_args.training_state_checkpointing_steps == 0:
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                self.sp_group.barrier()

                # Run full validation after checkpoint
                if self.training_args.log_validation:
                    self._log_validation(self.transformer, self.training_args,
                                         step)

                self.transformer.train()

        # Save final checkpoint and run validation
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        # Run final validation
        if self.training_args.log_validation:
            self._log_validation(self.transformer, self.training_args,
                                 self.training_args.max_train_steps)

        self.tracker.finish()

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def _log_training_info(self) -> None:
        """Log training information."""
        total_batch_size = (self.world_size *
                            self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size *
                            self.training_args.train_sp_batch_size)
        logger.info("***** Running HYWorld training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s", self.init_steps)
        logger.info("  Instantaneous batch size per device = %s",
                    self.training_args.train_batch_size)
        logger.info("  Total train batch size = %s", total_batch_size)
        logger.info("  Gradient Accumulation steps = %s",
                    self.training_args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s",
                    self.training_args.max_train_steps)
        logger.info("  Total training parameters = %s B",
                    round(count_trainable(self.transformer) / 1e9, 3))
        logger.info("  Master weight dtype: %s",
                    self.transformer.parameters().__next__().dtype)
        logger.info("  Action conditioning: %s", self.action)
        logger.info("  Causal training: %s", self.causal)

        gpu_memory_usage = current_platform.get_torch_device().memory_allocated(
        ) / 1024**2
        logger.info("GPU memory usage before training: %s MB", gpu_memory_usage)

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint."""
        logger.info("Loading checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        resumed_step = load_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.train_dataloader, self.lr_scheduler,
            self.noise_random_generator)
        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = 0


def main(args) -> None:
    """Main entry point for HYWorld training."""
    logger.info("Starting HYWorld training pipeline...")

    pipeline = HYWorldTrainingPipeline.from_pretrained(args.model_path,
                                                       args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("HYWorld training pipeline done")


if __name__ == "__main__":
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)

    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
