import gc
import os
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch
import torchvision
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader

import wandb
from fastvideo.distill.solver import EulerSolver
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset.parquet_datasets import ParquetVideoTextDataset
from fastvideo.v1.distributed import get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs, Mode, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


# from: https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps *
                                                  quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


def reshard_fsdp(model):
    """Reshard FSDP model for EMA updates."""
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


class DistillationPipeline(ComposedPipelineBase, ABC):
    """
    A pipeline for distillation training. All distillation pipelines should inherit from this class.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def initialize_distillation_pipeline(self, fastvideo_args: TrainingArgs):
        logger.info("Initializing distillation pipeline...")
        self.device = fastvideo_args.device
        self.sp_group = get_sp_group()
        self.world_size = self.sp_group.world_size
        self.rank = self.sp_group.rank
        self.local_rank = self.sp_group.local_rank

        # Initialize student model
        self.transformer = self.get_module("transformer")
        assert self.transformer is not None

        self.transformer.requires_grad_(True)
        self.transformer.train()

        # Initialize teacher model without deepcopy to avoid FSDP issues
        logger.info("Creating teacher model...")
        from fastvideo.v1.models.loader.component_loader import (
            TransformerLoader)
        teacher_loader = TransformerLoader()
        transformer_path = os.path.join(self.model_path, "transformer")
        self.teacher_transformer = teacher_loader.load(transformer_path, "",
                                                       fastvideo_args)
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        logger.info("Teacher model initialized")

        # Initialize EMA model if needed
        if fastvideo_args.use_ema:
            logger.info("Creating EMA model...")
            ema_loader = TransformerLoader()
            self.ema_transformer = ema_loader.load(transformer_path, "",
                                                   fastvideo_args)
            self.ema_transformer.requires_grad_(False)
            self.ema_transformer.eval()
            logger.info("EMA model initialized")
        else:
            self.ema_transformer = None

        noise_scheduler = self.get_module("scheduler")
        assert noise_scheduler is not None

        # Initialize solver for distillation
        if fastvideo_args.scheduler_type == "pcm_linear_quadratic":
            linear_steps = int(noise_scheduler.config.num_train_timesteps *
                               fastvideo_args.linear_range)
            sigmas = linear_quadratic_schedule(
                noise_scheduler.config.num_train_timesteps,
                fastvideo_args.linear_quadratic_threshold,
                linear_steps,
            )
            sigmas = torch.tensor(sigmas).to(dtype=torch.float32)
        else:
            sigmas = noise_scheduler.sigmas

        self.solver = EulerSolver(
            sigmas.numpy(),
            noise_scheduler.config.num_train_timesteps,
            euler_timesteps=fastvideo_args.num_euler_timesteps,
        )
        self.solver.to(self.device)

        # Setup optimizer
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=fastvideo_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=fastvideo_args.weight_decay,
            eps=1e-8,
        )

        init_steps = 0
        logger.info("optimizer: %s", optimizer)

        # Setup lr scheduler
        lr_scheduler = get_scheduler(
            fastvideo_args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=fastvideo_args.lr_warmup_steps * self.world_size,
            num_training_steps=fastvideo_args.max_train_steps * self.world_size,
            num_cycles=fastvideo_args.lr_num_cycles,
            power=fastvideo_args.lr_power,
            last_epoch=init_steps - 1,
        )

        # Setup dataset
        train_dataset = ParquetVideoTextDataset(
            fastvideo_args.data_path,
            batch_size=fastvideo_args.train_batch_size,
            cfg_rate=fastvideo_args.cfg,
            num_latent_t=fastvideo_args.num_latent_t)

        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=fastvideo_args.train_batch_size,
            num_workers=fastvideo_args.dataloader_num_workers,
            prefetch_factor=2,
            shuffle=False,
            pin_memory=True,
            drop_last=True)

        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.init_steps = init_steps
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler

        # Get unconditional embeddings
        self.uncond_prompt_embed = torch.zeros(512, 4096).to(torch.float32)
        self.uncond_prompt_mask = torch.zeros(1, 512).bool()
        # self.uncond_prompt_embed = train_dataset.uncond_prompt_embed
        # self.uncond_prompt_mask = train_dataset.uncond_prompt_mask

        if self.rank <= 0:
            project = fastvideo_args.tracker_project_name or "fastvideo"
            wandb.init(project=project, config=fastvideo_args)

    @abstractmethod
    def initialize_validation_pipeline(self, fastvideo_args: FastVideoArgs):
        raise NotImplementedError(
            "Distillation pipelines must implement this method")

    @abstractmethod
    def distill_one_step(self, transformer, model_type, teacher_transformer,
                         ema_transformer, optimizer, lr_scheduler, loader_iter,
                         noise_scheduler, solver, noise_random_generator,
                         gradient_accumulation_steps, sp_size, max_grad_norm,
                         uncond_prompt_embed, uncond_prompt_mask,
                         num_euler_timesteps, multiphase, not_apply_cfg_solver,
                         distill_cfg, ema_decay, pred_decay_weight,
                         pred_decay_type, hunyuan_teacher_disable_cfg):
        """
        Distill one step of the model.
        """
        raise NotImplementedError(
            "Distillation pipeline must implement this method")

    def log_validation(self, transformer, fastvideo_args, global_step):
        """Log validation results during training."""
        fastvideo_args.mode = Mode.INFERENCE
        fastvideo_args.use_cpu_offload = False
        if not fastvideo_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(
            fastvideo_args.model_path)

        # Prepare validation prompts
        validation_dataset = ParquetVideoTextDataset(
            fastvideo_args.validation_prompt_dir,
            batch_size=1,
            cfg_rate=0,
            num_latent_t=fastvideo_args.num_latent_t)

        validation_dataloader = StatefulDataLoader(validation_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   prefetch_factor=2,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   drop_last=False)

        transformer.requires_grad_(False)
        for p in transformer.parameters():
            p.requires_grad = False
        transformer.eval()

        # Add the transformer to the validation pipeline
        self.validation_pipeline.add_module("transformer", transformer)
        self.validation_pipeline.latent_preparation_stage.transformer = transformer
        self.validation_pipeline.denoising_stage.transformer = transformer

        # Process validation prompts
        videos = []
        captions = []
        for _, embeddings, masks, infos in validation_dataloader:
            logger.info(f"infos: {infos}")
            caption = infos['caption']
            captions.append(caption)
            prompt_embeds = embeddings.to(fastvideo_args.device)
            prompt_attention_mask = masks.to(fastvideo_args.device)

            # Calculate sizes
            latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                            sampling_param.height // 8,
                            sampling_param.width // 8]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

            # Prepare batch for validation
            batch = ForwardBatch(
                data_type="video",
                latents=None,
                prompt_embeds=[prompt_embeds],
                prompt_attention_mask=[prompt_attention_mask],
                height=fastvideo_args.num_height,
                width=fastvideo_args.num_width,
                num_frames=fastvideo_args.num_frames,
                num_inference_steps=10,
                guidance_scale=1,
                n_tokens=n_tokens,
                do_classifier_free_guidance=False,
                eta=0.0,
                extra={},
            )

            # Run validation inference
            with torch.inference_mode():
                output_batch = self.validation_pipeline.forward(
                    batch, fastvideo_args)
                samples = output_batch.output

            # Process outputs
            video = rearrange(samples, "b c t h w -> t b c h w")
            frames = []
            for x in video:
                x = torchvision.utils.make_grid(x, nrow=6)
                x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                frames.append((x * 255).numpy().astype(np.uint8))
            # videos.append(frames)
            videos = [frames]

            video_filenames = []
            video_captions = []
            for i, video in enumerate(videos):
                caption = captions[i]
                filename = os.path.join(
                    fastvideo_args.output_dir,
                    f"validation_step_{global_step}_video_{i}.mp4")
                imageio.mimsave(filename, video, fps=sampling_param.fps)
                video_filenames.append(filename)
                video_captions.append(caption)

        # Log validation results
        if self.rank == 0:
            video_filenames = []
            video_captions = []
            for i, video in enumerate(videos):
                caption = captions[i]
                filename = os.path.join(
                    fastvideo_args.output_dir,
                    f"validation_step_{global_step}_video_{i}.mp4")
                imageio.mimsave(filename, video, fps=sampling_param.fps)
                video_filenames.append(filename)
                video_captions.append(caption)

            logs = {
                "validation_videos": [
                    wandb.Video(filename,
                                caption=caption) for filename, caption in zip(
                                    video_filenames, video_captions)
                ]
            }
            wandb.log(logs, step=global_step)

        # Re-enable gradients for training
        fastvideo_args.mode = Mode.DISTILL
        transformer.requires_grad_(True)
        transformer.train()

        gc.collect()
        torch.cuda.empty_cache()
