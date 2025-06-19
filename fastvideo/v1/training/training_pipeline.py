# SPDX-License-Identifier: Apache-2.0
import gc
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

import imageio
import numpy as np
import torch
import torchvision
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.distributed import (get_sp_group, get_torch_device,
                                      get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

import wandb  # isort: skip

logger = init_logger(__name__)


class TrainingPipeline(ComposedPipelineBase, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      Dict[str, Any]]]
    current_epoch: int = 0

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = get_torch_device()
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        assert self.transformer is not None

        self.transformer.requires_grad_(True)
        self.transformer.train()

        noise_scheduler = self.modules["scheduler"]
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=training_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps * self.world_size,
            num_training_steps=training_args.max_train_steps * self.world_size,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            last_epoch=self.init_steps - 1,
        )

        self.train_dataset, self.train_dataloader = build_parquet_map_style_dataloader(
            training_args.data_path,
            training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            text_padding_length=training_args.pipeline_config.
            text_encoder_configs[0].arch_config.
            text_len,  # type: ignore[attr-defined]
            seed=training_args.seed)

        self.noise_scheduler = noise_scheduler

        assert training_args.gradient_accumulation_steps is not None
        assert training_args.sp_size is not None
        assert training_args.train_sp_batch_size is not None
        assert training_args.max_train_steps is not None
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                          self.num_update_steps_per_epoch)

        # TODO(will): is there a cleaner way to track epochs?
        self.current_epoch = 0

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "fastvideo"
            wandb.init(project=project,
                       config=training_args,
                       name=training_args.wandb_run_name)

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError(
            "Training pipelines must implement this method")

    @abstractmethod
    def train_one_step(self, transformer, model_type, optimizer, lr_scheduler,
                       loader, noise_scheduler, noise_random_generator,
                       gradient_accumulation_steps, sp_size,
                       precondition_outputs, max_grad_norm, weighting_scheme,
                       logit_mean, logit_std, mode_scale):
        """
        Train one step of the model.
        """
        raise NotImplementedError(
            "Training pipeline must implement this method")

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        assert training_args is not None
        training_args.inference_mode = True
        training_args.use_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Set deterministic seed for validation
        validation_seed = training_args.seed if training_args.seed is not None else 42
        torch.manual_seed(validation_seed)
        torch.cuda.manual_seed_all(validation_seed)

        logger.info("Using validation seed: %s", validation_seed)

        # Prepare validation prompts
        logger.info('fastvideo_args.validation_prompt_dir: %s',
                    training_args.validation_prompt_dir)
        validation_dataset, validation_dataloader = build_parquet_map_style_dataloader(
            training_args.validation_prompt_dir,
            batch_size=1,
            num_data_workers=0,
            drop_last=False,
            drop_first_row=sampling_param.negative_prompt is not None,
            cfg_rate=training_args.cfg)
        if sampling_param.negative_prompt:
            _, negative_prompt_embeds, negative_prompt_attention_mask, _ = validation_dataset.get_validation_negative_prompt(
            )

        transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]

        # Process each validation prompt for each validation step
        for num_inference_steps in validation_steps:
            step_videos: List[np.ndarray] = []
            step_captions: List[str | None] = []

            for _, embeddings, masks, infos in validation_dataloader:
                step_captions.extend([None])  # TODO(peiyuan): add caption
                prompt_embeds = embeddings.to(get_torch_device())
                prompt_attention_mask = masks.to(get_torch_device())

                # Calculate sizes
                latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                                sampling_param.height // 8,
                                sampling_param.width // 8]
                n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

                temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
                num_frames = (training_args.num_latent_t -
                              1) * temporal_compression_factor + 1

                # Prepare batch for validation
                batch = ForwardBatch(
                    data_type="video",
                    latents=None,
                    seed=validation_seed,  # Use deterministic seed
                    generator=torch.Generator(
                        device="cpu").manual_seed(validation_seed),
                    prompt_embeds=[prompt_embeds],
                    prompt_attention_mask=[prompt_attention_mask],
                    negative_prompt_embeds=[negative_prompt_embeds],
                    negative_attention_mask=[negative_prompt_attention_mask],
                    height=training_args.num_height,
                    width=training_args.num_width,
                    num_frames=num_frames,
                    num_inference_steps=
                    num_inference_steps,  # Use the current validation step
                    guidance_scale=sampling_param.guidance_scale,
                    n_tokens=n_tokens,
                    eta=0.0,
                    VSA_sparsity=training_args.VSA_sparsity,
                )

                # Run validation inference
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
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
                step_videos.append(frames)

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
                    for i, (video,
                            caption) in enumerate(zip(all_videos,
                                                      all_captions)):
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
                            f"validation_step_{global_step}_inference_steps_{num_inference_steps}_video_{i}.mp4"
                        )
                        imageio.mimsave(filename, video, fps=sampling_param.fps)
                        video_filenames.append(filename)

                    logs = {
                        f"validation_videos_{num_inference_steps}_steps": [
                            wandb.Video(filename, caption=caption)
                            for filename, caption in zip(
                                video_filenames, all_captions)
                        ]
                    }
                    wandb.log(logs, step=global_step)
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        # Re-enable gradients for training
        transformer.train()
        gc.collect()
        torch.cuda.empty_cache()
