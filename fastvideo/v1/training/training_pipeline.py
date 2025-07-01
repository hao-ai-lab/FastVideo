# SPDX-License-Identifier: Apache-2.0
import gc
import math
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterator, List

import imageio
import numpy as np
import torch
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadata)
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.dataset.dataloader.schema import (
    pyarrow_schema_t2v, pyarrow_schema_t2v_validation)
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch)
from fastvideo.v1.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp)
from fastvideo.v1.utils import is_vsa_available, set_random_seed

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class TrainingPipeline(ComposedPipelineBase, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[Dict[str, Any]]
    current_epoch: int = 0

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_t2v
        self.validation_dataset_schema = pyarrow_schema_t2v_validation

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.training_args = training_args
        self.device = get_torch_device()
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        assert training_args.seed is not None
        self.seed = training_args.seed
        assert self.transformer is not None
        self.set_schemas()

        self.transformer.requires_grad_(True)
        self.transformer.train()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

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
            parquet_schema=self.train_dataset_schema,
            num_data_workers=training_args.dataloader_num_workers,
            cfg_rate=training_args.training_cfg_rate,
            drop_last=True,
            text_padding_length=training_args.pipeline_config.
            text_encoder_configs[0].arch_config.
            text_len,  # type: ignore[attr-defined]
            seed=self.seed)

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

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.transformer.requires_grad_(True)
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        assert self.train_loader_iter is not None
        assert self.train_dataloader is not None

        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        # latents, encoder_hidden_states, encoder_attention_mask, infos = batch
        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            get_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            get_torch_device(), dtype=torch.bfloat16)
        training_batch.infos = infos

        return training_batch

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        # TODO(will): support other models
        training_batch.latents = normalize_dit_input('wan',
                                                     training_batch.latents)
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        assert training_batch.latents is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert self.noise_random_generator is not None

        batch_size = training_batch.latents.shape[0]
        noise = torch.randn_like(training_batch.latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=batch_size,
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(
            device=training_batch.latents.device)
        if self.training_args.sp_size > 1:
            # Make sure that the timesteps are the same across all sp processes.
            sp_group = get_sp_group()
            sp_group.broadcast(timesteps, src=0)
        sigmas = get_sigmas(
            self.noise_scheduler,
            training_batch.latents.device,
            timesteps,
            n_dim=training_batch.latents.ndim,
            dtype=training_batch.latents.dtype,
        )
        noisy_model_input = (1.0 -
                             sigmas) * training_batch.latents + sigmas * noise

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise

        return training_batch

    def _build_attention_metadata(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        latents = training_batch.latents
        assert latents is not None
        assert training_batch.timesteps is not None
        patch_size = self.training_args.pipeline_config.dit_config.patch_size
        current_vsa_sparsity = training_batch.current_vsa_sparsity

        if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            dit_seq_shape = [
                latents.shape[2] * self.sp_world_size // patch_size[0],
                latents.shape[3] // patch_size[1],
                latents.shape[4] // patch_size[2]
            ]
            training_batch.attn_metadata = VideoSparseAttentionMetadata(
                current_timestep=training_batch.timesteps,
                dit_seq_shape=dit_seq_shape,
                VSA_sparsity=current_vsa_sparsity)
        else:
            training_batch.attn_metadata = None

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None

        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "timestep":
            training_batch.timesteps.to(get_torch_device(),
                                        dtype=torch.bfloat16),
            "encoder_attention_mask":
            training_batch.encoder_attention_mask,
            "return_dict":
            False,
        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.transformer is not None
        assert self.training_args is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.latents is not None
        assert training_batch.noise is not None
        assert training_batch.sigmas is not None

        if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            assert training_batch.attn_metadata is not None
        else:
            assert training_batch.attn_metadata is None

        assert training_batch.input_kwargs is not None
        input_kwargs = training_batch.input_kwargs

        # if 'hunyuan' in self.training_args.model_type:
        #     input_kwargs["guidance"] = torch.tensor(
        #         [1000.0],
        #         device=training_batch.noisy_model_input.device,
        #         dtype=torch.bfloat16)

        with set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            model_pred = self.transformer(**input_kwargs)
            if self.training_args.precondition_outputs:
                model_pred = training_batch.noisy_model_input - model_pred * training_batch.sigmas
            target = training_batch.latents if self.training_args.precondition_outputs else training_batch.noise - training_batch.latents

            # make sure no implicit broadcasting happens
            assert model_pred.shape == target.shape, f"model_pred.shape: {model_pred.shape}, target.shape: {target.shape}"
            loss = (torch.mean((model_pred.float() - target.float())**2) /
                    self.training_args.gradient_accumulation_steps)

            loss.backward()
            avg_loss = loss.detach().clone()

        # logger.info(f"rank: {self.rank}, avg_loss: {avg_loss.item()}",
        #             local_main_process_only=False)
        world_group = get_world_group()
        world_group.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
        training_batch.total_loss += avg_loss.item()

        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
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
        assert self.training_args is not None

        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            # Normalize DIT input
            training_batch = self._normalize_dit_input(training_batch)
            # Create noisy model input
            training_batch = self._prepare_dit_inputs(training_batch)

            # Shard latents across sp groups
            training_batch.latents = shard_latents_across_sp(
                training_batch.latents,
                num_latent_t=self.training_args.num_latent_t)
            # shard noisy_model_input to match
            training_batch.noisy_model_input = shard_latents_across_sp(
                training_batch.noisy_model_input,
                num_latent_t=self.training_args.num_latent_t)
            # shard noise to match latents
            training_batch.noise = shard_latents_across_sp(
                training_batch.noise,
                num_latent_t=self.training_args.num_latent_t)

            training_batch = self._build_attention_metadata(training_batch)
            training_batch = self._build_input_kwargs(training_batch)
            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        training_batch = self._clip_grad_norm(training_batch)

        self.optimizer.step()
        self.lr_scheduler.step()

        training_batch.total_loss = training_batch.total_loss
        training_batch.grad_norm = training_batch.grad_norm
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        assert self.training_args is not None
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

    def train(self) -> None:
        assert self.training_args is not None

        # Set random seeds for deterministic training
        set_random_seed(self.seed)
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        logger.info("Initialized random seeds with seed: %s", self.seed)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.transformer, self.training_args, 1)

        # Train!
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )
        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
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
            if self.global_rank == 0:
                wandb.log(
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
            if step % self.training_args.checkpointing_steps == 0:
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                self.transformer.train()
                self.sp_group.barrier()
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self._log_validation(self.transformer, self.training_args, step)
                gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
                logger.info("GPU memory usage after validation: %s MB",
                            gpu_memory_usage)

        wandb.finish()
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def _log_training_info(self) -> None:
        assert self.training_args is not None
        assert self.training_args.sp_size is not None
        assert self.training_args.gradient_accumulation_steps is not None
        total_batch_size = (self.world_size *
                            self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size *
                            self.training_args.train_sp_batch_size)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s",
                    self.init_steps)  # type: ignore
        logger.info("  Instantaneous batch size per device = %s",
                    self.training_args.train_batch_size)
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size)
        logger.info("  Gradient Accumulation steps = %s",
                    self.training_args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s",
                    self.training_args.max_train_steps)
        logger.info(
            "  Total training parameters per FSDP shard = %s B",
            sum(p.numel()
                for p in self.transformer.parameters() if p.requires_grad) /
            1e9)
        # print dtype
        logger.info("  Master weight dtype: %s",
                    self.transformer.parameters().__next__().dtype)

        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)
        logger.info("VSA validation sparsity: %s",
                    self.training_args.VSA_sparsity)

    def _prepare_validation_inputs(
            self, sampling_param: SamplingParam, training_args: TrainingArgs,
            validation_batch: Dict[str, Any], num_inference_steps: int,
            negative_prompt_embeds: torch.Tensor | None,
            negative_prompt_attention_mask: torch.Tensor | None
    ) -> ForwardBatch:

        assert len(validation_batch['info_list']
                   ) == 1, "Only batch size 1 is supported for validation"
        prompt = validation_batch['info_list'][0]['prompt']
        prompt_embeds = validation_batch['text_embedding']
        prompt_attention_mask = validation_batch['text_attention_mask']

        prompt_embeds = prompt_embeds.to(get_torch_device())
        prompt_attention_mask = prompt_attention_mask.to(get_torch_device())

        # Calculate sizes
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1

        # Prepare batch for validation
        batch = ForwardBatch(
            prompt=prompt,
            data_type="video",
            latents=None,
            seed=self.seed,  # Use deterministic seed
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
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
        return batch

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
        set_random_seed(self.seed)
        logger.info("Using validation seed: %s", self.seed)

        # Prepare validation prompts
        logger.info('fastvideo_args.validation_preprocessed_path: %s',
                    training_args.validation_preprocessed_path)
        validation_dataset, validation_dataloader = build_parquet_map_style_dataloader(
            training_args.validation_preprocessed_path,
            batch_size=1,
            parquet_schema=self.validation_dataset_schema,
            num_data_workers=0,
            cfg_rate=0.0,
            drop_last=False,
            drop_first_row=sampling_param.negative_prompt is not None)
        if sampling_param.negative_prompt:
            negative_prompt_embeds, negative_prompt_attention_mask, negative_prompt = validation_dataset.get_validation_negative_prompt(
            )
            logger.info("Using negative_prompt: %s", negative_prompt)

        transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]

        # Process each validation prompt for each validation step
        for num_inference_steps in validation_steps:
            step_videos: List[np.ndarray] = []
            step_captions: List[str] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_inputs(
                    sampling_param, training_args, validation_batch,
                    num_inference_steps, negative_prompt_embeds,
                    negative_prompt_attention_mask)

                assert batch.prompt is not None and isinstance(
                    batch.prompt, str)
                step_captions.append(batch.prompt)

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
