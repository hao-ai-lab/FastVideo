# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
import math
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any
from fastvideo.profiler import profile_region
import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.envs as envs
try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionMetadataBuilder)
    from fastvideo.attention.backends.vmoba import VideoMobaAttentionMetadataBuilder
except Exception:
    pass
from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset import build_parquet_map_style_dataloader
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.dataset.validation_dataset import ValidationDataset
from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import (ComposedPipelineBase, ForwardBatch,
                                 LoRAPipeline, TrainingBatch)
from fastvideo.platforms import current_platform
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.training.trackers import (DummyTracker, TrackerType,
                                         initialize_trackers, Trackers)
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, count_trainable,
    count_trainable_total, get_scheduler, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp)
from fastvideo.utils import (is_vmoba_available, is_vsa_available,
                             set_random_seed, shallow_asdict)

try:
    vsa_available = is_vsa_available()
    vmoba_available = is_vmoba_available()
except Exception:
    vsa_available = False
    vmoba_available = False

logger = init_logger(__name__)


class TrainingPipeline(LoRAPipeline, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0
    train_transformer_2: bool = False
    tracker: TrackerType

    def __init__(
            self,
            model_path: str,
            fastvideo_args: TrainingArgs,
            required_config_modules: list[str] | None = None,
            loaded_modules: dict[str, torch.nn.Module] | None = None) -> None:
        fastvideo_args.inference_mode = False
        self.lora_training = fastvideo_args.lora_training
        if self.lora_training and fastvideo_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")

        set_random_seed(fastvideo_args.seed)  # for lora param init
        super().__init__(model_path, fastvideo_args, required_config_modules,
                         loaded_modules)  # type: ignore
        self.tracker = DummyTracker()

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_t2v

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
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
        self.transformer_2 = self.get_module("transformer_2", None)
        self.seed = training_args.seed
        self.set_schemas()

        # Set random seeds for deterministic training
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            if self.transformer_2 is not None:
                self.transformer_2 = apply_activation_checkpointing(
                    self.transformer_2,
                    checkpointing_type=training_args.
                    enable_gradient_checkpointing_type)

        noise_scheduler = self.modules["scheduler"]
        self.set_trainable()
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))
        # Parse betas from string format "beta1,beta2"
        betas_str = training_args.betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=training_args.learning_rate,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

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
        if self.transformer_2 is not None:
            # Ensure transformer_2 has trainable parameters before creating optimizer
            params_to_optimize_2 = self.transformer_2.parameters()
            params_to_optimize_2 = list(
                filter(lambda p: p.requires_grad, params_to_optimize_2))
            self.optimizer_2 = torch.optim.AdamW(
                params_to_optimize_2,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )
            self.lr_scheduler_2 = get_scheduler(
                training_args.lr_scheduler,
                optimizer=self.optimizer_2,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
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
            seed=self.seed,
            reshuffle_each_epoch=training_args.reshuffle_each_epoch)

        self.noise_scheduler = noise_scheduler
        if self.training_args.boundary_ratio is not None:
            self.boundary_timestep = self.training_args.boundary_ratio * self.noise_scheduler.num_train_timesteps
        else:
            self.boundary_timestep = None

        logger.info("train_dataloader length: %s", len(self.train_dataloader))
        logger.info("train_sp_batch_size: %s",
                    training_args.train_sp_batch_size)
        logger.info("gradient_accumulation_steps: %s",
                    training_args.gradient_accumulation_steps)
        logger.info("sp_size: %s", training_args.sp_size)

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                          self.num_update_steps_per_epoch)

        # TODO(will): is there a cleaner way to track epochs?
        self.current_epoch = 0

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
        project = training_args.tracker_project_name or "fastvideo"
        self.tracker = initialize_trackers(
            trackers,
            experiment_name=project,
            config=tracker_config,
            log_dir=tracker_log_dir,
            run_name=tracker_run_name,
        )

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError(
            "Training pipelines must implement this method")

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.optimizer.zero_grad()
        if self.transformer_2 is not None:
            self.optimizer_2.zero_grad()
        training_batch.total_loss = 0.0
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        with self.tracker.timed("timing/get_next_batch"):
            batch = next(self.train_loader_iter, None)  # type: ignore
            if batch is None:
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                # Reshuffle dataset order each epoch
                self.train_dataset.sampler.set_epoch(self.current_epoch)
                # Reset iterator for next epoch
                self.train_loader_iter = iter(self.train_dataloader)
                # Get first batch of new epoch
                batch = next(self.train_loader_iter)

            latents = batch['vae_latent']
            latents = latents[:, :, :self.training_args.num_latent_t]
            encoder_hidden_states = batch['text_embedding']
            encoder_attention_mask = batch['text_attention_mask']
            infos = batch['info_list']

            training_batch.latents = latents.to(get_local_torch_device(),
                                                dtype=torch.bfloat16)
            training_batch.encoder_hidden_states = encoder_hidden_states.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.encoder_attention_mask = encoder_attention_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.infos = infos

        return training_batch

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        # TODO(will): support other models
        with self.tracker.timed("timing/normalize_input"):
            training_batch.latents = normalize_dit_input(
                'wan',
                training_batch.latents,
                self.get_module("vae"),
            )
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None, "training_args must be set"
        with self.tracker.timed("timing/prepare_dit_inputs"):
            latents = training_batch.latents
            batch_size = latents.shape[0]
            noise = torch.randn(latents.shape,
                                generator=self.noise_gen_cuda,
                                device=latents.device,
                                dtype=latents.dtype)
            timesteps = self._sample_timesteps(batch_size, latents.device)

            if self.training_args.sp_size > 1:
                # Make sure that the timesteps are the same across all sp processes.
                sp_group = get_sp_group()
                sp_group.broadcast(timesteps, src=0)
            sigmas = get_sigmas(
                self.noise_scheduler,
                latents.device,
                timesteps,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            noisy_model_input = (
                1.0 - sigmas) * training_batch.latents + sigmas * noise

            training_batch.noisy_model_input = noisy_model_input
            training_batch.timesteps = timesteps
            training_batch.sigmas = sigmas
            training_batch.noise = noise
            training_batch.raw_latent_shape = training_batch.latents.shape

        return training_batch

    def _sample_timesteps(self, batch_size: int,
                          device: torch.device) -> torch.Tensor:
        # Determine which model to train based on the boundary timestep
        if (self.transformer_2 is not None
                and self.boundary_timestep is not None
                and torch.rand(1, generator=self.noise_random_generator).item()
                <= self.training_args.boundary_ratio):
            self.train_transformer_2 = True
        else:
            self.train_transformer_2 = False

        # Broadcast the decision to all processes
        decision = torch.tensor(1.0 if self.train_transformer_2 else 0.0,
                                device=self.device)
        dist.broadcast(decision, src=0)
        self.train_transformer_2 = decision.item() == 1.0

        # Sample u from the appropriate range
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=batch_size,
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        )

        boundary_ratio = self.training_args.boundary_ratio
        if self.train_transformer_2:
            u = (1 - boundary_ratio
                 ) + u * boundary_ratio  # min: 1 - boundary_ratio, max: 1
        # elif self.transformer_2 is not None:
        #     u = u * (1 - boundary_ratio)  # min: 0, max: 1 - boundary_ratio
        # else:  # patch for now to align with non-MoE timestep logic
        #     pass

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        return self.noise_scheduler.timesteps[indices].to(device=device)

    def _build_attention_metadata(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        latents_shape = training_batch.raw_latent_shape
        patch_size = self.training_args.pipeline_config.dit_config.patch_size
        current_vsa_sparsity = training_batch.current_vsa_sparsity
        assert latents_shape is not None
        assert training_batch.timesteps is not None
        if envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            if not vsa_available:
                raise ImportError(
                    "FASTVIDEO_ATTENTION_BACKEND is set to VIDEO_SPARSE_ATTN, "
                    "but fastvideo_kernel is not correctly installed or detected. "
                    "Please ensure fastvideo-kernel is installed.")
            training_batch.attn_metadata = VideoSparseAttentionMetadataBuilder(  # type: ignore
            ).build(  # type: ignore
                raw_latent_shape=latents_shape[2:5],
                current_timestep=training_batch.timesteps,
                patch_size=patch_size,
                VSA_sparsity=current_vsa_sparsity,
                device=get_local_torch_device())
        elif envs.FASTVIDEO_ATTENTION_BACKEND == "VMOBA_ATTN":
            if not vmoba_available:
                raise ImportError(
                    "FASTVIDEO_ATTENTION_BACKEND is set to VMOBA_ATTN, "
                    "but fastvideo_kernel (or flash_attn>=2.7.4) is not correctly installed."
                )
            moba_params = self.training_args.moba_config.copy()
            moba_params.update({
                "current_timestep":
                training_batch.timesteps,
                "raw_latent_shape":
                training_batch.raw_latent_shape[2:5],
                "patch_size":
                self.training_args.pipeline_config.dit_config.patch_size,
                "device":
                get_local_torch_device(),
            })
            training_batch.attn_metadata = VideoMobaAttentionMetadataBuilder(
            ).build(**moba_params)
        else:
            training_batch.attn_metadata = None

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "encoder_attention_mask":
            training_batch.encoder_attention_mask,
            "return_dict":
            False,
        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN" or vmoba_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VMOBA_ATTN":
            assert training_batch.attn_metadata is not None
        else:
            assert training_batch.attn_metadata is None
        input_kwargs = training_batch.input_kwargs

        # if 'hunyuan' in self.training_args.model_type:
        #     input_kwargs["guidance"] = torch.tensor(
        #         [1000.0],
        #         device=training_batch.noisy_model_input.device,
        #         dtype=torch.bfloat16)
        current_model = self.transformer_2 if self.train_transformer_2 else self.transformer

        with self.tracker.timed("timing/forward_backward"), set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            model_pred = current_model(**input_kwargs)
            if self.training_args.precondition_outputs:
                assert training_batch.sigmas is not None
                model_pred = training_batch.noisy_model_input - model_pred * training_batch.sigmas
            assert training_batch.latents is not None
            assert training_batch.noise is not None
            target = training_batch.latents if self.training_args.precondition_outputs else training_batch.noise - training_batch.latents

            # make sure no implicit broadcasting happens
            assert model_pred.shape == target.shape, f"model_pred.shape: {model_pred.shape}, target.shape: {target.shape}"

            # Defensive: avoid NaNs/div0 if an upstream bug ever produces empty tensors.
            # mean(empty) -> NaN; division by 0 -> inf/NaN. Keep a 0 loss with grad history.
            if model_pred.numel() == 0:
                loss = (model_pred.sum() *
                        0.0) / self.training_args.gradient_accumulation_steps
            else:
                # Compute MSE on one SP shard. We shard along the flattened token axis
                # (t*h*w) with optional padding so (t*h*w) need not be divisible by sp_size.
                sp_world_size = get_sp_group().world_size
                if sp_world_size > 1:
                    sharded_pred = shard_latents_across_sp(model_pred)
                    sharded_target = shard_latents_across_sp(target)
                    local_sse = ((sharded_pred.float() -
                                  sharded_target.float())**2).sum()
                    loss = (sp_world_size * local_sse / model_pred.numel()
                            ) / self.training_args.gradient_accumulation_steps
                else:
                    loss = (torch.mean(
                        (model_pred.float() - target.float())**2) /
                            self.training_args.gradient_accumulation_steps)

            loss.backward()

            avg_loss = loss.detach().clone()

        # logger.info(f"rank: {self.rank}, avg_loss: {avg_loss.item()}",
        #             local_main_process_only=False)
        with self.tracker.timed("timing/reduce_loss"):
            world_group = get_world_group()
            avg_loss = world_group.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        training_batch.total_loss += avg_loss.item()

        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            with self.tracker.timed("timing/clip_grad_norm"):
                # Only clip gradients for the model that is currently training
                if self.train_transformer_2 and self.transformer_2 is not None:
                    model_parts = [self.transformer_2]
                else:
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

    @profile_region("profiler_region_training_train_one_step")
    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            # Normalize DIT input
            training_batch = self._normalize_dit_input(training_batch)
            # Create noisy model input
            training_batch = self._prepare_dit_inputs(training_batch)

            # old sharding code, need to shard latents and noise but not input
            # Shard latents across sp groups
            training_batch.latents = training_batch.latents[:, :, :self.
                                                            training_args.
                                                            num_latent_t]
            # shard noisy_model_input to match
            training_batch.noisy_model_input = training_batch.noisy_model_input[:, :, :
                                                                                self
                                                                                .
                                                                                training_args
                                                                                .
                                                                                num_latent_t]
            # shard noise to match latents
            training_batch.noise = training_batch.noise[:, :, :self.
                                                        training_args.
                                                        num_latent_t]

            training_batch = self._build_attention_metadata(training_batch)
            training_batch = self._build_input_kwargs(training_batch)

            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        training_batch = self._clip_grad_norm(training_batch)

        # Only step the optimizer and scheduler for the model that is currently training
        with self.tracker.timed("timing/optimizer_step"):
            if self.train_transformer_2 and self.transformer_2 is not None:
                self.optimizer_2.step()
                self.lr_scheduler_2.step()
            else:
                self.optimizer.step()
                self.lr_scheduler.step()

        return training_batch

    def _resume_from_checkpoint(self) -> None:
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

    @profile_region("profiler_region_training_train")
    def train(self) -> None:
        assert self.seed is not None, "seed must be set"
        assert self.training_args is not None, "training_args must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info('rank: %s: start training',
                    self.global_rank,
                    local_main_process_only=False)
        if not self.post_init_called:
            self.post_init()
        local_trainable = count_trainable(self.transformer)
        total_trainable = count_trainable_total(
            self.transformer,
            get_local_torch_device(),
        )
        logger.info(
            "Starting training with %s B trainable parameters (total); "
            "this rank shard: %s B",
            round(total_trainable / 1e9, 3),
            round(local_trainable / 1e9, 3),
        )

        if getattr(self, "transformer_2", None) is not None:
            local_trainable_2 = count_trainable(self.transformer_2)
            total_trainable_2 = count_trainable_total(
                self.transformer_2,
                get_local_torch_device(),
            )
            logger.info(
                "Transformer 2: %s B trainable parameters (total); "
                "this rank shard: %s B",
                round(total_trainable_2 / 1e9, 3),
                round(local_trainable_2 / 1e9, 3),
            )

        # Set random seeds for deterministic training
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

        self._best_mf_angle_err_mean = float('inf')
        self._last_mf_angle_err_mean = float('inf')

        self._log_validation(self.transformer, self.training_args,
                             self.init_steps)

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
            elif vmoba_available:
                #TODO: add vmoba sparsity scheduling here
                current_vsa_sparsity = 0.0
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
                metrics = {
                    "train_loss": loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "vsa_sparsity": current_vsa_sparsity,
                }
                try:
                    metrics["batch_size"] = int(
                        training_batch.raw_latent_shape[0])

                    patch_t, patch_h, patch_w = self.training_args.pipeline_config.dit_config.patch_size
                    seq_len = (
                        training_batch.raw_latent_shape[2] // patch_t) * (
                            training_batch.raw_latent_shape[3] // patch_h) * (
                                training_batch.raw_latent_shape[4] // patch_w)
                    if training_batch.encoder_hidden_states is not None:
                        context_len = int(
                            training_batch.encoder_hidden_states.shape[1])
                    else:
                        context_len = 0

                    metrics["dit_seq_len"] = int(seq_len)
                    metrics["context_len"] = context_len

                    arch_config = self.training_args.pipeline_config.dit_config.arch_config

                    metrics["hidden_dim"] = arch_config.hidden_size
                    metrics["num_layers"] = arch_config.num_layers
                    metrics["ffn_dim"] = arch_config.ffn_dim
                except Exception:
                    pass

                self.tracker.log(metrics, step)
            if step % self.training_args.training_state_checkpointing_steps == 0:
                with self.profiler_controller.region(
                        "profiler_region_training_save_checkpoint"):
                    save_checkpoint(self.transformer, self.global_rank,
                                    self.training_args.output_dir, step,
                                    self.optimizer, self.train_dataloader,
                                    self.lr_scheduler,
                                    self.noise_random_generator)
                self.transformer.train()
                self.sp_group.barrier()

            if self.training_args.log_visualization and step % self.training_args.visualization_steps == 0:
                self.visualize_intermediate_latents(training_batch,
                                                    self.training_args, step)

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                with self.profiler_controller.region(
                        "profiler_region_training_validation"):
                    self._log_validation(self.transformer, self.training_args,
                                         step)
                    gpu_memory_usage = current_platform.get_torch_device(
                    ).memory_allocated() / 1024**2
                    trainable_params = round(
                        count_trainable(self.transformer) / 1e9, 3)
                    logger.info(
                        "GPU memory usage after validation: %s MB, trainable params: %sB",
                        gpu_memory_usage, trainable_params)

                best_start = self.training_args.best_checkpoint_start_step
                if (best_start > 0
                        and step >= best_start
                        and self._last_mf_angle_err_mean
                        < self._best_mf_angle_err_mean):
                    self._best_mf_angle_err_mean = (
                        self._last_mf_angle_err_mean)
                    logger.info(
                        "New best mf_angle_err_mean=%.6f at step %d, "
                        "saving best checkpoint",
                        self._best_mf_angle_err_mean, step)
                    save_checkpoint(
                        self.transformer, self.global_rank,
                        self.training_args.output_dir, "best",
                        self.optimizer, self.train_dataloader,
                        self.lr_scheduler,
                        self.noise_random_generator)
                    if self.global_rank == 0:
                        import json
                        meta_path = os.path.join(
                            self.training_args.output_dir,
                            "checkpoint-best",
                            "best_metric.json")
                        with open(meta_path, "w") as f:
                            json.dump({
                                "step": step,
                                "mf_angle_err_mean":
                                    self._best_mf_angle_err_mean,
                            }, f, indent=2)
                        self.tracker.log({
                            "best/mf_angle_err_mean":
                                self._best_mf_angle_err_mean,
                            "best/step": step,
                        }, step)
                    self.transformer.train()
                    self.sp_group.barrier()

        self.tracker.finish()
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if envs.FASTVIDEO_TORCH_PROFILER_DIR:
            logger.info("Stopping profiler...")
            self.profiler_controller.stop()
            logger.info("Profiler stopped.")

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def _log_training_info(self) -> None:
        assert self.training_args is not None, "training_args must be set"
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
        logger.info("  Total training parameters per FSDP shard = %s B",
                    round(count_trainable(self.transformer) / 1e9, 3))
        # print dtype
        logger.info("  Master weight dtype: %s",
                    self.transformer.parameters().__next__().dtype)

        gpu_memory_usage = current_platform.get_torch_device().memory_allocated(
        ) / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)
        logger.info("VSA validation sparsity: %s",
                    self.training_args.VSA_sparsity)

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        if training_args.validation_guidance_scale:
            sampling_param.guidance_scale = float(
                training_args.validation_guidance_scale)
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
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch

    def _post_process_validation_frames(self, frames: list[np.ndarray],
                                        batch: ForwardBatch) -> list[np.ndarray]:
        """Post-process validation frames before saving.
        
        Override this method in subclasses to add custom processing,
        e.g., overlay action indicators for action-conditioned models.
        
        Args:
            frames: List of numpy arrays (H, W, C) representing video frames
            batch: The ForwardBatch containing input data (may include action data)
            
        Returns:
            Processed frames (same format as input)
        """
        return frames

    def _evaluate_validation_video(
        self,
        video_path: str,
        caption: str,
        action_path: str | None,
        global_step: int,
        num_inference_steps: int,
    ) -> dict[str, float] | None:
        """Optionally evaluate a saved validation video and return scalars."""
        del video_path, caption, action_path, global_step
        del num_inference_steps
        return None

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        """
        Generate a validation video and log it to the configured tracker to check the quality during training.
        """
        self._last_mf_angle_err_mean = float('inf')
        training_args.inference_mode = True
        training_args.dit_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Prepare validation prompts
        logger.info('rank: %s: fastvideo_args.validation_dataset_file: %s',
                    self.global_rank,
                    training_args.validation_dataset_file,
                    local_main_process_only=False)
        validation_dataset = ValidationDataset(
            training_args.validation_dataset_file,
            num_samples=training_args.validation_num_samples)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=None,
                                           num_workers=0)

        self.transformer.eval()
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.eval()

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
            step_action_paths: list[str | None] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_batch(sampling_param,
                                                       training_args,
                                                       validation_batch,
                                                       num_inference_steps)
                action_path = validation_batch.get("action_path")
                if not isinstance(action_path, str):
                    action_path = None
                logger.info("rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                            self.global_rank,
                            self.rank_in_sp_group,
                            batch.prompt,
                            local_main_process_only=False)

                assert batch.prompt is not None and isinstance(
                    batch.prompt, str)
                step_captions.append(batch.prompt)

                # Run validation inference
                output_batch = self.validation_pipeline.forward(
                    batch, training_args)
                samples = output_batch.output.cpu()

                if self.rank_in_sp_group != 0:
                    continue

                # Process outputs
                video = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                
                # Apply optional post-processing (e.g., overlay for action-conditioned models)
                frames = self._post_process_validation_frames(frames, batch)
                step_videos.append(frames)
                step_action_paths.append(action_path)

            # Only sp_group leaders (rank_in_sp_group == 0) need to send their
            # results to global rank 0
            if self.rank_in_sp_group == 0:
                local_video_filenames: list[str] = []
                local_validation_metrics: list[dict[str, float]] = []
                local_eval_error: str | None = None

                for i, (video, caption, action_path) in enumerate(
                        zip(step_videos,
                            step_captions,
                            step_action_paths,
                            strict=True)):
                    os.makedirs(training_args.output_dir, exist_ok=True)
                    filename = os.path.join(
                        training_args.output_dir,
                        f"validation_step_{global_step}_inference_steps_{num_inference_steps}_rank_{self.global_rank}_video_{i}.mp4"
                    )
                    imageio.mimsave(filename, video, fps=sampling_param.fps)
                    local_video_filenames.append(filename)

                    try:
                        sample_metrics = self._evaluate_validation_video(
                            video_path=filename,
                            caption=caption,
                            action_path=action_path,
                            global_step=global_step,
                            num_inference_steps=num_inference_steps,
                        )
                        if sample_metrics:
                            local_validation_metrics.append(sample_metrics)
                    except Exception as e:
                        local_eval_error = (
                            f"rank {self.global_rank} validation eval failed "
                            f"for {filename}: {e}")
                        logger.exception(local_eval_error)
                        break

                if self.global_rank == 0:
                    all_video_filenames = local_video_filenames
                    all_captions = step_captions
                    validation_metrics = local_validation_metrics
                    eval_errors: list[str] = []
                    if local_eval_error:
                        eval_errors.append(local_eval_error)

                    # Receive from other sp_group leaders
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size
                        recv_video_filenames = world_group.recv_object(
                            src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        recv_metrics = world_group.recv_object(src=src_rank)
                        recv_error = world_group.recv_object(src=src_rank)

                        all_video_filenames.extend(recv_video_filenames)
                        all_captions.extend(recv_captions)
                        validation_metrics.extend(recv_metrics)
                        if recv_error:
                            eval_errors.append(str(recv_error))

                    if eval_errors:
                        raise RuntimeError(
                            "Validation flow evaluation failed:\n" +
                            "\n".join(eval_errors))

                    artifacts = []
                    for filename, caption in zip(all_video_filenames,
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

                    if validation_metrics:
                        metric_logs: dict[str, float] = {}
                        metric_keys = sorted(
                            {k for row in validation_metrics for k in row.keys()})
                        for metric_key in metric_keys:
                            metric_vals = [
                                row[metric_key] for row in validation_metrics
                                if metric_key in row
                                and np.isfinite(row[metric_key])
                            ]
                            if not metric_vals:
                                continue
                            metric_logs[f"metrics/{metric_key}"] = float(
                                np.mean(metric_vals))
                        self.tracker.log(metric_logs, global_step)

                        mf_val = metric_logs.get(
                            "metrics/mf_angle_err_mean")
                        if mf_val is not None:
                            self._last_mf_angle_err_mean = mf_val
                else:
                    # Other sp_group leaders send their local results to rank 0
                    world_group.send_object(local_video_filenames, dst=0)
                    world_group.send_object(step_captions, dst=0)
                    world_group.send_object(local_validation_metrics, dst=0)
                    world_group.send_object(local_eval_error, dst=0)
                    if local_eval_error:
                        raise RuntimeError(local_eval_error)

        # Broadcast the latest mf_angle_err_mean from rank 0 to all ranks
        _mf_tensor = torch.tensor(
            [self._last_mf_angle_err_mean], device=self.device)
        dist.broadcast(_mf_tensor, src=0)
        self._last_mf_angle_err_mean = _mf_tensor.item()

        # Re-enable gradients for training
        training_args.inference_mode = False
        self.transformer.train()
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.train()

    @staticmethod
    def _mux_audio(
        video_path: str,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Mux audio into video using PyAV."""
        try:
            import av
        except ImportError:
            logger.warning("PyAV not installed; cannot mux audio. "
                           "Install with: pip install av")
            return False

        if torch.is_tensor(audio):
            audio_np = audio.detach().cpu().float().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        elif audio_np.ndim == 2:
            if audio_np.shape[0] <= 8 and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T
        else:
            logger.warning("Unexpected audio shape %s; skipping mux.",
                           audio_np.shape)
            return False

        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767.0).astype(np.int16)
        num_channels = audio_int16.shape[1]
        layout = "stereo" if num_channels == 2 else "mono"

        try:
            import wave
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "muxed.mp4")
                wav_path = os.path.join(tmpdir, "audio.wav")

                # Write audio to WAV file
                with wave.open(wav_path, "wb") as wav_file:
                    wav_file.setnchannels(num_channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())

                # Open input video and audio
                input_video = av.open(video_path)
                input_audio = av.open(wav_path)

                # Create output with both streams
                output = av.open(out_path, mode="w")

                # Add video stream (copy codec from input)
                in_video_stream = input_video.streams.video[0]
                out_video_stream = output.add_stream(
                    codec_name=in_video_stream.codec_context.name,
                    rate=in_video_stream.average_rate,
                )
                out_video_stream.width = in_video_stream.width
                out_video_stream.height = in_video_stream.height
                out_video_stream.pix_fmt = in_video_stream.pix_fmt

                # Add audio stream (AAC)
                out_audio_stream = output.add_stream("aac", rate=sample_rate)
                out_audio_stream.layout = layout

                # Remux video (decode and re-encode to be safe)
                for frame in input_video.decode(video=0):
                    for packet in out_video_stream.encode(frame):
                        output.mux(packet)
                for packet in out_video_stream.encode():
                    output.mux(packet)

                # Encode audio
                for frame in input_audio.decode(audio=0):
                    frame.pts = None  # Let encoder assign PTS
                    for packet in out_audio_stream.encode(frame):
                        output.mux(packet)
                for packet in out_audio_stream.encode():
                    output.mux(packet)

                input_video.close()
                input_audio.close()
                output.close()
                shutil.move(out_path, video_path)
            return True
        except Exception as e:
            logger.warning("Audio mux failed: %s", e)
            return False

    def visualize_intermediate_latents(self, training_batch: TrainingBatch,
                                       training_args: TrainingArgs, step: int):
        """Add visualization data to tracker logging and save frames to disk."""
        raise NotImplementedError(
            "Visualize intermediate latents is not implemented for training pipeline"
        )
