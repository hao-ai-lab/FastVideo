import gc
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy

import imageio
import numpy as np
import torch
import torchvision
import wandb
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

from fastvideo.distill.solver import EulerSolver, extract_into_tensor
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset.parquet_datasets import ParquetVideoTextDataset
from fastvideo.v1.distributed import cleanup_dist_env_and_memory, get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, normalize_dit_input,
    save_checkpoint)
from fastvideo.v1.pipelines.wan.wan_pipeline import WanValidationPipeline

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


def get_norm(model_pred, norms, gradient_accumulation_steps):
    """Calculate and aggregate model prediction norms."""
    fro_norm = (
        torch.linalg.matrix_norm(model_pred, ord="fro") /  # codespell:ignore
        gradient_accumulation_steps)
    largest_singular_value = (torch.linalg.matrix_norm(model_pred, ord=2) /
                              gradient_accumulation_steps)
    absolute_mean = torch.mean(
        torch.abs(model_pred)) / gradient_accumulation_steps
    absolute_max = torch.max(
        torch.abs(model_pred)) / gradient_accumulation_steps

    sp_group = get_sp_group()
    sp_group.all_reduce(fro_norm, op=torch.distributed.ReduceOp.AVG)
    sp_group.all_reduce(largest_singular_value,
                        op=torch.distributed.ReduceOp.AVG)
    sp_group.all_reduce(absolute_mean, op=torch.distributed.ReduceOp.AVG)

    norms["fro"] += torch.mean(fro_norm).item()  # codespell:ignore
    norms["largest singular value"] += torch.mean(largest_singular_value).item()
    norms["absolute mean"] += absolute_mean.item()
    norms["absolute max"] += absolute_max.item()


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
        from fastvideo.v1.models.loader.component_loader import TransformerLoader
        teacher_loader = TransformerLoader()
        transformer_path = os.path.join(self.model_path, "transformer")
        self.teacher_transformer = teacher_loader.load(transformer_path, "", fastvideo_args)
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        logger.info("Teacher model initialized")

        # Initialize EMA model if needed
        if fastvideo_args.use_ema:
            logger.info("Creating EMA model...")
            ema_loader = TransformerLoader()
            self.ema_transformer = ema_loader.load(transformer_path, "", fastvideo_args)
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
            sigmas.numpy()[::-1],
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
            rank=self.rank,
            world_size=self.world_size,
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
        fastvideo_args.mode = "inference"
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
            rank=0,
            world_size=1,
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
            videos.append(frames)

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
        fastvideo_args.mode = "distill"
        transformer.requires_grad_(True)
        transformer.train()

        gc.collect()
        torch.cuda.empty_cache()


class WanDistillationPipeline(DistillationPipeline):
    """
    A distillation pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        pass

    def create_training_stages(self, fastvideo_args: FastVideoArgs):
        pass

    def initialize_validation_pipeline(self, fastvideo_args: FastVideoArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(fastvideo_args)

        args_copy.mode = "inference"
        args_copy.vae_config.load_encoder = False
        validation_pipeline = WanValidationPipeline.from_pretrained(
            fastvideo_args.model_path, args=args_copy)

        self.validation_pipeline = validation_pipeline

    def distill_one_step(
        self,
        transformer,
        model_type,
        teacher_transformer,
        ema_transformer,
        optimizer,
        lr_scheduler,
        loader_iter,
        noise_scheduler,
        solver,
        noise_random_generator,
        gradient_accumulation_steps,
        sp_size,
        max_grad_norm,
        uncond_prompt_embed,
        uncond_prompt_mask,
        num_euler_timesteps,
        multiphase,
        not_apply_cfg_solver,
        distill_cfg,
        ema_decay,
        pred_decay_weight,
        pred_decay_type,
        hunyuan_teacher_disable_cfg,
        weighting_scheme,
        logit_mean,
        logit_std,
        mode_scale,
    ):
        """Perform one step of distillation training."""
        total_loss = 0.0
        optimizer.zero_grad()
        model_pred_norm = {
            "fro": 0.0,  # codespell:ignore
            "largest singular value": 0.0,
            "absolute mean": 0.0,
            "absolute max": 0.0,
        }

        for _ in range(gradient_accumulation_steps):
            (
                latents,
                encoder_hidden_states,
                encoder_attention_mask,
                infos,
            ) = next(loader_iter)

            latents = latents.to(self.device, dtype=torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(
                self.device, dtype=torch.bfloat16)

            latents = normalize_dit_input(model_type, latents)
            batch_size = latents.shape[0]
            noise = torch.randn_like(latents)
            # u = compute_density_for_timestep_sampling(
            #     weighting_scheme=weighting_scheme,
            #     batch_size=batch_size,
            #     generator=noise_random_generator,
            #     logit_mean=logit_mean,
            #     logit_std=logit_std,
            #     mode_scale=mode_scale,
            # )
            # indices = (u * noise_scheduler.config.num_train_timesteps).long()
            # timesteps = noise_scheduler.timesteps[indices].to(
            #     device=latents.device)
            # indices = indices.to(latents.device)
            # if sp_size > 1:
            #     # Make sure that the timesteps are the same across all sp processes.
            #     sp_group = get_sp_group()
            #     sp_group.broadcast(timesteps, src=0)
            # sigmas = get_sigmas(
            #     noise_scheduler,
            #     latents.device,
            #     timesteps,
            #     n_dim=latents.ndim,
            #     dtype=latents.dtype,
            # )
            # noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
            
            indices = torch.randint(0,
                                  num_euler_timesteps, (batch_size, ),
                                  device=latents.device).long()

            if sp_size > 1:
                self.sp_group.broadcast(indices, src=0)

            # Add noise according to flow matching
            sigmas = extract_into_tensor(solver.sigmas, indices,
                                         latents.shape)
            sigmas_prev = extract_into_tensor(solver.sigmas_prev, indices,
                                              latents.shape)

            timesteps = (sigmas *
                         noise_scheduler.config.num_train_timesteps).view(-1)
            timesteps_prev = (
                sigmas_prev *
                noise_scheduler.config.num_train_timesteps).view(-1)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
            noisy_model_input = noisy_model_input.to(torch.bfloat16)

            # Get student model prediction
            with torch.autocast("cuda", dtype=torch.bfloat16):
                input_kwargs = {
                    "hidden_states": noisy_model_input,
                    "encoder_hidden_states": encoder_hidden_states,
                    "timestep": timesteps,
                    "encoder_attention_mask": encoder_attention_mask,
                    "return_dict": False,
                }
                if hunyuan_teacher_disable_cfg:
                    input_kwargs["guidance"] = torch.tensor(
                        [1000.0],
                        device=noisy_model_input.device,
                        dtype=torch.bfloat16)

                with set_forward_context(current_timestep=timesteps,
                                         attn_metadata=None):
                    model_pred = transformer(**input_kwargs)[0]

            # Apply multi-phase prediction
            model_pred, end_index = solver.euler_style_multiphase_pred(
                noisy_model_input, model_pred, indices, multiphase)

            # Get teacher model prediction
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with set_forward_context(current_timestep=timesteps,
                                             attn_metadata=None):
                        cond_teacher_output = teacher_transformer(
                            noisy_model_input,
                            encoder_hidden_states,
                            timesteps,
                            encoder_attention_mask,
                            return_dict=False,
                        )[0].float()

                if not_apply_cfg_solver:
                    uncond_teacher_output = cond_teacher_output
                else:
                    # Get teacher model prediction on unconditional embedding
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        input_kwargs = {
                            "hidden_states": noisy_model_input,
                            "encoder_hidden_states": uncond_prompt_embed.unsqueeze(0).expand(
                                    batch_size, -1, -1),
                            "timestep": timesteps,
                            "encoder_attention_mask": uncond_prompt_mask.unsqueeze(0).expand(batch_size, -1),
                            "return_dict": False,
                        }
                        with set_forward_context(current_timestep=timesteps,
                                                 attn_metadata=None):
                            uncond_teacher_output = teacher_transformer(**input_kwargs)[0]
                teacher_output = uncond_teacher_output + distill_cfg * (
                    cond_teacher_output - uncond_teacher_output)
                x_prev = solver.euler_step(noisy_model_input, teacher_output,
                                           indices).to(torch.bfloat16)

            # Get target prediction
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if ema_transformer is not None:
                        with set_forward_context(
                                current_timestep=timesteps_prev,
                                attn_metadata=None):
                            target_pred = ema_transformer(
                                x_prev,
                                encoder_hidden_states,
                                timesteps_prev,
                                encoder_attention_mask,
                                return_dict=False,
                            )[0]
                    else:
                        with set_forward_context(
                                current_timestep=timesteps_prev,
                                attn_metadata=None):
                            target_pred = transformer(
                                x_prev,
                                encoder_hidden_states,
                                timesteps_prev,
                                encoder_attention_mask,
                                return_dict=False,
                            )[0]

                target, end_index = solver.euler_style_multiphase_pred(
                    x_prev, target_pred, indices, multiphase, True)

            # Calculate loss
            huber_c = 0.001
            loss = (torch.mean(
                torch.sqrt((model_pred.float() - target.float())**2 +
                           huber_c**2) - huber_c) / gradient_accumulation_steps)

            if pred_decay_weight > 0:
                if pred_decay_type == "l1":
                    pred_decay_loss = (
                        torch.mean(torch.sqrt(model_pred.float()**2)) *
                        pred_decay_weight / gradient_accumulation_steps)
                    loss += pred_decay_loss
                elif pred_decay_type == "l2":
                    pred_decay_loss = (torch.mean(model_pred.float()**2) *
                                       pred_decay_weight /
                                       gradient_accumulation_steps)
                    loss += pred_decay_loss
                else:
                    raise NotImplementedError(
                        "pred_decay_type is not implemented")

            # Calculate model prediction norms
            get_norm(model_pred.detach().float(), model_pred_norm,
                     gradient_accumulation_steps)
            loss.backward()

            avg_loss = loss.detach().clone()
            self.sp_group.all_reduce(avg_loss,
                                     op=torch.distributed.ReduceOp.AVG)
            total_loss += avg_loss.item()

        # Update EMA
        if ema_transformer is not None:
            reshard_fsdp(ema_transformer)
            for p_averaged, p_model in zip(ema_transformer.parameters(),
                                           transformer.parameters()):
                with torch.no_grad():
                    p_averaged.copy_(
                        torch.lerp(p_averaged.detach(), p_model.detach(),
                                   1 - ema_decay))

        # Gradient clipping and optimization step
        if max_grad_norm is not None:
            model_parts = [transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0

        optimizer.step()
        lr_scheduler.step()

        return total_loss, grad_norm, model_pred_norm

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: TrainingArgs,
    ):
        assert self.training_args is not None
        train_dataloader = self.train_dataloader
        init_steps = self.init_steps
        lr_scheduler = self.lr_scheduler
        optimizer = self.optimizer
        noise_scheduler = self.noise_scheduler
        solver = self.solver
        noise_random_generator = None
        uncond_prompt_embed = self.uncond_prompt_embed
        uncond_prompt_mask = self.uncond_prompt_mask

        # Train!
        total_batch_size = (self.world_size * self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size * self.training_args.train_sp_batch_size)
        logger.info("***** Running distillation training *****")
        logger.info(f"  Resume training from step {init_steps}")
        logger.info(
            f"  Instantaneous batch size per device = {self.training_args.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.training_args.max_train_steps}")
        logger.info(
            f"  Total training parameters per FSDP shard = {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad) / 1e9} B"
        )
        logger.info(
            f"  Master weight dtype: {self.transformer.parameters().__next__().dtype}"
        )

        # Potentially load in the weights and states from a previous save
        if self.training_args.resume_from_checkpoint:
            raise NotImplementedError(
                "resume_from_checkpoint is not supported now.")

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )

        loader_iter = iter(train_dataloader)
        step_times = deque(maxlen=100)

        # Skip steps if resuming
        for i in range(init_steps):
            next(loader_iter)

        def get_num_phases(multi_phased_distill_schedule, step):
            # step-phase,step-phase
            multi_phases = multi_phased_distill_schedule.split(",")
            phase = multi_phases[-1].split("-")[-1]
            for step_phases in multi_phases:
                phase_step, phase = step_phases.split("-")
                if step <= int(phase_step):
                    return int(phase)
            return int(phase)

        for step in range(init_steps + 1, self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()

            assert self.training_args.multi_phased_distill_schedule is not None
            num_phases = get_num_phases(self.training_args.multi_phased_distill_schedule,
                                        step)
            loss, grad_norm, pred_norm = self.distill_one_step(
                self.transformer,
                "wan",  # model_type
                self.teacher_transformer,
                self.ema_transformer,
                optimizer,
                lr_scheduler,
                loader_iter,
                noise_scheduler,
                solver,
                noise_random_generator,
                self.training_args.gradient_accumulation_steps,
                self.training_args.sp_size,
                self.training_args.max_grad_norm,
                uncond_prompt_embed,
                uncond_prompt_mask,
                self.training_args.num_euler_timesteps,
                num_phases,
                self.training_args.not_apply_cfg_solver,
                self.training_args.distill_cfg,
                self.training_args.ema_decay,
                self.training_args.pred_decay_weight,
                self.training_args.pred_decay_type,
                self.training_args.hunyuan_teacher_disable_cfg,
                self.training_args.weighting_scheme,
                self.training_args.logit_mean,
                self.training_args.logit_std,
                self.training_args.mode_scale,
            )

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
                "phases": num_phases,
            })
            progress_bar.update(1)

            if self.rank <= 0:
                wandb.log(
                    {
                        "train_loss":
                        loss,
                        "learning_rate":
                        lr_scheduler.get_last_lr()[0],
                        "step_time":
                        step_time,
                        "avg_step_time":
                        avg_step_time,
                        "grad_norm":
                        grad_norm,
                        "pred_fro_norm":
                        pred_norm["fro"],  # codespell:ignore
                        "pred_largest_singular_value":
                        pred_norm["largest singular value"],
                        "pred_absolute_mean":
                        pred_norm["absolute mean"],
                        "pred_absolute_max":
                        pred_norm["absolute max"],
                        "phases":
                        num_phases,
                    },
                    step=step,
                )

            if step % self.training_args.checkpointing_steps == 0:
                if self.training_args.use_lora:
                    raise NotImplementedError("LoRA is not supported now")
                else:
                    if self.training_args.use_ema:
                        save_checkpoint(self.ema_transformer, self.rank,
                                           self.training_args.output_dir, step)
                    else:
                        save_checkpoint(self.transformer, self.rank,
                                           self.training_args.output_dir, step)
                self.sp_group.barrier()

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self.log_validation(self.transformer, self.training_args, step)

        # Final checkpoint
        if self.training_args.use_lora:
            raise NotImplementedError("LoRA is not supported now")
        else:
            save_checkpoint(self.transformer, self.rank, self.training_args.output_dir,
                               self.training_args.max_train_steps)

        if get_sp_group():
            cleanup_dist_env_and_memory()


def main(args):
    logger.info("Starting distillation pipeline...")

    pipeline = WanDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    pipeline.forward(None, args)
    logger.info("Distillation pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.v1.fastvideo_args import TrainingArgs
    from fastvideo.v1.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.use_cpu_offload = False
    print(args)
    main(args)
