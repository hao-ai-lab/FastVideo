import gc
import os
import traceback
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch
import torchvision
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset.parquet_datasets import ParquetVideoTextDataset
from fastvideo.v1.distributed import get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.training.training_utils import (
    compute_density_for_timestep_sampling, get_sigmas, normalize_dit_input)

import wandb  # isort: skip

logger = init_logger(__name__)

# Note: if checking with float32, cannot use flash-attn.
GRADIENT_CHECK_DTYPE = torch.bfloat16


class TrainingPipeline(ComposedPipelineBase, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = training_args.device
        self.sp_group = get_sp_group()
        self.world_size = self.sp_group.world_size
        self.rank = self.sp_group.rank
        self.local_rank = self.sp_group.local_rank
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

        self.train_dataset = ParquetVideoTextDataset(
            training_args.data_path,
            batch_size=training_args.train_batch_size,
            rank=self.rank,
            world_size=self.world_size,
            cfg_rate=training_args.cfg,
            num_latent_t=training_args.num_latent_t)

        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=training_args.train_batch_size,
            num_workers=training_args.
            dataloader_num_workers,  # Reduce number of workers to avoid memory issues
            prefetch_factor=2,
            shuffle=False,
            pin_memory=True,
            drop_last=True)

        self.noise_scheduler = noise_scheduler

        if self.rank <= 0:
            project = training_args.tracker_project_name or "fastvideo"
            wandb.init(project=project, config=training_args)

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

    def log_validation(self, transformer, training_args, global_step) -> None:
        assert training_args is not None
        training_args.inference_mode = True
        training_args.use_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Prepare validation prompts
        logger.info('fastvideo_args.validation_prompt_dir: %s',
                    training_args.validation_prompt_dir)
        validation_dataset = ParquetVideoTextDataset(
            training_args.validation_prompt_dir,
            batch_size=1,
            rank=0,
            world_size=1,
            cfg_rate=0,
            num_latent_t=training_args.num_latent_t)

        validation_dataloader = StatefulDataLoader(
            validation_dataset,
            batch_size=1,
            num_workers=1,  # Reduce number of workers to avoid memory issues
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
        self.validation_pipeline.latent_preparation_stage.transformer = transformer  # type: ignore[attr-defined]
        self.validation_pipeline.denoising_stage.transformer = transformer  # type: ignore[attr-defined]

        # Process each validation prompt
        videos = []
        captions = []
        for _, embeddings, masks, infos in validation_dataloader:
            logger.info("infos: %s", infos)
            caption = infos['caption']
            captions.append(caption)
            prompt_embeds = embeddings.to(training_args.device)
            prompt_attention_mask = masks.to(training_args.device)

            # Calculate sizes
            latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                            sampling_param.height // 8,
                            sampling_param.width // 8]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

            # Prepare batch for validation
            # print('shape of embeddings', prompt_embeds.shape)
            batch = ForwardBatch(
                data_type="video",
                latents=None,
                # seed=sampling_param.seed,
                prompt_embeds=[prompt_embeds],
                prompt_attention_mask=[prompt_attention_mask],
                # make sure we use the same height, width, and num_frames as the training pipeline
                height=training_args.num_height,
                width=training_args.num_width,
                num_frames=training_args.num_frames,
                # num_inference_steps=fastvideo_args.validation_sampling_steps,
                num_inference_steps=10,
                # guidance_scale=fastvideo_args.validation_guidance_scale,
                guidance_scale=1,
                n_tokens=n_tokens,
                do_classifier_free_guidance=False,
                eta=0.0,
                extra={},
            )

            # Run validation inference
            with torch.inference_mode():
                output_batch = self.validation_pipeline.forward(
                    batch, training_args)
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
        rank = int(os.environ.get("RANK", 0))

        if rank == 0:
            video_filenames = []
            video_captions = []
            for i, video in enumerate(videos):
                caption = captions[i]
                filename = os.path.join(
                    training_args.output_dir,
                    f"validation_step_{global_step}_video_{i}.mp4")
                imageio.mimsave(filename, video, fps=sampling_param.fps)
                video_filenames.append(filename)
                video_captions.append(
                    caption)  # Store the caption for each video

            logs = {
                "validation_videos": [
                    wandb.Video(filename,
                                caption=caption) for filename, caption in zip(
                                    video_filenames, video_captions)
                ]
            }
            wandb.log(logs, step=global_step)

        # Re-enable gradients for training
        transformer.requires_grad_(True)
        transformer.train()

        gc.collect()
        torch.cuda.empty_cache()

    def gradient_check_parameters(self,
                                  transformer,
                                  latents,
                                  encoder_hidden_states,
                                  encoder_attention_mask,
                                  timesteps,
                                  target,
                                  eps=5e-2,
                                  max_params_to_check=2000) -> float:
        """
        Verify gradients using finite differences for FSDP models with GRADIENT_CHECK_DTYPE.
        Uses standard tolerances for GRADIENT_CHECK_DTYPE precision.
        """
        assert self.training_args is not None
        # Move all inputs to CPU and clear GPU memory
        inputs_cpu = {
            'latents': latents.cpu(),
            'encoder_hidden_states': encoder_hidden_states.cpu(),
            'encoder_attention_mask': encoder_attention_mask.cpu(),
            'timesteps': timesteps.cpu(),
            'target': target.cpu()
        }
        del latents, encoder_hidden_states, encoder_attention_mask, timesteps, target
        torch.cuda.empty_cache()

        def compute_loss() -> torch.Tensor:
            assert self.training_args is not None
            # Move inputs to GPU, compute loss, cleanup
            inputs_gpu = {
                k:
                v.to(self.training_args.device,
                     dtype=GRADIENT_CHECK_DTYPE
                     if k != 'encoder_attention_mask' else None)
                for k, v in inputs_cpu.items()
            }

            # Use GRADIENT_CHECK_DTYPE for more accurate gradient checking
            # with torch.autocast(enabled=False, device_type="cuda"):
            with torch.autocast("cuda", dtype=GRADIENT_CHECK_DTYPE):
                with set_forward_context(
                        current_timestep=inputs_gpu['timesteps'],
                        attn_metadata=None):
                    model_pred = transformer(
                        hidden_states=inputs_gpu['latents'],
                        encoder_hidden_states=inputs_gpu[
                            'encoder_hidden_states'],
                        timestep=inputs_gpu['timesteps'],
                        encoder_attention_mask=inputs_gpu[
                            'encoder_attention_mask'],
                        return_dict=False)[0]

                if self.training_args.precondition_outputs:
                    sigmas = get_sigmas(self.noise_scheduler,
                                        inputs_gpu['latents'].device,
                                        inputs_gpu['timesteps'],
                                        n_dim=inputs_gpu['latents'].ndim,
                                        dtype=inputs_gpu['latents'].dtype)
                    model_pred = inputs_gpu['latents'] - model_pred * sigmas
                    target_adjusted = inputs_gpu['target']
                else:
                    target_adjusted = inputs_gpu['target']

                loss = torch.mean((model_pred - target_adjusted)**2)

            # Cleanup and return
            loss_cpu = loss.cpu()
            del inputs_gpu, model_pred, target_adjusted
            if 'sigmas' in locals():
                del sigmas
            torch.cuda.empty_cache()
            return loss_cpu.to(self.training_args.device)

        try:
            # Get analytical gradients
            transformer.zero_grad()
            analytical_loss = compute_loss()
            analytical_loss.backward()

            # Check gradients for selected parameters
            absolute_errors: list[float] = []
            param_count = 0

            for name, param in transformer.named_parameters():
                if not (param.requires_grad and param.grad is not None
                        and param_count < max_params_to_check
                        and param.grad.abs().max() > 5e-4):
                    continue

                # Get local parameter and gradient tensors
                local_param = param._local_tensor if hasattr(
                    param, '_local_tensor') else param
                local_grad = param.grad._local_tensor if hasattr(
                    param.grad, '_local_tensor') else param.grad

                # Find first significant gradient element
                flat_param = local_param.data.view(-1)
                flat_grad = local_grad.view(-1)
                check_idx = next((i for i in range(min(10, flat_param.numel()))
                                  if abs(flat_grad[i]) > 1e-4), 0)

                # Store original values
                orig_value = flat_param[check_idx].item()
                analytical_grad = flat_grad[check_idx].item()

                # Compute numerical gradient
                for delta in [eps, -eps]:
                    with torch.no_grad():
                        flat_param[check_idx] = orig_value + delta
                        loss = compute_loss()
                        if delta > 0:
                            loss_plus = loss.item()
                        else:
                            loss_minus = loss.item()

                # Restore parameter and compute error
                with torch.no_grad():
                    flat_param[check_idx] = orig_value

                numerical_grad = (loss_plus - loss_minus) / (2 * eps)
                abs_error = abs(analytical_grad - numerical_grad)
                rel_error = abs_error / max(abs(analytical_grad),
                                            abs(numerical_grad), 1e-3)
                absolute_errors.append(abs_error)

                logger.info(
                    "%s[%s]: analytical=%s, numerical=%s, abs_error=%s, rel_error=%s",
                    name, check_idx, analytical_grad, numerical_grad, abs_error,
                    rel_error)

                # param_count += 1

            # Compute and log statistics
            if absolute_errors:
                min_err, max_err, mean_err = min(absolute_errors), max(
                    absolute_errors
                ), sum(absolute_errors) / len(absolute_errors)
                logger.info("Gradient check stats: min=%s, max=%s, mean=%s",
                            min_err, max_err, mean_err)

                if self.rank <= 0:
                    wandb.log({
                        "grad_check/min_abs_error":
                        min_err,
                        "grad_check/max_abs_error":
                        max_err,
                        "grad_check/mean_abs_error":
                        mean_err,
                        "grad_check/analytical_loss":
                        analytical_loss.item(),
                    })
                return max_err

            return float('inf')

        except Exception as e:
            logger.error("Gradient check failed: %s", e)
            traceback.print_exc()
            return float('inf')

    def setup_gradient_check(self, args, loader_iter, noise_scheduler,
                             noise_random_generator) -> float | None:
        """
        Setup and perform gradient check on a fresh batch.
        Args:
            args: Training arguments
            loader_iter: Data loader iterator
            noise_scheduler: Noise scheduler for diffusion
            noise_random_generator: Random number generator for noise
        Returns:
            float or None: Maximum gradient error or None if check is disabled/fails
        """
        assert self.training_args is not None

        try:
            # Get a fresh batch and process it exactly like train_one_step
            check_latents, check_encoder_hidden_states, check_encoder_attention_mask, check_infos = next(
                loader_iter)

            # Process exactly like in train_one_step but use GRADIENT_CHECK_DTYPE
            check_latents = check_latents.to(self.training_args.device,
                                             dtype=GRADIENT_CHECK_DTYPE)
            check_encoder_hidden_states = check_encoder_hidden_states.to(
                self.training_args.device, dtype=GRADIENT_CHECK_DTYPE)
            check_latents = normalize_dit_input("wan", check_latents)
            batch_size = check_latents.shape[0]
            check_noise = torch.randn_like(check_latents)

            check_u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                generator=noise_random_generator,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            check_indices = (check_u *
                             noise_scheduler.config.num_train_timesteps).long()
            check_timesteps = noise_scheduler.timesteps[check_indices].to(
                device=check_latents.device)

            check_sigmas = get_sigmas(
                noise_scheduler,
                check_latents.device,
                check_timesteps,
                n_dim=check_latents.ndim,
                dtype=check_latents.dtype,
            )
            check_noisy_model_input = (
                1.0 - check_sigmas) * check_latents + check_sigmas * check_noise

            # Compute target exactly like train_one_step
            if args.precondition_outputs:
                check_target = check_latents
            else:
                check_target = check_noise - check_latents

            # Perform gradient check with the exact same inputs as training
            max_grad_error = self.gradient_check_parameters(
                transformer=self.transformer,
                latents=
                check_noisy_model_input,  # Use noisy input like in training
                encoder_hidden_states=check_encoder_hidden_states,
                encoder_attention_mask=check_encoder_attention_mask,
                timesteps=check_timesteps,
                target=check_target,
                max_params_to_check=100  # Check more parameters
            )

            if max_grad_error > 5e-2:
                logger.error("❌ Large gradient error detected: %s",
                             max_grad_error)
            else:
                logger.info("✅ Gradient check passed: max error %s",
                            max_grad_error)

            return max_grad_error

        except Exception as e:
            logger.error("Gradient check setup failed: %s", e)
            traceback.print_exc()
            return None
