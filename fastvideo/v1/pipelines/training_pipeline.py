import gc
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy

import imageio
import numpy as np
import torch
import torchvision
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import wandb
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset.parquet_datasets import ParquetVideoTextDataset
from fastvideo.v1.distributed import cleanup_dist_env_and_memory, get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.training_utils import (
    compute_density_for_timestep_sampling, get_sigmas, save_checkpoint)
from fastvideo.v1.pipelines.wan.wan_pipeline import WanValidationPipeline

logger = init_logger(__name__)

# Manual gradient checking flag - set to True to enable gradient verification
ENABLE_GRADIENT_CHECK = False
# Note: if checking with float32, cannot use flash-attn.
GRADIENT_CHECK_DTYPE = torch.bfloat16


class TrainingPipeline(ComposedPipelineBase, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def initialize_training_pipeline(self, fastvideo_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = fastvideo_args.device
        self.sp_group = get_sp_group()
        self.world_size = self.sp_group.world_size
        self.rank = self.sp_group.rank
        self.local_rank = self.sp_group.local_rank
        self.transformer = self.get_module("transformer")
        assert self.transformer is not None

        self.transformer.requires_grad_(True)
        self.transformer.train()

        args = fastvideo_args

        noise_scheduler = self.modules["scheduler"]
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

        init_steps = 0
        logger.info("optimizer: %s", optimizer)

        # todo add lr scheduler
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * self.world_size,
            num_training_steps=args.max_train_steps * self.world_size,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
            last_epoch=init_steps - 1,
        )

        train_dataset = ParquetVideoTextDataset(
            args.data_path,
            batch_size=args.train_batch_size,
            rank=self.rank,
            world_size=self.world_size,
            cfg_rate=args.cfg,
            num_latent_t=args.num_latent_t)

        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            num_workers=args.
            dataloader_num_workers,  # Reduce number of workers to avoid memory issues
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

        if self.rank <= 0:
            project = args.tracker_project_name or "fastvideo"
            wandb.init(project=project, config=args)

    @abstractmethod
    def initialize_validation_pipeline(self, fastvideo_args: FastVideoArgs):
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

    def log_validation(self, transformer, fastvideo_args, global_step):
        fastvideo_args.inference_mode = True
        fastvideo_args.use_cpu_offload = False
        if not fastvideo_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(
            fastvideo_args.model_path)

        # Prepare validation prompts
        logger.info('fastvideo_args.validation_prompt_dir: %s',
                    fastvideo_args.validation_prompt_dir)
        validation_dataset = ParquetVideoTextDataset(
            fastvideo_args.validation_prompt_dir,
            batch_size=1,
            rank=0,
            world_size=1,
            cfg_rate=0,
            num_latent_t=args.num_latent_t)

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
        self.validation_pipeline.latent_preparation_stage.transformer = transformer
        self.validation_pipeline.denoising_stage.transformer = transformer

        # Process each validation prompt
        videos = []
        captions = []
        for _, embeddings, masks, infos in validation_dataloader:
            logger.info("infos: %s", infos)
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
            # print('shape of embeddings', prompt_embeds.shape)
            batch = ForwardBatch(
                data_type="video",
                latents=None,
                # seed=sampling_param.seed,
                prompt_embeds=[prompt_embeds],
                prompt_attention_mask=[prompt_attention_mask],
                # make sure we use the same height, width, and num_frames as the training pipeline
                height=args.num_height,
                width=args.num_width,
                num_frames=args.num_frames,
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
        rank = int(os.environ.get("RANK", 0))

        if rank == 0:
            video_filenames = []
            video_captions = []
            for i, video in enumerate(videos):
                caption = captions[i]
                filename = os.path.join(
                    fastvideo_args.output_dir,
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
            # Move inputs to GPU, compute loss, cleanup
            inputs_gpu = {
                k:
                v.to(self.fastvideo_args.device,
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

                if self.fastvideo_args.precondition_outputs:
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
            return loss_cpu.to(self.fastvideo_args.device)

        try:
            # Get analytical gradients
            transformer.zero_grad()
            analytical_loss = compute_loss()
            analytical_loss.backward()

            # Check gradients for selected parameters
            absolute_errors = []
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
                             noise_random_generator):
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
        if not ENABLE_GRADIENT_CHECK:
            return None

        try:
            # Get a fresh batch and process it exactly like train_one_step
            check_latents, check_encoder_hidden_states, check_encoder_attention_mask, check_infos = next(
                loader_iter)

            # Process exactly like in train_one_step but use GRADIENT_CHECK_DTYPE
            check_latents = check_latents.to(self.fastvideo_args.device,
                                             dtype=GRADIENT_CHECK_DTYPE)
            check_encoder_hidden_states = check_encoder_hidden_states.to(
                self.fastvideo_args.device, dtype=GRADIENT_CHECK_DTYPE)
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


class WanTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        pass

    def create_training_stages(self, fastvideo_args: FastVideoArgs):
        pass

    def initialize_validation_pipeline(self, fastvideo_args: FastVideoArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(fastvideo_args)

        args_copy.inference_mode = True
        args_copy.vae_config.load_encoder = False
        validation_pipeline = WanValidationPipeline.from_pretrained(
            args.model_path, args=args_copy)

        self.validation_pipeline = validation_pipeline

    def train_one_step(
        self,
        transformer,
        model_type,
        optimizer,
        lr_scheduler,
        loader_iter,
        noise_scheduler,
        noise_random_generator,
        gradient_accumulation_steps,
        sp_size,
        precondition_outputs,
        max_grad_norm,
        weighting_scheme,
        logit_mean,
        logit_std,
        mode_scale,
    ):
        self.modules["transformer"].requires_grad_(True)
        self.modules["transformer"].train()

        total_loss = 0.0
        optimizer.zero_grad()
        for _ in range(gradient_accumulation_steps):
            (
                latents,
                encoder_hidden_states,
                encoder_attention_mask,
                infos,
            ) = next(loader_iter)
            latents = latents.to(self.fastvideo_args.device,
                                 dtype=torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(
                self.fastvideo_args.device, dtype=torch.bfloat16)
            latents = normalize_dit_input(model_type, latents)
            batch_size = latents.shape[0]
            noise = torch.randn_like(latents)
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=batch_size,
                generator=noise_random_generator,
                logit_mean=logit_mean,
                logit_std=logit_std,
                mode_scale=mode_scale,
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(
                device=latents.device)
            if sp_size > 1:
                # Make sure that the timesteps are the same across all sp processes.
                sp_group = get_sp_group()
                sp_group.broadcast(timesteps, src=0)
            sigmas = get_sigmas(
                noise_scheduler,
                latents.device,
                timesteps,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                input_kwargs = {
                    "hidden_states": noisy_model_input,
                    "encoder_hidden_states": encoder_hidden_states,
                    "timestep": timesteps,
                    "encoder_attention_mask": encoder_attention_mask,  # B, L
                    "return_dict": False,
                }
                if 'hunyuan' in model_type:
                    input_kwargs["guidance"] = torch.tensor(
                        [1000.0],
                        device=noisy_model_input.device,
                        dtype=torch.bfloat16)
                with set_forward_context(current_timestep=timesteps,
                                         attn_metadata=None):
                    model_pred = transformer(**input_kwargs)[0]

                if precondition_outputs:
                    model_pred = noisy_model_input - model_pred * sigmas
                target = latents if precondition_outputs else noise - latents

                loss = (torch.mean((model_pred.float() - target.float())**2) /
                        gradient_accumulation_steps)

            loss.backward()

            avg_loss = loss.detach().clone()
            sp_group = get_sp_group()
            sp_group.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
            total_loss += avg_loss.item()

        # TODO(will): clip grad norm
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        return total_loss, 0.0
        # return total_loss, grad_norm.item()

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: TrainingArgs,
    ):
        args = fastvideo_args
        self.fastvideo_args = args
        train_dataloader = self.train_dataloader
        init_steps = self.init_steps
        lr_scheduler = self.lr_scheduler
        optimizer = self.optimizer
        noise_scheduler = self.noise_scheduler
        noise_random_generator = None

        from diffusers import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # Train!
        total_batch_size = (self.world_size * args.gradient_accumulation_steps /
                            args.sp_size * args.train_sp_batch_size)
        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(train_dataset)}")
        # logger.info(f"  Dataloader size = {len(train_dataloader)}")
        # logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info("  Resume training from step %s", init_steps)
        logger.info("  Instantaneous batch size per device = %s",
                    args.train_batch_size)
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size)
        logger.info("  Gradient Accumulation steps = %s",
                    args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s", args.max_train_steps)
        logger.info(
            "  Total training parameters per FSDP shard = %s B",
            sum(p.numel()
                for p in self.transformer.parameters() if p.requires_grad) /
            1e9)
        # print dtype
        logger.info("  Master weight dtype: %s",
                    self.transformer.parameters().__next__().dtype)

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            assert NotImplementedError(
                "resume_from_checkpoint is not supported now.")
            # TODO

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )

        loader_iter = iter(train_dataloader)

        step_times = deque(maxlen=100)

        # todo future
        for i in range(init_steps):
            next(loader_iter)
        # get gpu memory usage
        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)

        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.perf_counter()

            loss, grad_norm = self.train_one_step(
                self.transformer,
                # args.model_type,
                "wan",
                optimizer,
                lr_scheduler,
                loader_iter,
                noise_scheduler,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.precondition_outputs,
                args.max_grad_norm,
                args.weighting_scheme,
                args.logit_mean,
                args.logit_std,
                args.mode_scale,
            )
            gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
            logger.info("GPU memory usage after train_one_step: %s MB",
                        gpu_memory_usage)

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            # Manual gradient checking - only at first step
            if step == 1 and ENABLE_GRADIENT_CHECK:
                logger.info("Performing gradient check at step %s", step)
                self.setup_gradient_check(args, loader_iter, noise_scheduler,
                                          noise_random_generator)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            if self.rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )
            if step % args.checkpointing_steps == 0:
                # Your existing checkpoint saving code
                save_checkpoint(self.transformer, self.rank, args.output_dir,
                                step)
                self.transformer.train()
                self.sp_group.barrier()
            if args.log_validation and step % args.validation_steps == 0:
                self.log_validation(self.transformer, args, step)

        save_checkpoint(self.transformer, self.rank, args.output_dir,
                        args.max_train_steps)

        if get_sp_group():
            cleanup_dist_env_and_memory()


def main(args):
    logger.info("Starting training pipeline...")

    pipeline = WanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.fastvideo_args
    pipeline.forward(None, args)
    logger.info("Training pipeline done")


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
