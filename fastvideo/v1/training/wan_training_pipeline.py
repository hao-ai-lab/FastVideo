# SPDX-License-Identifier: Apache-2.0
import importlib.util
import random
import sys
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadata)
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.wan.wan_pipeline import WanValidationPipeline
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp)

import wandb  # isort: skip

vsa_available = False
if importlib.util.find_spec("vsa") is not None:
    vsa_available = True

logger = init_logger(__name__)

# Manual gradient checking flag - set to True to enable gradient verification
ENABLE_GRADIENT_CHECK = False


class WanTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.pipeline_config.vae_config.load_encoder = False
        validation_pipeline = WanValidationPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus)

        self.validation_pipeline = validation_pipeline

    def train_one_step(  # type: ignore[override]
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
        patch_size,
        current_vsa_sparsity,
    ) -> tuple[float, float]:
        assert self.training_args is not None
        self.modules["transformer"].requires_grad_(True)
        self.modules["transformer"].train()

        total_loss = 0.0
        optimizer.zero_grad()

        for _ in range(gradient_accumulation_steps):
            # Get next batch, handling epoch boundaries gracefully
            batch = next(self.train_loader_iter, None)  # type: ignore
            if batch is None:
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                # Reset iterator for next epoch
                self.train_loader_iter = iter(self.train_dataloader)
                # Get first batch of new epoch
                batch = next(self.train_loader_iter)

            latents, encoder_hidden_states, encoder_attention_mask, _ = batch

            latents = latents.to(get_torch_device(), dtype=torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(
                get_torch_device(), dtype=torch.bfloat16)
            latents = shard_latents_across_sp(
                latents, num_latent_t=self.training_args.num_latent_t)

            dit_seq_shape = [
                latents.shape[2] // patch_size[0],
                latents.shape[3] // patch_size[1],
                latents.shape[4] // patch_size[2]
            ]

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

                if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
                    attn_metadata = VideoSparseAttentionMetadata(
                        current_timestep=timesteps,
                        dit_seq_shape=dit_seq_shape,
                        VSA_sparsity=current_vsa_sparsity)
                else:
                    attn_metadata = None

                with set_forward_context(current_timestep=timesteps,
                                         attn_metadata=attn_metadata):
                    model_pred = transformer(**input_kwargs)

                if precondition_outputs:
                    model_pred = noisy_model_input - model_pred * sigmas
                target = latents if precondition_outputs else noise - latents

                loss = (torch.mean((model_pred.float() - target.float())**2) /
                        gradient_accumulation_steps)

            loss.backward()

            avg_loss = loss.detach().clone()
            # logger.info(f"rank: {self.rank}, avg_loss: {avg_loss.item()}",
            #             local_main_process_only=False)
            world_group = get_world_group()
            world_group.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
            total_loss += avg_loss.item()

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
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0

        optimizer.step()
        lr_scheduler.step()
        return total_loss, grad_norm

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ):
        assert self.training_args is not None

        # Set random seeds for deterministic training
        seed = self.training_args.seed if self.training_args.seed is not None else 42

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        noise_random_generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info("Initialized random seeds with seed: %s", seed)

        noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # Train!
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

        if self.training_args.resume_from_checkpoint:
            logger.info("Loading checkpoint from %s",
                        self.training_args.resume_from_checkpoint)
            resumed_step = load_checkpoint(
                self.transformer, self.global_rank,
                self.training_args.resume_from_checkpoint, self.optimizer,
                self.train_dataloader, self.lr_scheduler,
                noise_random_generator)
            if resumed_step > 0:
                self.init_steps = resumed_step
                logger.info("Successfully resumed from step %s", resumed_step)
            else:
                logger.warning(
                    "Failed to load checkpoint, starting from step 0")
                self.init_steps = 0

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        # TODO(will): fix this
        # for i in range(self.init_steps):
        #     next(loader_iter)
        # get gpu memory usage
        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)
        logger.info("VSA validation sparsity: %s",
                    self.training_args.VSA_sparsity)
        self._log_validation(self.transformer, self.training_args, 1)
        if vsa_available:
            vsa_sparsity = self.training_args.VSA_sparsity
            vsa_decay_rate = self.training_args.VSA_decay_rate
            vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps

        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
            if vsa_available:
                current_decay_times = min(step // vsa_decay_interval_steps,
                                          vsa_sparsity // vsa_decay_rate)
                current_vsa_sparsity = current_decay_times * vsa_decay_rate
            else:
                current_vsa_sparsity = 0.0
            loss, grad_norm = self.train_one_step(
                self.transformer,
                # args.model_type,
                "wan",
                self.optimizer,
                self.lr_scheduler,
                self.train_loader_iter,
                noise_scheduler,
                noise_random_generator,
                self.training_args.gradient_accumulation_steps,
                self.training_args.sp_size,
                self.training_args.precondition_outputs,
                self.training_args.max_grad_norm,
                self.training_args.weighting_scheme,
                self.training_args.logit_mean,
                self.training_args.logit_std,
                self.training_args.mode_scale,
                self.training_args.pipeline_config.dit_config.patch_size,
                current_vsa_sparsity,
            )

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            # Manual gradient checking - only at first step
            if step == 1 and ENABLE_GRADIENT_CHECK:
                logger.info("Performing gradient check at step %s", step)
                self.setup_gradient_check(args, self.train_loader_iter,
                                          noise_scheduler,
                                          noise_random_generator)

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
                                self.lr_scheduler, noise_random_generator)
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
                        noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
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
    main(args)
