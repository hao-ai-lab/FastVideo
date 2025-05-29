import sys
import time
from collections import deque
from copy import deepcopy

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

from fastvideo.v1.distributed import cleanup_dist_env_and_memory, get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.wan.wan_pipeline import WanValidationPipeline
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    compute_density_for_timestep_sampling, get_sigmas, normalize_dit_input,
    save_checkpoint)

import wandb  # isort: skip

logger = init_logger(__name__)

# Manual gradient checking flag - set to True to enable gradient verification
ENABLE_GRADIENT_CHECK = False


class WanTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.vae_config.load_encoder = False
        validation_pipeline = WanValidationPipeline.from_pretrained(
            args.model_path, args=None, inference_mode=True)

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
    ) -> tuple[float, float]:
        assert self.training_args is not None
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
            latents = latents.to(self.training_args.device,
                                 dtype=torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(
                self.training_args.device, dtype=torch.bfloat16)
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
        fastvideo_args: FastVideoArgs,
    ):
        assert self.training_args is not None
        noise_random_generator = None

        noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # Train!
        assert self.training_args.sp_size is not None
        assert self.training_args.gradient_accumulation_steps is not None
        total_batch_size = (self.world_size *
                            self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size *
                            self.training_args.train_sp_batch_size)
        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(train_dataset)}")
        # logger.info(f"  Dataloader size = {len(train_dataloader)}")
        # logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info("  Resume training from step %s", self.init_steps)
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

        # Potentially load in the weights and states from a previous save
        if self.training_args.resume_from_checkpoint:
            assert NotImplementedError(
                "resume_from_checkpoint is not supported now.")
            # TODO

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )

        loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        # TODO(will): fix this
        # for i in range(self.init_steps):
        #     next(loader_iter)
        # get gpu memory usage
        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)

        for step in range(self.init_steps + 1, args.max_train_steps + 1):
            start_time = time.perf_counter()

            loss, grad_norm = self.train_one_step(
                self.transformer,
                # args.model_type,
                "wan",
                self.optimizer,
                self.lr_scheduler,
                loader_iter,
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
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )
            if step % self.training_args.checkpointing_steps == 0:
                # Your existing checkpoint saving code
                save_checkpoint(self.transformer, self.rank,
                                self.training_args.output_dir, step)
                self.transformer.train()
                self.sp_group.barrier()
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self.log_validation(self.transformer, self.training_args, step)

        save_checkpoint(self.transformer, self.rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps)

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
