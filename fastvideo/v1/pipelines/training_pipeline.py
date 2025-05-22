import math
import os
import sys
import time
from collections import deque

import torch
from tqdm.auto import tqdm

# import torch.distributed as dist
import wandb
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.checkpoint import save_checkpoint_v1
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
from fastvideo.v1.distributed import cleanup_dist_env_and_memory, get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)

logger = init_logger(__name__)


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler,
               device,
               timesteps,
               n_dim=4,
               dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class WanTrainingPipeline(ComposedPipelineBase):  # == distill_one_step
    _required_config_modules = ["scheduler", "transformer"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def create_training_stages(self, fastvideo_args: FastVideoArgs):
        pass
        # self.add_stage(stage_name="training_stage",
        #                stage=TrainingStage(
        #                    transformer=self.get_module("transformer"),
        #                    scheduler=self.get_module("scheduler")))

    def train_one_step(
        self,
        transformer,
        model_type,
        optimizer,
        lr_scheduler,
        loader,
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
        total_loss = 0.0
        optimizer.zero_grad()
        for _ in range(gradient_accumulation_steps):
            (
                latents,
                encoder_hidden_states,
                latents_attention_mask,
                encoder_attention_mask,
            ) = next(loader)
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
            if precondition_outputs:
                target = latents
            else:
                target = noise - latents

            loss = (torch.mean((model_pred.float() - target.float())**2) /
                    gradient_accumulation_steps)

            loss.backward()

            avg_loss = loss.detach().clone()
            sp_group = get_sp_group()
            sp_group.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
            # dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()

        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        grad_norm = 0.0
        optimizer.step()
        lr_scheduler.step()
        return total_loss, 0.0
        # return total_loss, grad_norm.item()

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ):
        device = fastvideo_args.device
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        rank = int(os.environ.get("RANK", -1))
        assert rank != -1
        assert local_rank != -1
        sp_group = get_sp_group()
        world_size = sp_group.world_size
        rank = sp_group.rank
        args = fastvideo_args
        transformer = self.get_module("transformer")
        # teacher_transformer = self.get_module("teacher_transformer")
        # ema_transformer = None
        assert not fastvideo_args.use_ema, "ema is not supported now"
        # assert teacher_transformer is not None
        assert transformer is not None
        train_dataset = self.train_dataset
        train_dataloader = self.train_dataloader
        init_steps = self.init_steps
        lr_scheduler = self.lr_scheduler
        optimizer = self.optimizer
        noise_scheduler = self.noise_scheduler
        solver = self.solver
        noise_random_generator = None
        uncond_prompt_embed = self.uncond_prompt_embed
        uncond_prompt_mask = self.uncond_prompt_mask

        from diffusers import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # params_to_optimize = transformer.parameters()
        # params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

        # optimizer = torch.optim.AdamW(
        #     params_to_optimize,
        #     lr=args.learning_rate,
        #     betas=(0.9, 0.999),
        #     weight_decay=args.weight_decay,
        #     eps=1e-8,
        # )

        # init_steps = 0
        # if args.resume_from_lora_checkpoint:
        #     transformer, optimizer, init_steps = resume_lora_optimizer(transformer, args.resume_from_lora_checkpoint,
        #                                                             optimizer)
        # main_print(f"optimizer: {optimizer}")

        # lr_scheduler = get_scheduler(
        #     args.lr_scheduler,
        #     optimizer=optimizer,
        #     num_warmup_steps=args.lr_warmup_steps,
        #     num_training_steps=args.max_train_steps,
        #     num_cycles=args.lr_num_cycles,
        #     power=args.lr_power,
        #     last_epoch=init_steps - 1,
        # )

        # Train!
        total_batch_size = (world_size * args.gradient_accumulation_steps /
                            args.sp_size * args.train_sp_batch_size)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Dataloader size = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Resume training from step {init_steps}")
        logger.info(
            f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(
            f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
        )
        # print dtype
        logger.info(
            f"  Master weight dtype: {transformer.parameters().__next__().dtype}"
        )

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
            disable=local_rank > 0,
        )

        loader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )

        step_times = deque(maxlen=100)

        # todo future
        for i in range(init_steps):
            next(loader)
        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.perf_counter()
            loss, grad_norm = self.train_one_step(
                transformer,
                # args.model_type,
                "wan",
                optimizer,
                lr_scheduler,
                loader,
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

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            if rank <= 0:
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
                if args.use_lora:
                    raise NotImplementedError("LoRA is not supported now")
                    # Save LoRA weights
                    save_lora_checkpoint(transformer, optimizer, rank,
                                         args.output_dir, step, pipe)
                else:
                    # Your existing checkpoint saving code
                    save_checkpoint_v1(transformer, rank, args.output_dir, step)
                    transformer.train()
                sp_group.barrier()
            # if args.log_validation and step % args.validation_steps == 0:
            #     log_validation(args, transformer, device, torch.bfloat16, step, shift=args.flow_shift)

        if args.use_lora:
            raise NotImplementedError("LoRA is not supported now")
            # save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, args.max_train_steps, pipe)
        else:
            save_checkpoint_v1(transformer, rank, args.output_dir,
                            args.max_train_steps)

        if get_sp_group():
            cleanup_dist_env_and_memory()


def main(args):
    logger.info("Starting training pipeline...")
    pipeline = WanTrainingPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", args=args)
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
    print(args)
    main(args)
