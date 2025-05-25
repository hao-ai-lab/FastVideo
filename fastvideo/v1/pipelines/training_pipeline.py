import gc
import math
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
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

# import torch.distributed as dist
import wandb
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.checkpoint import save_checkpoint_v1
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset.parquet_datasets import ParquetVideoTextDataset
from fastvideo.v1.distributed import cleanup_dist_env_and_memory, get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.wan.wan_pipeline import WanValidationPipeline

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
        # self.noise_random_generator = noise_random_generator

        # num_update_steps_per_epoch = math.ceil(
        #     len(train_dataloader) / args.gradient_accumulation_steps *
        #     args.sp_size / args.train_sp_batch_size)
        # args.num_train_epochs = math.ceil(args.max_train_steps /
        #                                   num_update_steps_per_epoch)

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
        print('fastvideo_args.validation_prompt_dir',
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
            # print('shape of embeddings', prompt_embeds.shape)
            batch = ForwardBatch(
                # **shallow_asdict(sampling_param),
                data_type="video",
                latents=None,
                # seed=sampling_param.seed,
                # data_type="video",
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
        args = fastvideo_args
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
            f"  Total training parameters per FSDP shard = {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad) / 1e9} B"
        )
        # print dtype
        logger.info(
            f"  Master weight dtype: {self.transformer.parameters().__next__().dtype}"
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
            disable=self.local_rank > 0,
        )

        # loader = sp_parallel_dataloader_wrapper(
        #     train_dataloader,
        #     device,
        #     args.train_batch_size,
        #     args.sp_size,
        #     args.train_sp_batch_size,
        # )
        loader_iter = iter(train_dataloader)

        step_times = deque(maxlen=100)

        # todo future
        for i in range(init_steps):
            next(loader_iter)
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

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

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
                if args.use_lora:
                    raise NotImplementedError("LoRA is not supported now")
                    # Save LoRA weights
                    save_lora_checkpoint(transformer, optimizer, rank,
                                         args.output_dir, step, pipe)
                else:
                    # Your existing checkpoint saving code
                    save_checkpoint_v1(self.transformer, self.rank,
                                       args.output_dir, step)
                    self.transformer.train()
                self.sp_group.barrier()
            if args.log_validation and step % args.validation_steps == 0:
                self.log_validation(self.transformer, args, step)

        if args.use_lora:
            raise NotImplementedError("LoRA is not supported now")
            # save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, args.max_train_steps, pipe)
        else:
            save_checkpoint_v1(self.transformer, self.rank, args.output_dir,
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
    print(args)
    main(args)
