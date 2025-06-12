import os
import random
import sys
import time
from collections import deque
from copy import deepcopy
import gc

import torchvision
import imageio
from einops import rearrange
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

from fastvideo import SamplingParam
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_world_group, get_torch_device)
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.wan.wan_i2v_pipeline import WanImageToVideoValidationPipeline
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint)
# from fastvideo.v1.dataset.parquet_datasets import ParquetVideoTextDataset
from fastvideo.v1.dataset import build_parquet_map_style_dataloader

import wandb  # isort: skip

logger = init_logger(__name__)

# Manual gradient checking flag - set to True to enable gradient verification
ENABLE_GRADIENT_CHECK = False


class WanI2VTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.flow_shift)

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
        args_copy.log_validation = False
        validation_pipeline = WanImageToVideoValidationPipeline(
            training_args.model_path, 
            fastvideo_args=args_copy, 
            # inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
        )

        self.validation_pipeline = validation_pipeline
        self.latents = None

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
                self.latents,
                self.encoder_hidden_states,
                self.encoder_attention_mask,
                self.infos,
                self.extra_latents
            ) = next(self.train_loader_iter, None)
            if self.latents is None:
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                # Reset iterator for next epoch
                self.train_loader_iter = iter(self.train_dataloader)
                # Get first batch of new epoch
                (
                    self.latents,
                    self.encoder_hidden_states,
                    self.encoder_attention_mask,
                    self.infos,
                    self.extra_latents
                ) = next(self.train_loader_iter)

            latents = self.latents
            encoder_hidden_states = self.encoder_hidden_states
            encoder_attention_mask = self.encoder_attention_mask
            infos = self.infos
            extra_latents = self.extra_latents
            

            # logger.info("rank: %s, caption: %s",
            #             self.rank,
            #             infos['caption'],
            #             local_main_process_only=False)
            # TODO(will): don't hardcode bfloat16
            latents = latents.to(get_torch_device(),
                                 dtype=torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(
                get_torch_device(), dtype=torch.bfloat16)
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

            input_kwargs = {}

            # I2V
            if extra_latents:
                image_embeds, image_latents = extra_latents["img_embed"], extra_latents["img_lat"]
                # Image Embeds
                assert torch.isnan(image_embeds).sum() == 0
                image_embeds = image_embeds.to(get_torch_device(),
                                               dtype=torch.bfloat16)
                input_kwargs["encoder_hidden_states_image"] = image_embeds

                # Image Latents
                assert torch.isnan(image_latents).sum() == 0
                image_latents = image_latents.to(get_torch_device(),
                                                            dtype=torch.bfloat16)
                noisy_model_input = torch.cat(
                    [noisy_model_input, image_latents],
                    dim=1)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                input_kwargs.update({
                    "hidden_states": noisy_model_input,
                    "encoder_hidden_states": encoder_hidden_states,
                    "timestep": timesteps,
                    "encoder_attention_mask": encoder_attention_mask,  # B, L
                    "return_dict": False,
                })
                if 'hunyuan' in model_type:
                    input_kwargs["guidance"] = torch.tensor(
                        [1000.0],
                        device=noisy_model_input.device,
                        dtype=torch.bfloat16)
                with set_forward_context(current_timestep=timesteps,
                                         attn_metadata=None):
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
        self._log_validation(self.transformer, self.training_args, 1)
        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()

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

        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

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
        logger.info('fastvideo_args.validation_path: %s',
                    training_args.validation_path)

        # validation_dataset, validation_dataloader = build_parquet_map_style_dataloader(
        #     training_args.validation_path,
        #     batch_size=1,
        #     num_data_workers=0,
        #     drop_last=False,
        #     cfg_rate=training_args.cfg)

        if sampling_param.negative_prompt:
            _, negative_prompt_embeds, negative_prompt_attention_mask, _ = validation_dataset.get_validation_negative_prompt(
            )
        # validation_dataset = ParquetVideoTextDataset(
        #     training_args.validation_prompt_dir,
        #     batch_size=1,
        #     cfg_rate=training_args.cfg,
        #     num_latent_t=training_args.num_latent_t,
        #     validation=True)
        # if sampling_param.negative_prompt:
        #     _, negative_prompt_embeds, negative_prompt_attention_mask, _ = validation_dataset.get_validation_negative_prompt(
        #     )


        validation_loader_iter = iter(validation_dataloader)
        transformer.eval()

        # Process each validation prompt
        videos = []
        captions = []
        for batch in validation_loader_iter:
            logger.info(batch)
            (
                latents,
                encoder_hidden_states,
                encoder_attention_mask,
                infos,
                extra_latents
            ) = batch
            # caption = infos['caption']
            # captions.extend(caption)
            prompt_embeds = encoder_hidden_states.to(get_torch_device())
            prompt_attention_mask = encoder_attention_mask.to(get_torch_device())
            image_embeds = extra_latents["img_embed"].to(get_torch_device())
            image_latent = extra_latents["img_lat"].to(get_torch_device())

            # Calculate sizes
            latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                            sampling_param.height // 8,
                            sampling_param.width // 8]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

            temporal_compression_factor = training_args.vae_config.arch_config.temporal_compression_ratio
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
                image_embeds=[image_embeds],
                # image_latent=image_latent,
                # make sure we use the same height, width, and num_frames as the training pipeline
                height=training_args.num_height,
                width=training_args.num_width,
                num_frames=num_frames,
                # TODO(will): validation_sampling_steps and
                # validation_guidance_scale are actually passed in as a list of
                # values, like "10,20,30". The validation should be run for each
                # combination of values.
                # num_inference_steps=fastvideo_args.validation_sampling_steps,
                num_inference_steps=sampling_param.num_inference_steps,
                # guidance_scale=fastvideo_args.validation_guidance_scale,
                guidance_scale=sampling_param.guidance_scale,
                n_tokens=n_tokens,
                eta=0.0,
            )

            # Run validation inference
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch = self.validation_pipeline.forward(
                    batch, training_args)
                samples = output_batch.output

            # Re-enable gradients for training
            transformer.requires_grad_(True)
            transformer.train()

            # Process outputs
            video = rearrange(samples, "b c t h w -> t b c h w")
            frames = []
            for x in video:
                x = torchvision.utils.make_grid(x, nrow=6)
                x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                frames.append((x * 255).numpy().astype(np.uint8))
            videos.append(frames)

            # Log validation results
            world_group = get_world_group()
            num_sp_groups = world_group.world_size // self.sp_group.world_size

            # Only sp_group leaders (rank_in_sp_group == 0) need to send their
            # results to global rank 0
            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    # Global rank 0 collects results from all sp_group leaders
                    all_videos = videos  # Start with own results
                    all_captions = captions

                    # Receive from other sp_group leaders
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size  # Global rank of other sp_group leaders
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    video_filenames = []
                    for i, (video,
                            caption) in enumerate(zip(all_videos, all_captions)):
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
                            f"validation_step_{global_step}_video_{i}.mp4")
                        imageio.mimsave(filename, video, fps=sampling_param.fps)
                        video_filenames.append(filename)

                    logs = {
                        "validation_videos": [
                            wandb.Video(filename, caption=caption) for filename,
                            caption in zip(video_filenames, all_captions)
                        ]
                    }
                    wandb.log(logs, step=global_step)
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(videos, dst=0)
                    world_group.send_object(captions, dst=0)

        gc.collect()
        torch.cuda.empty_cache()

def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanI2VTrainingPipeline.from_pretrained(
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