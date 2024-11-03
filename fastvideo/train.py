# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
import gc
import wandb
from einops import rearrange
from tqdm import tqdm
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, \
    destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state
from fastvideo.utils.communications import prepare_parallel_data, broadcast
import time
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
from torch.nn import functional as F
import transformers
from accelerate import Accelerator, init_empty_weights, DistributedType
from diffusers.utils.torch_utils import is_compiled_module
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed, DummyOptim, DummyScheduler
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKLMochi,
    FlowMatchEulerDiscreteScheduler,
    MochiTransformer3DModel,
    MochiPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from transformers import AutoTokenizer
from fastvideo.utils.ema import EMAModel
from fastvideo.dataset.latent_datasets import LatentDataset, latent_collate_function


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
logger = get_logger(__name__)


import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            

@torch.inference_mode()
def log_validation(args, transformer,vae,  accelerator, weight_dtype, global_step,  ema=False):
    #TODO
    logger.info(f"Running validation....\n")
    transformer = accelerator.unwrap_model(transformer)
    mochi_pipeline = MochiPipeline.from_pretrained(args.pretrained_model_name_or_path,vae=vae, transformer=transformer, torch_dtype=weight_dtype).to(device=accelerator.device)
    videos = []
    for prompt in args.validaiton_prompt:
        logger.info('Processing the ({}) prompt'.format(prompt))
        video = mochi_pipeline(
                                args.prompt,
                                num_frames=args.num_frames,
                                height=args.max_height,
                                width=args.max_width,
                                num_inference_steps=args.validation_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                ).images
        videos.append(video[0])
    # import ipdb;ipdb.set_trace()
    gc.collect()
    torch.cuda.empty_cache()
    videos = torch.stack(videos).numpy()
    videos = rearrange(videos, 'b t h w c -> b t c h w')
    for tracker in accelerator.trackers:
        logs = {
            f"{'ema_' if ema else ''}validation": [
                wandb.Video(video, caption=f"{i}: {prompt}", fps=24)
                for i, (video, prompt) in enumerate(zip(videos, args.validaiton_prompt))
            ]
        }
        tracker.log(logs, step=global_step)

    del mochi_pipeline
    gc.collect()
    torch.cuda.empty_cache()


class ProgressInfo:
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # use LayerNorm, GeLu, SiLu always as fp32 mode
    # TODO: 
    if args.enable_stable_fp32:
        raise NotImplementedError("enable_stable_fp32 is not supported now.")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    initialize_sequence_parallel_state(args.sp_size)

    if not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Create model:
    
    vae = AutoencoderKLMochi.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    if args.enable_tiling:
        vae.enable_tiling()
        vae.tile_sample_stride_height = args.tile_sample_stride
        vae.tile_sample_stride_width = args.tile_sample_stride
    
    load_dtype = torch.bfloat16 # TODO
    transformer = MochiTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    # Set model as trainable.
    transformer.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=weight_dtype)
    # ae.vae.to(accelerator.device, dtype=weight_dtype)
    # transformer.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_transformer = deepcopy(transformer)
        ema_transformer = EMAModel(ema_transformer.parameters(), decay=args.ema_decay, update_after_step=args.ema_start_step,
                             model_cls=MochiTransformer3DModel, model_config=ema_transformer.config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                ema_transformer.save_pretrained(os.path.join(output_dir, "model_ema"))

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model: MochiTransformer3DModel = unwrap_model(model)
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()


    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), MochiTransformer3DModel)
            ema_transformer.load_state_dict(load_model.state_dict())
            ema_transformer.to(accelerator.device)
            del load_model

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = MochiTransformer3DModel.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True
        
        load_model = MochiTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model
        
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
    #     cast_training_params([transformer], dtype=torch.float32)
        

    params_to_optimize = transformer.parameters()

    optimizer = DummyOptim(params_to_optimize, lr=args.learning_rate)
    lr_scheduler = DummyScheduler(
        optimizer, 
        warmup_num_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, 
        total_num_steps=args.max_train_steps * args.gradient_accumulation_steps
    )

    logger.info(f"optimizer: {optimizer}")
    
    train_dataset = LatentDataset(args.data_merge_path, args.num_latent_t)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True, 
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True



    # Prepare everything with our `accelerator`.
    # model.requires_grad_(False)
    # model.pos_embed.requires_grad_(True)
    logger.info(f'before accelerator.prepare')
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    logger.info(f'after accelerator.prepare')
    if args.use_ema:
        ema_transformer.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_project_name = args.tracker_project_name or "fastvideo"
        accelerator.init_trackers(tracker_project_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_info = ProgressInfo(global_step, train_loss=0.0)

    def sync_gradients_info(loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if args.use_ema:
            ema_transformer.step(transformer.parameters())
        progress_bar.update(1)
        progress_info.global_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
        logs = {"step_loss": progress_info.train_loss, "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        
        progress_info.train_loss = 0.0

        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
            if progress_info.global_step % args.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    def run(latents, encoder_hidden_states,encoder_attention_mask, prof):
        global start_time
        start_time = time.time()
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        model_pred = transformer(
            noisy_model_input,
            encoder_hidden_states,
            noise_scheduler.config.num_train_timesteps - timesteps, # the timestep for mochi is invereted
            encoder_attention_mask, # B, L
            return_dict= False
        )[0]
        
        if args.precondition_outputs:
            model_pred = model_pred * sigmas + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
        if args.precondition_outputs:
            target = latents
        else:
            target = latents - noise

        loss = torch.mean(weighting.float() * (model_pred.float() - target.float()) ** 2)

        # Gather the losses across all processes for logging (if we use distributed training).
        # TODO Why repeat here? Weird
        avg_loss = accelerator.reduce(loss.clone().detach(), "mean")
        progress_info.train_loss += avg_loss.item() / args.gradient_accumulation_steps

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params_to_clip = transformer.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            sync_gradients_info(loss)

        if accelerator.is_main_process:

            if progress_info.global_step % args.checkpointing_steps == 0:

                if args.enable_tracker:
                    log_validation(args, transformer, vae,  accelerator,
                                   weight_dtype, progress_info.global_step)

                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer.store(transformer.parameters())
                        ema_transformer.copy_to(transformer.parameters())
                        log_validation(args, transformer,vae,  accelerator,
                                    weight_dtype, progress_info.global_step, ema=True)
                        # Switch back to the original UNet parameters.
                        ema_transformer.restore(transformer.parameters())

        if prof is not None:
            prof.step()


        return loss

    def train_one_step( data_item_, prof_=None):
        latents, cond,attn_mask, cond_mask = data_item_    
        # TODO
        latents = latents.to(dtype=torch.bfloat16)
        cond = cond.to(dtype=torch.bfloat16)
        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        )

        latents = (latents - latents_mean) * vae.config.scaling_factor / latents_std
        
            
        if get_sequence_parallel_state():
            latents, cond, attn_mask, cond_mask = prepare_parallel_data(latents, cond, attn_mask, cond_mask)
            for iter in range(args.train_batch_size * args.sp_size // args.train_sp_batch_size):
                with accelerator.accumulate(transformer):
                    st_idx = iter * args.train_sp_batch_size
                    ed_idx = (iter + 1) * args.train_sp_batch_size
                    encoder_hidden_states=cond[st_idx: ed_idx]
                    # TODO
                    attention_mask=attn_mask[st_idx: ed_idx]
                    encoder_attention_mask=cond_mask[st_idx: ed_idx]
                    run(latents[st_idx: ed_idx], encoder_hidden_states, encoder_attention_mask, prof_)
        else:
            with accelerator.accumulate(transformer):
                encoder_hidden_states=cond
                attention_mask=attn_mask,
                encoder_attention_mask=cond_mask
                run(latents, encoder_hidden_states, encoder_attention_mask, prof_)

        if progress_info.global_step >= args.max_train_steps:
            return True

        return False

    def train_all_epoch(prof_=None):
        for epoch in range(first_epoch, args.num_train_epochs):
            progress_info.train_loss = 0.0
            if progress_info.global_step >= args.max_train_steps:
                return True

            for step, data_item in enumerate(train_dataloader):
                if train_one_step(data_item, prof_):
                    break


    train_all_epoch()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true") # TODO
    parser.add_argument("--group_resolution", action="store_true") # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument('--tile_sample_stride', type=int, default=192)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--downsampler", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument('--enable_stable_fp32', action='store_true') # TODO

    # diffusion setting
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument("--prediction_type", type=str, default=None, help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
            
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument("--precondition_outputs", action="store_true", help="Whether to precondition the outputs of the model.")
    
    # validation & logs
    parser.add_argument("--validaiton_prompt", nargs='+', help="List of prompts to use for validation.")
    parser.add_argument("--validation_sampling_steps", type=int, default=64)
    parser.add_argument('--guidance_scale', type=float, default=4.5)
    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
                        )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
                        )
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
                        )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                        ),
                        )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")

    args = parser.parse_args()
    main(args)
