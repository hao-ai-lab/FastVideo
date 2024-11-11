import argparse
import math
import os
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, \
    destroy_sequence_parallel_group, get_sequence_parallel_state, nccl_info
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper, broadcast
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from fastvideo.model.mochi_latents_utils import normalize_mochi_dit_input
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
import json
from torch.utils.data.distributed import DistributedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.fsdp_util import get_dit_fsdp_kwargs, get_discriminator_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
)
import torch.nn.functional as F

import numpy as np
from fastvideo.model.modeling_mochi import MochiTransformer3DModel
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_datasets import LatentDataset, latent_collate_function
from fastvideo.gan.discriminator import LADDDiscriminator
import torch.distributed as dist
from safetensors.torch import save_file
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


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
            


def main_print(content):
    if int(os.environ['LOCAL_RANK']) <= 0: 
        print(content)

def save_checkpoint(transformer: MochiTransformer3DModel, rank, output_dir, step):
    main_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
        transformer, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        cpu_state = transformer.state_dict()

    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors 
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(transformer.config)
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    main_print(f"--> checkpoint saved at step {step}")
    
                
def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def student_forward(transformer, latents, encoder_hidden_states, encoder_attention_mask, sp_size):
    batch_size = latents.shape[0]
    noise = torch.randn_like(latents)
    # uniformly sample from [1.0, 0.75, 0.5, 0.25]
    p = [0.25, 0.25, 0.25, 0.25]
    sigmas = np.random.choice([1.0, 0.75, 0.5, 0.25], batch_size, p=p)
    sigmas = torch.tensor(sigmas, device=latents.device, dtype=latents.dtype)
    student_timesteps = (sigmas * 1000).long()
    if sp_size > 1:
        # Make sure that the timesteps are the same across all sp processes.
        broadcast(student_timesteps)

    while len(sigmas.shape) < latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    model_pred = transformer(
        noisy_model_input,
        encoder_hidden_states,
        1000 - student_timesteps,
        encoder_attention_mask, # B, L
        return_dict= False,
    )[0]
    # calculate X1
    model_pred = model_pred * sigmas + noisy_model_input
    return model_pred
        
        
def teacher_forward(transformer, latents, encoder_hidden_states, encoder_attention_mask, sp_size, noise_scheduler, weighting_scheme, logit_mean, logit_std, mode_scale):
    batch_size = latents.shape[0]
    noise = torch.randn_like(latents)
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
    if sp_size > 1:
        # Make sure that the timesteps are the same across all sp processes.
        broadcast(timesteps)
    sigmas = get_sigmas(noise_scheduler, latents.device, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    teacher_features = transformer(
            noisy_model_input,
            encoder_hidden_states,
            noise_scheduler.config.num_train_timesteps - timesteps,
            encoder_attention_mask, # B, L
            return_dict= False,
            output_attn=True
    )[1]
    return teacher_features
    
def train_one_step_generator(student_transformer, teacher_transformer, discriminator, student_optimizer, loader,noise_scheduler, gradient_accumulation_steps, sp_size,  max_grad_norm, weighting_scheme, logit_mean, logit_std, mode_scale):
    total_loss = 0.0
    teacher_transformer.requires_grad_(False)
    discriminator.requires_grad_(False)
    student_transformer.requires_grad_(True)
    student_optimizer.zero_grad()
    for _ in range(gradient_accumulation_steps):
        latents, encoder_hidden_states, latents_attention_mask, encoder_attention_mask = next(loader)
        latents = normalize_mochi_dit_input(latents)

        student_pred = student_forward(student_transformer, latents, encoder_hidden_states, encoder_attention_mask, sp_size)

        teacher_features = teacher_forward(teacher_transformer, student_pred, encoder_hidden_states, encoder_attention_mask, sp_size,  noise_scheduler, weighting_scheme, logit_mean, logit_std, mode_scale)

        logits = discriminator(teacher_features, encoder_hidden_states)
        
        loss = (F.relu(torch.ones_like(logits) - logits)).mean()  / gradient_accumulation_steps
        loss.backward()
        
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item() 
        

    grad_norm = student_transformer.clip_grad_norm_(max_grad_norm)
    student_optimizer.step()
    return total_loss, grad_norm.item()

def train_one_step_discriminator(student_transformer, teacher_transformer, discriminator, discriminator_optimizer, loader,noise_scheduler, gradient_accumulation_steps, sp_size,  max_grad_norm, weighting_scheme, logit_mean, logit_std, mode_scale):
    total_loss = 0.0
    teacher_transformer.requires_grad_(False)
    student_transformer.requires_grad_(False)
    discriminator.requires_grad_(True)
    discriminator_optimizer.zero_grad()
    for _ in range(gradient_accumulation_steps):
        latents, encoder_hidden_states, latents_attention_mask, encoder_attention_mask = next(loader)
        latents = normalize_mochi_dit_input(latents)
        with torch.no_grad():
            student_pred = student_forward(student_transformer, latents, encoder_hidden_states, encoder_attention_mask, sp_size)
            teacher_features_fake = teacher_forward(teacher_transformer, student_pred, encoder_hidden_states, encoder_attention_mask, sp_size,  noise_scheduler, weighting_scheme, logit_mean, logit_std, mode_scale)
            teacher_features_real = teacher_forward(teacher_transformer, latents,      encoder_hidden_states, encoder_attention_mask, sp_size,  noise_scheduler, weighting_scheme, logit_mean, logit_std, mode_scale)
        fake_logits = discriminator(teacher_features_fake, encoder_hidden_states)
        real_logits = discriminator(teacher_features_real, encoder_hidden_states)
        fake_loss = (F.relu(torch.ones_like(fake_logits) + fake_logits)).mean() 
        real_loss = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

        loss = (fake_loss + real_loss)/ 2 / gradient_accumulation_steps
        
        loss.backward()
        
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item() 
        

    grad_norm = discriminator.clip_grad_norm_(max_grad_norm)
    discriminator_optimizer.step()
    return total_loss, grad_norm.item()
        
    
def main(args):
    # use LayerNorm, GeLu, SiLu always as fp32 mode
    # TODO: 
    if args.enable_stable_fp32:
        raise NotImplementedError("enable_stable_fp32 is not supported now.")
    torch.backends.cuda.matmul.allow_tf32 = True
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)




    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)

    # Handle the repository creation
    if rank <=0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.


    # Create model:
    
    main_print(f"--> loading model...")
    student_transformer = MochiTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    teacher_transformer = MochiTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        load_dtype=torch.bfloat16, # TODO: FP8 or INT8 weight type
    )
    # feature_dim, cnn_channel_size, text_cond_dim, num_layers, layers_stride):
    discriminator = LADDDiscriminator(feature_dim=3072,  text_cond_dim=4096, num_layers=48, layers_stride=4)
    # get num of discriminator parameters in billion
    main_print(f"--> discriminator parameters: {sum(p.numel() for p in discriminator.parameters()) / 1e9} B")
    
    teacher_transformer.requires_grad_(False)
    
    
    
    main_print(f"--> Initializing FSDP...")
    dit_fsdp_kwargs = get_dit_fsdp_kwargs("full")
    student_transformer = FSDP(
        student_transformer,
        **dit_fsdp_kwargs,
    )
    teacher_transformer = FSDP(
        teacher_transformer,
        **dit_fsdp_kwargs,
    )
    discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs("full")
    discriminator = FSDP(
        discriminator,
        **discriminator_fsdp_kwargs,
    )
    
    
    main_print(f"--> model loaded")
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(teacher_transformer)
        apply_fsdp_checkpointing(student_transformer)
        
        
    # Set model as trainable.
    student_transformer.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()


    student_optimizer = torch.optim.AdamW(
        student_transformer.parameters(),
        lr=args.student_learning_rate,
        betas=(0.9,0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        betas=(0.9,0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True, 
    )



    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if rank <= 0:
        project = args.tracker_project_name or "fastvideo_distill"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps / args.sp_size * args.train_sp_batch_size
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(f"  Total student training parameters per FSDP shard = {sum(p.numel() for p in student_transformer.parameters() if p.requires_grad) / 1e9} B")
    main_print(f"  Total discriminator training parameters per FSDP shard = {sum(p.numel() for p in discriminator.parameters() if p.requires_grad) / 1e9} B")

    main_print(f"  Master weight dtype: {student_transformer.parameters().__next__().dtype}")


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO 


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable= local_rank > 0,
    )

            

    loader = sp_parallel_dataloader_wrapper(train_dataloader, device, args.train_batch_size, args.sp_size, args.train_sp_batch_size)
    
    for step in range(1, args.discriminator_warmup_steps+1):
        discriminator_loss, discriminator_grad_norm = train_one_step_discriminator(student_transformer, teacher_transformer, discriminator, \
            discriminator_optimizer, loader, noise_scheduler, args.gradient_accumulation_steps, args.sp_size,  \
                args.max_grad_norm, args.weighting_scheme, args.logit_mean, args.logit_std, args.mode_scale)
        progress_bar.set_postfix({"d_loss": discriminator_loss, "d_grad_n": discriminator_grad_norm})
        progress_bar.update(1)
        if rank <= 0:
            wandb.log({"discriminator_loss": discriminator_loss, "discriminator_grad_norm": discriminator_grad_norm}, step=step)
    
    for step in range(args.discriminator_warmup_steps+1, args.max_train_steps+1):
        stage = (step - args.discriminator_warmup_steps) // args.gan_switch_n_steps
        if stage % 2 == 1:
            # Train generator
            generator_loss, student_grad_norm= train_one_step_generator(student_transformer, teacher_transformer, discriminator, \
                student_optimizer, loader, noise_scheduler, args.gradient_accumulation_steps, args.sp_size,  \
                    args.max_grad_norm, args.weighting_scheme, args.logit_mean, args.logit_std, args.mode_scale)
            progress_bar.set_postfix({"g_loss": generator_loss, "g_grad_n": student_grad_norm})
            if rank <= 0:
                wandb.log({"generator_loss": generator_loss, "student_grad_norm": student_grad_norm}, step=step)
            
        else:
            # Train discriminator 
            discriminator_loss, discriminator_grad_norm = train_one_step_discriminator(student_transformer, teacher_transformer, discriminator, \
                discriminator_optimizer, loader, noise_scheduler, args.gradient_accumulation_steps, args.sp_size,  \
                    args.max_grad_norm, args.weighting_scheme, args.logit_mean, args.logit_std, args.mode_scale)
            progress_bar.set_postfix({"d_loss": discriminator_loss, "d_grad_n": discriminator_grad_norm})
            if rank <= 0:
                wandb.log({"discriminator_loss": discriminator_loss, "discriminator_grad_norm": discriminator_grad_norm}, step=step)
                
        progress_bar.update(1)

        if args.log_validation and step  % args.validation_steps == 0:
            log_validation(args, student_transformer, device,
                            torch.bfloat16, step)
                    
        if step  % args.checkpointing_steps == 0:
            save_checkpoint(student_transformer, rank, args.output_dir, step)
        dist.barrier()

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true") # TODO
    parser.add_argument("--group_resolution", action="store_true") # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument('--enable_stable_fp32', action='store_true') # TODO

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
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
    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument("--validation_sampling_steps", type=int, default=64)
    parser.add_argument('--validation_guidance_scale', type=float, default=4.5)
    parser.add_argument('--validation_steps', type=float, default=4.5)
    parser.add_argument("--log_validation", action="store_true")
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
    parser.add_argument("--student_learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--discriminator_learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--discriminator_warmup_steps", type=int, default=0)
    parser.add_argument("--gan_switch_n_steps", type=int, default=1)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
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
