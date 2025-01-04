import argparse
from email.policy import strict
import logging
import math
import numpy as np
import os
import shutil
from pathlib import Path
from einops import rearrange
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper, broadcast
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from fastvideo.utils.load import load_transformer

from fastvideo.models.mochi_hf.pipeline_mochi import linear_quadratic_schedule, retrieve_timesteps
import json
from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import (
    get_dit_fsdp_kwargs,
    apply_fsdp_checkpointing,
    get_discriminator_fsdp_kwargs,
)
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from fastvideo.distill.discriminator import Discriminator, DiscriminatorHead
from fastvideo.distill.solver import EulerSolver, extract_into_tensor
from copy import deepcopy
from diffusers.optimization import get_scheduler
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_lora_checkpoint,
    resume_lora_optimizer,
    resume_training_generator_fake_transformer,
)
from fastvideo.utils.communications import all_gather
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.validation import prepare_latents
from fastvideo.distill.solver import PCMFMScheduler

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
from typing import Optional, Union, List

from torch.distributed.fsdp import FullOptimStateDictConfig
from safetensors.torch import save_file
from fastvideo.utils.logging_ import ForkedPdb
#ForkedPdb().set_trace()

def save_checkpoint(model, fake_model, rank, output_dir, step, discriminator=False):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.state_dict()
    with FSDP.state_dict_type(
        fake_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state2 = fake_model.state_dict()
    # todo move to get_state_dict
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    # save using safetensors
    if rank <= 0 and not discriminator:
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        weight_path2 = os.path.join(save_dir, "diffusion_pytorch_model2.safetensors")
        save_file(cpu_state, weight_path)
        save_file(cpu_state2, weight_path2)
        config_dict = dict(model.config)
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    else:
        weight_path = os.path.join(save_dir, "discriminator_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        weight_path2 = os.path.join(save_dir, "discriminator_pytorch_model2.safetensors")
        save_file(cpu_state2, weight_path2)
    

def get_flow_scheduler(
    scheduler_type,
    shift=1.0,
    num_euler_timesteps=100,
    linear_quadratic_threshold=0.025,
    linear_range=0.5,
):
    if scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        linear_quadraic = True if scheduler_type == "pcm_linear_quadratic" else False
        scheduler = PCMFMScheduler(
            1000,
            shift,
            num_euler_timesteps,
            linear_quadraic,
            linear_quadratic_threshold,
            linear_range,
        )
    return scheduler

def sample_model_latent(
    transformer,
    prompt_embeds,
    prompt_attention_mask,
    model_type,
    scheduler_type="euler",
    height: Optional[int] = 784,
    width: Optional[int] = 1280,
    num_frames: int = 29,
    num_inference_steps: int = 4,
    timesteps: List[int] = None,
    guidance_scale: float = 4.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
):
    if model_type == "mochi":
        vae_spatial_scale_factor = 8
        vae_temporal_scale_factor = 6
        num_channels_latents = 12
    elif model_type == "hunyuan":
        vae_spatial_scale_factor = 8
        vae_temporal_scale_factor = 4
        num_channels_latents = 16
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    scheduler = get_flow_scheduler(scheduler_type)
    device = transformer.device
    do_classifier_free_guidance = guidance_scale > 1.0

    batch_size = prompt_embeds.shape[0]

    # 4. Prepare latent variables
    # TODO: Remove hardcore
    latents = prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        prompt_embeds.dtype,
        device,
        generator,
        vae_spatial_scale_factor,
        vae_temporal_scale_factor,
    )
    
    world_size, rank = nccl_info.sp_size, nccl_info.rank_within_group
    if get_sequence_parallel_state():
        latents = rearrange(
            latents, "b t (n s) h w -> b t n s h w", n=world_size
        ).contiguous()
        latents = latents[:, :, rank, :, :, :]
    
    # 5. Prepare timestep
    # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
    threshold_noise = 0.025
    sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
    sigmas = np.array(sigmas)
    if scheduler_type == "euler":
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas,
        )
    else:
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device,
        )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

    # 6. Denoising loop
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    # write with tqdm instead
    # only enable if nccl_info.global_rank == 0
    
    # sample [1000, 750, 500, 250]
    indice = torch.randint(0, num_inference_steps, (batch_size,), device=device)
    t = timesteps[indice]
    
    timestep = t.expand(latent_model_input.shape[0])    
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
        
    # forward with grad
    with torch.autocast("cuda", dtype=torch.bfloat16):
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
        )[0]

    # Mochi CFG + Sampling runs in FP32
    noise_pred = noise_pred.to(torch.float32)
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # compute the previous noisy sample x_t -> x_t-1
    latents_dtype = latents.dtype
    #if nccl_info.rank_within_group == 0:
    #    print("noise_pred ", noise_pred.shape)
    #    print("latents ", latents.shape)
    #    print("timestep ", t, type(t), t.shape)
    latents = scheduler.step(
        noise_pred, t, latents.to(torch.float32), return_dict=False
    )[0]
    latents = latents.to(latents_dtype)

    if latents.dtype != latents_dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            latents = latents.to(latents_dtype)

    #if get_sequence_parallel_state():
    #    latents = all_gather(latents, dim=2)

    return latents

def predict_noise(
    transformer,
    latents,
    encoder_hidden_states,
    timesteps,
    encoder_attention_mask, 
    guidance_scale=1,
    solver=None,
    index=None,
    multiphase=None,
):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        noise_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        
    latents, end_index = solver.euler_style_multiphase_pred(
        latents, noise_pred, index, multiphase
    )
            
    return noise_pred, latents

def distill_one_step_dmd(
    transformer,
    optimizer,
    model_type,
    real_transformer,
    fake_transformer,
    guidance_optimizer,
    discriminator,
    lr_scheduler,
    loader,
    noise_scheduler,
    solver,
    noise_random_generator,
    sp_size,
    max_grad_norm,
    uncond_prompt_embed,
    uncond_prompt_mask,
    num_euler_timesteps,
    multiphase,
    not_apply_cfg_solver,
    distill_cfg,
    adv_weight,
    discriminator_head_stride,
    generator_turn,
    args,
):
    optimizer.zero_grad()
    guidance_optimizer.zero_grad()
    fake_transformer.requires_grad_(False)
    discriminator.requires_grad_(False)

    (
        latents,
        encoder_hidden_states,
        latents_attention_mask,
        encoder_attention_mask,
    ) = next(loader)
    model_input = normalize_dit_input(model_type, latents)

    generator_pred = sample_model_latent(
        transformer,
        encoder_hidden_states,
        encoder_attention_mask,
        model_type,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
    )
    
    # assert pred has grad
    assert generator_pred.requires_grad
    
    def compute_cls_logits(
        feature,
        encoder_hidden_states,
        discriminator,
    ):
        bsz = feature.shape[0]
        index = torch.randint(
            0, num_euler_timesteps, (bsz,), device=feature.device
        ).long()
        if sp_size > 1:
            broadcast(index)
            
        sigmas = extract_into_tensor(solver.sigmas, index, feature.shape)
        timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).view(-1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            fake_features = fake_transformer(
                feature,
                encoder_hidden_states,
                timesteps,
                encoder_attention_mask,  # B, L'
                output_features=True,
                output_features_stride=discriminator_head_stride,
                return_dict=False,
            )[1]
        #print("fake_features shape: ", fake_features.shape)
        logits = discriminator(fake_features)
        return logits
    
    if generator_turn:
        
        # 1.1 DM loss
        with torch.no_grad():
            noise = torch.randn_like(generator_pred)
            bsz = generator_pred.shape[0]
            index = torch.randint(
                0, num_euler_timesteps, (bsz,), device=generator_pred.device
            ).long()
            if sp_size > 1:
                broadcast(index)
                
            sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
            timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).view(-1)

            noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

            # run at full precision as autocast and no_grad doesn't work well together 
            real_noise, real_pred = predict_noise(
                real_transformer,
                noisy_model_input,
                encoder_hidden_states,
                timesteps,
                encoder_attention_mask,  # B, L
                guidance_scale=1, # TODO
                solver=solver,
                index=index,
                multiphase=multiphase,            
            )
            
            fake_noise, fake_pred = predict_noise(
                fake_transformer,
                noisy_model_input,
                encoder_hidden_states,
                timesteps,
                encoder_attention_mask,  # B, L
                guidance_scale=1,
                solver=solver,
                index=index,
                multiphase=multiphase,      
            )
            
            grad = (real_pred - fake_pred) / torch.abs(real_pred).mean(dim=[1, 2, 3, 4], keepdim=True) 
            grad = torch.nan_to_num(grad)

        dm_loss = 0.5 * F.mse_loss(generator_pred.float(), (generator_pred-grad).detach().float(), reduction="mean")         

        # 1.2 GAN loss
        pred_realism_on_fake_with_grad = compute_cls_logits(
            generator_pred,
            encoder_hidden_states,
            discriminator,
        )
        ForkedPdb().set_trace()
        gan_loss = torch.tensor(0.0, device=generator_pred.device)
        for fake in pred_realism_on_fake_with_grad:
            gan_loss += (
                F.softplus(-fake.float()).mean() 
            ) / (discriminator.head_num * discriminator.num_h_per_head)
        
        assert dm_loss.requires_grad
        assert gan_loss.requires_grad
        print("dm_loss: ", dm_loss)
        print("gan_loss: ", gan_loss)
        dm_loss_weight = 1
        gen_cls_loss_weight = 5e-3
        g_loss = dm_loss * dm_loss_weight + gan_loss * gen_cls_loss_weight
        g_loss.backward()

        g_loss = g_loss.detach().clone()
        dist.all_reduce(g_loss, op=dist.ReduceOp.AVG)

        g_grad_norm = transformer.clip_grad_norm_(max_grad_norm).item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 2. Update Fake Transformer (Discriminator)
    guidance_optimizer.zero_grad()
    fake_transformer.requires_grad_(True)
    discriminator.requires_grad_(True)
    
    # 2.1 finetune loss
    
    noise = torch.randn_like(generator_pred)
    bsz = generator_pred.shape[0]
    index = torch.randint(
        0, num_euler_timesteps, (bsz,), device=generator_pred.device
    ).long()
    if sp_size > 1:
        broadcast(index)
        
    sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
    timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).view(-1)

    noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

    with torch.autocast("cuda", dtype=torch.bfloat16):
        fake_noise, fake_pred = predict_noise(
            fake_transformer,
            noisy_model_input,
            encoder_hidden_states,
            timesteps,
            encoder_attention_mask,  # B, L
            guidance_scale=1,
            solver=solver,
            index=index,
            multiphase=multiphase,      
        )

    loss_fake = torch.mean(
        (fake_noise.float() - noise.float())**2
    )
    
    # 2.2 GAN loss
    
    # Be aware of CFG
    pred_realism_on_real = compute_cls_logits(
        model_input,
        encoder_hidden_states,
        discriminator,
    )
    pred_realism_on_fake = compute_cls_logits(
        generator_pred,
        encoder_hidden_states,
        discriminator,
    )
    
    guidance_cls_loss = 0
    for real, fake in zip(pred_realism_on_real, pred_realism_on_fake):
        guidance_cls_loss += (
            F.softplus(fake).mean() 
            + F.softplus(-real).mean()
        ) / (discriminator.head_num * discriminator.num_h_per_head)
        
    guidance_cls_loss_weight = 1e-2
    d_loss = loss_fake + guidance_cls_loss * guidance_cls_loss_weight
    
    d_loss.backward()
    d_grad_norm = fake_transformer.clip_grad_norm_(max_grad_norm).item()
    guidance_optimizer.step()
    guidance_optimizer.zero_grad()

    return g_loss, g_grad_norm, d_loss, d_grad_norm


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Create model:

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    
    # keep the master weight to float32
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )
    real_transformer = deepcopy(transformer)
    fake_transformer = deepcopy(transformer)
    
    discriminator = Discriminator(
        args.discriminator_head_stride,
        total_layers=48 if args.model_type == "mochi" else 40,
    )
    
    if args.use_lora:
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)
        
        # fake transformer
        fake_transformer.requires_grad_(False)
        fake_transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        fake_transformer.add_adapter(fake_transformer_lora_config)

    main_print(
        f"  Total transformer parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs(args.master_weight_type)
    if args.use_lora:
        assert args.model_type == "mochi", "LoRA is only supported for Mochi model."
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        transformer._no_split_modules = no_split_modules
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)

    transformer = FSDP(transformer, **fsdp_kwargs,)
    real_transformer = FSDP(real_transformer, **fsdp_kwargs,)
    fake_transformer = FSDP(fake_transformer, **fsdp_kwargs,)
    discriminator = FSDP(discriminator, **discriminator_fsdp_kwargs,)
    
    main_print(f"--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
        apply_fsdp_checkpointing(
            fake_transformer, no_split_modules, args.selective_checkpointing
        )
    # Set model as trainable.
    transformer.train()
    real_transformer.requires_grad_(False)
    fake_transformer.train()
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    if args.scheduler_type == "pcm_linear_quadratic":
        sigmas = linear_quadratic_schedule(
            noise_scheduler.config.num_train_timesteps, args.linear_quadratic_threshold
        )
        sigmas = torch.tensor(sigmas).to(dtype=torch.float32)
    else:
        sigmas = noise_scheduler.sigmas
    solver = EulerSolver(
        sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
    )
    solver.to(device)
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-8,
    )

    guidance_optimizer = torch.optim.AdamW(
        fake_transformer.parameters(),
        lr=args.guidance_learning_rate,
        betas=(0, 0.999),
        weight_decay=1e-3,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )
    elif args.resume_from_checkpoint:
        (
            transformer,
            optimizer,
            fake_transformer,
            guidance_optimizer,
            init_steps,
        ) = resume_training_generator_fake_transformer( # TODO:
            transformer,
            optimizer,
            fake_transformer,
            guidance_optimizer,
            args.resume_from_checkpoint,
            rank,
        )

    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    uncond_prompt_embed = train_dataset.uncond_prompt_embed
    uncond_prompt_mask = train_dataset.uncond_prompt_mask
    sampler = (
        LengthGroupedSampler(
            args.train_batch_size,
            rank=rank,
            world_size=world_size,
            lengths=train_dataset.lengths,
            group_frame=args.group_frame,
            group_resolution=args.group_resolution,
        )
        if (args.group_frame or args.group_resolution)
        else DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=False
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    assert args.gradient_accumulation_steps == 1
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        * args.sp_size
        / args.train_sp_batch_size
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

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
    # log_validation(args, transformer, device,
    #             torch.bfloat16, 0, scheduler_type=args.scheduler_type, shift=args.shift, num_euler_timesteps=args.num_euler_timesteps, linear_quadratic_threshold=args.linear_quadratic_threshold,ema=False)
    def get_num_phases(multi_phased_distill_schedule, step):
        # step-phase,step-phase
        multi_phases = multi_phased_distill_schedule.split(",")
        phase = multi_phases[-1].split("-")[-1]
        for step_phases in multi_phases:
            phase_step, phase = step_phases.split("-")
            if step <= int(phase_step):
                return int(phase)
        return phase

    for i in range(init_steps):
        _ = next(loader)
    for step in range(init_steps + 1, args.max_train_steps + 1):
        assert args.multi_phased_distill_schedule is not None
        num_phases = get_num_phases(args.multi_phased_distill_schedule, step)
        start_time = time.time()
        (
            generator_loss,
            generator_grad_norm,
            guidance_loss,
            guidance_grad_norm,
        ) = distill_one_step_dmd(
            transformer,
            optimizer,
            args.model_type,
            real_transformer,
            fake_transformer,
            guidance_optimizer,
            discriminator,
            lr_scheduler,
            loader,
            noise_scheduler,
            solver,
            noise_random_generator,
            args.sp_size,
            args.max_grad_norm,
            uncond_prompt_embed,
            uncond_prompt_mask,
            args.num_euler_timesteps,
            num_phases,
            args.not_apply_cfg_solver,
            args.distill_cfg,
            args.adv_weight,
            args.discriminator_head_stride,
            step % args.generator_update_steps == 0,
            args,
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix(
            {
                "g_loss": f"{generator_loss:.4f}",
                "d_loss": f"{guidance_loss:.4f}",
                "g_grad_norm": generator_grad_norm,
                "d_grad_norm": guidance_grad_norm,
                "step_time": f"{step_time:.2f}s",
            }
        )
        progress_bar.update(1)
        if rank <= 0:
            wandb.log(
                {
                    "generator_loss": generator_loss,
                    "guidance_loss": guidance_loss,
                    "generator_grad_norm": generator_grad_norm,
                    "guidance_grad_norm": guidance_grad_norm,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                },
                step=step,
            )
        if step % args.checkpointing_steps == 0:
            main_print(f"--> saving checkpoint at step {step}")
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(
                    transformer, optimizer, rank, args.output_dir, step
                )
            else:
                # Your existing checkpoint saving code
                # TODO
                # save_checkpoint_generator_discriminator(
                #     transformer,
                #     optimizer,
                #     discriminator,
                #     discriminator_optimizer,
                #     rank,
                #     args.output_dir,
                #     step,
                # )
                save_checkpoint(
                    transformer, fake_transformer, rank, args.output_dir, step, discriminator
                )
            main_print(f"--> checkpoint saved at step {step}")
            dist.barrier()
        if args.log_validation and step % args.validation_steps == 0:
            log_validation(
                args,
                transformer,
                device,
                torch.bfloat16,
                step,
                scheduler_type=args.scheduler_type,
                shift=args.shift,
                num_euler_timesteps=args.num_euler_timesteps,
                linear_quadratic_threshold=args.linear_quadratic_threshold,
                linear_range=args.linear_range,
                ema=False,
            )

    if args.use_lora:
        save_lora_checkpoint(
            transformer, optimizer, rank, args.output_dir, args.max_train_steps
        )
    else:
        save_checkpoint(transformer, rank, args.output_dir, args.max_train_steps)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", type=str, default="mochi", help="The type of model to train."
    )
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t", type=int, default=28, help="Number of latent timesteps."
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    # validation & logs
    parser.add_argument("--validation_sampling_steps", type=str, default="64")
    parser.add_argument("--validation_guidance_scale", type=str, default="4.5")
    parser.add_argument("--validation_steps", type=float, default=64)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )
    parser.add_argument(
        "--generator_update_steps",
        type=int,
        default=1,
        help="Number of steps to update the generator.",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="LoRA rank parameter. "
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument("--multi_phased_distill_schedule", type=str, default=None)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=6,
        help="Number of inference steps.",
    )
        
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--not_apply_cfg_solver",
        action="store_true",
        help="Whether to apply the cfg_solver.",
    )
    parser.add_argument(
        "--distill_cfg", type=float, default=3.0, help="Distillation coefficient."
    )
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument(
        "--scheduler_type", type=str, default="pcm", help="The scheduler type to use."
    )
    parser.add_argument(
        "--adv_weight",
        type=float,
        default=0.1,
        help="The weight of the adversarial loss.",
    )
    parser.add_argument(
        "--discriminator_head_stride",
        type=int,
        default=2,
        help="The stride of the discriminator head.",
    )
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="The threshold of the linear quadratic scheduler.",
    )
    parser.add_argument(
        "--linear_range",
        type=float,
        default=0.5,
        help="Range for linear quadratic scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)
