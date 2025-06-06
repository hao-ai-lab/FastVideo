# isort: skip_file
import gc
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

import wandb
from fastvideo.distill.solver import PCMFMScheduler
from fastvideo.models.mochi_hf.pipeline_mochi import (linear_quadratic_schedule, retrieve_timesteps)
from fastvideo.utils.communications import all_gather
from fastvideo.utils.load import load_vae
from fastvideo.utils.parallel_states import (get_sequence_parallel_state, nccl_info)


def prepare_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    num_frames,
    dtype,
    device,
    generator,
    vae_spatial_scale_factor,
    vae_temporal_scale_factor,
):
    height = height // vae_spatial_scale_factor
    width = width // vae_spatial_scale_factor
    num_frames = (num_frames - 1) // vae_temporal_scale_factor + 1

    shape = (batch_size, num_channels_latents, num_frames, height, width)

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return latents


def sample_validation_video(
    model_type,
    transformer,
    vae,
    scheduler,
    scheduler_type="euler",
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 16,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 4.5,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    vae_spatial_scale_factor=8,
    vae_temporal_scale_factor=6,
    num_channels_latents=12,
):
    device = vae.device

    batch_size = prompt_embeds.shape[0]

    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # 4. Prepare latent variables
    # TODO: Remove hardcore
    latents = prepare_latents(
        batch_size * num_videos_per_prompt,
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
        latents = rearrange(latents, "b t (n s) h w -> b t n s h w", n=world_size).contiguous()
        latents = latents[:, :, rank, :, :, :]

    # 5. Prepare timestep
    # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
    threshold_noise = 0.025
    sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
    sigmas = np.array(sigmas)
    if scheduler_type == "euler" and model_type == "mochi":  #todo
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
    else:
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
        )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

    # 6. Denoising loop
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    # write with tqdm instead
    # only enable if nccl_info.global_rank == 0

    with tqdm(
            total=num_inference_steps,
            disable=nccl_info.rank_within_group != 0,
            desc="Validation sampling...",
    ) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if model_type == "wan":
                    pred_kwargs = {
                        "hidden_states": latent_model_input,
                        "encoder_hidden_states": prompt_embeds,
                        "timestep":timestep,
                        "return_dict":False,
                    }
                else:
                    pred_kwargs = {
                        "hidden_states": latent_model_input,
                        "encoder_hidden_states": prompt_embeds,
                        "timestep":timestep,
                        "encoder_attention_mask":prompt_attention_mask,
                        "return_dict":False,
                    }
                noise_pred = transformer(**pred_kwargs)[0]

            # Mochi CFG + Sampling runs in FP32
            noise_pred = noise_pred.to(torch.float32)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents.to(torch.float32), return_dict=False)[0]
            latents = latents.to(latents_dtype)

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                progress_bar.update()

    if get_sequence_parallel_state():
        latents = all_gather(latents, dim=2)

    if output_type == "latent":
        video = latents
    else:
        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = (hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None)
        has_latents_std = (hasattr(vae.config, "latents_std") and vae.config.latents_std is not None)
        if model_type == "wan":
            vae.config.scaling_factor = 1
        if has_latents_mean and has_latents_std:
            latents_mean = (torch.tensor(vae.config.latents_mean).view(1, num_channels_latents, 1, 1,
                                                                       1).to(latents.device, latents.dtype))
            latents_std = (torch.tensor(vae.config.latents_std).view(1, num_channels_latents, 1, 1, 1).to(latents.device, latents.dtype))
            latents = latents * latents_std / vae.config.scaling_factor + latents_mean
        else:
            latents = latents / vae.config.scaling_factor
        with torch.autocast("cuda", dtype=vae.dtype):
            video = vae.decode(latents, return_dict=False)[0]
        video_processor = VideoProcessor(vae_scale_factor=vae_spatial_scale_factor)
        video = video_processor.postprocess_video(video, output_type=output_type)

    return (video, )


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def log_validation(
    args,
    transformer,
    device,
    weight_dtype,  # TODO
    global_step,
    scheduler_type="euler",
    shift=1.0,
    num_euler_timesteps=100,
    linear_quadratic_threshold=0.025,
    linear_range=0.5,
    ema=False,
):
    # TODO
    print("Running validation....\n")
    if args.model_type == "mochi":
        vae_spatial_scale_factor = 8
        vae_temporal_scale_factor = 6
        num_channels_latents = 12
    elif args.model_type == "hunyuan" or "hunyuan_hf" or "wan":
        vae_spatial_scale_factor = 8
        vae_temporal_scale_factor = 4
        num_channels_latents = 16
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    vae, autocast_type, fps = load_vae(args.model_type, args.pretrained_model_name_or_path)
    if args.model_type != "wan":
        vae.enable_tiling()
    if scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
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
    # args.validation_prompt_dir

    validation_guidance_scale_ls = args.validation_guidance_scale.split(",")
    validation_guidance_scale_ls = [float(scale) for scale in validation_guidance_scale_ls]
    for validation_sampling_step in args.validation_sampling_steps.split(","):
        validation_sampling_step = int(validation_sampling_step)
        for validation_guidance_scale in validation_guidance_scale_ls:
            videos = []
            # prompt_embed are named embed0 to embedN
            # check how many embeds are there
            embe_dir = os.path.join(args.validation_prompt_dir, "prompt_embed")
            mask_dir = os.path.join(args.validation_prompt_dir, "prompt_attention_mask")
            embeds = sorted([f for f in os.listdir(embe_dir)])
            masks = sorted([f for f in os.listdir(mask_dir)])
            num_embeds = len(embeds)
            validation_prompt_ids = list(range(num_embeds))
            num_sp_groups = int(os.getenv("WORLD_SIZE", "1")) // nccl_info.sp_size
            # pad to multiple of groups
            if num_embeds % num_sp_groups != 0:
                validation_prompt_ids += [0] * (num_sp_groups - num_embeds % num_sp_groups)
            num_embeds_per_group = len(validation_prompt_ids) // num_sp_groups
            local_prompt_ids = validation_prompt_ids[nccl_info.group_id *
                                                     num_embeds_per_group:(nccl_info.group_id + 1) *
                                                     num_embeds_per_group]

            for i in local_prompt_ids:
                prompt_embed_path = os.path.join(embe_dir, f"{embeds[i]}")
                prompt_mask_path = os.path.join(mask_dir, f"{masks[i]}")
                prompt_embeds = (torch.load(prompt_embed_path, map_location="cpu",
                                            weights_only=True).to(device).unsqueeze(0))
                prompt_attention_mask = (torch.load(prompt_mask_path, map_location="cpu",
                                                    weights_only=True).to(device).unsqueeze(0))
                negative_prompt_embeds = torch.zeros(256, 4096).to(device).unsqueeze(0)
                negative_prompt_attention_mask = (torch.zeros(256).bool().to(device).unsqueeze(0))
                generator = torch.Generator(device="cpu").manual_seed(12345)
                video = sample_validation_video(
                    args.model_type,
                    transformer,
                    vae,
                    scheduler,
                    scheduler_type=scheduler_type,
                    num_frames=args.num_frames,
                    height=args.num_height,
                    width=args.num_width,
                    num_inference_steps=validation_sampling_step,
                    guidance_scale=validation_guidance_scale,
                    generator=generator,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    vae_spatial_scale_factor=vae_spatial_scale_factor,
                    vae_temporal_scale_factor=vae_temporal_scale_factor,
                    num_channels_latents=num_channels_latents,
                )[0]
                if nccl_info.rank_within_group == 0:
                    videos.append(video[0])
            # collect videos from all process to process zero

            gc.collect()
            torch.cuda.empty_cache()
            # log if main process
            torch.distributed.barrier()
            all_videos = [None for i in range(int(os.getenv("WORLD_SIZE", "1")))]  # remove padded videos
            torch.distributed.all_gather_object(all_videos, videos)
            if nccl_info.global_rank == 0:
                # remove padding
                videos = [video for videos in all_videos for video in videos]
                videos = videos[:num_embeds]
                # linearize all videos
                video_filenames = []
                for i, video in enumerate(videos):
                    filename = os.path.join(
                        args.output_dir,
                        f"validation_step_{global_step}_sample_{validation_sampling_step}_guidance_{validation_guidance_scale}_video_{i}.mp4",
                    )
                    export_to_video(video, filename, fps=fps)
                    video_filenames.append(filename)

                logs = {
                    f"{'ema_' if ema else ''}validation_sample_{validation_sampling_step}_guidance_{validation_guidance_scale}":
                    [wandb.Video(filename) for i, filename in enumerate(video_filenames)]
                }
                wandb.log(logs, step=global_step)
