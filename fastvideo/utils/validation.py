

from typing import  Optional, Union, List
import numpy as np
import torch
from einops import rearrange
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
from fastvideo.utils.communications import  all_gather
from diffusers.utils.torch_utils import randn_tensor
from fastvideo.model.pipeline_mochi import linear_quadratic_schedule, retrieve_timesteps
from tqdm import tqdm
from diffusers.video_processor import VideoProcessor
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLMochi,
)
from fastvideo.distill.solver import PCMFMDeterministicScheduler
from diffusers.utils import export_to_video
import os
import wandb
import gc
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
    vae_spatial_scale_factor = 8,
    vae_temporal_scale_factor = 6, 
):
    device = vae.device

    batch_size = prompt_embeds.shape[0]

    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # 4. Prepare latent variables
    # TODO: Remove hardcore
    num_channels_latents = 12
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
        vae_temporal_scale_factor
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
    if scheduler_type == "euler":
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
    
    with tqdm(total=num_inference_steps, disable= nccl_info.global_rank != 0, desc="Validation sampling...") as progress_bar:
        for i, t in enumerate(timesteps):


            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            noise_pred = transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=prompt_attention_mask,
                return_dict=False,
            )[0]
            

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) %  scheduler.order == 0):
                progress_bar.update()



    if get_sequence_parallel_state():
        latents = all_gather(latents, dim=2)


    if output_type == "latent":
        video = latents
    else:
        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
        has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / vae.config.scaling_factor + latents_mean
        else:
            latents = latents / vae.config.scaling_factor

        video = vae.decode(latents, return_dict=False)[0]
        video_processor = VideoProcessor(vae_scale_factor=vae_spatial_scale_factor)
        video = video_processor.postprocess_video(video, output_type=output_type)
        


    return (video,)



@torch.no_grad()
def log_validation(args, transformer, device, weight_dtype, global_step,  scheduler_type="euler",shift=1.0, num_euler_timesteps=100,  ema=False):
    #TODO
    print(f"Running validation....\n")
    generator = torch.Generator(device="cuda").manual_seed(12345)
    vae = AutoencoderKLMochi.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype).to("cuda")
    vae.enable_tiling()
    if scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        scheduler = PCMFMDeterministicScheduler(1000, shift, num_euler_timesteps)
    # args.validation_prompt_dir
    prompt_embed_path = os.path.join(args.validation_prompt_dir, "embed.pt")
    prompt_mask_path = os.path.join(args.validation_prompt_dir, "mask.pt")
    negative_prompt_embed_path = os.path.join(args.uncond_prompt_dir, "embed.pt")
    negative_prompt_mask_path = os.path.join(args.uncond_prompt_dir, "mask.pt")
    prompt_embeds = torch.load(prompt_embed_path, map_location="cpu", weights_only=True).to(device).to(weight_dtype).unsqueeze(0)
    prompt_attention_mask = torch.load(prompt_mask_path, map_location="cpu", weights_only=True).to(device).to(weight_dtype).unsqueeze(0)
    negative_prompt_embeds = torch.load(negative_prompt_embed_path, map_location="cpu", weights_only=True).to(device).to(weight_dtype).unsqueeze(0)
    negative_prompt_attention_mask = torch.load(negative_prompt_mask_path, map_location="cpu", weights_only=True).to(device).to(weight_dtype).unsqueeze(0)
    

    videos = []
    video = sample_validation_video(
                transformer,
                vae,
                scheduler,
                scheduler_type=scheduler_type,
                num_frames=args.num_frames,
                # Peiyuan TODO: remove hardcode
                height=480,
                width=848,
                num_inference_steps=args.validation_sampling_steps,
                guidance_scale=args.validation_guidance_scale,
                generator=generator, 
                prompt_embeds = prompt_embeds,
                prompt_attention_mask = prompt_attention_mask,
                negative_prompt_embeds = negative_prompt_embeds,
                negative_prompt_attention_mask = negative_prompt_attention_mask,
                )[0]
    videos.append(video[0])
    # import ipdb;ipdb.set_trace()
    gc.collect()
    torch.cuda.empty_cache()
    # log if main process
    if int(os.environ['RANK']) <= 0:
        video_filenames = []
        for i, video in enumerate(videos):
            filename = os.path.join(args.output_dir, f"validation_step_{global_step}_video_{i}.mp4")
            export_to_video(video, filename, fps=30)
            video_filenames.append(filename)

        logs = {
            f"{'ema_' if ema else ''}validation": [
                wandb.Video(filename)
                for i, filename in enumerate(video_filenames)
            ]
        }
        wandb.log(logs, step=global_step)

    del vae
    del prompt_embeds
    del prompt_attention_mask
    del negative_prompt_embeds
    del negative_prompt_attention_mask
    gc.collect()
    torch.cuda.empty_cache()

