import os
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.distributed
import random
import torchvision.transforms as transforms
from PIL import Image
import cv2

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from hymm_sp.config import parse_args
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.data_kits.video_dataset import VideoCSVDataset
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

class CropResize:
    """
    Custom transform to resize and crop images to a target size while preserving aspect ratio.
    
    Resizes the image to ensure it covers the target dimensions, then center-crops to the exact size.
    Useful for preparing consistent input dimensions for video generation models.
    """
    def __init__(self, size=(704, 1216)):
        """
        Args:
            size (tuple): Target dimensions (height, width) for the output image
        """
        self.target_h, self.target_w = size  

    def __call__(self, img):
        """
        Apply the transform to an image.
        
        Args:
            img (PIL.Image): Input image to transform
            
        Returns:
            PIL.Image: Resized and cropped image with target dimensions
        """
        # Get original image dimensions
        w, h = img.size
        
        # Calculate scaling factor to ensure image covers target size
        scale = max(  
            self.target_w / w,  # Scale needed to cover target width
            self.target_h / h   # Scale needed to cover target height
        )
        
        # Resize image while preserving aspect ratio
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(
            new_size, 
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        resized_img = resize_transform(img)
        
        # Center-crop to exact target dimensions
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)


def main():
    """
    Main function for video generation using the Hunyuan multimodal model.
    
    Handles argument parsing, distributed setup, model loading, data preparation,
    and video generation with action-controlled transitions. Supports both image-to-video
    and video-to-video generation tasks.
    """
    # Parse command-line arguments and configuration
    args = parse_args()
    models_root_path = Path(args.ckpt)
    action_list = args.action_list
    action_speed_list = args.action_speed_list
    negative_prompt = args.add_neg_prompt

    # Initialize distributed training/evaluation environment
    logger.info("*" * 20) 
    initialize_distributed(args.seed)
    
    # Validate model checkpoint path exists
    if not models_root_path.exists():
        raise ValueError(f"Model checkpoint path does not exist: {models_root_path}")
    logger.info("+" * 20)

    # Set up output directory
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Generated videos will be saved to: {save_path}")

    # Initialize device configuration for distributed processing
    rank = 0
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        # Use specific GPU based on process rank in distributed setup
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = torch.distributed.get_rank()

    # Load the Hunyuan video sampler model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.ckpt}")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        args.ckpt, 
        args=args, 
        device=device if not args.cpu_offload else torch.device("cpu")
    )
    # Update args with model-specific configurations from the checkpoint
    args = hunyuan_video_sampler.args

    # Enable CPU offloading if specified to reduce GPU memory usage
    if args.cpu_offload:
        from diffusers.hooks import apply_group_offloading
        onload_device = torch.device("cuda")
        apply_group_offloading(
            hunyuan_video_sampler.pipeline.transformer, 
            onload_device=onload_device, 
            offload_type="block_level", 
            num_blocks_per_group=1
        )
        logger.info("Enabled CPU offloading for transformer blocks")

    # Process each batch in the dataset

    prompt = args.prompt
    image_paths = [args.image_path]
    logger.info(f"Prompt: {prompt}, Image Path {args.image_path}")
    # Generate random seed for reproducibility
    seed = args.seed if args.seed else random.randint(0, 1_000_000)
    
    # Define image transformation pipeline for input reference images
    closest_size = (704, 1216)
    ref_image_transform = transforms.Compose([
        CropResize(closest_size),
        transforms.CenterCrop(closest_size),
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
    ])
    
    # Handle image-based generation (start from a single image)
    if args.image_start:
        # Load and preprocess reference images
        raw_ref_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        
        # Apply transformations and prepare tensor for model input
        ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in raw_ref_images]
        ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(device)
        
        # Encode reference images to latent space using VAE
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if args.cpu_offload:
                # Move VAE components to GPU temporarily for encoding
                hunyuan_video_sampler.vae.quant_conv.to('cuda')
                hunyuan_video_sampler.vae.encoder.to('cuda')
            
            # Enable tiling for VAE to handle large images efficiently
            hunyuan_video_sampler.pipeline.vae.enable_tiling()
            
            # Encode image to latents and scale by VAE's scaling factor
            raw_last_latents = hunyuan_video_sampler.vae.encode(
                ref_images_pixel_values
            ).latent_dist.sample().to(dtype=torch.float16)  # Shape: (B, C, F, H, W)
            raw_last_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
            raw_ref_latents = raw_last_latents.clone()
            
            # Clean up
            hunyuan_video_sampler.pipeline.vae.disable_tiling()
            if args.cpu_offload:
                # Move VAE components back to CPU after encoding
                hunyuan_video_sampler.vae.quant_conv.to('cpu')
                hunyuan_video_sampler.vae.encoder.to('cpu')


    # Handle video-based generation (start from an existing video)
    else:
        from decord import VideoReader  # Lazy import for video handling
        
        # Validate video file exists
        video_path = args.video_path
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")
        
        # Load reference images from video metadata
        raw_ref_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]

        # Load video and extract frames
        ref_video = VideoReader(video_path)
        ref_frames_length = len(ref_video)
        logger.info(f"Loaded reference video with {ref_frames_length} frames")
        
        # Preprocess video frames
        transformed_images = []
        for index in range(ref_frames_length):
            # Convert video frame to PIL image and apply transformations
            video_image = ref_video[index].numpy()
            transformed_image = ref_image_transform(Image.fromarray(video_image))
            transformed_images.append(transformed_image)
        
        # Prepare tensor for model input
        transformed_images = torch.stack(transformed_images, dim=1).unsqueeze(0).to(
            device=hunyuan_video_sampler.device, 
            dtype=torch.float16
        )
        
        # Encode video frames to latent space using VAE
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if args.cpu_offload:
                hunyuan_video_sampler.vae.quant_conv.to('cuda')
                hunyuan_video_sampler.vae.encoder.to('cuda')
            
            hunyuan_video_sampler.pipeline.vae.enable_tiling()
            
            # Encode last 33 frames of video (model-specific requirement)
            raw_last_latents = hunyuan_video_sampler.vae.encode(
                transformed_images[:, :, -33:, ...]
            ).latent_dist.sample().to(dtype=torch.float16)
            raw_last_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
            
            # Encode a single reference frame from the video
            raw_ref_latents = hunyuan_video_sampler.vae.encode(
                transformed_images[:, :, -33:-32, ...]
            ).latent_dist.sample().to(dtype=torch.float16)
            raw_ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
            
            # Clean up
            hunyuan_video_sampler.pipeline.vae.disable_tiling()
            if args.cpu_offload:
                hunyuan_video_sampler.vae.quant_conv.to('cpu')
                hunyuan_video_sampler.vae.encoder.to('cpu')
    
    # Store references for generation loop
    ref_images = raw_ref_images
    last_latents = raw_last_latents
    ref_latents = raw_ref_latents
    
    # Generate video segments for each action in the action list
    for idx, action_id in enumerate(action_list):
        # Determine if this is the first action and using image start
        is_image = (idx == 0 and args.image_start)
        
        logger.info(f"Generating segment {idx+1}/{len(action_list)} with action ID: {action_id}")
        # Generate video segment with the current action
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            action_id=action_id,
            action_speed=action_speed_list[idx],                    
            is_image=is_image,
            size=(704, 1216),
            seed=seed,
            last_latents=last_latents,  # Previous frame latents for continuity
            ref_latents=ref_latents,    # Reference latents for style consistency
            video_length=args.sample_n_frames,
            guidance_scale=args.cfg_scale,
            num_images_per_prompt=args.num_images,
            negative_prompt=negative_prompt,
            infer_steps=args.infer_steps,
            flow_shift=args.flow_shift_eval_video,
            use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
            linear_schedule_end=args.linear_schedule_end,
            use_deepcache=args.use_deepcache,
            cpu_offload=args.cpu_offload,
            ref_images=ref_images,
            output_dir=save_path,
            return_latents=True,
            use_sage=args.use_sage,
        )
        
        # Update latents for next iteration (maintain temporal consistency)
        ref_latents = outputs["ref_latents"]
        last_latents = outputs["last_latents"]
        
        # Save generated video segments if this is the main process (rank 0)
        if rank == 0:
            sub_samples = outputs['samples'][0]

            # Initialize or concatenate video segments
            if idx == 0:
                if args.image_start:
                    out_cat = sub_samples
                else:
                    # Combine original video frames with generated frames
                    out_cat = torch.cat([(transformed_images.detach().cpu() + 1) / 2.0, sub_samples], dim=2)
            else:
                # Append new segment to existing video
                out_cat = torch.cat([out_cat, sub_samples], dim=2)

            # Save final combined video
            save_path_mp4 = f"{save_path}/{os.path.basename(args.image_path).split('.')[0]}.mp4"
            save_videos_grid(out_cat, save_path_mp4, n_rows=1, fps=24)
            logger.info(f"Saved generated video to: {save_path_mp4}")

if __name__ == "__main__":
    main()
