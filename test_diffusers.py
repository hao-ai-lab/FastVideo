

import torch
import logging
import time
import os
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from PIL import Image
import numpy as np


# Configure logging to capture all output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Also configure the diffusers logger to output to stdout
import diffusers.utils.logging as diffusers_logging
diffusers_logging.set_verbosity_info()


def test_diffusers_cosmos():
    logger.info("🔍 [DIFFUSERS] Starting diffusers Cosmos2VideoToWorld test")
   
    # Create output directory
    os.makedirs("outputs/diffusers", exist_ok=True)
   
    # Load model
    model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
    logger.info(f"🔍 [DIFFUSERS] Loading model: {model_id}")
   
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
   
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        logger.info("🔍 [DIFFUSERS] Model moved to CUDA")
   
    # Load test image
    image_path = "tennis.jpg"
    logger.info(f"🔍 [DIFFUSERS] Loading image: {image_path}")
   
    # Use load_image utility like the official example
    image = load_image(image_path)
    logger.info(f"🔍 [DIFFUSERS] Image loaded: {image.size}")
   
    # Test parameters (matching the official example)
    prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
    num_frames = 21
    height = 720
    width = 1280
    num_inference_steps = 35  # Use default like the example
    guidance_scale = 7.0  # Use default like the example
    fps = 16
   
    logger.info(f"🔍 [DIFFUSERS] Test parameters:")
    logger.info(f"🔍 [DIFFUSERS] - prompt: {prompt[:100]}...")
    logger.info(f"🔍 [DIFFUSERS] - negative_prompt: {negative_prompt[:100]}...")
    logger.info(f"🔍 [DIFFUSERS] - num_frames: {num_frames}")
    logger.info(f"🔍 [DIFFUSERS] - height: {height}, width: {width}")
    logger.info(f"🔍 [DIFFUSERS] - num_inference_steps: {num_inference_steps}")
    logger.info(f"🔍 [DIFFUSERS] - guidance_scale: {guidance_scale}")
    logger.info(f"🔍 [DIFFUSERS] - fps: {fps}")
   
    # Generate video (matching the official example exactly)
    logger.info("🔍 [DIFFUSERS] Starting generation...")
   
    try:
        # Use the exact same pattern as the official example
        video = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            generator=torch.Generator().manual_seed(42)
        ).frames[0]
        
        logger.info("🔍 [DIFFUSERS] Generation completed successfully")
        logger.info(f"🔍 [DIFFUSERS] Video type: {type(video)}")
        logger.info(f"🔍 [DIFFUSERS] Video length: {len(video)} frames")
        if len(video) > 0:
            logger.info(f"🔍 [DIFFUSERS] First frame type: {type(video[0])}")
            logger.info(f"🔍 [DIFFUSERS] First frame shape: {video[0].shape if hasattr(video[0], 'shape') else 'No shape'}")
        
        # Save video using the exact same pattern as the official example
        output_path = f"outputs/diffusers/{prompt[:50].replace(' ', '_').replace(',', '')}.mp4"
        logger.info(f"🔍 [DIFFUSERS] Saving video to: {output_path}")
        
        try:
            export_to_video(video, output_path, fps=fps)
            logger.info(f"✅ [DIFFUSERS] Video saved successfully to: {output_path}")
        except Exception as e:
            logger.error(f"🔍 [DIFFUSERS] Failed to save video: {e}")
            output_path = None
        
        logger.info("✅ [DIFFUSERS] Diffusers test completed successfully")
        logger.info("🔍 [DIFFUSERS] Test completed - logs captured for comparison")
        return output_path
        
    except Exception as e:
        logger.error(f"🔍 [DIFFUSERS] Generation failed: {e}")
        raise


if __name__ == "__main__":
    test_diffusers_cosmos()


