import torch
import logging
import sys
import os
import time


# Add FastVideo to path
sys.path.insert(0, "FastVideo")


from fastvideo.v1.entrypoints.video_generator import VideoGenerator


# Configure logging to capture all output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fastvideo_cosmos():
    logger.info("🔍 [FASTVIDEO] Starting FastVideo Cosmos2VideoToWorld test")
   
    logger.info("🔍 [FASTVIDEO] Test parameters:")
   
    # Create video generator using the simpler from_pretrained interface
    logger.info("🔍 [FASTVIDEO] Creating video generator...")
    generator = VideoGenerator.from_pretrained(
        model_path="nvidia/Cosmos-Predict2-2B-Video2World",
        num_gpus=1,
    )
   
    # Test parameters
    prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
    image_path = "tennis.jpg"
   
    logger.info(f"🔍 [FASTVIDEO] - prompt: {prompt}")
    logger.info(f"🔍 [FASTVIDEO] - negative_prompt: {negative_prompt}")
   
    # Generate video using the simplified interface
    logger.info("🔍 [FASTVIDEO] Starting generation...")
    start_time = time.time()
   
    try:
        result = generator.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
            num_inference_steps=35,
            guidance_scale=7.0,
            num_frames=21,
            height=720,
            width=1280,
            fps=16,
            seed=42,
            save_video=True,
            output_path="outputs/fastvideo"
        )
       
        gen_time = time.time() - start_time
        logger.info(f"🔍 [FASTVIDEO] Generation completed in {gen_time:.2f} seconds")
        logger.info(f"🔍 [FASTVIDEO] Result type: {type(result)}")
       
        if isinstance(result, dict):
            logger.info(f"🔍 [FASTVIDEO] Result keys: {list(result.keys())}")
            if 'samples' in result:
                samples = result['samples']
                logger.info(f"🔍 [FASTVIDEO] Samples shape: {samples.shape}")
                
                # Calculate and log per_frame_means like comparison_test.py does
                if len(samples.shape) == 5:  # [B, C, T, H, W]
                    logger.info("🔍 [FASTVIDEO] Calculating per_frame_means...")
                    per_frame_means = []
                    for frame_idx in range(samples.shape[2]):  # T dimension
                        frame_mean = samples[0, :, frame_idx, :, :].mean().item()
                        per_frame_means.append(frame_mean)
                    
                    logger.info(f"🔍 [FASTVIDEO] Per frame means: {per_frame_means}")
                    
                    # Analyze the conditioning (first frame should be different from others)
                    if len(per_frame_means) > 1:
                        first_frame_mean = per_frame_means[0]
                        other_frames_mean = sum(per_frame_means[1:]) / len(per_frame_means[1:])
                        conditioning_diff = abs(first_frame_mean - other_frames_mean)
                        
                        logger.info(f"🔍 [FASTVIDEO] First frame mean: {first_frame_mean:.6f}")
                        logger.info(f"🔍 [FASTVIDEO] Other frames mean: {other_frames_mean:.6f}")
                        logger.info(f"🔍 [FASTVIDEO] Conditioning difference: {conditioning_diff:.6f}")
                        
                        if conditioning_diff < 0.1:
                            logger.warning("⚠️ [FASTVIDEO] First frame and other frames have very similar means - conditioning may not be working!")
                        else:
                            logger.info("✅ [FASTVIDEO] Conditioning appears to be working (first frame differs from others)")
                    
                    # Log additional statistics
                    logger.info(f"🔍 [FASTVIDEO] Samples mean: {samples.mean().item():.6f}")
                    logger.info(f"🔍 [FASTVIDEO] Samples std: {samples.std().item():.6f}")
                    logger.info(f"🔍 [FASTVIDEO] Samples min: {samples.min().item():.6f}")
                    logger.info(f"🔍 [FASTVIDEO] Samples max: {samples.max().item():.6f}")
                else:
                    logger.warning("🔍 [FASTVIDEO] Unexpected samples shape, cannot calculate per_frame_means")
       
        logger.info("✅ [FASTVIDEO] FastVideo test completed successfully")
        return "outputs/fastvideo/A tennis ball bouncing on a racquet.mp4"
       
    except Exception as e:
        logger.error(f"❌ [FASTVIDEO] Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    test_fastvideo_cosmos()

