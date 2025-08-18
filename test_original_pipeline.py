#!/usr/bin/env python3
"""
Simple script to generate a video using the original Cosmos2 pipeline.
"""

import os
import sys

# Add cosmos-predict2 to path
sys.path.insert(0, "/workspace/cosmos-predict2")

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline


def generate_video():
    """Generate a video using the Cosmos2 pipeline."""
    
    # Change to cosmos-predict2 directory for relative paths to work
    original_cwd = os.getcwd()
    cosmos_predict2_dir = "/workspace/cosmos-predict2"
    os.chdir(cosmos_predict2_dir)
    
    try:
        # Configuration
        input_image_path = "/workspace/FastVideo/tennis.jpg"
        prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
        num_inference_steps = 10
        output_path = "/workspace/FastVideo/cosmos2_output.mp4"
        
        # Check if input image exists
        if not os.path.exists(input_image_path):
            print(f"Error: Input image not found: {input_image_path}")
            return False
        
        # Create pipeline
        print("Creating Video2World pipeline...")
        config = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="720", fps=16)
        pipe = Video2WorldPipeline.from_config(
            config=config,
            dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
            text_encoder_path="checkpoints/google-t5/t5-11b",
        )
        
        print("Pipeline created successfully")
        
        # Run inference
        print("Generating video...")
        video = pipe(
            input_path=input_image_path,
            prompt=prompt,
            num_sampling_step=num_inference_steps,
            guidance=7.0,
            seed=42
        )
        
        # Save video
        if video is not None:
            print(f"Inference completed successfully. Output shape: {video.shape}")
            
            from imaginaire.utils.io import save_image_or_video
            save_image_or_video(video, output_path, fps=16)
            print(f"Video saved to {output_path}")
            return True
        else:
            print("Error: Inference failed - no output generated")
            return False
        
    except Exception as e:
        print(f"Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    success = generate_video()
    if success:
        print("✅ Video generation completed successfully")
        sys.exit(0)
    else:
        print("❌ Video generation failed")
        sys.exit(1)