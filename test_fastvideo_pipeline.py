#!/usr/bin/env python3
"""
Simple script to generate a video using the FastVideo generator.
"""

import os
import sys

# Add FastVideo to path
sys.path.insert(0, "/workspace/FastVideo")

from fastvideo.entrypoints.video_generator import VideoGenerator


def generate_video():
    """Generate a video using the FastVideo generator."""
    
    # Configuration
    input_image_path = "/workspace/FastVideo/tennis.jpg"
    prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
    num_inference_steps = 10
    output_path = "/workspace/FastVideo/cosmos2_fastvideo_output.mp4"
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found: {input_image_path}")
        return False
    
    try:
        # Create video generator
        print("Creating FastVideo generator...")
        generator = VideoGenerator.from_pretrained(
            model_path="nvidia/Cosmos-Predict2-2B-Video2World",
            num_gpus=1,
        )
        
        print("Generator created successfully")
        
        # Run inference
        print("Generating video...")
        result = generator.generate_video(
            prompt=prompt,
            input_path=input_image_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            seed=42,
            save_video=True,
            output_path=output_path
        )
        
        if result:
            print("Video generation completed successfully!")
            return True
        else:
            print("Video generation failed - no result returned")
            return False
        
    except Exception as e:
        print(f"Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_video()
    if success:
        print("✅ Video generation completed successfully")
        sys.exit(0)
    else:
        print("❌ Video generation failed")
        sys.exit(1)