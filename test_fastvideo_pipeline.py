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
    #input_image_path = "/workspace/FastVideo/tennis.jpg"
    #prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
    input_image_path = "/workspace/FastVideo/yellow-scrubber.png"
    prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
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
            num_gpus=2,
        )
        
        print("Generator created successfully")
        
        # Run inference
        print("Generating video...")
        result = generator.generate_video(
            height=704,
            width=1280,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=21,
            image_path=input_image_path,
            num_inference_steps=35,
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