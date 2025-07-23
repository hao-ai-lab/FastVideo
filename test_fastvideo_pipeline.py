#!/usr/bin/env python3
"""
Test script for FastVideo Cosmos2 pipeline with hidden state logging.

This script runs inference using the FastVideo Cosmos2 implementation
and captures intermediate hidden states from various pipeline stages.
"""

import torch
import json
import os
import sys
import logging
from typing import Dict, Any

# Add FastVideo to path
sys.path.insert(0, "/workspace/workspace-cosmos/FastVideo")

from fastvideo.v1.entrypoints.video_generator import VideoGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiddenStateLogger:
    """Utility class to capture and log hidden states from the FastVideo pipeline."""
    
    def __init__(self):
        self.hidden_states = {}
        self.hooks = []
    
    def capture_tensor_stats(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Capture statistics from a tensor."""
        if tensor is None:
            return {"name": name, "data": None}
        
        tensor_float = tensor.float()
        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": tensor_float.mean().item(),
            "std": tensor_float.std().item(),
            "min": tensor_float.min().item(),
            "max": tensor_float.max().item(),
            "sum": tensor_float.sum().item(),
            "abs_sum": tensor_float.abs().sum().item(),
            "abs_max": tensor_float.abs().max().item(),
            "norm": tensor_float.norm().item(),
        }
    
    def register_forward_hook(self, module, name: str):
        """Register a forward hook to capture outputs."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.hidden_states[name] = self.capture_tensor_stats(output, name)
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    self.hidden_states[name] = self.capture_tensor_stats(output[0], name)
            logger.info(f"Captured hidden state from {name}")
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def save_states(self, filepath: str):
        """Save captured hidden states to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.hidden_states, f, indent=2)
        logger.info(f"Saved hidden states to {filepath}")


def run_fastvideo_inference_with_logging(generator, state_logger, prompt, image_path, num_inference_steps):
    """Run FastVideo inference and capture intermediate stage outputs."""
    
    logger.info("=== FASTVIDEO INFERENCE ===")
    
    # Run inference with negative prompt
    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
    
    result = generator.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.0,
        num_frames=21,
        height=720,
        width=1280,
        fps=16,
        seed=42,
        save_video=False  # Don't save video file, just return tensor
    )
    
    return result


def test_fastvideo_cosmos2_pipeline():
    """Test the FastVideo Cosmos2 pipeline and capture hidden states."""
    
    logger.info("Starting FastVideo Cosmos2 pipeline test...")
    
    # Test configuration
    input_image_path = "/workspace/workspace-cosmos/FastVideo/tennis.jpg"
    prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
    num_inference_steps = 10
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        logger.error(f"Input image not found: {input_image_path}")
        return False
    
    try:
        # Create video generator
        logger.info("Creating FastVideo generator...")
        generator = VideoGenerator.from_pretrained(
            model_path="nvidia/Cosmos-Predict2-2B-Video2World",
            num_gpus=1,
        )
        
        logger.info("Generator created successfully")
        
        # Create hidden state logger
        state_logger = HiddenStateLogger()
        
        # Run inference with hidden state logging
        logger.info("Starting inference with hidden state logging...")
        
        # Log pipeline configuration
        state_logger.hidden_states["config"] = {
            "input_image_path": input_image_path,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "pipeline_type": "fastvideo_cosmos2",
            "model_path": "nvidia/Cosmos-Predict2-2B-Video2World",
            "guidance_scale": 7.0,
            "seed": 42
        }
        
        # Run inference
        result = run_fastvideo_inference_with_logging(generator, state_logger, prompt, input_image_path, num_inference_steps)
        
        # Extract stage outputs if available
        if isinstance(result, dict) and "stage_outputs" in result and result["stage_outputs"]:
            logger.info(f"Found {len(result['stage_outputs'])} stage outputs")
            state_logger.hidden_states.update(result["stage_outputs"])
            for stage_key in result["stage_outputs"].keys():
                logger.info(f"Captured stage output: {stage_key}")
        else:
            logger.warning("No stage outputs found in result")
        
        # Log final output
        if result is not None and isinstance(result, dict) and 'samples' in result:
            samples = result['samples']
            state_logger.hidden_states["final_output"] = state_logger.capture_tensor_stats(samples, "final_output")
            logger.info(f"Inference completed successfully. Output shape: {samples.shape}")
            
            # Save video
            import imageio
            import numpy as np
            
            # Convert tensor to video format and save (handle [-1, 1] range)
            video_np = samples[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
            video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)  # Map [-1,1] to [0,255]
            video_output_path = "/workspace/workspace-cosmos/comparison_videos/fastvideo/cosmos2_fastvideo_output.mp4"
            imageio.mimsave(video_output_path, video_np, fps=16)
            logger.info(f"Video saved to {video_output_path}")
            
            # Log per-frame statistics like the reference test
            if len(samples.shape) == 5:  # [B, C, T, H, W]
                per_frame_means = []
                for frame_idx in range(samples.shape[2]):  # T dimension
                    frame_mean = samples[0, :, frame_idx, :, :].mean().item()
                    per_frame_means.append(frame_mean)
                
                state_logger.hidden_states["per_frame_means"] = per_frame_means
                
                # Analyze conditioning
                if len(per_frame_means) > 1:
                    first_frame_mean = per_frame_means[0]
                    other_frames_mean = sum(per_frame_means[1:]) / len(per_frame_means[1:])
                    conditioning_diff = abs(first_frame_mean - other_frames_mean)
                    
                    state_logger.hidden_states["conditioning_analysis"] = {
                        "first_frame_mean": first_frame_mean,
                        "other_frames_mean": other_frames_mean,
                        "conditioning_difference": conditioning_diff
                    }
        else:
            logger.error("Inference failed - no output generated")
            return False
        
        # Save hidden states
        output_path = "/workspace/workspace-cosmos/comparison_logs/fastvideo_pipeline_output.json"
        state_logger.save_states(output_path)
        
        logger.info("FastVideo Cosmos2 pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fastvideo_cosmos2_pipeline()
    if success:
        print("✅ FastVideo pipeline test completed successfully")
        sys.exit(0)
    else:
        print("❌ FastVideo pipeline test failed")
        sys.exit(1)