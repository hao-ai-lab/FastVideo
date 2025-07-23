#!/usr/bin/env python3
"""
Test script for original Cosmos2 pipeline with hidden state logging.

This script runs inference using the original Cosmos2 implementation
and captures intermediate hidden states from various pipeline stages.
"""

import torch
import json
import os
import sys
import logging
from contextlib import contextmanager
from typing import Dict, Any, List

# Add cosmos-predict2 to path
sys.path.insert(0, "/workspace/workspace-cosmos/cosmos-predict2")

from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiddenStateLogger:
    """Utility class to capture and log hidden states from the pipeline."""
    
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


@contextmanager
def patch_pipeline_for_logging(pipeline, state_logger):
    """Context manager to patch pipeline methods for hidden state logging."""
    
    # Store original methods
    original_encode = pipeline.encode
    original_denoise = pipeline.denoise
    original_decode = pipeline.decode
    original_text_encoder_encode = pipeline.text_encoder.encode_prompts if pipeline.text_encoder else None
    
    # Patch encode method
    def patched_encode(state):
        logger.info("=== ENCODE STAGE ===")
        result = original_encode(state)
        state_logger.hidden_states["vae_encode_input"] = state_logger.capture_tensor_stats(state, "vae_encode_input")
        state_logger.hidden_states["vae_encode_output"] = state_logger.capture_tensor_stats(result, "vae_encode_output")
        return result
    
    # Patch denoise method
    def patched_denoise(xt_B_C_T_H_W, sigma, condition, use_cuda_graphs=False):
        logger.info("=== DENOISE STAGE ===")
        
        # Log input to denoise
        state_logger.hidden_states["denoise_input_xt"] = state_logger.capture_tensor_stats(xt_B_C_T_H_W, "denoise_input_xt")
        state_logger.hidden_states["denoise_input_sigma"] = state_logger.capture_tensor_stats(sigma, "denoise_input_sigma")
        
        # Register hooks on the DiT model to capture intermediate states
        if hasattr(pipeline.dit, 'blocks') or hasattr(pipeline.dit, 'transformer_blocks'):
            # Try to capture from transformer blocks
            blocks = getattr(pipeline.dit, 'blocks', None) or getattr(pipeline.dit, 'transformer_blocks', None)
            if blocks and len(blocks) > 0:
                # Capture from first and last blocks
                state_logger.register_forward_hook(blocks[0], "dit_block_0_output")
                if len(blocks) > 1:
                    state_logger.register_forward_hook(blocks[-1], f"dit_block_{len(blocks)-1}_output")
                # Capture from middle block if available
                if len(blocks) > 2:
                    mid_idx = len(blocks) // 2
                    state_logger.register_forward_hook(blocks[mid_idx], f"dit_block_{mid_idx}_output")
        
        # Register hook on the main DiT model
        state_logger.register_forward_hook(pipeline.dit, "dit_main_output")
        
        # Call original denoise
        result = original_denoise(xt_B_C_T_H_W, sigma, condition, use_cuda_graphs)
        
        # Log denoise output
        if hasattr(result, 'x0'):
            state_logger.hidden_states["denoise_output_x0"] = state_logger.capture_tensor_stats(result.x0, "denoise_output_x0")
        if hasattr(result, 'eps_pred'):
            state_logger.hidden_states["denoise_output_eps"] = state_logger.capture_tensor_stats(result.eps_pred, "denoise_output_eps")
        
        return result
    
    # Patch decode method
    def patched_decode(latent):
        logger.info("=== DECODE STAGE ===")
        state_logger.hidden_states["vae_decode_input"] = state_logger.capture_tensor_stats(latent, "vae_decode_input")
        result = original_decode(latent)
        state_logger.hidden_states["vae_decode_output"] = state_logger.capture_tensor_stats(result, "vae_decode_output")
        return result
    
    # Patch text encoder if available
    def patched_text_encode(prompts, max_length=512, return_mask=False):
        logger.info("=== TEXT ENCODING STAGE ===")
        result = original_text_encoder_encode(prompts, max_length, return_mask)
        state_logger.hidden_states["text_encoder_output"] = state_logger.capture_tensor_stats(result, "text_encoder_output")
        return result
    
    try:
        # Apply patches
        pipeline.encode = patched_encode
        pipeline.denoise = patched_denoise  
        pipeline.decode = patched_decode
        if pipeline.text_encoder:
            pipeline.text_encoder.encode_prompts = patched_text_encode
        
        yield
        
    finally:
        # Restore original methods
        pipeline.encode = original_encode
        pipeline.denoise = original_denoise
        pipeline.decode = original_decode
        if pipeline.text_encoder and original_text_encoder_encode:
            pipeline.text_encoder.encode_prompts = original_text_encoder_encode
        
        # Remove hooks
        state_logger.remove_hooks()


def test_original_cosmos2_pipeline():
    """Test the original Cosmos2 pipeline and capture hidden states."""
    
    logger.info("Starting original Cosmos2 pipeline test...")
    
    # Change to cosmos-predict2 directory for relative paths to work
    original_cwd = os.getcwd()
    cosmos_predict2_dir = "/workspace/workspace-cosmos/cosmos-predict2"
    os.chdir(cosmos_predict2_dir)
    
    try:
        # Test configuration
        input_image_path = "/workspace/workspace-cosmos/FastVideo/tennis.jpg"
        prompt = "A tennis ball bouncing on a racquet, the ball moves in a smooth arc as it hits the strings and rebounds with natural physics. The racquet strings vibrate slightly from the impact, and the ball continues its trajectory with realistic motion."
        num_inference_steps = 10
        
        # Check if input image exists
        if not os.path.exists(input_image_path):
            logger.error(f"Input image not found: {input_image_path}")
            return False
        
        # Create pipeline
        logger.info("Creating Video2World pipeline...")
        pipe = Video2WorldPipeline.from_config(
            config=PREDICT2_VIDEO2WORLD_PIPELINE_2B,
            dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
            text_encoder_path="checkpoints/google-t5/t5-11b",
        )
        
        logger.info("Pipeline created successfully")
        
        # Create hidden state logger
        state_logger = HiddenStateLogger()
        
        # Run inference with hidden state logging
        logger.info("Starting inference with hidden state logging...")
        
        with patch_pipeline_for_logging(pipe, state_logger):
            # Log pipeline configuration
            state_logger.hidden_states["config"] = {
                "input_image_path": input_image_path,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "pipeline_type": "original_cosmos2",
                "model_path": "/workspace/workspace-cosmos/cosmos-predict2/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
                "text_encoder_path": "/workspace/workspace-cosmos/cosmos-predict2/checkpoints/google-t5/t5-11b"
            }
            
            # Run inference
            video = pipe(
                input_path=input_image_path,
                prompt=prompt,
                num_sampling_step=num_inference_steps,
                guidance=7.0,
                seed=42
            )
            
            # Log final output
            if video is not None:
                state_logger.hidden_states["final_output"] = state_logger.capture_tensor_stats(video, "final_output")
                logger.info(f"Inference completed successfully. Output shape: {video.shape}")
                
                # Save video
                from imaginaire.utils.io import save_image_or_video
                video_output_path = "/workspace/workspace-cosmos/comparison_videos/original/cosmos2_original_output.mp4"
                save_image_or_video(video, video_output_path, fps=16)
                logger.info(f"Video saved to {video_output_path}")
                
            else:
                logger.error("Inference failed - no output generated")
                return False
        
        # Save hidden states
        output_path = "/workspace/workspace-cosmos/comparison_logs/original_pipeline_output.json"
        state_logger.save_states(output_path)
        
        logger.info("Original Cosmos2 pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    success = test_original_cosmos2_pipeline()
    if success:
        print("✅ Original pipeline test completed successfully")
        sys.exit(0)
    else:
        print("❌ Original pipeline test failed")
        sys.exit(1)