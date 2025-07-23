"""
Stage logging utility for capturing intermediate hidden states in FastVideo pipelines.
"""

import torch
import json
import threading
from typing import Dict, Any, Optional

class StageLogger:
    """Logger for capturing intermediate hidden states from pipeline stages."""
    
    def __init__(self):
        self.stage_outputs = {}
        
    def log_stage_output(self, stage_name: str, batch, description: str = ""):
        """Log output from a pipeline stage."""
        stage_key = f"{stage_name}_{description}" if description else stage_name
        
        # Log output if available (final stage)
        if hasattr(batch, 'output') and batch.output is not None:
            self.stage_outputs[f"{stage_key}_output"] = self._capture_tensor_stats(batch.output, f"{stage_key}_output")
        
        # Log latents if available
        if hasattr(batch, 'latents') and batch.latents is not None:
            self.stage_outputs[f"{stage_key}_latents"] = self._capture_tensor_stats(batch.latents, f"{stage_key}_latents")
        
        # Log prompt embeddings if available
        if hasattr(batch, 'prompt_embeds') and batch.prompt_embeds:
            if isinstance(batch.prompt_embeds, list) and len(batch.prompt_embeds) > 0:
                self.stage_outputs[f"{stage_key}_prompt_embeds"] = self._capture_tensor_stats(batch.prompt_embeds[0], f"{stage_key}_prompt_embeds")
        
        # Log negative prompt embeddings if available
        if hasattr(batch, 'negative_prompt_embeds') and batch.negative_prompt_embeds:
            if isinstance(batch.negative_prompt_embeds, list) and len(batch.negative_prompt_embeds) > 0:
                self.stage_outputs[f"{stage_key}_negative_prompt_embeds"] = self._capture_tensor_stats(batch.negative_prompt_embeds[0], f"{stage_key}_negative_prompt_embeds")
        
        # Log samples if available
        if hasattr(batch, 'samples') and batch.samples is not None:
            self.stage_outputs[f"{stage_key}_samples"] = self._capture_tensor_stats(batch.samples, f"{stage_key}_samples")
        
        # Log conditioning-specific outputs
        if hasattr(batch, 'extra') and batch.extra:
            if 'conditioning_latents' in batch.extra and batch.extra['conditioning_latents'] is not None:
                self.stage_outputs[f"{stage_key}_conditioning_latents"] = self._capture_tensor_stats(batch.extra['conditioning_latents'], f"{stage_key}_conditioning_latents")
            
            if 'condition_mask' in batch.extra and batch.extra['condition_mask'] is not None:
                self.stage_outputs[f"{stage_key}_condition_mask"] = self._capture_tensor_stats(batch.extra['condition_mask'], f"{stage_key}_condition_mask")
    
    def _capture_tensor_stats(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
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
    
    def get_outputs(self) -> Dict[str, Any]:
        """Get all captured stage outputs."""
        return self.stage_outputs
    
    def save_outputs(self, filepath: str):
        """Save captured stage outputs to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.stage_outputs, f, indent=2)


# Thread-local storage for stage loggers to work with multiprocessing
_thread_local = threading.local()

def set_global_stage_logger(logger: Optional[StageLogger]):
    """Set the global stage logger for the current thread/process."""
    _thread_local.stage_logger = logger

def get_global_stage_logger() -> Optional[StageLogger]:
    """Get the global stage logger for the current thread/process."""
    return getattr(_thread_local, 'stage_logger', None)

def log_stage_output(stage_name: str, batch, description: str = ""):
    """Convenience function to log stage output using global logger."""
    logger = get_global_stage_logger()
    if logger is not None:
        logger.log_stage_output(stage_name, batch, description)