# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Dict, Optional
from fastvideo.loader.loader import get_model_loader
from fastvideo.inference_args import InferenceArgs
from fastvideo.logger import init_logger
import os

logger = init_logger(__name__)

def load_transformers_model(architecture: str, model_path: str, inference_args: InferenceArgs, component_name: Optional[str] = None) -> nn.Module:
    """
    Load a transformers model.
    
    Args:
        architecture: The model architecture name
        model_path: Path to the model directory
        inference_args: Inference arguments
        component_name: Optional component name (e.g., "text_encoder", "vae")
            If provided, will load the config from the component subdirectory
    
    Returns:
        The loaded model
    """
    model_loader = get_model_loader(inference_args)
    return model_loader.load_model(architecture, model_path, inference_args, component_name=component_name)

def load_diffusers_model(model_name: str) -> nn.Module:
    """
    Load a diffusers model.
    """
    raise NotImplementedError("Diffusers models are not supported yet.")

def get_model(modules_config: Dict, inference_args: InferenceArgs) -> Dict:
    return model

def get_scheduler(module_path: str, architecture: str, inference_args: InferenceArgs) -> Dict:
    """Create a scheduler based on the inference args. Can be overridden by subclasses."""
    if hasattr(inference_args, 'denoise_type') and inference_args.denoise_type == "flow":
        # TODO(will): add schedulers to register or create a new scheduler registry
        # TODO(will): default to config file but allow override through
        # inference args. Currently only uses inference args.
        from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchDiscreteScheduler
        return FlowMatchDiscreteScheduler(
            shift=inference_args.flow_shift,
            reverse=inference_args.flow_reverse,
            solver=inference_args.flow_solver,
        )
    else:
        raise ValueError(f"Invalid denoise type: {inference_args.denoise_type}")

__all__ = [
    "set_random_seed",
    "BasevLLMParameter",
    "PackedvLLMParameter",
    "get_model",
    "get_scheduler",
]
