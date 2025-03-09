"""
Diffusion pipelines for FastVideo.

This package contains diffusion pipelines for generating videos and images.
"""
import os
import json
from copy import deepcopy

from typing import Dict, Optional, Type, Any

# First, import the registry
from fastvideo.pipelines.pipeline_registry import PipelineRegistry, register_pipeline
from fastvideo.inference_args import InferenceArgs
from fastvideo.logger import init_logger
from huggingface_hub import snapshot_download

logger = init_logger(__name__)

# Then import the base classes
from fastvideo.pipelines.composed.composed_pipeline_base import (
    ComposedPipelineBase, 
    DiffusionPipelineOutput
)

def get_pipeline_type(inference_args: InferenceArgs) -> str:
    # hardcode for now
    return "hunyuan_video"

def maybe_download_model(model_path: str) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.
    
    Args:
        model_path: Local path or Hugging Face Hub model ID
        
    Returns:
        Local path to the model
    """
    
    # If the path exists locally, return it
    if os.path.exists(model_path):
        logger.info(f"Model already exists locally at {model_path}")
        return model_path
    
    # Otherwise, assume it's a HF Hub model ID and try to download it
    try:
        logger.info(f"Downloading model snapshot from HF Hub...")
        local_path = snapshot_download(
            repo_id=model_path,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.pt", "*.pth", "*.ckpt"],
            ignore_patterns=["*.onnx", "*.msgpack"],
        )
        logger.info(f"Downloaded model to {local_path}")
        return local_path
    except Exception as e:
        raise ValueError(f"Could not find model at {model_path} and failed to download from HF Hub: {e}")

def verify_model_config_and_directory(model_path: str) -> dict:
    """
    Verify that the model directory contains a valid diffusers configuration.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        The loaded model configuration as a dictionary
    """
    
    # Check for model_index.json which is required for diffusers models
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        raise ValueError(
            f"Model directory {model_path} does not contain model_index.json. "
            "Only Hugging Face diffusers format is supported."
        )
    
    # Check for transformer and vae directories
    transformer_dir = os.path.join(model_path, "transformer")
    vae_dir = os.path.join(model_path, "vae")
    
    if not os.path.exists(transformer_dir):
        raise ValueError(f"Model directory {model_path} does not contain a transformer/ directory.")
    
    if not os.path.exists(vae_dir):
        raise ValueError(f"Model directory {model_path} does not contain a vae/ directory.")
    
    # Load the config
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load model configuration from {config_path}: {e}")

    # Verify diffusers version exists
    if "_diffusers_version" not in config:
        raise ValueError(f"model_index.json does not contain _diffusers_version")
    
    logger.info(f"Diffusers version: {config['_diffusers_version']}")
    return config

def _load_pipeline_modules(config: Dict) -> dict[str, Any]:
    """
    Load the pipeline modules from the config.
    """
    modules = {}
    logger.info(f"Loading pipeline modules from config: {config}")
    modules_config = deepcopy(config)

    # remove keys that are not pipeline modules
    modules_config = modules_config.pop("_class_name")
    modules_config = modules_config.pop("_diffusers_version")

    # some sanity checks
    assert len(modules_config) > 1, "model_index.json must contain at least one pipeline module"

    assert "vae" in modules_config, "model_index.json must contain a vae module"
    assert "text_encoder" in modules_config, "model_index.json must contain a text_encoder module"
    assert "transformer" in modules_config, "model_index.json must contain a transformer module"
    assert "scheduler" in modules_config, "model_index.json must contain a scheduler module"
    assert "tokenizer" in modules_config, "model_index.json must contain a tokenizer module"

    for module_name, (transformers_or_diffusers, architecture) in modules_config.items():
        print(f"module_name: {module_name}, transformers_or_diffusers: {transformers_or_diffusers}, architecture: {architecture}")

    return modules_config
    # pass

def build_pipeline(inference_args: InferenceArgs) -> ComposedPipelineBase:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class 
    4. parse the config to get the model component (vae, text_encoders, etc...)
    5. the pipeline loader class will use the model component names and paths to load
    6. the pipeline class will be composed of the models returned by the pipeline loader
    """
    # Get pipeline type
    model_path = inference_args.model_path
    model_path = maybe_download_model(model_path)
    logger.info(f"Model path: {model_path}")
    config = verify_model_config_and_directory(model_path)
    print(f"config: {config}")

    pipeline_architecture = config["_class_name"]
    if pipeline_architecture is None:
        raise ValueError("Model config does not contain a _class_name attribute. "
                         "Only diffusers format is supported.")
    
    pipeline_cls = PipelineRegistry.resolve_pipeline_cls(pipeline_architecture)

    # instantiate the pipeline
    pipeline = pipeline_cls()

    pipeline_modules = _load_pipeline_modules(config)

    logger.info(f"Registering modules")
    pipeline.register_modules(pipeline_modules)
    logger.info(f"Setting up pipeline")
    pipeline.setup_pipeline(inference_args)

    logger.info(f"Initializing pipeline")
    pipeline.initialize_pipeline(inference_args)
    
    # pipeline is now initialized and ready to use
    return pipeline



def list_available_pipelines() -> Dict[str, Type[Any]]:
    """
    List all available pipeline types.
    
    Returns:
        A dictionary of pipeline names to pipeline classes.
    """
    return PipelineRegistry.list()


# Import all pipeline implementations to register them
# These imports should be at the end to avoid circular imports
from fastvideo.pipelines.implementations.hunyuan import HunyuanVideoPipeline

__all__ = [
    "create_pipeline",
    "list_available_pipelines",
    "ComposedPipelineBase",
    "DiffusionPipelineOutput",
    "register_pipeline",
    "HunyuanVideoPipeline",
] 