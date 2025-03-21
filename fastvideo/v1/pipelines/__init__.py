"""
Diffusion pipelines for fastvideo.v1.

This package contains diffusion pipelines for generating videos and images.
"""
import os
import json
from copy import deepcopy
import tempfile
import filelock
import hashlib

from typing import Dict, Optional, Type, Any

from fastvideo.v1.pipelines.pipeline_registry import PipelineRegistry
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from huggingface_hub import snapshot_download
from fastvideo.v1.models.loader.component_loader import PipelineComponentLoader

logger = init_logger(__name__)

# Then import the base classes
from fastvideo.v1.pipelines.composed.composed_pipeline_base import (
    ComposedPipelineBase, 
    DiffusionPipelineOutput
)


def get_pipeline_type(inference_args: InferenceArgs) -> str:
    # hardcode for now
    return "hunyuan_video"



def get_lock(model_name_or_path: str):
    lock_dir = tempfile.gettempdir()
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock



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
        logger.info(f"Downloading model snapshot from HF Hub for {model_path}...")
        with get_lock(model_path):
            local_path = snapshot_download(
                repo_id=model_path,
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

def load_pipeline_module(module_name: str, component_model_path: str, transformers_or_diffusers: str, architecture: str, inference_args: InferenceArgs) -> Any:
    """
    Load a pipeline module using the appropriate loader.
    
    Args:
        module_name: Name of the module (e.g., "vae", "text_encoder", "transformer", "scheduler")
        component_model_path: Path to the component model
        transformers_or_diffusers: Whether the module is from transformers or diffusers
        architecture: Architecture of the component model
        inference_args: Inference arguments
        
    Returns:
        The loaded module
    """
    return PipelineComponentLoader.load_module(
        module_name=module_name,
        component_model_path=component_model_path,
        transformers_or_diffusers=transformers_or_diffusers,
        architecture=architecture,
        inference_args=inference_args
    )

def load_pipeline_modules(model_path: str, config: Dict, inference_args: InferenceArgs) -> dict[str, Any]:
    """
    Load the pipeline modules from the config.
    
    Args:
        config: The model_index.json config
        inference_args: Inference arguments
        
    Returns:
        Dictionary mapping module names to loaded modules
    """
    logger.info(f"Loading pipeline modules from config: {config}")
    modules_config = deepcopy(config)
    
    # remove keys that are not pipeline modules
    modules_config.pop("_class_name")
    modules_config.pop("_diffusers_version")
    
    # some sanity checks
    assert len(modules_config) > 1, "model_index.json must contain at least one pipeline module"
    
    required_modules = ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]
    for module_name in required_modules:
        if module_name not in modules_config:
            raise ValueError(f"model_index.json must contain a {module_name} module")
    logger.info(f"Diffusers config passed sanity checks")
    

    # all the component models used by the pipeline
    pipeline_modules = {}
    for module_name, (transformers_or_diffusers, architecture) in modules_config.items():
        component_model_path = os.path.join(model_path, module_name)
        module = load_pipeline_module(
            module_name,
            component_model_path,
            transformers_or_diffusers,
            architecture,
            inference_args,
        )

        pipeline_modules[module_name] = module


    # Check if all required modules were loaded
    for module_name in required_modules:
        if module_name not in pipeline_modules or pipeline_modules[module_name] is None:
            logger.warning(f"Required module {module_name} was not loaded properly")
    
    return pipeline_modules

def build_pipeline(inference_args: InferenceArgs) -> ComposedPipelineBase:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class 
    4. parse the config to get the model components (vae, text_encoders, etc...)
    5. the pipeline loader class will use the model component names and paths to load
    6. the pipeline class will be composed of the models returned by the pipeline loader
    """
    # Get pipeline type
    model_path = inference_args.model_path
    model_path = maybe_download_model(model_path)
    # inference_args.downloaded_model_path = model_path
    logger.info(f"Model path: {model_path}")
    config = verify_model_config_and_directory(model_path)

    pipeline_architecture = config.get("_class_name")
    if pipeline_architecture is None:
        raise ValueError("Model config does not contain a _class_name attribute. "
                         "Only diffusers format is supported.")
    
    pipeline_cls, pipeline_architecture = PipelineRegistry.resolve_pipeline_cls(pipeline_architecture)

    # instantiate the pipeline
    pipeline = pipeline_cls()
    

    pipeline_modules = load_pipeline_modules(model_path, config, inference_args)


    logger.info(f"Initializing encoders")
    pipeline.initialize_encoders(pipeline_modules, inference_args)

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

__all__ = [
    "build_pipeline",
    "list_available_pipelines",
    "ComposedPipelineBase",
    "DiffusionPipelineOutput",
] 