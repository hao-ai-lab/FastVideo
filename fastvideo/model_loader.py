"""
Model loader for diffusion models.

This module provides a base class and implementations for loading model components
needed by diffusion pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
import os
import torch
import traceback

from fastvideo.inference_args import InferenceArgs
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    A model loader is responsible for loading the components needed by a diffusion
    pipeline, such as the UNet, VAE, text encoder, etc.
    """
    
    @abstractmethod
    def load_components(
        self,
        inference_args: InferenceArgs,
    ) -> Dict[str, Any]:
        """
        Load the components needed by a diffusion pipeline.
        
        Args:
            inference_args: The inference arguments.
            
        Returns:
            A dictionary of component names to component objects.
        """
        pass


class HunYuanModelLoader(ModelLoader):
    """
    Model loader for HunYuan models.
    
    This loader loads the components needed by HunYuan diffusion pipelines.
    """
    
    def load_components(
        self,
        inference_args: InferenceArgs,
    ) -> Dict[str, Any]:
        """
        Load the components needed by a HunYuan diffusion pipeline.
        
        Args:
            inference_args: The inference arguments.
            
        Returns:
            A dictionary of component names to component objects.
            
        Raises:
            FileNotFoundError: If a required model file is not found.
            RuntimeError: If there is an error loading a model component.
        """
        try:
            from fastvideo.models.unet import UNet3DConditionModel
            from fastvideo.models.vae import AutoencoderKLCausal3D
            from fastvideo.models.text_encoder import TextEncoder
            from fastvideo.models.tokenizer import Tokenizer
            from fastvideo.models.scheduler import DDIMScheduler
            
            logger.info(f"Loading model components from {inference_args.model_path}")
            
            # Check if model path exists
            if not os.path.exists(inference_args.model_path):
                raise FileNotFoundError(f"Model path not found: {inference_args.model_path}")
            
            # Check if DiT weight exists
            if inference_args.dit_weight and not os.path.exists(inference_args.dit_weight):
                raise FileNotFoundError(f"DiT weight not found: {inference_args.dit_weight}")
            
            # Load UNet
            try:
                logger.info("Loading UNet...")
                unet = UNet3DConditionModel.from_pretrained(
                    inference_args.dit_weight,
                    subfolder="",
                    load_key=inference_args.load_key,
                )
                logger.info("UNet loaded successfully")
            except Exception as e:
                logger.error(f"Error loading UNet: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load UNet: {e}")
            
            # Load VAE
            try:
                logger.info("Loading VAE...")
                vae = AutoencoderKLCausal3D.from_pretrained(
                    inference_args.model_path,
                    subfolder="vae",
                    use_safetensors=True,
                )
                logger.info("VAE loaded successfully")
            except Exception as e:
                logger.error(f"Error loading VAE: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load VAE: {e}")
            
            # Configure VAE for sequence parallelism if needed
            if inference_args.vae_sp:
                try:
                    logger.info("Configuring VAE for sequence parallelism")
                    vae.enable_sequence_parallel()
                    logger.info("VAE configured for sequence parallelism")
                except Exception as e:
                    logger.error(f"Error configuring VAE for sequence parallelism: {e}")
                    logger.error(traceback.format_exc())
                    logger.warning("Continuing without sequence parallelism for VAE")
            
            # Load text encoders
            try:
                logger.info("Loading primary text encoder...")
                text_encoder = TextEncoder.from_pretrained(
                    inference_args.model_path,
                    subfolder="text_encoder",
                    use_safetensors=True,
                )
                logger.info("Primary text encoder loaded successfully")
            except Exception as e:
                logger.error(f"Error loading primary text encoder: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load primary text encoder: {e}")
            
            try:
                logger.info("Loading secondary text encoder...")
                text_encoder_2 = TextEncoder.from_pretrained(
                    inference_args.model_path,
                    subfolder="text_encoder_2",
                    use_safetensors=True,
                )
                logger.info("Secondary text encoder loaded successfully")
            except Exception as e:
                logger.error(f"Error loading secondary text encoder: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load secondary text encoder: {e}")
            
            # Load tokenizers
            try:
                logger.info("Loading primary tokenizer...")
                tokenizer = Tokenizer.from_pretrained(
                    inference_args.model_path,
                    subfolder="tokenizer",
                )
                logger.info("Primary tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading primary tokenizer: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load primary tokenizer: {e}")
            
            try:
                logger.info("Loading secondary tokenizer...")
                tokenizer_2 = Tokenizer.from_pretrained(
                    inference_args.model_path,
                    subfolder="tokenizer_2",
                )
                logger.info("Secondary tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading secondary tokenizer: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load secondary tokenizer: {e}")
            
            # Load scheduler
            try:
                logger.info("Loading scheduler...")
                scheduler = DDIMScheduler.from_pretrained(
                    inference_args.model_path,
                    subfolder="scheduler",
                )
                logger.info("Scheduler loaded successfully")
            except Exception as e:
                logger.error(f"Error loading scheduler: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load scheduler: {e}")
            
            # Move models to the appropriate device
            device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
            logger.info(f"Moving models to device: {device}")
            
            try:
                unet.to(device)
                vae.to(device)
                text_encoder.to(device)
                text_encoder_2.to(device)
                logger.info("Models moved to device successfully")
            except Exception as e:
                logger.error(f"Error moving models to device: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to move models to device: {e}")
            
            # Set precision
            try:
                logger.info(f"Setting UNet precision to {inference_args.precision}")
                if inference_args.precision == "bf16":
                    unet = unet.to(torch.bfloat16)
                elif inference_args.precision == "fp16":
                    unet = unet.to(torch.float16)
                
                logger.info(f"Setting VAE precision to {inference_args.vae_precision}")
                if inference_args.vae_precision == "fp16":
                    vae = vae.to(torch.float16)
                
                logger.info(f"Setting text encoder precision to {inference_args.text_encoder_precision}")
                if inference_args.text_encoder_precision == "fp16":
                    text_encoder = text_encoder.to(torch.float16)
                
                logger.info(f"Setting secondary text encoder precision to {inference_args.text_encoder_precision_2}")
                if inference_args.text_encoder_precision_2 == "fp16":
                    text_encoder_2 = text_encoder_2.to(torch.float16)
                
                logger.info("Precision set successfully")
            except Exception as e:
                logger.error(f"Error setting precision: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Continuing with default precision")
            
            # Return the components
            return {
                "unet": unet,
                "vae": vae,
                "text_encoder": text_encoder,
                "text_encoder_2": text_encoder_2,
                "tokenizer": tokenizer,
                "tokenizer_2": tokenizer_2,
                "scheduler": scheduler,
            }
        except Exception as e:
            logger.error(f"Unhandled error in load_components: {e}")
            logger.error(traceback.format_exc())
            raise


# Factory function to get the appropriate model loader
def get_model_loader(model_type: str) -> Type[ModelLoader]:
    """
    Get the appropriate model loader for the given model type.
    
    Args:
        model_type: The type of model to load.
        
    Returns:
        The model loader class.
        
    Raises:
        ValueError: If the model type is not recognized.
    """
    model_loaders = {
        "hunyuan": HunYuanModelLoader,
        "HYVideo-T/2-cfgdistill": HunYuanModelLoader,
    }
    
    if model_type not in model_loaders:
        available_models = list(model_loaders.keys())
        raise ValueError(
            f"Model type '{model_type}' not recognized. "
            f"Available model types: {available_models}"
        )
    
    return model_loaders[model_type] 