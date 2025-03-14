from abc import ABC, abstractmethod

import torch
from fastvideo.inference_args import InferenceArgs
from fastvideo.logger import init_logger 
import os
from fastvideo.distributed.parallel_state import initialize_sequence_parallel_group
import glob
from fastvideo.loader.fsdp_load import load_fsdp_model
from fastvideo.loader.loader import get_model_loader
from transformers import PretrainedConfig, AutoTokenizer
from fastvideo.models.hf_transformer_utils import get_hf_config, get_diffusers_config
from fastvideo.models import get_scheduler
from fastvideo.models.registry import ModelRegistry
from safetensors.torch import load_file as safetensors_load_file

import json


logger = init_logger(__name__)


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""
    
    def __init__(self, device=None):
        self.device = device
    
    @abstractmethod
    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """
        Load the component based on the model path, architecture, and inference args.
        
        Args:
            model_path: Path to the component model
            architecture: Architecture of the component model
            inference_args: Inference arguments
            
        Returns:
            The loaded component
        """
        raise NotImplementedError
    
    @classmethod
    def for_module_type(cls, module_type: str, transformers_or_diffusers: str) -> 'ComponentLoader':
        """
        Factory method to create a component loader for a specific module type.
        
        Args:
            module_type: Type of module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the module is from transformers or diffusers
            
        Returns:
            A component loader for the specified module type
        """
        # Map of module types to their loader classes and expected library
        module_loaders = {
            "scheduler": (SchedulerLoader, "diffusers"),
            "transformer": (TransformerLoader, "diffusers"),
            "vae": (VAELoader, "diffusers"),
            "text_encoder": (TextEncoderLoader, "transformers"),
            "text_encoder_2": (TextEncoderLoader, "transformers"),
            "tokenizer": (TokenizerLoader, "transformers"),
            "tokenizer_2": (TokenizerLoader, "transformers"),
        }
        
        if module_type in module_loaders:
            loader_cls, expected_library = module_loaders[module_type]
            # Assert that the library matches what's expected for this module type
            assert transformers_or_diffusers == expected_library, f"{module_type} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()
        
        # For unknown module types, use a generic loader
        logger.warning(f"No specific loader found for module type: {module_type}. Using generic loader.")
        return GenericComponentLoader(transformers_or_diffusers)

class TextEncoderLoader(ComponentLoader):
    """Loader for text encoders."""

    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load the text encoders based on the model path, architecture, and inference args."""
        use_v1 = True
        if not use_v1:
            return self.load_v0(model_path, architecture, inference_args)
        else:
            return self.load_v1(model_path, architecture, inference_args)
    
    def load_v1(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load the text encoders based on the model path, architecture, and inference args."""
        model_config: PretrainedConfig = get_hf_config(
            model=model_path,
            trust_remote_code=inference_args.trust_remote_code,
            revision=inference_args.revision,
            model_override_args=None,
            inference_args=inference_args,
        )
        logger.info(f"HF Model config: {model_config}")
        model_loader = get_model_loader(inference_args)
        model = model_loader.load_model(model_path, model_config, inference_args)
        return model
    
    def load_v0(self, model_path: str, architecture: str, inference_args: InferenceArgs):        
        from fastvideo.pipelines.hunyuan.constants import PROMPT_TEMPLATE
        from fastvideo.models.hunyuan.text_encoder import TextEncoder
        # Text encoder
        if inference_args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[inference_args.prompt_template_video].get("crop_start", 0)
        elif inference_args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[inference_args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = inference_args.text_len + crop_start

        # prompt_template
        prompt_template = (PROMPT_TEMPLATE[inference_args.prompt_template] if inference_args.prompt_template is not None else None)

        # prompt_template_video
        prompt_template_video = (PROMPT_TEMPLATE[inference_args.prompt_template_video]
                                 if inference_args.prompt_template_video is not None else None)

        text_encoder = TextEncoder(
            text_encoder_type=inference_args.text_encoder,
            text_encoder_path=model_path,
            max_length=max_length,
            text_encoder_precision=inference_args.text_encoder_precision,
            tokenizer_type=inference_args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=inference_args.hidden_state_skip_layer,
            apply_final_norm=inference_args.apply_final_norm,
            reproduce=inference_args.reproduce,
            logger=logger,
            device=self.device if not inference_args.use_cpu_offload else "cpu",
        )
        return text_encoder

class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""
    
    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load the tokenizer based on the model path, architecture, and inference args."""
        logger.info(f"Loading tokenizer from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            # TODO(will): pass these tokenizer kwargs from inference args? Maybe
            # other method of config?
            padding_size='right',
        )
        logger.info(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
        return tokenizer

class VAELoader(ComponentLoader):
    """Loader for VAE."""
    
    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load the VAE based on the model path, architecture, and inference args."""
        # TODO(will): move this to a constants file
        from fastvideo.pipelines.hunyuan.constants import PRECISION_TO_TYPE

        config = get_diffusers_config(model=model_path)
        
        class_name = config.pop("_class_name")
        assert class_name is not None, "Model config does not contain a _class_name attribute. Only diffusers format is supported."
        config.pop("_diffusers_version")

        vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        vae = vae_cls(**config).to(inference_args.device)
        
        # Find all safetensors files
        safetensors_list = glob.glob(os.path.join(str(model_path), "*.safetensors"))
        # TODO(PY)
        assert len(safetensors_list) == 1, f"Found {len(safetensors_list)} safetensors files in {path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded)
        dtype = PRECISION_TO_TYPE[inference_args.vae_precision]
        vae = vae.eval().to(dtype)

        # TODO(will):  should we define hunyuan vae config class?
        vae_kwargs = {
            "s_ratio": config["spatial_compression_ratio"],
            "t_ratio": config["temporal_compression_ratio"],
        }

        vae.kwargs = vae_kwargs
        
        return vae

class TransformerLoader(ComponentLoader):
    """Loader for transformer."""
    
    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load the transformer based on the model path, architecture, and inference args."""
        
        model_config = get_diffusers_config(model=model_path)
        cls_name = model_config.pop("_class_name")
        if cls_name is None:
            raise ValueError(f"Model config does not contain a _class_name attribute. "
                         "Only diffusers format is supported.")
        model_config.pop("_diffusers_version")

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = glob.glob(os.path.join(str(model_path), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")
        
        logger.info(f"Loading model from {len(safetensors_list)} safetensors files in {model_path}")

        # initialize_sequence_parallel_group(inference_args.sp_size)
        
        # Load the model using FSDP loader
        logger.info(f"Loading model from {cls_name}")
        model = load_fsdp_model(
            model_cls=model_cls,
            init_params=model_config,
            weight_dir_list=safetensors_list,
            device=inference_args.device,
            cpu_offload=inference_args.use_cpu_offload
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded model with {total_params / 1e9:.2f}B parameters")
        
        model.eval()
        return model

class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""
    
    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load the scheduler based on the model path, architecture, and inference args."""
        
        scheduler = get_scheduler(
            module_path=model_path,
            architecture=architecture,
            inference_args=inference_args,
        )
        logger.info(f"Scheduler loaded: {scheduler}")
        return scheduler

class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""
    
    def __init__(self, library="transformers"):
        super().__init__()
        self.library = library
    
    def load(self, model_path: str, architecture: str, inference_args: InferenceArgs):
        """Load a generic component based on the model path, architecture, and inference args."""
        logger.warning(f"Using generic loader for {model_path} with library {self.library}")
        
        if self.library == "transformers":
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=inference_args.trust_remote_code,
                revision=inference_args.revision,
            )
            logger.info(f"Loaded generic transformers model: {model.__class__.__name__}")
            return model
        elif self.library == "diffusers":
            logger.warning(f"Generic loading for diffusers components is not fully implemented")
            from fastvideo.models.hf_transformer_utils import get_diffusers_config
            
            model_config = get_diffusers_config(model=model_path)
            logger.info(f"Diffusers Model config: {model_config}")
            # This is a placeholder - in a real implementation, you'd need to handle this properly
            return None
        else:
            raise ValueError(f"Unsupported library: {self.library}")

class PipelineComponentLoader:
    """
    Utility class for loading pipeline components.
    This replaces the chain of if-else statements in load_pipeline_module.
    """
    
    @staticmethod
    def load_module(module_name: str, component_model_path: str, transformers_or_diffusers: str, 
                   architecture: str, inference_args: InferenceArgs):
        """
        Load a pipeline module.
        
        Args:
            module_name: Name of the module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the module is from transformers or diffusers
            architecture: Architecture of the component model
            inference_args: Inference arguments
            
        Returns:
            The loaded module
        """
        logger.info(f"Loading {module_name} using {transformers_or_diffusers} from {component_model_path}")
        
        # Get the appropriate loader for this module type
        loader = ComponentLoader.for_module_type(module_name, transformers_or_diffusers)
        
        # Load the module
        return loader.load(component_model_path, architecture, inference_args)
