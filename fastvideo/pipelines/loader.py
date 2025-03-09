from abc import ABC, abstractmethod

import torch
from fastvideo.inference_args import InferenceArgs
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
from fastvideo.logger import init_logger
from fastvideo.platforms import current_platform
from fastvideo.distributed.parallel_state import get_sp_group
from fastvideo.pipelines.composed import ComposedPipelineBase
import os

import json


logger = init_logger(__name__)

T2V_PIPLINES = {
    "HunyuanVideo": ("hunyuan_video", "HunyuanVideoPipeline"),
}

FASTVIDEO_PIPLINES = {
    **T2V_PIPLINES,
}

def resolve_pipeline_cls(inference_args: InferenceArgs):
    """Resolve the pipeline class based on the inference args."""
    # TODO(will): resolve the pipeline class based on the model path
    # read from hf config file _class_name
    from fastvideo.pipelines.hunyuan.hunyuan_video import HunyuanVideoPipeline
    return HunyuanVideoPipeline

def resolve_pipeline_cls_v2(inference_args: InferenceArgs):
    """Resolve the pipeline class based on the inference args."""
    # TODO(will): resolve the pipeline class based on the model path
    # read from hf config file _class_name
    from fastvideo.pipelines.implementations.hunyuan import HunyuanVideoPipeline
    return HunyuanVideoPipeline


class ModelComponents:
    """Container for model components used by diffusion pipelines"""
    
    def __init__(self, 
                 vae=None, 
                 text_encoder=None, 
                 text_encoder_2=None, 
                 transformer=None, 
                 scheduler=None):
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.transformer = transformer
        self.scheduler = scheduler

HF_PIPELINE_CONFIG = \
"""
{
  "_class_name": "HunyuanVideoPipeline",
  "_diffusers_version": "0.32.0.dev0",
  "scheduler": [
    "diffusers",
    "FlowMatchEulerDiscreteScheduler"
  ],
  "text_encoder": [
    "transformers",
    "LlamaModel"
  ],
  "text_encoder_2": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "LlamaTokenizerFast"
  ],
  "tokenizer_2": [
    "transformers",
    "CLIPTokenizer"
  ],
  "transformer": [
    "diffusers",
    "HunyuanVideoTransformer3DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKLHunyuanVideo"
  ]
}
"""


class PipelineLoader(ABC):
    """
    Base class for loading model components that can be used across different pipelines.
    This separates model loading from pipeline creation and inference.
    """

    def __init__(self, inference_args: InferenceArgs, use_v0=False):
        assert current_platform.device_type == "cuda"
        if use_v0:
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            logger.info(f"Using device: cuda:{local_rank}")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            sp_group = get_sp_group()
            # device = "cuda"
            if sp_group.world_size > 1:
                local_rank = sp_group.rank
                logger.info(f"Using device: cuda:{local_rank}")
                self.device = torch.device(f"cuda:{local_rank}")
    
    @abstractmethod
    def load_text_encoders(self, inference_args: InferenceArgs):
        """Load the text encoders based on the inference args."""
        raise NotImplementedError

    @abstractmethod
    def load_vae(self, inference_args: InferenceArgs):
        """Load the VAE based on the inference args."""
        raise NotImplementedError

    @abstractmethod
    def load_transformer(self, inference_args: InferenceArgs):
        """Load the transformer based on the inference args."""
        raise NotImplementedError

    @abstractmethod
    def load_scheduler(self, inference_args: InferenceArgs):
        """Load the scheduler based on the inference args."""
        raise NotImplementedError
    
    def load_components(self, inference_args: InferenceArgs):
        logger.info(f"Loading components for {inference_args.model_path}")
        logger.info("Loading transformer")
        transformer = self.load_transformer(inference_args)

        logger.info("Loading VAE")
        vae, vae_kwargs = self.load_vae(inference_args)

        logger.info("Loading text encoders")
        text_encoders = self.load_text_encoders(inference_args)
        logger.info(f"\tLoaded {len(text_encoders)} text encoders")
        logger.info(f"\ttext_encoders: {[type(e) for e in text_encoders]}")
        text_encoder = text_encoders[0]
        text_encoder_2 = text_encoders[1] # may be None
        
        logger.info("Loading scheduler")
        scheduler = self.load_scheduler(inference_args)
        logger.info("Model components loaded")

        return vae, vae_kwargs, text_encoder, text_encoder_2, transformer, scheduler

    def load_pipeline_v2(self, inference_args: InferenceArgs):
        """Load the staged pipeline based on the inference args."""
        # Load configuration
        # TODO(will): load config from model path. also move to better place.
        pipeline_cls = resolve_pipeline_cls_v2(inference_args)
        assert issubclass(pipeline_cls, ComposedPipelineBase)
        logger.info(f"Pipeline class: {pipeline_cls}")

        config = json.loads(HF_PIPELINE_CONFIG)
        logger.info(f"Config: {config}")

        logger.info(f"Loading components for {inference_args.model_path}")
        logger.info("Loading transformer")
        transformer = self.load_transformer(inference_args)

        logger.info("Loading VAE")
        vae, vae_kwargs = self.load_vae(inference_args)

        logger.info("Loading text encoders")
        text_encoders = self.load_text_encoders(inference_args)
        logger.info(f"\tLoaded {len(text_encoders)} text encoders")
        logger.info(f"\ttext_encoders: {[type(e) for e in text_encoders]}")
        text_encoder = text_encoders[0]
        text_encoder_2 = text_encoders[1] # may be None
        
        logger.info("Loading scheduler")
        scheduler = self.load_scheduler(inference_args)

        # Create pipeline
        pipeline = pipeline_cls()

        # TODO(will): the keys of this dict should be from the hf config
        pipeline_modules = {
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "transformer": transformer,
            "scheduler": scheduler
        }

        logger.info(f"Registering modules")
        pipeline.register_modules(pipeline_modules)
        logger.info(f"Setting up pipeline")
        pipeline.setup_pipeline(inference_args)

        logger.info(f"Initializing pipeline")
        pipeline.initialize_pipeline(inference_args)
        
        return pipeline

    def load_pipeline(self, inference_args: InferenceArgs):
        """Load the pipeline based on the inference args."""
        # Load configuration
        # TODO(will): load config from model path. also move to better place.
        pipeline_cls = resolve_pipeline_cls(inference_args)
        logger.info(f"Pipeline class: {pipeline_cls}")

        logger.info(f"Loading components for {inference_args.model_path}")
        logger.info("Loading transformer")
        transformer = self.load_transformer(inference_args)

        logger.info("Loading VAE")
        vae, vae_kwargs = self.load_vae(inference_args)

        logger.info("Loading text encoders")
        text_encoders = self.load_text_encoders(inference_args)
        logger.info(f"\tLoaded {len(text_encoders)} text encoders")
        logger.info(f"\ttext_encoders: {[type(e) for e in text_encoders]}")
        text_encoder = text_encoders[0]
        text_encoder_2 = text_encoders[1] # may be None
        
        logger.info("Loading scheduler")
        scheduler = self.load_scheduler(inference_args)
        logger.info("Model components loaded")

        # Create pipeline
        pipeline = pipeline_cls(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            scheduler=scheduler
        )
        
        return pipeline
    
    def load_scheduler(self, inference_args: InferenceArgs):
        """Create a scheduler based on the inference args. Can be overridden by subclasses."""
        if hasattr(inference_args, 'denoise_type') and inference_args.denoise_type == "flow":
            return FlowMatchDiscreteScheduler(
                shift=inference_args.flow_shift,
                reverse=inference_args.flow_reverse,
                solver=inference_args.flow_solver,
            )
        return None
    

def get_pipeline_loader(inference_args: InferenceArgs) -> PipelineLoader:
    """Get a pipeline loader based on the inference args."""
    from fastvideo.pipelines.implementations.hunyuan.hunyuan_loader import HunyuanPipelineLoader
    # Map model types to their specific loaders
    loader_map = {
        "hunyuan_video": HunyuanPipelineLoader,
        # Add other model types and their loaders here
    }
    
    # Determine which loader to use based on model type
    # model_type = inference_args.model_type if hasattr(inference_args, 'model_type') else None
    
    # if model_type and model_type in loader_map:
    #     return loader_map[model_type]()
    
    # TODO(will): add more model types and their loaders here instead of
    # hardcoding
    return HunyuanPipelineLoader(inference_args, use_v0=True)
    
    # Default to base loader
    return PipelineLoader()
