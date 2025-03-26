"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import numpy as np
from copy import deepcopy
import os

from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.stages import PipelineStage
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import maybe_download_model, verify_model_config_and_directory
from fastvideo.v1.models.loader.component_loader import PipelineComponentLoader

logger = init_logger(__name__)


@dataclass
class DiffusionPipelineOutput:
    """Output from a diffusion pipeline."""
    videos: Union[torch.Tensor, np.ndarray]


class ComposedPipelineBase(ABC):
    """
    Base class for pipelines composed of multiple stages.
    
    This class provides the framework for creating pipelines by composing multiple
    stages together. Each stage is responsible for a specific part of the diffusion
    process, and the pipeline orchestrates the execution of these stages.
    """

    is_video_pipeline: bool = False  # To be overridden by video pipelines

    # TODO(will): args should support both inference args and training args
    def __init__(self,
                 model_path: str,
                 inference_args: InferenceArgs,
                 config: Dict[str, Any] = None):
        """
        Initialize the pipeline. After __init__, the pipeline should be ready to
        use. The pipeline should be stateless and not hold any batch state.
        """
        self.model_path = model_path
        self._stages = []
        self._stage_name_mapping = {}

        if config is None:
            # Load configuration
            logger.info(f"Loading pipeline configuration...")
            self.config = self._load_config(model_path)
        else:
            self.config = config

        # Load modules directly in initialization
        logger.info(f"Loading pipeline modules...")
        self.modules = self.load_modules(inference_args)
        print(f"keys: {self.modules.keys()}")

        self.initialize_pipeline(inference_args)

        logger.info(f"Creating pipeline stages...")
        self.create_pipeline_stages(inference_args)

    def get_module(self, module_name: str) -> Any:
        return self.modules[module_name]

    def add_module(self, module_name: str, module: Any):
        self.modules[module_name] = module

    def _load_config(self, model_path: str) -> Dict[str, Any]:
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        # inference_args.downloaded_model_path = model_path
        logger.info(f"Model path: {model_path}")
        config = verify_model_config_and_directory(model_path)
        return config

    @abstractmethod
    def required_config_modules(self) -> List[str]:
        """
        List of modules that are required by the pipeline. The names should match
        the diffusers directory and model_index.json file. These modules will be
        loaded using the PipelineComponentLoader and made available in the
        modules dictionary. Access these modules using the get_module method.

        Example:
        def required_config_modules(self) -> List[str]:
            return ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]
        """
        raise NotImplementedError

    @abstractmethod
    def create_pipeline_stages(self, inference_args: InferenceArgs):
        """
        Create the pipeline stages.
        """
        raise NotImplementedError
    
    
    def load_modules(self, inference_args: InferenceArgs) -> Dict[str, Any]:
        """
        Load the modules from the config.
        """
        logger.info(f"Loading pipeline modules from config: {self.config}")
        modules_config = deepcopy(self.config)

        # remove keys that are not pipeline modules
        modules_config.pop("_class_name")
        modules_config.pop("_diffusers_version")

        # some sanity checks
        assert len(
            modules_config
        ) > 1, "model_index.json must contain at least one pipeline module"

        required_modules = [
            "vae", "text_encoder", "transformer", "scheduler", "tokenizer"
        ]
        for module_name in required_modules:
            if module_name not in modules_config:
                raise ValueError(
                    f"model_index.json must contain a {module_name} module")
        logger.info(f"Diffusers config passed sanity checks")

        # all the component models used by the pipeline
        modules = {}
        for module_name, (transformers_or_diffusers,
                          architecture) in modules_config.items():
            component_model_path = os.path.join(self.model_path, module_name)
            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                inference_args=inference_args,
            )
            logger.info(
                f"Loaded module {module_name} from {component_model_path}")

            if module_name in modules:
                logger.warning(f"Overwriting module {module_name}")
            modules[module_name] = module

        required_modules = self.required_config_modules()
        # Check if all required modules were loaded
        for module_name in required_modules:
            if module_name not in modules or modules[module_name] is None:
                raise ValueError(
                    f"Required module {module_name} was not loaded properly")

        return modules

    def add_stage(self, stage_name: str, stage: PipelineStage):
        assert self.modules is not None, "No modules are registered"
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

    # TODO(will): don't hardcode no_grad
    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> DiffusionPipelineOutput:
        """
        Generate a video or image using the pipeline.
        
        Args:
            prompt: The prompt(s) to guide generation.
            negative_prompt: The negative prompt(s) to guide generation.
            height: The height of the generated video/image.
            width: The width of the generated video/image.
            num_frames: The number of frames to generate (for video).
            num_inference_steps: The number of inference steps.
            guidance_scale: The scale for classifier-free guidance.
            num_videos_per_prompt: The number of videos to generate per prompt.
            generator: The random number generator.
            latents: The initial latents.
            output_type: The output type.
            **kwargs: Additional arguments.
            
        Returns:
            The generated video or image.
        """
        # Execute each stage
        for stage in self._stages:
            batch = stage(batch, inference_args)

        # Return the output
        return DiffusionPipelineOutput(videos=batch.output)
