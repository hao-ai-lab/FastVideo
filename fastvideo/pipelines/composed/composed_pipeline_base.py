"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from tqdm.auto import tqdm

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.pipelines.stages import PipelineStage
from fastvideo.logger import init_logger

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
    
    def __init__(self):
        """
        Initialize the pipeline.
        The pipeline should be completely stateless and not hold any batch
        state.
        """
        self._stages: List[PipelineStage] = []
        self._modules: Dict[str, Any] = {}
        self._stage_name_mapping: Dict[str, PipelineStage] = {}


    @property
    def device(self) -> torch.device:
        """Get the device for this pipeline."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def modules(self) -> Dict[str, Any]:
        """Get all modules used by this pipeline."""
        return self._modules

    @abstractmethod
    def setup_pipeline(self, inference_args: InferenceArgs):
        """
        Setup the pipeline.
        """
        ...
    
    @abstractmethod
    def initialize_encoders(self, modules: Dict[str, Any], inference_args: InferenceArgs):
        """
        Initialize the encoders. Will remove the encoders/tokenizers modules from the
        modules. Will add the TextEncoder or ImageEncoder to the modules.
        """
        ...
    
    def register_modules(self, modules: Dict[str, Any]):
        """
        Register modules with the pipeline and its stages.
        
        We will use the _module_name_mapping to map the module names used
        in the Diffusers config to the internal names (how it can be accessed
        in the pipeline).
        
        Args:
            modules: The modules to register.
        """
        self._modules.update(modules)
        # Register modules with self
        for name, module in modules.items():
            setattr(self, name, module)
            # self._modules[name] = module
        
        # Register modules with stages that need them
        for stage in self._stages:
            stage.register_modules(modules)
            # TODO(will): perhaps we should not register all modules with the
            # stage. See below.
            
            # stage_modules = {}
            # for name, module in mapped_modules.items():
            #     if hasattr(stage, f"needs_{name}") and getattr(stage, f"needs_{name}"):
            #         stage_modules[name] = module
            
            # if stage_modules:
            #     stage.register_modules(**stage_modules)
    
    def add_stage(self, name: str, stage: PipelineStage):
        assert self._modules is not None, "No modules are registered"
        stage.register_modules(self._modules)
        self._stages.append(stage)
        self._stage_name_mapping[name] = stage
        setattr(self, name, stage)



    
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