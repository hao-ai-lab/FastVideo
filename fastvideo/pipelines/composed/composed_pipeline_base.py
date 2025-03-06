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
        """Initialize the pipeline."""
        self._stages: List[PipelineStage] = []
        self._progress_bar_config = {}
    
    @property
    def device(self) -> torch.device:
        """Get the device for this pipeline."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def components(self) -> Dict[str, Any]:
        """Get all components used by this pipeline."""
        return {}
    
    def register_modules(self, **kwargs):
        """
        Register modules with the pipeline and its stages.
        
        Args:
            **kwargs: The modules to register.
        """
        for name, module in kwargs.items():
            setattr(self, name, module)
        
        # Register modules with stages that need them
        for stage in self._stages:
            stage_components = {}
            for name, module in kwargs.items():
                if hasattr(stage, f"needs_{name}") and getattr(stage, f"needs_{name}"):
                    stage_components[name] = module
            
            if stage_components:
                stage.register_modules(**stage_components)
    
    def add_stage(self, stage: PipelineStage):
        """
        Add a stage to the pipeline.
        
        Args:
            stage: The stage to add.
        """
        self._stages.append(stage)
    
    def progress_bar(self, iterable=None, total=None):
        """
        Create a progress bar for the pipeline.
        
        Args:
            iterable: The iterable to create a progress bar for.
            total: The total number of items.
            
        Returns:
            A progress bar.
        """
        if self._progress_bar_config.get("disable", False):
            return iterable
        
        return tqdm(
            iterable=iterable,
            total=total,
            **{k: v for k, v in self._progress_bar_config.items() if k != "disable"}
        )
    
    def set_progress_bar_config(self, **kwargs):
        """
        Set the configuration for the progress bar.
        
        Args:
            **kwargs: The configuration options.
        """
        self._progress_bar_config = kwargs
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs
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
        # Create inference args
        inference_args = InferenceArgs(
            model_path="",  # This will be overridden by the actual model path
            height=height or 720,
            width=width or 1280,
            num_frames=num_frames or (117 if self.is_video_pipeline else 1),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type,
            batch_size=1,
            num_videos=num_videos_per_prompt,
        )
        
        # Update inference args with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(inference_args, key):
                setattr(inference_args, key, value)
        
        # Create batch
        batch = ForwardBatch(
            data_type="video" if self.is_video_pipeline else "image",
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            latents=latents,
            num_videos_per_prompt=num_videos_per_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames or (117 if self.is_video_pipeline else 1),
            device=self.device,
        )
        
        # Execute each stage
        for stage in self._stages:
            batch = stage(batch, inference_args)
        
        # Return the output
        return DiffusionPipelineOutput(videos=batch.output) 