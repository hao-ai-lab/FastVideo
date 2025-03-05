"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch


@dataclass
class TextData:
    """Text inputs and encoded embeddings for the pipeline."""
    # Text inputs
    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    
    # Primary encoder embeddings
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    negative_attention_mask: Optional[torch.Tensor] = None
    
    # Secondary encoder embeddings (for dual-encoder models)
    prompt_embeds_2: Optional[torch.Tensor] = None
    negative_prompt_embeds_2: Optional[torch.Tensor] = None
    attention_mask_2: Optional[torch.Tensor] = None
    negative_attention_mask_2: Optional[torch.Tensor] = None
    
    # Additional text-related parameters
    max_sequence_length: Optional[int] = None
    prompt_template: Optional[Dict[str, Any]] = None
    do_classifier_free_guidance: bool = True
    data_type: Optional[str] = None
    
    # Batch info
    batch_size: Optional[int] = None
    num_videos_per_prompt: int = 1
    
    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False


@dataclass
class LatentData:
    """Latent representations and related parameters."""
    # Latent tensors
    latents: Optional[torch.Tensor] = None
    noise_pred: Optional[torch.Tensor] = None
    
    # Latent dimensions
    num_channels_latents: Optional[int] = None
    height_latents: Optional[int] = None
    width_latents: Optional[int] = None
    num_frames: Optional[int] = 1  # Default for image models
    
    # Original dimensions (before VAE scaling)
    height: Optional[int] = None
    width: Optional[int] = None


@dataclass
class SchedulerData:
    """Scheduler state and parameters."""
    # Timesteps
    timesteps: Optional[torch.Tensor] = None
    timestep: Optional[Union[torch.Tensor, float, int]] = None
    step_index: Optional[int] = None
    
    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    
    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: Dict[str, Any] = field(default_factory=dict)


# @dataclass
# class GenerationParameters:
#     """Parameters controlling the generation process."""
#     # Device and dtype
#     device: torch.device = field(default_factory=lambda: torch.device("cpu"))
#     dtype: Optional[torch.dtype] = None
    
#     # Random generation
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    
#     # Output configuration
#     output_type: str = "pil"
#     return_dict: bool = True
    
#     # Video-specific parameters
#     fps: Optional[int] = None
    
#     # Callback configuration
#     callback: Optional[Callable] = None
#     callback_steps: int = 1
#     callback_on_step_end: Optional[Callable] = None
#     callback_on_step_end_tensor_inputs: Optional[List[str]] = None


@dataclass
class ForwardBatch:
    """
    Complete state passed through the pipeline execution.
    
    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """
    # Core components of the pipeline state
    text: TextData = field(default_factory=TextData)
    latent: LatentData = field(default_factory=LatentData)
    scheduler: SchedulerData = field(default_factory=SchedulerData)
    # params: GenerationParameters = field(default_factory=GenerationParameters)
    
    # Component modules (populated by the pipeline)
    modules: Dict[str, Any] = field(default_factory=dict)
    
    # Final output (after pipeline completion)
    output: Any = None
    
    # Extra parameters that might be needed by specific pipeline implementations
    extra: Dict[str, Any] = field(default_factory=dict)

    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    
    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        # Set batch size based on prompt if not explicitly set
        if self.text.batch_size is None and self.text.prompt is not None:
            if isinstance(self.text.prompt, str):
                self.text.batch_size = 1
            else:
                self.text.batch_size = len(self.text.prompt)
        
        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.scheduler.guidance_scale <= 1.0 or self.text.negative_prompt is None:
            self.text.do_classifier_free_guidance = False
            
    def update(self, **kwargs):
        """
        Update specific fields in the batch.
        
        Args:
            **kwargs: Key-value pairs where keys follow dot notation to access nested fields.
                     For example: "text.prompt" to update the prompt field in TextData.
        
        Returns:
            The updated ForwardBatch instance.
        """
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested fields
                parent, child = key.split(".", 1)
                if hasattr(self, parent) and hasattr(getattr(self, parent), child):
                    setattr(getattr(self, parent), child, value)
            else:
                # Handle top-level fields
                if hasattr(self, key):
                    setattr(self, key, value)
        
        return self
    
    def to_device(self, device: torch.device):
        """
        Move tensors in the batch to the specified device.
        
        Args:
            device: The device to move tensors to.
            
        Returns:
            The updated ForwardBatch instance.
        """
        # Update device parameter
        self.params.device = device
        
        # Move text embeddings
        if self.text.prompt_embeds is not None:
            self.text.prompt_embeds = self.text.prompt_embeds.to(device)
        if self.text.negative_prompt_embeds is not None:
            self.text.negative_prompt_embeds = self.text.negative_prompt_embeds.to(device)
        
        # Move latent tensors
        if self.latent.latents is not None:
            self.latent.latents = self.latent.latents.to(device)
        if self.latent.noise_pred is not None:
            self.latent.noise_pred = self.latent.noise_pred.to(device)
        
        # Move scheduler timesteps
        if self.scheduler.timesteps is not None:
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        if isinstance(self.scheduler.timestep, torch.Tensor):
            self.scheduler.timestep = self.scheduler.timestep.to(device)
        
        return self
    
    @classmethod
    def from_pipeline_inputs(cls, prompt, negative_prompt=None, height=None, width=None, 
                            num_inference_steps=50, guidance_scale=7.5, 
                            num_images_per_prompt=1, eta=0.0, generator=None,
                            latents=None, prompt_embeds=None, negative_prompt_embeds=None,
                            output_type="pil", return_dict=True, callback=None,
                            callback_steps=1, callback_on_step_end=None,
                            callback_on_step_end_tensor_inputs=None, device=None,
                            dtype=None, **kwargs):
        """
        Create a ForwardBatch instance from the standard pipeline __call__ parameters.
        
        Args:
            prompt: The prompt to guide generation
            negative_prompt: The prompt to avoid in generation
            height: Height of generated images/videos
            width: Width of generated images/videos
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            num_images_per_prompt: Number of samples per prompt
            eta: Weight for noise in DDIM sampling
            generator: Random number generator
            latents: Pre-generated latent inputs
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            output_type: Output format ("pil", "np", "pt", "latent")
            return_dict: Whether to return dict outputs
            callback: Callback function for each step
            callback_steps: Frequency of callback calls
            callback_on_step_end: Callback at the end of each step
            callback_on_step_end_tensor_inputs: Tensor inputs for callback
            device: Device to place tensors on
            dtype: Data type for tensors
            **kwargs: Additional arguments
            
        Returns:
            A ForwardBatch instance with populated fields
        """
        # Create component dataclasses
        text_data = TextData(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            do_classifier_free_guidance=guidance_scale > 1.0 and negative_prompt is not None,
            num_images_per_prompt=num_images_per_prompt,
        )
        
        latent_data = LatentData(
            latents=latents,
            height=height,
            width=width,
            num_frames=kwargs.get("num_frames", 1),  # Default to 1 for image models
        )
        
        scheduler_data = SchedulerData(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
        )
        
        params = GenerationParameters(
            device=device if device is not None else torch.device("cpu"),
            dtype=dtype,
            generator=generator,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            fps=kwargs.get("fps", None),
        )
        
        # Create the ForwardBatch
        batch = cls(
            text=text_data,
            latent=latent_data,
            scheduler=scheduler_data,
            params=params,
            extra=kwargs,  # Store any additional kwargs
        )
        
        return batch 