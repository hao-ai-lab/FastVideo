import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import importlib

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

@dataclass
class AbstractPipelineOutput:
    """
    Base class for pipeline outputs.
    
    Pipeline-specific outputs should inherit from this class and add 
    their own specific attributes.
    """
    pass

class AbstractDiffusionPipeline(ABC):
    """
    Abstract base class for diffusion pipelines that provides a common structure.
    This class assumes that all abstract components are implemented by child classes.
    
    The pipeline implements a general diffusion process with the following steps:
    1. Input validation and preparation
    2. Encoding the input prompt(s)
    3. Preparing timesteps for the diffusion process
    4. Preparing initial latent variables
    5. Running the denoising loop
    6. Decoding the results
    
    Child classes should implement the abstract methods to provide the specific
    functionality needed for their particular use case.
    """
    
    # Define configuration properties
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    model_cpu_offload_seq = None  # Should be defined by child classes if needed
    is_video_pipeline: bool = False  # To be overridden by video pipelines
    
    def __init__(self):
        # Properties to track during inference
        self._guidance_scale = None
        self._attention_kwargs = None
        self._current_timestep = None
        self._num_timesteps = None
        self._interrupt = False
        
        # Optional utility for video processing - set by child classes if needed
        self.video_processor = None
    
    @property
    def device(self) -> torch.device:
        """Returns the device on which the pipeline is running."""
        for module in self.components.values():
            if isinstance(module, torch.nn.Module):
                return next(module.parameters()).device
        return torch.device("cpu")
    
    @property
    def components(self) -> Dict[str, Any]:
        """Returns all the components of the pipeline."""
        components = {}
        for name, value in self.__dict__.items():
            if not name.startswith("_"):
                components[name] = value
        return components
    
    @property
    def guidance_scale(self):
        """The guidance scale for classifier-free guidance."""
        return self._guidance_scale
    
    @property
    def do_classifier_free_guidance(self):
        """Whether to use classifier-free guidance."""
        return self._guidance_scale > 1.0
    
    @property
    def num_timesteps(self):
        """The number of timesteps in the current generation process."""
        return self._num_timesteps
    
    @property
    def attention_kwargs(self):
        """Additional keyword arguments for attention modules."""
        return self._attention_kwargs
    
    @property
    def current_timestep(self):
        """The current timestep in the generation process."""
        return self._current_timestep
    
    @property
    def interrupt(self):
        """Whether the generation has been interrupted."""
        return self._interrupt
    
    @property
    def _execution_device(self):
        """The device used for execution."""
        return self.device
    
    def register_modules(self, **kwargs):
        """
        Register components of the pipeline.
        
        Args:
            **kwargs: Components to register, as name=component pairs.
        """
        for name, module in kwargs.items():
            setattr(self, name, module)
    
    def maybe_free_model_hooks(self):
        """
        Free memory used by model hooks if applicable.
        This is a placeholder that can be implemented by child classes.
        """
        pass
    
    @abstractmethod
    def check_inputs(self, **kwargs):
        """
        Validate inputs to ensure they meet the requirements for the pipeline.
        Raise appropriate errors for invalid inputs.
        
        Args:
            **kwargs: Input parameters to validate.
        """
        pass
    
    @abstractmethod
    def encode_prompt(self, **kwargs):
        """
        Encode the input prompt into embeddings that can be used by the model.
        
        Args:
            **kwargs: Input parameters for prompt encoding.
            
        Returns:
            Tuple containing prompt embeddings and optional negative prompt embeddings.
        """
        pass
    
    @abstractmethod
    def prepare_latents(self, **kwargs):
        """
        Prepare the initial latent variables for the diffusion process.
        
        Args:
            **kwargs: Input parameters for latent preparation.
            
        Returns:
            Initial latent variables.
        """
        pass
    
    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Prepares the timestep sequence for the diffusion process based on the scheduler.
        
        Args:
            scheduler: The scheduler to get timesteps from.
            num_inference_steps: The number of diffusion steps.
            device: The device to move timesteps to.
            timesteps: Custom timesteps to override the scheduler's default.
            sigmas: Custom sigmas to override the scheduler's default.
            **kwargs: Additional arguments for the scheduler.
            
        Returns:
            Tuple of timesteps tensor and number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
            
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The scheduler {scheduler.__class__} does not support custom timestep schedules."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The scheduler {scheduler.__class__} does not support custom sigma schedules."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            
        return timesteps, num_inference_steps
    
    def progress_bar(self, iterable=None, total=None):
        """
        Provides a progress bar for the denoising process.
        
        Args:
            iterable: The iterable to iterate over.
            total: The total number of items.
            
        Returns:
            A tqdm progress bar.
        """
        return tqdm(iterable=iterable, total=total)
    
    def set_progress_bar_config(self, **kwargs):
        """
        Configure the progress bar with additional parameters.
        
        Args:
            **kwargs: Arguments to pass to tqdm.
        """
        tqdm.set_options(**kwargs)
    
    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
    ):
        # TODO(will): see if callbacks are needed
        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(batch, inference_args=self.inference_args)

        # Set pipeline properties
        self._guidance_scale = self.inference_args.guidance_scale
        self._guidance_rescale = getattr(self.inference_args, "guidance_rescale", 0.0)
        self._clip_skip = getattr(self.inference_args, "clip_skip", None)
        self._attention_kwargs = getattr(self.inference_args, "attention_kwargs", None)
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        
        # Get batch size
        if batch.text_data.prompt is not None and isinstance(batch.text_data.prompt, str):
            batch_size = 1
        elif batch.text_data.prompt is not None and isinstance(batch.text_data.prompt, list):
            batch_size = len(batch.text_data.prompt)
        else:
            batch_size = batch.text_data.prompt_embeds.shape[0]
        
        # Encode input prompt
        lora_scale = (self._attention_kwargs.get("scale", None)
                     if self._attention_kwargs is not None else None)
        
        # Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        ) = self.encode_prompt(
            prompt=batch.text_data.prompt,
            device=device,
            num_videos_per_prompt=self.inference_args.num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=batch.text_data.negative_prompt,
            prompt_embeds=batch.text_data.prompt_embeds,
            attention_mask=batch.text_data.attention_mask,
            negative_prompt_embeds=batch.text_data.negative_prompt_embeds,
            negative_attention_mask=batch.text_data.negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self._clip_skip,
            data_type=getattr(self.inference_args, "data_type", "image"),
        )
        
        # Handle text_encoder_2 if available
        if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                attention_mask_2,
                negative_attention_mask_2,
            ) = self.encode_prompt(
                prompt=batch.text_data.prompt,
                device=device,
                num_videos_per_prompt=self.inference_args.num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=batch.text_data.negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self._clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=getattr(self.inference_args, "data_type", "image"),
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            attention_mask_2 = None
            negative_attention_mask_2 = None

        # For classifier free guidance, concatenate unconditional and text embeddings
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if attention_mask is not None:
                attention_mask = torch.cat([negative_attention_mask, attention_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if attention_mask_2 is not None:
                attention_mask_2 = torch.cat([negative_attention_mask_2, attention_mask_2])

        # Prepare timesteps
        scheduler = getattr(self, "scheduler", batch.scheduler_data.scheduler)
        if scheduler is None:
            raise ValueError("No scheduler found. Make sure to pass a scheduler or initialize with one.")
        
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            scheduler.set_timesteps,
            {"n_tokens": getattr(self.inference_args, "n_tokens", None)}
        )
        
        timesteps, num_inference_steps = self.retrieve_timesteps(
            scheduler=scheduler,
            num_inference_steps=self.inference_args.num_inference_steps,
            device=device,
            timesteps=batch.scheduler_data.timesteps,
            sigmas=batch.scheduler_data.sigmas,
            **extra_set_timesteps_kwargs,
        )
        
        # Adjust video length if needed based on VAE version
        video_length = batch.latent_data.num_frames
        vae_ver = getattr(self.inference_args, "vae_ver", None)
        if vae_ver:
            if "884" in vae_ver:
                video_length = (video_length - 1) // 4 + 1
            elif "888" in vae_ver:
                video_length = (video_length - 1) // 8 + 1
        
        # Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * self.inference_args.num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=batch.latent_data.height,
            width=batch.latent_data.width,
            num_frames=video_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=batch.latent_data.generator,
            latents=batch.latent_data.latents,
        )
        
        # Handle sequence parallelism if enabled
        if hasattr(self, "nccl_info") and get_sequence_parallel_state():
            world_size, rank = self.nccl_info.sp_size, self.nccl_info.rank_within_group
            latents = rearrange(latents, "b t (n s) h w -> b t n s h w", n=world_size).contiguous()
            latents = latents[:, :, rank, :, :, :]
        
        # Prepare extra step kwargs for the scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            scheduler.step,
            {
                "generator": batch.latent_data.generator,
                "eta": getattr(self.inference_args, "eta", 0.0)
            },
        )
        
        # Prepare for autocast if needed
        target_dtype = PRECISION_TO_TYPE[self.inference_args.precision] if hasattr(self.inference_args, "precision") else None
        autocast_enabled = (target_dtype is not None and target_dtype != torch.float32) and not getattr(self.inference_args, "disable_autocast", False)
        
        vae_dtype = PRECISION_TO_TYPE[self.inference_args.vae_precision] if hasattr(self.inference_args, "vae_precision") else None
        vae_autocast_enabled = (vae_dtype is not None and vae_dtype != torch.float32) and not getattr(self.inference_args, "disable_autocast", False)
        
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        self._num_timesteps = len(timesteps)
        
        # Process mask strategy if provided
        mask_strategy = getattr(self.inference_args, "mask_strategy", None)
        if mask_strategy:
            mask_strategy = self.process_mask_strategy(mask_strategy)
        
        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                # Prepare timestep and guidance tensors
                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = None
                if hasattr(self.inference_args, "embedded_guidance_scale") and self.inference_args.embedded_guidance_scale is not None:
                    guidance_expand = torch.tensor(
                        [self.inference_args.embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype) * 1000.0
                    
                # Predict the noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    # Handle multiple text encoders if present
                    if prompt_embeds_2 is not None:
                        if prompt_embeds_2.shape[-1] != prompt_embeds.shape[-1]:
                            prompt_embeds_2 = F.pad(
                                prompt_embeds_2,
                                (0, prompt_embeds.shape[2] - prompt_embeds_2.shape[1]),
                                value=0,
                            ).unsqueeze(1)
                        encoder_hidden_states = torch.cat([prompt_embeds_2, prompt_embeds], dim=1)
                    else:
                        encoder_hidden_states = prompt_embeds
                        
                    # Get current mask strategy for this step if available
                    current_mask = mask_strategy[i] if mask_strategy else None
                    
                    # Forward pass through transformer
                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states,
                        t_expand,
                        attention_mask,
                        mask_strategy=current_mask,
                        guidance=guidance_expand,
                        return_dict=False,
                    )[0]
                    
                # Apply guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Apply guidance rescale if needed
                    if self._guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self._guidance_rescale,
                        )
                        
                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
        
        # Gather results if using sequence parallelism
        if hasattr(self, "nccl_info") and get_sequence_parallel_state():
            latents = all_gather(latents, dim=2)
        
        # Decode latents if needed
        output_type = getattr(self.inference_args, "output_type", "pil")
        if output_type != "latent":
            # Process latents for VAE
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) != 5:
                raise ValueError(f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")
            
            # Scale latents for VAE decoding
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents / self.vae.config.scaling_factor + self.vae.config.shift_factor)
            else:
                latents = latents / self.vae.config.scaling_factor
            
            # Decode with VAE
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if getattr(self.inference_args, "enable_tiling", False):
                    self.vae.enable_tiling()
                if getattr(self.inference_args, "enable_vae_sp", False):
                    self.vae.enable_parallel()
                image = self.vae.decode(latents, return_dict=False, generator=batch.latent_data.generator)[0]
            
            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)
        else:
            image = latents
        
        # Normalize image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()
        
        # Offload models
        self.maybe_free_model_hooks()
        
        # Return results
        if not self.inference_args.return_dict:
            return image
        
        return self.create_output_object(image)
    
    @abstractmethod
    def denoising_step(self, latents, timestep, index, prompt_embeds, negative_prompt_embeds, guidance_scale, **kwargs):
        """
        Perform a single denoising step.
        
        Args:
            latents: The current latent variables.
            timestep: The current timestep.
            index: The current step index.
            prompt_embeds: The encoded prompts.
            negative_prompt_embeds: The encoded negative prompts.
            guidance_scale: The guidance scale for classifier-free guidance.
            **kwargs: Additional arguments.
            
        Returns:
            Updated latents for the next step.
        """
        pass
    
    @abstractmethod
    def decode_latents(self, latents, output_type="pil", **kwargs):
        """
        Decode the generated latents into the desired output format.
        
        Args:
            latents: The final latent representation.
            output_type: The desired output type.
            **kwargs: Additional arguments.
            
        Returns:
            The decoded output in the specified format.
        """
        pass
    
    def create_output_object(self, output):
        """
        Create the output object for the pipeline.
        This method should be overridden by child classes if they need custom output objects.
        
        Args:
            output: The raw output from the pipeline.
            
        Returns:
            Dictionary with the output or a custom output object.
        """
        # Default implementation just returns a dictionary
        return {"frames" if self.is_video_pipeline else "images": output}

    def process_mask_strategy(self, mask_strategy, t_max=50, l_max=60, h_max=24):
        """Convert mask strategy dict to 3D list for easier access during inference"""
        result = [[[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)]
        if mask_strategy is None:
            return result
        for key, value in mask_strategy.items():
            t, l, h = map(int, key.split('_'))
            result[t][l][h] = value
        return result 