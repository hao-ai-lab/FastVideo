# Copyright 2025 The FastVideo Authors. All rights reserved.

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.image_processor import VaeImageProcessor
from fastvideo.pipelines.pipeline_base import DiffusionPipelineBase
# from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.logger import init_logger


# TODO(will): temporary import
from diffusers.models import AutoencoderKL
# from fastvideo.models.encoders.encoder import TextEncoder
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from fastvideo.models.hunyuan.modules import HYVideoDiffusionTransformer
from fastvideo.inference_args import InferenceArgs

logger = init_logger(__name__)


class HunyuanVideoPipeline(DiffusionPipelineBase):
    """
    Pipeline for text-to-video generation using HunyuanVideo.
    This is a concrete implementation of the AbstractDiffusionPipeline.
    
    This implementation assumes the availability of:
    - text_encoder: LlamaModel for prompt encoding
    - tokenizer: LlamaTokenizerFast for prompt tokenization
    - transformer: HunyuanVideoTransformer3DModel for latent diffusion
    - vae: AutoencoderKLHunyuanVideo for encoding/decoding latents
    - scheduler: FlowMatchEulerDiscreteScheduler for diffusion timesteps
    - text_encoder_2: CLIPTextModel for additional prompt encoding
    - tokenizer_2: CLIPTokenizer for additional prompt tokenization
    """
    
    # Overriding this to match hunyuan_video pipeline settings
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    is_video_pipeline = True
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        inferece_args: InferenceArgs = None,
    ):
        super().__init__()

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.inference_args = inferece_args

        # TODO(will): add scheduler stuff
        
        # Register all the components using the provided method
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )

        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        # Set additional properties specific to Hunyuan Video
        # self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        # self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 8
        
        # Initialize video processor if needed
        # try:
        #     from diffusers.video_processor import VideoProcessor
        #     self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        # except ImportError:
        #     self.video_processor = None
    
    def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings from Llama model.
        
        Args:
            prompt: Text prompt(s) to encode
            prompt_template: Template to format prompts
            num_videos_per_prompt: Number of videos per prompt
            device: Device to place tensors on
            dtype: Data type for tensors
            max_sequence_length: Maximum sequence length for tokenization
            num_hidden_layers_to_skip: Number of hidden layers to skip in embedding extraction
            
        Returns:
            Tuple of prompt embeddings and attention mask
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Format prompts with template
        prompt = [prompt_template["template"].format(p) for p in prompt]

        # Get crop start position (if specified in template)
        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        # Tokenize inputs
        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        # Get embeddings from text encoder
        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # Repeat for videos per prompt if needed
        if num_videos_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_videos_per_prompt, dim=0)

        return prompt_embeds, prompt_attention_mask
    
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        """
        Get prompt embeddings from CLIP model.
        
        Args:
            prompt: Text prompt(s) to encode
            num_videos_per_prompt: Number of videos per prompt
            device: Device to place tensors on
            dtype: Data type for tensors
            max_sequence_length: Maximum sequence length for tokenization
            
        Returns:
            Tensor of CLIP prompt embeddings
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        # Tokenize with CLIP tokenizer
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Get CLIP text embeddings
        prompt_embeds_clip = self.text_encoder_2(
            text_input_ids,
            output_hidden_states=True,
        ).pooler_output

        # Convert to expected dtype
        prompt_embeds_clip = prompt_embeds_clip.to(dtype=dtype)
        
        # Repeat for videos per prompt if needed
        if num_videos_per_prompt > 1:
            prompt_embeds_clip = prompt_embeds_clip.repeat_interleave(num_videos_per_prompt, dim=0)
            
        return prompt_embeds_clip
    
    def encode_prompt(self, batch: ForwardBatch) -> ForwardBatch:
        """
        Encode the input prompt(s) to embeddings.
        
        Args:
            batch: ForwardBatch with prompt to encode
            
        Returns:
            ForwardBatch with encoded prompt embeddings
        """
        device = batch.params.device
        
        # Skip if embeddings are already provided
        if batch.text.prompt_embeds is not None:
            if batch.text.do_classifier_free_guidance and batch.text.negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds must be provided if prompt_embeds is provided and do_classifier_free_guidance is True")
            return batch
            
        # Process prompts
        prompt = batch.text.prompt
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        # Process negative prompts
        negative_prompt = batch.text.negative_prompt
        if batch.text.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
                
            if len(negative_prompt) != batch_size:
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompt)} must match prompt batch size {batch_size}"
                )
        
        # Encode with tokenizer and text encoder
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = self.text_encoder(
            text_input_ids,
        ).last_hidden_state
        
        # Get unconditional embeddings for classifier-free guidance
        if batch.text.do_classifier_free_guidance:
            uncond_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(device)
            
            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
            ).last_hidden_state
        else:
            negative_prompt_embeds = None
            
        # Duplicate for each video per prompt
        bs_embed = prompt_embeds.shape[0]
        prompt_embeds = prompt_embeds.repeat(1, batch.text.num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * batch.text.num_images_per_prompt, -1, prompt_embeds.shape[-1])
        
        if batch.text.do_classifier_free_guidance and negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, batch.text.num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                bs_embed * batch.text.num_images_per_prompt, -1, negative_prompt_embeds.shape[-1]
            )
            
        # Update batch with embeddings
        batch.text.prompt_embeds = prompt_embeds
        batch.text.negative_prompt_embeds = negative_prompt_embeds
        batch.text.is_prompt_processed = True
        
        return batch
    
    def check_inputs(self, batch: ForwardBatch, inference_args: InferenceArgs):
        height = batch.latent.height
        width = batch.latent.width
        video_length = batch.latent.num_frames
        vae_ver = inference_args.vae
        prompt = batch.text.prompt
        prompt_embeds = batch.text.prompt_embeds
        negative_prompt = batch.text.negative_prompt
        negative_prompt_embeds = batch.text.negative_prompt_embeds
        
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(f"`video_length` has to be 1 or a multiple of 4 but is {video_length}.")
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(f"`video_length` has to be 1 or a multiple of 8 but is {video_length}.")

        # if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
        #     raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
        #                      f" {type(callback_steps)}.")
        # if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs
        #                                                               for k in callback_on_step_end_tensor_inputs):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                             f" {negative_prompt_embeds}. Please make sure to only forward one of the two.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}.")
            
    
    def prepare_latents(self, batch: ForwardBatch) -> ForwardBatch:
        """
        Prepare initial latent variables for the diffusion process.
        
        Args:
            batch: ForwardBatch with latent configuration
            
        Returns:
            ForwardBatch with prepared latents
        """
        device = batch.params.device
        
        # Set num_channels_latents if not already set
        if batch.latent.num_channels_latents is None:
            batch.latent.num_channels_latents = self.transformer.config.in_channels
            
        # Set default num_frames if not specified
        if batch.latent.num_frames is None:
            batch.latent.num_frames = 16  # Default for Hunyuan
            
        # Calculate latent dimensions
        batch.latent.height_latents = batch.latent.height // 8
        batch.latent.width_latents = batch.latent.width // 8
        
        # Calculate total batch size
        if batch.text.batch_size is not None:
            batch_size = batch.text.batch_size * batch.text.num_images_per_prompt
        else:
            batch_size = batch.text.prompt_embeds.shape[0]
            
        # Use provided latents if available
        if batch.latent.latents is not None:
            return batch
            
        # Generate random latents
        latents_shape = (
            batch_size,
            batch.latent.num_channels_latents,
            batch.latent.num_frames // 4,  # Temporal downscaling for Hunyuan
            batch.latent.height_latents, 
            batch.latent.width_latents,
        )
        
        # Special handling for MPS
        dtype = batch.params.dtype or next(self.transformer.parameters()).dtype
        
        if device.type == "mps":
            # MPS: Use CPU for RNG, then move to device
            latents = torch.randn(
                latents_shape,
                generator=batch.params.generator,
                device="cpu",
                dtype=dtype,
            ).to(device)
        else:
            # Other devices: Generate directly on device
            latents = torch.randn(
                latents_shape,
                generator=batch.params.generator,
                device=device,
                dtype=dtype,
            )
            
        # Scale latents according to scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Update batch with latents
        batch.latent.latents = latents
        
        return batch
    
    def denoising_step(self, batch: ForwardBatch) -> ForwardBatch:
        """
        Perform a single denoising step.
        
        Args:
            batch: ForwardBatch with current latents and timestep
            
        Returns:
            ForwardBatch with updated latents
        """
        # Cast to correct dtype for transformer
        latents = batch.latent.latents
        timestep = batch.scheduler.timestep
        transformer_dtype = next(self.transformer.parameters()).dtype
        latent_model_input = latents.to(dtype=transformer_dtype)
        
        # Expand timestep to batch dimension
        timestep_tensor = torch.tensor([timestep], device=latents.device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        
        # UNet forward pass
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=batch.text.prompt_embeds,
            encoder_attention_mask=batch.text.prompt_attention_mask,
            pooled_projections=batch.text.pooled_prompt_embeds,
            guidance=batch.scheduler.guidance_scale * 1000.0,
            attention_kwargs=self.attention_kwargs,
            return_dict=False,
        )[0]
        
        # Classifier-free guidance
        if batch.text.do_classifier_free_guidance and batch.text.negative_prompt_embeds is not None:
            latent_uncond = latent_model_input
            noise_pred_uncond = self.transformer(
                hidden_states=latent_uncond,
                timestep=timestep_tensor,
                encoder_hidden_states=batch.text.negative_prompt_embeds,
                encoder_attention_mask=batch.text.negative_prompt_attention_mask,
                pooled_projections=batch.text.negative_pooled_prompt_embeds,
                guidance=batch.scheduler.guidance_scale * 1000.0,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            
            # Perform guidance
            noise_pred = noise_pred_uncond + batch.scheduler.guidance_scale * (noise_pred - noise_pred_uncond)
            
        # Update latents with scheduler step
        updated_latents = self.scheduler.step(
            noise_pred, timestep, latents, **batch.scheduler.extra_step_kwargs
        ).prev_sample
        
        # Update batch with new latents
        batch.latent.latents = updated_latents
        batch.latent.noise_pred = noise_pred
        
        return batch
    
    def decode_latents(self, batch: ForwardBatch) -> ForwardBatch:
        """
        Decode the latents to video frames.
        
        Args:
            batch: ForwardBatch with final latents
            
        Returns:
            ForwardBatch with decoded video frames
        """
        if batch.params.output_type == "latent":
            batch.output = batch.latent.latents
            return batch
            
        # Scale latents for VAE input
        vae_dtype = next(self.vae.parameters()).dtype
        latents = 1 / 0.18215 * batch.latent.latents.to(vae_dtype)
        
        # Decode with VAE
        video = self.vae.decode(latents).sample
        
        # Process video based on output type
        if batch.params.output_type == "pt":
            batch.output = video
        elif batch.params.output_type == "np":
            video = video.cpu().permute(0, 2, 3, 4, 1).float().numpy()
            batch.output = video
        elif batch.params.output_type == "pil":
            video = video.cpu().permute(0, 2, 3, 4, 1).float().numpy()
            # Rescale from [-1, 1] to [0, 1]
            video = (video + 1.0) / 2.0
            # Clip to [0, 1]
            video = np.clip(video, 0.0, 1.0)
            # Convert to [0, 255] uint8
            video = (video * 255).round().astype("uint8")
            
            # Convert to PIL if needed
            if batch.params.fps is not None:
                try:
                    from PIL import Image
                    frames = []
                    for i in range(video.shape[0]):
                        for j in range(video.shape[1]):
                            frames.append(Image.fromarray(video[i, j]))
                    batch.output = frames
                except ImportError:
                    batch.output = video
            else:
                batch.output = video
                
        else:
            raise ValueError(f"Unknown output_type: {batch.params.output_type}")
        
        return batch