# SPDX-License-Identifier: Apache-2.0
"""
Image and video encoding stages for diffusion pipelines.

This module contains implementations of encoding stages for diffusion pipelines:
- ImageEncodingStage: Encodes images using image encoders (e.g., CLIP)
- RefImageEncodingStage: Encodes reference image for Wan2.1 control pipeline
- ImageVAEEncodingStage: Encodes images to latent space using VAE for I2V generation
- VideoVAEEncodingStage: Encodes videos to latent space using VAE for V2V and control tasks
"""

import PIL
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vision_utils import (get_default_height_width, normalize,
                                           numpy_to_pt, pil_to_numpy, resize,
                                           create_default_image,
                                           preprocess_reference_image_for_clip)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.
    
    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, image_encoder, image_processor) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary image encoder.
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into image encoder hidden states.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """
        self.image_encoder = self.image_encoder.to(get_local_torch_device())

        image = batch.pil_image

        image_inputs = self.image_processor(
            images=image, return_tensors="pt").to(get_local_torch_device())
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.image_encoder(**image_inputs)
            image_embeds = outputs.last_hidden_state

        batch.image_embeds.append(image_embeds)

        if fastvideo_args.image_encoder_cpu_offload:
            self.image_encoder.to('cpu')

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        result.add_check("image_embeds", batch.image_embeds,
                         V.list_of_tensors_dims(3))
        return result


class Hy15ImageEncodingStage(ImageEncodingStage):
    """
    Stage for encoding image prompts into embeddings for HunyuanVideo1.5 models.
    
    Uses SigLIP (or other vision encoder) to encode reference images for I2V tasks.
    Also encodes reference image with VAE for conditional latent.
    """
    
    # SigLIP normalization constants
    SIGLIP_MEAN = [0.5, 0.5, 0.5]
    SIGLIP_STD = [0.5, 0.5, 0.5]
    
    def __init__(self, image_encoder=None, image_processor=None, vae=None):
        super().__init__(image_encoder=image_encoder, image_processor=image_processor)
        self.vae = vae

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        return VerificationResult()

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """
        Encode the prompt into image encoder hidden states and VAE latents.
        
        For I2V: 
            - encodes the reference image using SigLIP → image_embeds
            - encodes the reference image using VAE → image_latent (expanded to full temporal dim)
        For T2V: creates zero embeddings
        
        The image_latent is expanded to match the full temporal dimension of the video latent,
        following the original HunyuanVideo-1.5 implementation where:
        - First frame contains the encoded reference image
        - All other frames are zeros
        - Mask channel is 1 for first frame, 0 for rest
        """
        device = get_local_torch_device()
        
        # Default vision embed dimensions for HunyuanVideo1.5/HyWorld
        num_vision_tokens = 729  # (384/14)^2 for SigLIP
        vision_dim = 1152  # SigLIP hidden size
        
        # Get temporal dimension from raw_latent_shape (set by LatentPreparationStage)
        raw_latent_shape = list(batch.raw_latent_shape)
        target_temporal = raw_latent_shape[2]  # T dimension
        latent_height = raw_latent_shape[3]
        latent_width = raw_latent_shape[4]
        
        if batch.pil_image is None:
            # T2V case: create zero embeddings for image_embeds
            batch.image_embeds = [
                torch.zeros(1, num_vision_tokens, vision_dim, device=device)
            ]
            # T2V: create zero latents for image_latent with full temporal dimension
            # Shape: [B, 33, T, H, W] where 33 = 32 latent channels + 1 mask
            batch.image_latent = torch.zeros(
                1, 33, target_temporal, latent_height, latent_width,
                device=device
            )
        else:
            image = batch.pil_image
            
            # 1. Encode with SigLIP for image_embeds
            if self.image_encoder is not None:
                self.image_encoder = self.image_encoder.to(device)
                
                # Preprocess image for SigLIP
                if hasattr(self.image_processor, 'preprocess'):
                    image_inputs = self.image_processor.preprocess(
                        images=image, return_tensors="pt"
                    ).to(device=device)
                    pixel_values = image_inputs['pixel_values']
                else:
                    from torchvision import transforms as T
                    import numpy as np
                    
                    transform = T.Compose([
                        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=self.SIGLIP_MEAN, std=self.SIGLIP_STD),
                    ])
                    
                    if isinstance(image, np.ndarray):
                        from PIL import Image
                        image = Image.fromarray(image)
                        
                    pixel_values = transform(image).unsqueeze(0).to(device)
                
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    outputs = self.image_encoder(pixel_values=pixel_values)
                    image_embeds = outputs.last_hidden_state
                
                batch.image_embeds = [image_embeds]
                
                if fastvideo_args.image_encoder_cpu_offload:
                    self.image_encoder.to('cpu')
            else:
                batch.image_embeds = [
                    torch.zeros(1, num_vision_tokens, vision_dim, device=device)
                ]
            
            # 2. Encode with VAE for image_latent (conditional latent for I2V)
            if self.vae is not None:
                from torchvision import transforms as T
                import numpy as np
                from PIL import Image as PILImage
                
                # Get target size from batch
                height = batch.height
                width = batch.width
                
                # Preprocess image for VAE
                if isinstance(image, np.ndarray):
                    image = PILImage.fromarray(image)
                
                # Resize and center crop to target resolution
                # Use the same preprocessing as HunyuanVideo-1.5
                original_width, original_height = image.size
                scale_factor = max(width / original_width, height / original_height)
                resize_width = int(round(original_width * scale_factor))
                resize_height = int(round(original_height * scale_factor))
                
                transform = T.Compose([
                    T.Resize((resize_height, resize_width), interpolation=T.InterpolationMode.LANCZOS),
                    T.CenterCrop((height, width)),
                    T.ToTensor(),  # [0, 1]
                    T.Normalize([0.5], [0.5]),  # to [-1, 1]
                ])
                
                ref_image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
                ref_image_tensor = ref_image_tensor.to(device)
                # Add temporal dimension: [1, 3, 1, H, W]
                ref_image_tensor = ref_image_tensor.unsqueeze(2)
                
                # Encode with VAE
                self.vae = self.vae.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    cond_latents = self.vae.encode(ref_image_tensor).mode()
                    if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'scaling_factor'):
                        cond_latents = cond_latents * self.vae.config.scaling_factor
                
                # cond_latents shape: [1, 32, 1, H//compression, W//compression]
                # Expand to full temporal dimension: [1, 32, T, H, W]
                # First frame contains the encoded image, rest are zeros
                expanded_latent = cond_latents.repeat(1, 1, target_temporal, 1, 1)
                expanded_latent[:, :, 1:, :, :] = 0.0  # Zero out all frames except first
                
                # Create mask: [1, 1, T, H, W]
                # First frame mask = 1 (conditional), rest = 0
                mask = torch.zeros(
                    1, 1, target_temporal, latent_height, latent_width,
                    device=device, dtype=expanded_latent.dtype
                )
                mask[:, :, 0, :, :] = 1.0  # First frame is conditional
                
                # Concatenate latent and mask: [1, 33, T, H, W]
                batch.image_latent = torch.cat([expanded_latent, mask], dim=1)
                
                if fastvideo_args.vae_cpu_offload:
                    self.vae.to('cpu')
            else:
                # No VAE available, create zero latents with full temporal dimension
                batch.image_latent = torch.zeros(
                    1, 33, target_temporal, latent_height, latent_width,
                    device=device
                )

        # Initialize video latent placeholder
        raw_latent_shape[1] = 1
        batch.video_latent = torch.zeros(tuple(raw_latent_shape), device=device)
        
        return batch


class MatrixGameImageEncodingStage(ImageEncodingStage):
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        assert batch.pil_image is not None
        self.image_encoder = self.image_encoder.to(get_local_torch_device())

        image = batch.pil_image

        if isinstance(image, torch.Tensor):
            if image.dim() == 5:
                image = image[:, :,
                              0]  # Extract first frame: [B, C, T, H, W] -> [B, C, H, W]
        else:
            from torchvision import transforms as T
            transform = T.Compose([
                T.ToTensor(),  # [0, 1]
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                       0.5]),  # -> [-1, 1]
            ])
            image = transform(image).unsqueeze(0)  # [1, C, H, W]

        device = get_local_torch_device()
        image = image.to(device)

        # F.interpolate with bicubic
        image = torch.nn.functional.interpolate(image,
                                                size=(224, 224),
                                                mode='bicubic',
                                                align_corners=False)

        #  [-1, 1] to [0, 1]
        image = image * 0.5 + 0.5

        # CLIP normalization
        mean = torch.tensor(self.CLIP_MEAN, device=device,
                            dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.CLIP_STD, device=device,
                           dtype=image.dtype).view(1, 3, 1, 1)
        image = (image - mean) / std

        with set_forward_context(current_timestep=0, attn_metadata=None):
            # WAN2_1ControlCLIPVisionConfig sets num_hidden_layers_override=31
            # so last_hidden_state is the second-to-last layer output
            outputs = self.image_encoder(pixel_values=image)
            image_embeds = outputs.last_hidden_state

        batch.image_embeds.append(image_embeds)
        if fastvideo_args.image_encoder_cpu_offload:
            self.image_encoder.to('cpu')
        return batch


class RefImageEncodingStage(ImageEncodingStage):
    """
    Stage for encoding reference image prompts into embeddings for Wan2.1 Control models.

    This stage extends ImageEncodingStage with specialized preprocessing for reference images.
    """

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into image encoder hidden states.

        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.

        Returns:
            The batch with encoded prompt embeddings.
        """
        self.image_encoder = self.image_encoder.to(get_local_torch_device())

        image = batch.pil_image
        if image is None:
            image = create_default_image()
        # Preprocess reference image for CLIP encoder
        image_tensor = preprocess_reference_image_for_clip(
            image, get_local_torch_device())

        image_inputs = self.image_processor(images=image_tensor,
                                            return_tensors="pt").to(
                                                get_local_torch_device())
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.image_encoder(**image_inputs)
            image_embeds = outputs.last_hidden_state
        batch.image_embeds.append(image_embeds)

        if batch.pil_image is None:
            batch.image_embeds = [
                torch.zeros_like(x) for x in batch.image_embeds
            ]

        return batch


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding image pixel representations into latent space.

    This stage handles the encoding of image pixel representations into the final
    input format (e.g., latents) for image-to-video generation.
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        self.vae: ParallelTiledVAE = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode pixel representations into latent space.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded outputs.
        """
        assert batch.pil_image is not None
        if fastvideo_args.mode == ExecutionMode.INFERENCE:
            assert batch.pil_image is not None and isinstance(
                batch.pil_image, PIL.Image.Image)
            assert batch.height is not None and isinstance(batch.height, int)
            assert batch.width is not None and isinstance(batch.width, int)
            assert batch.num_frames is not None and isinstance(
                batch.num_frames, int)
            height = batch.height
            width = batch.width
            num_frames = batch.num_frames
        elif fastvideo_args.mode == ExecutionMode.PREPROCESS:
            assert batch.pil_image is not None and isinstance(
                batch.pil_image, torch.Tensor)
            assert batch.height is not None and isinstance(batch.height, list)
            assert batch.width is not None and isinstance(batch.width, list)
            assert batch.num_frames is not None and isinstance(
                batch.num_frames, list)
            num_frames = batch.num_frames[0]
            height = batch.height[0]
            width = batch.width[0]

        self.vae = self.vae.to(get_local_torch_device())

        # Process single image for I2V
        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio
        image = batch.pil_image
        image = self.preprocess(
            image,
            vae_scale_factor=self.vae.spatial_compression_ratio,
            height=height,
            width=width).to(get_local_torch_device(), dtype=torch.float32)

        # (B, C, H, W) -> (B, C, 1, H, W)
        image = image.unsqueeze(2)

        video_condition = torch.cat([
            image,
            image.new_zeros(image.shape[0], image.shape[1], num_frames - 1,
                            image.shape[3], image.shape[4])
        ],
                                    dim=2)
        video_condition = video_condition.to(device=get_local_torch_device(),
                                             dtype=torch.float32)

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode Image
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            # if fastvideo_args.vae_sp:
            #     self.vae.enable_parallel()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        if fastvideo_args.mode == ExecutionMode.PREPROCESS:
            latent_condition = encoder_output.mean
        else:
            generator = batch.generator
            if generator is None:
                raise ValueError("Generator must be provided")
            latent_condition = self.retrieve_latents(encoder_output, generator)

        # Apply shifting if needed
        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        if fastvideo_args.mode == ExecutionMode.PREPROCESS:
            batch.image_latent = latent_condition
        else:
            mask_lat_size = torch.ones(1, 1, num_frames, latent_height,
                                       latent_width)
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask,
                dim=2,
                repeats=self.vae.temporal_compression_ratio)
            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            mask_lat_size = mask_lat_size.view(
                1, -1, self.vae.temporal_compression_ratio, latent_height,
                latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latent_condition.device)

            batch.image_latent = torch.concat([mask_lat_size, latent_condition],
                                              dim=1)

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch

    def retrieve_latents(self,
                         encoder_output: torch.Tensor,
                         generator: torch.Generator | None = None,
                         sample_mode: str = "sample"):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError(
                "Could not access latents of provided encoder_output")

    def preprocess(
            self,
            image: torch.Tensor | PIL.Image.Image,
            vae_scale_factor: int,
            height: int | None = None,
            width: int | None = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:

        if isinstance(image, PIL.Image.Image):
            height, width = get_default_height_width(image, vae_scale_factor,
                                                     height, width)
            image = resize(image, height, width, resize_mode=resize_mode)
            image = pil_to_numpy(image)  # to np
            image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        if fastvideo_args.mode == ExecutionMode.PREPROCESS:
            result.add_check("height", batch.height, V.list_not_empty)
            result.add_check("width", batch.width, V.list_not_empty)
            result.add_check("num_frames", batch.num_frames, V.list_not_empty)
        else:
            result.add_check("height", batch.height, V.positive_int)
            result.add_check("width", batch.width, V.positive_int)
            result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent,
                         [V.is_tensor, V.with_dims(5)])
        return result


class VideoVAEEncodingStage(ImageVAEEncodingStage):
    """
    Stage for encoding video pixel representations into latent space.

    This stage handles the encoding of video pixel representations for video-to-video generation and control.
    Inherits from ImageVAEEncodingStage to reuse common functionality.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode video pixel representations into latent space.

        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.

        Returns:
            The batch with encoded outputs.
        """
        assert batch.video_latent is not None, "Video latent input is required for VideoVAEEncodingStage"

        if fastvideo_args.mode == ExecutionMode.INFERENCE:
            assert batch.height is not None and isinstance(batch.height, int)
            assert batch.width is not None and isinstance(batch.width, int)
            assert batch.num_frames is not None and isinstance(
                batch.num_frames, int)
            height = batch.height
            width = batch.width
            num_frames = batch.num_frames
        elif fastvideo_args.mode == ExecutionMode.PREPROCESS:
            assert batch.height is not None and isinstance(batch.height, list)
            assert batch.width is not None and isinstance(batch.width, list)
            assert batch.num_frames is not None and isinstance(
                batch.num_frames, list)
            num_frames = batch.num_frames[0]
            height = batch.height[0]
            width = batch.width[0]

        self.vae = self.vae.to(get_local_torch_device())

        # Prepare video tensor from control video
        video_condition = self._prepare_control_video_tensor(
            batch.video_latent, num_frames, height,
            width).to(get_local_torch_device(), dtype=torch.float32)

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode control video
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")
        latent_condition = self.retrieve_latents(encoder_output, generator)

        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        batch.video_latent = latent_condition

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch

    def _prepare_control_video_tensor(self, control_video, num_frames: int,
                                      height: int, width: int) -> torch.Tensor:
        """
        Prepare video tensor from control video input.
        """
        if isinstance(control_video, list):
            processed_frames = []
            for i, frame in enumerate(control_video):
                if i >= num_frames:
                    break
                processed_frame = self.preprocess(
                    frame,
                    vae_scale_factor=self.vae.spatial_compression_ratio,
                    height=height,
                    width=width).to(get_local_torch_device(),
                                    dtype=torch.float32)
                processed_frames.append(processed_frame)

            if processed_frames:
                video_tensor = torch.cat(
                    [f.unsqueeze(2) for f in processed_frames], dim=2)
            else:
                video_tensor = torch.zeros(1,
                                           3,
                                           0,
                                           height,
                                           width,
                                           device=get_local_torch_device(),
                                           dtype=torch.float32)
        elif isinstance(control_video, torch.Tensor):
            # Handle tensor input [batch, channels, frames, height, width]
            video_tensor = control_video.to(get_local_torch_device(),
                                            dtype=torch.float32)

            if video_tensor.shape[2] > num_frames:
                video_tensor = video_tensor[:, :, :num_frames]
        else:
            raise ValueError(
                f"Unsupported control_video type: {type(control_video)}. "
                "Expected list of PIL Images or torch.Tensor.")

        # Pad with zeros if we have fewer frames than required
        current_frames = video_tensor.shape[2]
        if current_frames < num_frames:
            padding_frames = num_frames - current_frames
            zero_padding = torch.zeros(video_tensor.shape[0],
                                       video_tensor.shape[1],
                                       padding_frames,
                                       height,
                                       width,
                                       device=video_tensor.device,
                                       dtype=video_tensor.dtype)
            video_tensor = torch.cat([video_tensor, zero_padding], dim=2)

        return video_tensor

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify video encoding stage inputs."""
        result = VerificationResult()
        result.add_check("video_latent", batch.video_latent, V.not_none)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        if fastvideo_args.mode == ExecutionMode.PREPROCESS:
            result.add_check("height", batch.height, V.list_not_empty)
            result.add_check("width", batch.width, V.list_not_empty)
            result.add_check("num_frames", batch.num_frames, V.list_not_empty)
        else:
            result.add_check("height", batch.height, V.positive_int)
            result.add_check("width", batch.width, V.positive_int)
            result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify video encoding stage outputs."""
        result = VerificationResult()
        result.add_check("video_latent", batch.video_latent,
                         [V.is_tensor, V.with_dims(5)])
        return result


class MatrixGameImageVAEEncodingStage(ImageVAEEncodingStage):

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        assert batch.pil_image is not None

        if fastvideo_args.mode == ExecutionMode.INFERENCE:
            # Accept both PIL.Image and torch.Tensor
            # Causal pipeline (is_causal=True) converts PIL to tensor in InputValidationStage
            assert batch.pil_image is not None and isinstance(
                batch.pil_image, PIL.Image.Image | torch.Tensor)
            assert batch.height is not None and isinstance(batch.height, int)
            assert batch.width is not None and isinstance(batch.width, int)
            assert batch.num_frames is not None and isinstance(
                batch.num_frames, int)
            height = batch.height
            width = batch.width
            num_frames = batch.num_frames
        elif fastvideo_args.mode == ExecutionMode.PREPROCESS:
            assert batch.pil_image is not None and isinstance(
                batch.pil_image, torch.Tensor)
            assert batch.height is not None and isinstance(batch.height, list)
            assert batch.width is not None and isinstance(batch.width, list)
            assert batch.num_frames is not None and isinstance(
                batch.num_frames, list)
            num_frames = batch.num_frames[0]
            height = batch.height[0]
            width = batch.width[0]
        else:
            # Fallback for other modes
            height = batch.height if isinstance(batch.height,
                                                int) else batch.height[0]
            width = batch.width if isinstance(batch.width,
                                              int) else batch.width[0]
            num_frames = batch.num_frames if isinstance(
                batch.num_frames, int) else batch.num_frames[0]

        self.vae = self.vae.to(get_local_torch_device())

        # Process single image for I2V (latent dimensions computed but not used directly)

        image = batch.pil_image

        # Handle tensor input from causal pipeline
        if isinstance(image, torch.Tensor):
            # Causal pipeline provides tensor in [B, C, F, H, W] format
            if image.dim() == 5:
                # Already 5D, extract first frame for conditioning
                # Shape: [B, C, F, H, W] -> use first frame [B, C, 1, H, W]
                first_frame = image[:, :, :1]  # Keep dim, [B, C, 1, H, W]
                # Create video condition with first frame + zeros
                video_condition = torch.cat([
                    first_frame,
                    first_frame.new_zeros(
                        first_frame.shape[0], first_frame.shape[1], num_frames -
                        1, first_frame.shape[3], first_frame.shape[4])
                ],
                                            dim=2)
            elif image.dim() == 4:
                # [B, C, H, W] -> need to add frame dim
                image = image.unsqueeze(2)  # [B, C, 1, H, W]
                video_condition = torch.cat([
                    image,
                    image.new_zeros(image.shape[0], image.shape[1], num_frames -
                                    1, image.shape[3], image.shape[4])
                ],
                                            dim=2)
            else:
                raise ValueError(f"Unexpected tensor dimensions: {image.dim()}")
            video_condition = video_condition.to(get_local_torch_device(),
                                                 dtype=torch.float32)
        else:
            # PIL Image input - use preprocess
            image = self.preprocess(
                image,
                vae_scale_factor=self.vae.spatial_compression_ratio,
                height=height,
                width=width).to(get_local_torch_device(), dtype=torch.float32)

            # (B, C, H, W) -> (B, C, 1, H, W)
            image = image.unsqueeze(2)

            # Create video tensor with first frame as image, rest as zeros
            video_condition = torch.cat([
                image,
                image.new_zeros(image.shape[0], image.shape[1], num_frames - 1,
                                image.shape[3], image.shape[4])
            ],
                                        dim=2)
            video_condition = video_condition.to(
                device=get_local_torch_device(), dtype=torch.float32)

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode Image
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        # MatrixGame uses deterministic VAE encode for the first-frame conditioning.
        # Sampling would inject random noise into the cond_concat tensor and destroy the action guidance.
        img_cond = encoder_output.mode()

        # manually using latents_mean and latents_std from config...
        if (hasattr(self.vae.config, 'latents_mean')
                and hasattr(self.vae.config, 'latents_std')):
            # Convert config values to tensors
            latents_mean = torch.tensor(self.vae.config.latents_mean,
                                        device=img_cond.device,
                                        dtype=img_cond.dtype).view(
                                            1, -1, 1, 1, 1)

            latents_std = torch.tensor(self.vae.config.latents_std,
                                       device=img_cond.device,
                                       dtype=img_cond.dtype).view(
                                           1, -1, 1, 1, 1)

            # Apply normalization: (latent - mean) * (1/std)
            img_cond = (img_cond - latents_mean) / latents_std
        elif (hasattr(self.vae, "shift_factor")
              and self.vae.shift_factor is not None):
            # Fallback to shift_factor/scaling_factor if available
            if isinstance(self.vae.shift_factor, torch.Tensor):
                img_cond -= self.vae.shift_factor.to(img_cond.device,
                                                     img_cond.dtype)
            else:
                img_cond -= self.vae.shift_factor

            if hasattr(self.vae, 'scaling_factor'):
                if isinstance(self.vae.scaling_factor, torch.Tensor):
                    img_cond = img_cond * self.vae.scaling_factor.to(
                        img_cond.device, img_cond.dtype)
                else:
                    img_cond = img_cond * self.vae.scaling_factor

        # Create mask_cond: ones for first frame, zeros for rest
        # Shape: (B, 16, latent_frames, latent_height, latent_width)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0  # Set all frames except first to 0

        # Create cond_concat: first 4 channels of mask + all 16 channels of img_cond
        # Shape: (B, 20, latent_frames, latent_height, latent_width)
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)

        # Store cond_concat in batch.image_latent
        # This will be concatenated with noise latents in DenoisingStage
        batch.image_latent = cond_concat

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch
