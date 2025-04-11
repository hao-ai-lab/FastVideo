# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.v1.distributed import (init_distributed_environment,
                                      initialize_model_parallel)
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    build_pipeline)
from fastvideo.v1.utils import align_to

logger = init_logger(__name__)


def initialize_distributed_and_parallelism(inference_args: InferenceArgs):
    """Initialize distributed environment and model parallelism."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    device_str = f"cuda:{local_rank}"
    inference_args.device_str = device_str
    inference_args.device = torch.device(device_str)
    initialize_model_parallel(
        sequence_model_parallel_size=inference_args.sp_size,
        tensor_model_parallel_size=inference_args.tp_size,
    )


class VideoGenerator:
    """
    A unified class for generating videos using diffusion models.
    
    This class provides a simple interface for video generation with rich
    customization options, similar to popular frameworks like HF Diffusers.
    """

    def __init__(self, pipeline: ComposedPipelineBase,
                 inference_args: InferenceArgs):
        """
        Initialize the video generator.
        
        Args:
            pipeline: The pipeline to use for inference
            inference_args: The inference arguments
        """
        self.pipeline = pipeline
        self.inference_args = inference_args
        self._is_initialized = True

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        device: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        **kwargs) -> "VideoGenerator":
        """
        Create a video generator from a pretrained model.
        
        Args:
            model_path: Path or identifier for the pretrained model
            device: Device to load the model on (e.g., "cuda", "cuda:0", "cpu")
            torch_dtype: Data type for model weights (e.g., torch.float16)
            **kwargs: Additional arguments to customize model loading
                
        Returns:
            The created video generator
        """
        inference_args = InferenceArgs(
            model_path=model_path,
            device_str=device or "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs)

        if torch_dtype is not None:
            inference_args.dtype = torch_dtype

        return cls.create_generator(inference_args)

    @classmethod
    def create_generator(cls,
                         inference_args: InferenceArgs) -> "VideoGenerator":
        """
        Create a video generator with the specified arguments.
        
        Args:
            inference_args: The inference arguments
                
        Returns:
            The created video generator
        """
        # Initialize distributed environment if needed
        initialize_distributed_and_parallelism(inference_args)

        logger.info("Building pipeline...")
        pipeline = build_pipeline(inference_args)
        logger.info("Pipeline Ready")

        return cls(pipeline, inference_args)

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        save_video: bool = True,
        return_frames: bool = False,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[Dict[str, Any], List[np.ndarray]]:
        """
        Generate a video based on the given prompt.
        
        Args:
            prompt: The prompt to use for generation
            negative_prompt: The negative prompt to use (overrides the one in inference_args)
            output_path: Path to save the video (overrides the one in inference_args)
            save_video: Whether to save the video to disk
            return_frames: Whether to return the raw frames
            num_inference_steps: Number of denoising steps (overrides inference_args)
            guidance_scale: Classifier-free guidance scale (overrides inference_args)
            num_frames: Number of frames to generate (overrides inference_args)
            height: Height of generated video (overrides inference_args)
            width: Width of generated video (overrides inference_args)
            fps: Frames per second for saved video (overrides inference_args)
            seed: Random seed for generation (overrides inference_args)
            callback: Callback function called after each step
            callback_steps: Number of steps between each callback
            
        Returns:
            Either the output dictionary or the list of frames depending on return_frames
        """
        # Create a copy of inference args to avoid modifying the original
        inference_args = self.inference_args.copy()

        # Override parameters if provided
        if negative_prompt is not None:
            inference_args.neg_prompt = negative_prompt
        if num_inference_steps is not None:
            inference_args.num_inference_steps = num_inference_steps
        if guidance_scale is not None:
            inference_args.guidance_scale = guidance_scale
        if num_frames is not None:
            inference_args.num_frames = num_frames
        if height is not None:
            inference_args.height = height
        if width is not None:
            inference_args.width = width
        if fps is not None:
            inference_args.fps = fps
        if seed is not None:
            inference_args.seed = seed

        # Store callback info
        inference_args.callback = callback
        inference_args.callback_steps = callback_steps

        # Validate inputs
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = prompt.strip()

        # Process negative prompt
        if inference_args.neg_prompt is not None:
            inference_args.neg_prompt = inference_args.neg_prompt.strip()

        # Validate dimensions
        if (inference_args.height <= 0 or inference_args.width <= 0
                or inference_args.num_frames <= 0):
            raise ValueError(
                f"Height, width, and num_frames must be positive integers, got "
                f"height={inference_args.height}, width={inference_args.width}, "
                f"num_frames={inference_args.num_frames}")

        if (inference_args.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames-1 must be a multiple of 4, got {inference_args.num_frames}"
            )

        # Calculate sizes
        target_height = align_to(inference_args.height, 16)
        target_width = align_to(inference_args.width, 16)

        # Calculate latent sizes
        latents_size = [(inference_args.num_frames - 1) // 4 + 1,
                        inference_args.height // 8, inference_args.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # Log parameters
        debug_str = f"""
                      height: {target_height}
                       width: {target_width}
                video_length: {inference_args.num_frames}
                      prompt: {prompt}
                  neg_prompt: {inference_args.neg_prompt}
                        seed: {inference_args.seed}
                 infer_steps: {inference_args.num_inference_steps}
       num_videos_per_prompt: {inference_args.num_videos}
              guidance_scale: {inference_args.guidance_scale}
                    n_tokens: {n_tokens}
                  flow_shift: {inference_args.flow_shift}
     embedded_guidance_scale: {inference_args.embedded_cfg_scale}"""
        logger.info(debug_str)

        # Prepare batch
        device = torch.device(inference_args.device_str)
        batch = ForwardBatch(
            prompt=prompt,
            negative_prompt=inference_args.neg_prompt,
            num_videos_per_prompt=inference_args.num_videos,
            height=inference_args.height,
            width=inference_args.width,
            num_frames=inference_args.num_frames,
            num_inference_steps=inference_args.num_inference_steps,
            guidance_scale=inference_args.guidance_scale,
            eta=0.0,
            n_tokens=n_tokens,
            data_type="video" if inference_args.num_frames > 1 else "image",
            device=device,
            extra={},
        )

        # Run inference
        start_time = time.time()
        samples = self.pipeline.forward(
            batch=batch,
            inference_args=inference_args,
        ).output

        gen_time = time.time() - start_time
        logger.info(f"Generated successfully in {gen_time:.2f} seconds")

        # Process outputs
        videos = rearrange(samples, "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Save video if requested
        if save_video:
            save_path = output_path or inference_args.output_path
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                video_path = os.path.join(save_path, f"{prompt[:100]}.mp4")
                imageio.mimsave(video_path, frames, fps=inference_args.fps)
                logger.info(f"Saved video to {video_path}")
            else:
                logger.warning("No output path provided, video not saved")

        if return_frames:
            return frames
        else:
            return {
                "samples": samples,
                "prompts": prompt,
                "size":
                (target_height, target_width, inference_args.num_frames),
                "generation_time": gen_time
            }

    def batch_generate(self,
                       prompts: List[str],
                       output_path: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """
        Generate videos for a batch of prompts.
        
        Args:
            prompts: List of prompts to generate videos for
            output_path: Path to save the videos (overrides the one in inference_args)
            **kwargs: Additional parameters to pass to generate_video
            
        Returns:
            List of output dictionaries from each generation
        """
        if output_path:
            self.inference_args.output_path = output_path

        results = []
        for prompt in prompts:
            result = self.generate_video(prompt=prompt, **kwargs)
            results.append(result)

        return results

    def image_to_video(self,
                       image: Union[str, torch.Tensor, np.ndarray],
                       prompt: Optional[str] = None,
                       strength: float = 0.8,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate a video from an initial image.
        
        Args:
            image: Input image (path, tensor, or numpy array)
            prompt: Text prompt to guide the generation
            strength: How much to transform the original image (0-1)
            **kwargs: Additional parameters to pass to generate_video
            
        Returns:
            Output dictionary from the generation
        """
        # Load image if path is provided
        if isinstance(image, str):
            # Implementation would depend on your image loading utilities
            # This is a placeholder for the concept
            image_tensor = self._load_image(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        else:
            image_tensor = image

        # Add image to inference args
        inference_args = self.inference_args.copy()
        inference_args.init_image = image_tensor
        inference_args.strength = strength

        # Generate video
        return self.generate_video(prompt=prompt or "",
                                   inference_args=inference_args,
                                   **kwargs)

    def to(self, device: Union[str, torch.device]) -> "VideoGenerator":
        """
        Move the model to the specified device.
        
        Args:
            device: The device to move the model to
            
        Returns:
            Self for chaining
        """
        device_str = str(device)
        self.inference_args.device_str = device_str
        self.inference_args.device = torch.device(device_str)

        # Move pipeline components to device
        self.pipeline.to(device)

        return self

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from a path and convert to tensor.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tensor representation of the image
        """
        # Placeholder implementation - would need to be implemented
        # based on your image loading utilities
        import PIL.Image
        from torchvision import transforms

        image = PIL.Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0)


def load_prompts_from_file(prompt_path: str) -> List[str]:
    """
    Load prompts from a file.
    
    Args:
        prompt_path: Path to the file containing prompts
        
    Returns:
        List of prompts
    """
    with open(prompt_path) as f:
        return [line.strip() for line in f.readlines()]
