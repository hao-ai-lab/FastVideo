# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import asdict

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ForwardBatch)
from fastvideo.v1.configs import get_pipeline_config_cls_for_name

from fastvideo.v1.utils import align_to
from fastvideo.v1.worker.executor import Executor

logger = init_logger(__name__)


class VideoGenerator:
    """
    A unified class for generating videos using diffusion models.
    
    This class provides a simple interface for video generation with rich
    customization options, similar to popular frameworks like HF Diffusers.
    """

    def __init__(self, fastvideo_args: FastVideoArgs,
                 executor_class: type[Executor], log_stats: bool):
        """
        Initialize the video generator.
        
        Args:
            pipeline: The pipeline to use for inference
            fastvideo_args: The inference arguments
        """
        self.fastvideo_args = fastvideo_args
        self.executor = executor_class(fastvideo_args)

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

        config_cls = get_pipeline_config_cls_for_name(model_path)
        config = config_cls()

        if config is None:
            logger.warning(f"No config found for model {model_path}, using default config")
            config_args = {}
        else:
            config_args = asdict(config)

        # override config_args with kwargs
        config_args.update(kwargs)

        fastvideo_args = FastVideoArgs(
            model_path=model_path,
            device_str=device or "cuda" if torch.cuda.is_available() else "cpu",
            **config_args)


        if torch_dtype is not None:
            fastvideo_args.dtype = torch_dtype

        return cls.from_fastvideo_args(fastvideo_args)

    @classmethod
    def from_fastvideo_args(cls,
                            fastvideo_args: FastVideoArgs) -> "VideoGenerator":
        """
        Create a video generator with the specified arguments.
        
        Args:
            fastvideo_args: The inference arguments
                
        Returns:
            The created video generator
        """
        # Initialize distributed environment if needed
        # initialize_distributed_and_parallelism(fastvideo_args)

        executor_class = Executor.get_class(fastvideo_args)

        return cls(
            fastvideo_args=fastvideo_args,
            executor_class=executor_class,
            log_stats=False,  # TODO: implement
        )

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
            negative_prompt: The negative prompt to use (overrides the one in fastvideo_args)
            output_path: Path to save the video (overrides the one in fastvideo_args)
            save_video: Whether to save the video to disk
            return_frames: Whether to return the raw frames
            num_inference_steps: Number of denoising steps (overrides fastvideo_args)
            guidance_scale: Classifier-free guidance scale (overrides fastvideo_args)
            num_frames: Number of frames to generate (overrides fastvideo_args)
            height: Height of generated video (overrides fastvideo_args)
            width: Width of generated video (overrides fastvideo_args)
            fps: Frames per second for saved video (overrides fastvideo_args)
            seed: Random seed for generation (overrides fastvideo_args)
            callback: Callback function called after each step
            callback_steps: Number of steps between each callback
            
        Returns:
            Either the output dictionary or the list of frames depending on return_frames
        """
        # Create a copy of inference args to avoid modifying the original
        fastvideo_args = self.fastvideo_args.copy()

        # Override parameters if provided
        if negative_prompt is not None:
            fastvideo_args.neg_prompt = negative_prompt
        if num_inference_steps is not None:
            fastvideo_args.num_inference_steps = num_inference_steps
        if guidance_scale is not None:
            fastvideo_args.guidance_scale = guidance_scale
        if num_frames is not None:
            fastvideo_args.num_frames = num_frames
        if height is not None:
            fastvideo_args.height = height
        if width is not None:
            fastvideo_args.width = width
        if fps is not None:
            fastvideo_args.fps = fps
        if seed is not None:
            fastvideo_args.seed = seed

        # Store callback info
        fastvideo_args.callback = callback
        fastvideo_args.callback_steps = callback_steps

        # Validate inputs
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = prompt.strip()

        # Process negative prompt
        if fastvideo_args.neg_prompt is not None:
            fastvideo_args.neg_prompt = fastvideo_args.neg_prompt.strip()

        # Validate dimensions
        if (fastvideo_args.height <= 0 or fastvideo_args.width <= 0
                or fastvideo_args.num_frames <= 0):
            raise ValueError(
                f"Height, width, and num_frames must be positive integers, got "
                f"height={fastvideo_args.height}, width={fastvideo_args.width}, "
                f"num_frames={fastvideo_args.num_frames}")

        if (fastvideo_args.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames-1 must be a multiple of 4, got {fastvideo_args.num_frames}"
            )

        # Calculate sizes
        target_height = align_to(fastvideo_args.height, 16)
        target_width = align_to(fastvideo_args.width, 16)

        # Calculate latent sizes
        latents_size = [(fastvideo_args.num_frames - 1) // 4 + 1,
                        fastvideo_args.height // 8, fastvideo_args.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # Log parameters
        debug_str = f"""
                      height: {target_height}
                       width: {target_width}
                video_length: {fastvideo_args.num_frames}
                      prompt: {prompt}
                  neg_prompt: {fastvideo_args.neg_prompt}
                        seed: {fastvideo_args.seed}
                 infer_steps: {fastvideo_args.num_inference_steps}
       num_videos_per_prompt: {fastvideo_args.num_videos}
              guidance_scale: {fastvideo_args.guidance_scale}
                    n_tokens: {n_tokens}
                  flow_shift: {fastvideo_args.flow_shift}
     embedded_guidance_scale: {fastvideo_args.embedded_cfg_scale}"""
        logger.info(debug_str)

        # Prepare batch
        device = torch.device(fastvideo_args.device_str)
        batch = ForwardBatch(
            prompt=prompt,
            negative_prompt=fastvideo_args.neg_prompt,
            num_videos_per_prompt=fastvideo_args.num_videos,
            height=fastvideo_args.height,
            width=fastvideo_args.width,
            num_frames=fastvideo_args.num_frames,
            num_inference_steps=fastvideo_args.num_inference_steps,
            guidance_scale=fastvideo_args.guidance_scale,
            eta=0.0,
            n_tokens=n_tokens,
            data_type="video" if fastvideo_args.num_frames > 1 else "image",
            device=device,
            extra={},
        )

        # Run inference
        start_time = time.time()
        samples = self.pipeline.forward(
            batch=batch,
            fastvideo_args=fastvideo_args,
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
            save_path = output_path or fastvideo_args.output_path
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                video_path = os.path.join(save_path, f"{prompt[:100]}.mp4")
                imageio.mimsave(video_path, frames, fps=fastvideo_args.fps)
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
                (target_height, target_width, fastvideo_args.num_frames),
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
            output_path: Path to save the videos (overrides the one in fastvideo_args)
            **kwargs: Additional parameters to pass to generate_video
            
        Returns:
            List of output dictionaries from each generation
        """
        if output_path:
            self.fastvideo_args.output_path = output_path

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
        fastvideo_args = self.fastvideo_args.copy()
        fastvideo_args.init_image = image_tensor
        fastvideo_args.strength = strength

        # Generate video
        return self.generate_video(prompt=prompt or "",
                                   fastvideo_args=fastvideo_args,
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
        self.fastvideo_args.device_str = device_str
        self.fastvideo_args.device = torch.device(device_str)

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
