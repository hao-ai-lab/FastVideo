# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Union

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.v1.configs import get_pipeline_config_cls_for_name
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ForwardBatch
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

        config = None
        config_cls = get_pipeline_config_cls_for_name(model_path)
        if config_cls is not None:
            config = config_cls()

        if config is None:
            logger.warning("No config found for model %s, using default config",
                           model_path)
            config_args = {}
        else:
            config_args = asdict(config)

        # override config_args with kwargs
        config_args.update(kwargs)

        fastvideo_args = FastVideoArgs(
            model_path=model_path,
            device_str=device or "cuda" if torch.cuda.is_available() else "cpu",
            **config_args)
        fastvideo_args.check_fastvideo_args()

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
        fastvideo_args = self.fastvideo_args

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
        output_batch = self.executor.execute_forward(batch, fastvideo_args)
        samples = output_batch.output

        gen_time = time.time() - start_time
        logger.info("Generated successfully in %.2f seconds", gen_time)

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
                logger.info("Saved video to %s", video_path)
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
