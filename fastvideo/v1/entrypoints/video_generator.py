# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import gc
import os
import time
from typing import Any, Dict, List, Optional, Union

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.v1.configs.pipelines import (PipelineConfig,
                                            get_pipeline_config_cls_for_name)
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ForwardBatch
from fastvideo.v1.utils import align_to, shallow_asdict
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
                        pipeline_config: Optional[
                            Union[str
                                  | PipelineConfig]] = None,
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

        Priority level: Default pipeline config < User's pipeline config < User's kwargs
        """
        config = None
        # 1. If users provide a pipeline config, it will override the default pipeline config
        if isinstance(pipeline_config, PipelineConfig):
            config = pipeline_config
        else:
            config_cls = get_pipeline_config_cls_for_name(model_path)
            if config_cls is not None:
                config = config_cls()
                if isinstance(pipeline_config, str):
                    config.load_from_json(pipeline_config)

        # 2. If users also provide some kwargs, it will override the pipeline config.
        # The user kwargs shouldn't contain model config parameters!
        if config is None:
            logger.warning("No config found for model %s, using default config",
                           model_path)
            config_args = kwargs
        else:
            config_args = shallow_asdict(config)
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
        sampling_param: Optional[SamplingParam] = None,
        **kwargs,
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

        # Validate inputs
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = prompt.strip()

        if sampling_param is None:
            sampling_param = SamplingParam.from_pretrained(
                fastvideo_args.model_path)
        kwargs["prompt"] = prompt
        sampling_param.update(kwargs)

        # Process negative prompt
        if sampling_param.negative_prompt is not None:
            sampling_param.negative_prompt = sampling_param.negative_prompt.strip(
            )

        # Validate dimensions
        if (sampling_param.height <= 0 or sampling_param.width <= 0
                or sampling_param.num_frames <= 0):
            raise ValueError(
                f"Height, width, and num_frames must be positive integers, got "
                f"height={sampling_param.height}, width={sampling_param.width}, "
                f"num_frames={sampling_param.num_frames}")

        if (
                sampling_param.num_frames - 1
        ) % fastvideo_args.vae_config.arch_config.temporal_compression_ratio != 0:
            raise ValueError(
                f"num_frames-1 must be a multiple of {fastvideo_args.vae_config.arch_config.temporal_compression_ratio}, got {sampling_param.num_frames}"
            )

        # Calculate sizes
        target_height = align_to(sampling_param.height, 16)
        target_width = align_to(sampling_param.width, 16)

        # Calculate latent sizes
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # Log parameters
        debug_str = f"""
                      height: {target_height}
                       width: {target_width}
                video_length: {sampling_param.num_frames}
                      prompt: {prompt}
                  neg_prompt: {sampling_param.negative_prompt}
                        seed: {sampling_param.seed}
                 infer_steps: {sampling_param.num_inference_steps}
       num_videos_per_prompt: {sampling_param.num_videos_per_prompt}
              guidance_scale: {sampling_param.guidance_scale}
                    n_tokens: {n_tokens}
                  flow_shift: {fastvideo_args.flow_shift}
     embedded_guidance_scale: {fastvideo_args.embedded_cfg_scale}
                  save_video: {sampling_param.save_video}
                  output_path: {sampling_param.output_path}
        """ # type: ignore[attr-defined]
        logger.info(debug_str)

        # Prepare batch
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            extra={},
        )

        # Run inference
        start_time = time.perf_counter()
        output_batch = self.executor.execute_forward(batch, fastvideo_args)
        samples = output_batch

        gen_time = time.perf_counter() - start_time
        logger.info("Generated successfully in %.2f seconds", gen_time)

        # Process outputs
        videos = rearrange(samples, "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Save video if requested
        if batch.save_video:
            save_path = batch.output_path
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                video_path = os.path.join(save_path, f"{prompt[:100]}.mp4")
                imageio.mimsave(video_path, frames, fps=batch.fps, format="mp4")
                logger.info("Saved video to %s", video_path)
            else:
                logger.warning("No output path provided, video not saved")

        if batch.return_frames:
            return frames
        else:
            return {
                "samples": samples,
                "prompts": prompt,
                "size": (target_height, target_width, batch.num_frames),
                "generation_time": gen_time
            }

    def shutdown(self):
        """
        Shutdown the video generator.
        """
        self.executor.shutdown()
        del self.executor
        gc.collect()
        torch.cuda.empty_cache()
