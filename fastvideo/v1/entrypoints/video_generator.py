# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import gc
import math
import os
import time
from typing import Any, Dict, List, Optional, Union

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

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
            fastvideo_args: The inference arguments
            executor_class: The executor class to use for inference
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
            pipeline_config: Pipeline config to use for inference
            **kwargs: Additional arguments to customize model loading, set any FastVideoArgs or PipelineConfig attributes here.
                
        Returns:
            The created video generator

        Priority level: Default pipeline config < User's pipeline config < User's kwargs
        """
        # If users also provide some kwargs, it will override the FastVideoArgs and PipelineConfig.
        kwargs['model_path'] = model_path
        fastvideo_args = FastVideoArgs.from_kwargs(kwargs)

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
        pipeline_config = fastvideo_args.pipeline_config

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

        temporal_scale_factor = pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = sampling_param.num_frames
        num_gpus = fastvideo_args.num_gpus
        use_temporal_scaling_frames = pipeline_config.vae_config.use_temporal_scaling_frames

        # Adjust number of frames based on number of GPUs
        if use_temporal_scaling_frames:
            orig_latent_num_frames = (num_frames -
                                      1) // temporal_scale_factor + 1
        else:  # stepvideo only
            orig_latent_num_frames = sampling_param.num_frames // 17 * 3

        if orig_latent_num_frames % fastvideo_args.num_gpus != 0:
            # Adjust latent frames to be divisible by number of GPUs
            if sampling_param.num_frames_round_down:
                # Ensure we have at least 1 batch per GPU
                new_latent_num_frames = max(
                    1, (orig_latent_num_frames // num_gpus)) * num_gpus
            else:
                new_latent_num_frames = math.ceil(
                    orig_latent_num_frames / num_gpus) * num_gpus

            if use_temporal_scaling_frames:
                # Convert back to number of frames, ensuring num_frames-1 is a multiple of temporal_scale_factor
                new_num_frames = (new_latent_num_frames -
                                  1) * temporal_scale_factor + 1
            else:  # stepvideo only
                # Find the least common multiple of 3 and num_gpus
                divisor = math.lcm(3, num_gpus)
                # Round up to the nearest multiple of this LCM
                new_latent_num_frames = (
                    (new_latent_num_frames + divisor - 1) // divisor) * divisor
                # Convert back to actual frames using the StepVideo formula
                new_num_frames = new_latent_num_frames // 3 * 17

            logger.info(
                "Adjusting number of frames from %s to %s based on number of GPUs (%s)",
                sampling_param.num_frames, new_num_frames,
                fastvideo_args.num_gpus)
            sampling_param.num_frames = new_num_frames

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
                  flow_shift: {fastvideo_args.pipeline_config.flow_shift}
     embedded_guidance_scale: {fastvideo_args.pipeline_config.embedded_cfg_scale}
                  save_video: {sampling_param.save_video}
                  output_path: {sampling_param.output_path}
        """ # type: ignore[attr-defined]
        logger.info(debug_str)
        # Prepare batch
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
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
            output_path = batch.output_path
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                video_path = os.path.join(output_path, f"{prompt[:100]}.mp4")
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

    def set_lora_adapter(self, lora_nickname: str, lora_path: str) -> None:
        self.executor.set_lora_adapter(lora_nickname, lora_path)

    def shutdown(self):
        """
        Shutdown the video generator.
        """
        self.executor.shutdown()
        del self.executor
        gc.collect()
        torch.cuda.empty_cache()
