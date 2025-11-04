# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import math
import os
import re
import time
from copy import deepcopy
from typing import Any

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import align_to, shallow_asdict
from fastvideo.worker.executor import Executor

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
                        device: str | None = None,
                        torch_dtype: torch.dtype | None = None,
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
        fastvideo_args = FastVideoArgs.from_kwargs(**kwargs)

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
        prompt: str | None = None,
        sampling_param: SamplingParam | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[np.ndarray] | list[dict[str, Any]]:
        """
        Generate a video based on the given prompt.
        
        Args:
            prompt: The prompt to use for generation (optional if prompt_txt is provided)
            negative_prompt: The negative prompt to use (overrides the one in fastvideo_args)
            output_path: Path to save the video (overrides the one in fastvideo_args)
            prompt_path: Path to prompt file
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
            Either the output dictionary, list of frames, or list of results for batch processing
        """
        # Handle batch processing from text file
        if sampling_param is None:
            sampling_param = SamplingParam.from_pretrained(
                self.fastvideo_args.model_path)
        sampling_param.update(kwargs)

        if self.fastvideo_args.prompt_txt is not None or sampling_param.prompt_path is not None:
            prompt_txt_path = sampling_param.prompt_path or self.fastvideo_args.prompt_txt
            if not os.path.exists(prompt_txt_path):
                raise FileNotFoundError(
                    f"Prompt text file not found: {prompt_txt_path}")

            # Read prompts from file
            with open(prompt_txt_path, encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]

            if not prompts:
                raise ValueError(f"No prompts found in file: {prompt_txt_path}")

            logger.info("Found %d prompts in %s", len(prompts), prompt_txt_path)

            results = []
            for i, batch_prompt in enumerate(prompts):
                logger.info("Processing prompt %d/%d: %s...", i + 1,
                            len(prompts), batch_prompt[:100])
                try:
                    # Generate video for this prompt using the same logic below
                    output_path = self._prepare_output_path(
                        sampling_param.output_path, batch_prompt)
                    kwargs["output_path"] = output_path
                    result = self._generate_single_video(
                        prompt=batch_prompt,
                        sampling_param=sampling_param,
                        **kwargs)

                    # Add prompt info to result
                    if isinstance(result, dict):
                        result["prompt_index"] = i
                        result["prompt"] = batch_prompt

                    results.append(result)
                    logger.info("Successfully generated video for prompt %d",
                                i + 1)

                except Exception as e:
                    logger.error("Failed to generate video for prompt %d: %s",
                                 i + 1, e)
                    continue

            logger.info(
                "Completed batch processing. Generated %d videos successfully.",
                len(results))
            return results

        # Single prompt generation (original behavior)
        if prompt is None:
            raise ValueError("Either prompt or prompt_txt must be provided")
        output_path = self._prepare_output_path(sampling_param.output_path,
                                                prompt)
        kwargs["output_path"] = output_path
        return self._generate_single_video(prompt=prompt,
                                           sampling_param=sampling_param,
                                           **kwargs)

    def _prepare_output_path(
        self,
        output_path: str,
        prompt: str,
    ) -> str:
        """Build a unique, sanitized .mp4 output file path.

        - If `output_path` ends with .mp4 (case-insensitive), treat it as a file path.
        - Otherwise, treat `output_path` as a directory and derive the filename
          from the prompt.
        - Invalid filename characters are removed; if the name changes, a
          warning is logged.
        - If the target path already exists, a numeric suffix is appended.
        """

        def _sanitize_filename_component(name: str) -> str:
            # Remove characters invalid on common filesystems, strip spaces/dots
            sanitized = re.sub(r'[\\/:*?"<>|]', '', name)
            sanitized = sanitized.strip().strip('.')
            sanitized = re.sub(r'\s+', ' ', sanitized)
            return sanitized or "video"

        base_path, extension = os.path.splitext(output_path)
        extension_lower = extension.lower()

        if extension_lower == ".mp4":
            output_dir = os.path.dirname(output_path)
            base_name = os.path.basename(
                base_path)  # filename without extension
            sanitized_base = _sanitize_filename_component(base_name)
            if sanitized_base != base_name:
                logger.warning(
                    "The video name '%s' contained invalid characters. It has been renamed to '%s.mp4'",
                    os.path.basename(output_path),
                    sanitized_base,
                )
            video_name = f"{sanitized_base}.mp4"
        else:
            # Treat as directory; inform if an unexpected extension was provided.
            if extension:
                logger.info(
                    "Output path '%s' has non-mp4 extension '%s'; treating it as a directory and using a .mp4 filename derived from the prompt",
                    output_path,
                    extension,
                )
            output_dir = output_path
            prompt_component = _sanitize_filename_component(prompt[:100])
            video_name = f"{prompt_component}.mp4"

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        new_output_path = os.path.join(output_dir, video_name)
        counter = 1
        while os.path.exists(new_output_path):
            name_part, ext_part = os.path.splitext(video_name)
            new_video_name = f"{name_part}_{counter}{ext_part}"
            new_output_path = os.path.join(output_dir, new_video_name)
            counter += 1
        return new_output_path

    def _generate_single_video(
        self,
        prompt: str,
        sampling_param: SamplingParam | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[np.ndarray]:
        """Internal method for single video generation"""
        # Create a copy of inference args to avoid modifying the original
        fastvideo_args = self.fastvideo_args
        pipeline_config = fastvideo_args.pipeline_config

        # Validate inputs
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = prompt.strip()
        sampling_param = deepcopy(sampling_param)
        output_path = kwargs["output_path"]
        sampling_param.prompt = prompt
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
                      prompt: {sampling_param.prompt}
                      image_path: {sampling_param.image_path}
                  neg_prompt: {sampling_param.negative_prompt}
                        seed: {sampling_param.seed}
                 infer_steps: {sampling_param.num_inference_steps}
       num_videos_per_prompt: {sampling_param.num_videos_per_prompt}
              guidance_scale: {sampling_param.guidance_scale}
                    n_tokens: {n_tokens}
                  flow_shift: {fastvideo_args.pipeline_config.flow_shift}
     embedded_guidance_scale: {fastvideo_args.pipeline_config.embedded_cfg_scale}
                  save_video: {sampling_param.save_video}
                  output_path: {output_path}
        """ # type: ignore[attr-defined]
        logger.info(debug_str)

        # Prepare batch
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
        )

        # Run inference
        start_time = time.perf_counter()
        output_batch = self.executor.execute_forward(batch, fastvideo_args)
        samples = output_batch.output
        logging_info = output_batch.logging_info

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
            imageio.mimsave(output_path, frames, fps=batch.fps, format="mp4")
            logger.info("Saved video to %s", output_path)

        if batch.return_frames:
            return frames
        else:
            return {
                "samples": samples,
                "frames": frames,
                "prompts": prompt,
                "size": (target_height, target_width, batch.num_frames),
                "generation_time": gen_time,
                "logging_info": logging_info,
                "trajectory": output_batch.trajectory_latents,
                "trajectory_timesteps": output_batch.trajectory_timesteps,
                "trajectory_decoded": output_batch.trajectory_decoded,
            }

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        self.executor.set_lora_adapter(lora_nickname, lora_path)

    def unmerge_lora_weights(self) -> None:
        """
        Use unmerged weights for inference to produce videos that align with 
        validation videos generated during training.
        """
        self.executor.unmerge_lora_weights()

    def merge_lora_weights(self) -> None:
        self.executor.merge_lora_weights()

    def shutdown(self):
        """
        Shutdown the video generator.
        """
        self.executor.shutdown()
        del self.executor
