# type: ignore

# SPDX-License-Identifier: Apache-2.0
"""
Inference module for diffusion models.

This module provides classes and functions for running inference with diffusion models.
"""

import time
from typing import Any, Dict

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    build_pipeline)
# TODO(will): remove, check if this is hunyuan specific
from fastvideo.v1.utils import align_to

logger = init_logger(__name__)


class InferenceEngine:
    """
    Engine for running inference with diffusion models.
    """

    def __init__(
        self,
        pipeline: ComposedPipelineBase,
        fastvideo_args: FastVideoArgs,
    ):
        """
        Initialize the inference engine.
        
        Args:
            pipeline: The pipeline to use for inference.
            fastvideo_args: The inference arguments.
            default_negative_prompt: The default negative prompt to use.
        """
        self.pipeline = pipeline
        self.fastvideo_args = fastvideo_args

    @classmethod
    def create_engine(
        cls,
        fastvideo_args: FastVideoArgs,
    ) -> "InferenceEngine":
        """
        Create an inference engine with the specified arguments.
        
        Args:
            fastvideo_args: The inference arguments.
            model_loader_cls: The model loader class to use. If None, it will be
                determined from the model type.
            pipeline_type: The type of pipeline to create. If None, it will be
                determined from the model type.
                
        Returns:
            The created inference engine.
            
        Raises:
            ValueError: If the model type is not recognized or if the pipeline type
                is not recognized.
        """

        logger.info("Building pipeline...")

        # TODO(will): I don't really like this api.
        # it should be something closer to pipeline_cls.from_pretrained(...)
        # this way for training we can just do pipeline_cls.from_pretrained(
        # checkpoint_path) and have it handle everything.
        # TODO(Peiyuan): Then maybe we should only pass in model path and device, not the entire inference args?
        pipeline = build_pipeline(fastvideo_args)
        logger.info("Pipeline Ready")

        # Create the inference engine
        return cls(pipeline, fastvideo_args)

    def run(
        self,
        prompt: str,
        fastvideo_args: FastVideoArgs,
    ) -> Dict[str, Any]:
        """
        Run inference with the pipeline.
        
        Args:
            prompt: The prompt to use for generation.
            negative_prompt: The negative prompt to use. If None, the default will be used.
            seed: The random seed to use. If None, a random seed will be used.
            **kwargs: Additional arguments to pass to the pipeline.
            
        Returns:
            A dictionary containing the generated videos and metadata.
        """
        out_dict: Dict[str, Any] = dict()

        num_videos_per_prompt = fastvideo_args.num_videos
        seed = fastvideo_args.seed
        height = fastvideo_args.height
        width = fastvideo_args.width
        video_length = fastvideo_args.num_frames
        negative_prompt = fastvideo_args.neg_prompt
        infer_steps = fastvideo_args.num_inference_steps
        guidance_scale = fastvideo_args.guidance_scale
        flow_shift = fastvideo_args.flow_shift
        embedded_guidance_scale = fastvideo_args.embedded_cfg_scale
        image_path = fastvideo_args.image_path

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}")

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = prompt.strip()

        # negative prompt
        if negative_prompt is not None:
            negative_prompt = negative_prompt.strip()

        # TODO(PY): move to hunyuan stage
        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}"""
        logger.info(debug_str)
        # return
        # sp_group = get_sp_group()
        # local_rank = sp_group.rank
        device = torch.device(fastvideo_args.device_str)
        batch = ForwardBatch(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            height=fastvideo_args.height,
            width=fastvideo_args.width,
            num_frames=fastvideo_args.num_frames,
            num_inference_steps=fastvideo_args.num_inference_steps,
            guidance_scale=fastvideo_args.guidance_scale,
            # generator=generator,
            eta=0.0,
            n_tokens=n_tokens,
            data_type="video" if fastvideo_args.num_frames > 1 else "image",
            device=device,
            extra={},  # Any additional parameters
        )

        print('===============================================')
        print(batch)
        print('===============================================')
        print('===============================================')
        print(fastvideo_args)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        start_time = time.perf_counter()
        samples = self.pipeline.forward(
            batch=batch,
            fastvideo_args=fastvideo_args,
        ).output
        # TODO(will): fix and move to hunyuan stage
        # out_dict["seeds"] = batch.seeds
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.perf_counter() - start_time
        logger.info("Success, time: %s", gen_time)

        return out_dict
