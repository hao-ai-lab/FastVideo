"""
Inference module for diffusion models.

This module provides classes and functions for running inference with diffusion models.
"""

import os
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Type
from PIL import Image
import uuid
import random

from fastvideo.inference_args import InferenceArgs
from fastvideo.pipelines import ComposedPipelineBase, build_pipeline
from fastvideo.pipelines.pipeline_registry import PipelineRegistry
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.logger import init_logger
# TODO(will): remove
from fastvideo.models.hunyuan.utils.data_utils import align_to
from fastvideo.models.hunyuan.constants import NEGATIVE_PROMPT


logger = init_logger(__name__)

class InferenceEngine:
    """
    Engine for running inference with diffusion models.
    """
    
    def __init__(
        self,
        pipeline: ComposedPipelineBase,
        inference_args: InferenceArgs,
    ):
        """
        Initialize the inference engine.
        
        Args:
            pipeline: The pipeline to use for inference.
            inference_args: The inference arguments.
            default_negative_prompt: The default negative prompt to use.
        """
        self.pipeline = pipeline
        self.inference_args = inference_args
        # TODO(will): this is a hack to get the default negative prompt
        self.default_negative_prompt = NEGATIVE_PROMPT
    
    @classmethod
    def create_engine(
        cls,
        inference_args: InferenceArgs,
    ) -> "InferenceEngine":
        """
        Create an inference engine with the specified arguments.
        
        Args:
            inference_args: The inference arguments.
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
        try:
            logger.info(f"Building pipeline...")
            # TODO(will): I don't really like this api.
            # it should be something closer to pipeline_cls.from_pretrained(...)
            # this way for training we can just do pipeline_cls.from_pretrained(
            # checkpoint_path) and have it handle everything.
            pipeline = build_pipeline(inference_args)
            logger.info(f"Pipeline Ready")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise RuntimeError(f"Failed to load pipeline: {e}")
        
        # Create the inference engine
        return cls(pipeline, inference_args)
    
    def run(
        self,
        prompt: str,
        inference_args: InferenceArgs,
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
        out_dict = dict()

        num_videos_per_prompt = inference_args.num_videos
        seed = inference_args.seed
        height = inference_args.height
        width = inference_args.width
        video_length = inference_args.num_frames
        negative_prompt = inference_args.neg_prompt
        infer_steps = inference_args.num_inference_steps
        guidance_scale = inference_args.guidance_scale
        flow_shift = inference_args.flow_shift
        embedded_guidance_scale = inference_args.embedded_cfg_scale

        # generated from inference_args.seed
        # seeds = inference_args.seeds
        # generator = inference_args.generator
        

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(f"`video_length-1` must be a multiple of 4, got {video_length}")

        logger.info(f"Input (height, width, video_length) = ({height}, {width}, {video_length})")

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
        negative_prompt = [negative_prompt.strip()]


        # from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
        # scheduler = FlowMatchDiscreteScheduler(
        #     shift=flow_shift,
        #     reverse=self.inference_args.flow_reverse,
        #     solver=self.inference_args.flow_solver,
        # )

        # # reset scheduler
        # self.pipeline.scheduler = scheduler

        # TODO(will): move to hunyuan stage
        if "884" in self.inference_args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.inference_args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
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
        import os
        local_rank = os.environ.get("LOCAL_RANK", 0)
        device = torch.device(f"cuda:{local_rank}")
        batch = ForwardBatch(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            height=inference_args.height,
            width=inference_args.width,
            num_frames=inference_args.num_frames,
            num_inference_steps=inference_args.num_inference_steps,
            guidance_scale=inference_args.guidance_scale,
            # generator=generator,
            eta=0.0,
            n_tokens=n_tokens,
            data_type="video" if inference_args.num_frames > 1 else "image",
            device=device,
            extra={},  # Any additional parameters
        )

        print('===============================================')
        print(batch)
        print('===============================================')
        print('===============================================')
        print(inference_args)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        start_time = time.time()
        samples = self.pipeline.forward(
            batch=batch,
            inference_args=inference_args,
        )[0]
        # TODO(will): fix and move to hunyuan stage
        # out_dict["seeds"] = batch.seeds
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict