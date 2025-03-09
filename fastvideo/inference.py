import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from safetensors.torch import load_file as safetensors_load_file
import json

from fastvideo.models.hunyuan.constants import NEGATIVE_PROMPT, PRECISION_TO_TYPE, PROMPT_TEMPLATE
# from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline
from fastvideo.pipelines.pipeline_base import DiffusionPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
# from fastvideo.models.hunyuan.modules import load_model
# from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.utils.data_utils import align_to
# from fastvideo.models.hunyuan.vae import load_vae
# from fastvideo.utils.parallel_states import nccl_info
from fastvideo.inference_args import InferenceArgs
from fastvideo.platforms import current_platform
from fastvideo.logger import init_logger
from fastvideo.pipelines.loader import PipelineLoader, get_pipeline_loader
from fastvideo.distributed.parallel_state import get_sp_group

logger = init_logger(__name__)


class DiffusionInference:
    """
    Unified inference class that works with any diffusion pipeline.
    This combines model loading, pipeline creation, and inference in a flexible way.
    """
    
    def __init__(self, 
                args: InferenceArgs,
                pipeline: DiffusionPipelineBase,
                default_negative_prompt: str = NEGATIVE_PROMPT):
        """
        Initialize the inference class with a pipeline and args.
        
        Args:
            args: The inference arguments
            pipeline: The diffusion pipeline to use
            default_negative_prompt: The default negative prompt to use
        """
        self.args = args
        self.pipeline = pipeline
        self.default_negative_prompt = default_negative_prompt

    
        
    @classmethod
    def load_pipeline(cls, inference_args: InferenceArgs):
        """
        Create an inference instance from pretrained model components.
        
        Args:
            inference_args: The inference arguments containing model path and other settings
            
        Returns:
            A DiffusionInference instance ready for inference
        """
        logger.debug("start loading pipeline")
        models_root_path = Path(inference_args.model_path)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")

        # Create save folder to save the samples
        save_path = inference_args.output_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Log which platform we're using
        print(f"Using platform: {current_platform.__class__.__name__}")
        assert current_platform.is_cuda(), "CUDA is not available"

        # init distributed
        # _initialize_distributed()

        
        # get loader for the pipeline

        # Create pipeline using the components
        pipeline_loader = get_pipeline_loader(inference_args)
        pipeline = pipeline_loader.load_pipeline(
            inference_args=inference_args,
        )
        
        return cls(inference_args, pipeline, default_negative_prompt=NEGATIVE_PROMPT)
    
    def predict(self, prompt: str, inference_args: InferenceArgs) -> Dict[str, Any]:
        """
        Run inference with the pipeline.
            
        Returns:
            Dictionary with generated samples and metadata
        """

        out_dict = dict()

        batch_size = inference_args.batch_size
        num_videos_per_prompt = inference_args.num_videos
        seed = inference_args.seed
        height = inference_args.height
        width = inference_args.width
        video_length = inference_args.num_frames
        # prompt = inference_args.prompt
        negative_prompt = inference_args.neg_prompt
        infer_steps = inference_args.num_inference_steps
        guidance_scale = inference_args.guidance_scale
        flow_shift = inference_args.flow_shift
        # mask_strategy = inference_args.mask_strategy
        embedded_guidance_scale = inference_args.embedded_cfg_scale
        

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}.")
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
        generator = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

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

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver,
        )
        self.pipeline.scheduler = scheduler

        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
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
            generator=generator,
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
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict