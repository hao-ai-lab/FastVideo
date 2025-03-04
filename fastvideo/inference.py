import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from safetensors.torch import load_file as safetensors_load_file
import json

from fastvideo.models.hunyuan.constants import NEGATIVE_PROMPT, PRECISION_TO_TYPE, PROMPT_TEMPLATE
from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_base import AbstractDiffusionPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TextData, LatentData, SchedulerData
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.modules import load_model
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.utils.data_utils import align_to
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.utils.parallel_states import nccl_info
from fastvideo.inference_args import InferenceArgs
from fastvideo.platforms import current_platform
from fastvideo.logger import init_logger
from fastvideo.pipelines.loader import PipelineLoader

logger = init_logger(__name__)





class DiffusionInference:
    """
    Unified inference class that works with any diffusion pipeline.
    This combines model loading, pipeline creation, and inference in a flexible way.
    """
    
    def __init__(self, 
                args: InferenceArgs,
                pipeline: AbstractDiffusionPipeline,
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
        pipeline = PipelineLoader.load_pipeline(
            inference_args=inference_args,
        )
        
        return cls(inference_args, pipeline, default_negative_prompt=NEGATIVE_PROMPT)
    
    @staticmethod
    def _load_model_components(args: InferenceArgs, model_path: Path) -> ModelComponents:
        """
        Load model components from the model path.
        
        Args:
            args: The inference arguments
            model_path: Path to the model directory
            
        Returns:
            ModelComponents containing the loaded models
        """
        # Load VAE
        vae = load_vae(args.vae, args.precision)
        
        # Load text encoder
        text_encoder = TextEncoder.from_pretrained(args.text_encoder)
        
        # Load transformer model
        transformer = load_model(
            model_path=str(model_path),
            precision=PRECISION_TO_TYPE[args.precision]
        )
        
        # Create scheduler
        scheduler = FlowMatchDiscreteScheduler(
            shift=args.flow_shift if hasattr(args, 'flow_shift') else 0.0,
            reverse=args.flow_reverse if hasattr(args, 'flow_reverse') else False,
            solver=args.flow_solver if hasattr(args, 'flow_solver') else "midpoint",
        )
        
        return ModelComponents(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler
        )
        
    def predict(self, inference_args: InferenceArgs) -> Dict[str, Any]:
        """
        Run inference with the pipeline using ForwardBatch API.
        
        Args:
            prompt: The text prompt for generation
            height: Output height (defaults to args value)
            width: Output width (defaults to args value)
            video_length: Number of frames (defaults to args value)
            seed: Random seed (defaults to random)
            negative_prompt: Negative prompt (defaults to default_negative_prompt)
            infer_steps: Number of inference steps (defaults to args value)
            guidance_scale: Guidance scale (defaults to args value)
            batch_size: Batch size for generation
            num_videos_per_prompt: Number of videos per prompt
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Dictionary with generated samples and metadata
        """
        out_dict = {}
        
        # Set defaults from args if not provided
        height = height or self.args.height
        width = width or self.args.width
        video_length = video_length or self.args.num_frames
        infer_steps = infer_steps or self.args.num_inference_steps
        guidance_scale = guidance_scale or self.args.guidance_scale
        
        # Validate dimensions
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"height, width, and video_length must be positive integers"
            )
            
        # Align dimensions
        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length
        
        # Handle seeds
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
                raise ValueError(f"Invalid seed length: {len(seed)}")
        else:
            raise ValueError(f"Invalid seed type: {type(seed)}")
            
        # Create generators
        generator = [torch.Generator("cpu").manual_seed(s) for s in seeds]
        out_dict["seeds"] = seeds
        out_dict["size"] = (target_height, target_width, target_video_length)
        
        # Handle prompts
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt)}")
        prompt = [prompt.strip()]
        
        # Handle negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"negative_prompt must be a string, got {type(negative_prompt)}")
        negative_prompt = [negative_prompt.strip()]
        
        # Create scheduler if needed
        if hasattr(self.args, 'flow_shift') and hasattr(self.args, 'denoise_type') and self.args.denoise_type == "flow":
            flow_shift = kwargs.get('flow_shift', self.args.flow_shift)
            scheduler = FlowMatchDiscreteScheduler(
                shift=flow_shift,
                reverse=self.args.flow_reverse,
                solver=self.args.flow_solver,
            )
            self.pipeline.scheduler = scheduler
        
        # Calculate number of tokens
        if "884" in self.args.vae:
            latents_size = [(target_video_length - 1) // 4 + 1, target_height // 8, target_width // 8]
        elif "888" in self.args.vae:
            latents_size = [(target_video_length - 1) // 8 + 1, target_height // 8, target_width // 8]
        else:
            latents_size = [target_video_length, target_height // 8, target_width // 8]
        
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        
        # Create batch data
        batch = ForwardBatch(
            text_data=TextData(
                prompt=prompt, 
                negative_prompt=negative_prompt
            ),
            latent_data=LatentData(
                height=target_height,
                width=target_width, 
                num_frames=target_video_length,
                generator=generator
            ),
            scheduler_data=SchedulerData()
        )
        
        # Log inference parameters
        logger.info(f"Running inference with parameters: {inference_args}")
        
        # Run inference
        start_time = time.time()
        # Store inference args on the pipeline
        self.pipeline.inference_args = inference_args
        # Call forward instead of __call__
        samples = self.pipeline.forward(batch)
        
        gen_time = time.time() - start_time
        logger.info(f"Generation complete in {gen_time:.2f}s")
        
        # Return results
        out_dict["samples"] = samples.videos if hasattr(samples, 'videos') else samples
        out_dict["prompts"] = prompt
        
        return out_dict
