import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.models.hunyuan.constants import NEGATIVE_PROMPT, PRECISION_TO_TYPE, PROMPT_TEMPLATE
from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_base import AbstractDiffusionPipeline
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_batch import ForwardBatch, TextData, LatentData, SchedulerData, GenerationParameters
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.modules import load_model
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.utils.data_utils import align_to
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.utils.parallel_states import nccl_info
from fastvideo.inference_args import InferenceArgs


class ModelComponents:
    """Container for model components used by diffusion pipelines"""
    
    def __init__(self, 
                 vae=None, 
                 text_encoder=None, 
                 text_encoder_2=None, 
                 transformer=None, 
                 scheduler=None):
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.transformer = transformer
        self.scheduler = scheduler


class ModelLoader:
    """
    Base class for loading model components that can be used across different pipelines.
    This separates model loading from pipeline creation and inference.
    """
    
    @staticmethod
    def load_state_dict(args, model, model_path):
        """Load model state dict from checkpoint"""
        logger.info(f'Loading state dict from {model_path}')
        
        load_key = args.load_key
        
        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
            
        if model_path.suffix == ".safetensors":
            # Use safetensors library for .safetensors files
            state_dict = safetensors_load_file(model_path)
        elif model_path.suffix == ".pt":
            # Use torch for .pt files
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        else:
            raise ValueError(f"Unsupported file format: {model_path}")

        # Check if model is a bare model or has a specific key structure
        bare_model = True
        if "ema" in state_dict or "module" in state_dict:
            bare_model = False
            
        # Extract the model weights based on the structure
        if not bare_model:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                               f"are: {list(state_dict.keys())}.")
                
        model.load_state_dict(state_dict, strict=True)
        return model
    
    @classmethod
    def load_components(cls, args: InferenceArgs, model_path: Union[str, Path], device=None) -> ModelComponents:
        """
        Load all model components based on the args configuration.
        This method can be overridden by subclasses to load different model components.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
        # Build transformer model
        logger.info("Building transformer model...")
        factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
        transformer = load_model(
            args,
            in_channels=args.latent_channels,
            out_channels=args.latent_channels,
            factor_kwargs=factor_kwargs,
        ).to(device)
        
        # Load weights
        transformer_path = cls._find_transformer_checkpoint(args, model_path)
        transformer = cls.load_state_dict(args, transformer, transformer_path)
        
        if args.enable_torch_compile:
            transformer = torch.compile(transformer)
        transformer.eval()
        
        # Load VAE
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            args.vae_precision,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        
        # Prepare text encoder parameters
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # Get prompt templates
        prompt_template = (PROMPT_TEMPLATE[args.prompt_template] 
                           if args.prompt_template is not None else None)
        prompt_template_video = (PROMPT_TEMPLATE[args.prompt_template_video]
                                 if args.prompt_template_video is not None else None)
        
        # Load primary text encoder
        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        
        # Load secondary text encoder if specified
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            )
            
        # Create default scheduler
        scheduler = None
        if hasattr(args, 'denoise_type') and args.denoise_type == "flow":
            scheduler = FlowMatchDiscreteScheduler(
                shift=args.flow_shift,
                reverse=args.flow_reverse,
                solver=args.flow_solver,
            )
            
        return ModelComponents(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            scheduler=scheduler
        )
    
    @staticmethod
    def _find_transformer_checkpoint(args, model_path):
        """Find the transformer checkpoint file"""
        if args.dit_weight is not None:
            dit_weight = Path(args.dit_weight)
            if dit_weight.is_file():
                return dit_weight
                
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                    
                if any(str(f).startswith("pytorch_model_") for f in files):
                    return dit_weight / f"pytorch_model_{args.load_key}.pt"
                    
                if any(str(f).endswith("_model_states.pt") for f in files):
                    files = [f for f in files if str(f).endswith("_model_states.pt")]
                    if len(files) > 1:
                        logger.warning(f"Multiple model weights found in {dit_weight}, using {files[0]}")
                    return files[0]
                    
                raise ValueError(f"Invalid model path: {dit_weight} with unrecognized weight format")
        
        # Look in default location
        model_dir = model_path / f"t2v_{args.model_resolution}"
        files = list(model_dir.glob("*.pt"))
        if len(files) == 0:
            raise ValueError(f"No model weights found in {model_dir}")
            
        if any(str(f).startswith("pytorch_model_") for f in files):
            return model_dir / f"pytorch_model_{args.load_key}.pt"
            
        if any(str(f).endswith("_model_states.pt") for f in files):
            files = [f for f in files if str(f).endswith("_model_states.pt")]
            if len(files) > 1:
                logger.warning(f"Multiple model weights found in {model_dir}, using {files[0]}")
            return files[0]
            
        raise ValueError(f"Invalid model path: {model_dir} with unrecognized weight format")


class PipelineFactory:
    """
    Factory class for creating different diffusion pipelines based on model type.
    This separates pipeline creation from model loading and inference.
    """
    
    @staticmethod
    def create_pipeline(model_components: ModelComponents, 
                       pipeline_cls: type = HunyuanVideoPipeline,
                       args: Optional[InferenceArgs] = None,
                       device=None) -> AbstractDiffusionPipeline:
        """
        Create a pipeline instance based on the model components and pipeline class.
        
        Args:
            model_components: The model components to use in the pipeline
            pipeline_cls: The pipeline class to instantiate
            args: Additional arguments for the pipeline
            device: The device to place the pipeline on
            
        Returns:
            An instance of the specified pipeline class
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Create the pipeline with the model components
        pipeline = pipeline_cls(
            vae=model_components.vae,
            text_encoder=model_components.text_encoder,
            text_encoder_2=model_components.text_encoder_2,
            transformer=model_components.transformer,
            scheduler=model_components.scheduler,
            args=args
        )
        
        # Handle CPU offload or device placement
        if args and args.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)
            
        return pipeline


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
    def from_pretrained(cls, 
                       model_path: Union[str, Path],
                       args: InferenceArgs,
                       model_loader_cls: type = ModelLoader,
                       pipeline_cls: type = HunyuanVideoPipeline,
                       device=None,
                       **kwargs):
        """
        Create an inference instance from pretrained model components.
        
        Args:
            model_path: Path to the model directory or checkpoint
            args: The inference arguments
            model_loader_cls: The model loader class to use
            pipeline_cls: The pipeline class to use
            device: The device to place the models on
            
        Returns:
            A DiffusionInference instance ready for inference
        """
        if device is None:
            # Handle distributed setup
            if nccl_info.sp_size > 1:
                device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model components
        model_components = model_loader_cls.load_components(args, model_path, device)
        
        # Create pipeline
        pipeline = PipelineFactory.create_pipeline(
            model_components, 
            pipeline_cls=pipeline_cls,
            args=args,
            device=device
        )
        
        return cls(args, pipeline)
        
    def predict(self, 
               prompt: str,
               height: int = None,
               width: int = None,
               video_length: int = None,
               seed: int = None,
               negative_prompt: str = None,
               infer_steps: int = None,
               guidance_scale: float = None,
               batch_size: int = 1,
               num_videos_per_prompt: int = 1,
               **kwargs) -> Dict[str, Any]:
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
        
        # Create inference args
        inference_args = GenerationParameters(
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            num_videos_per_prompt=num_videos_per_prompt,
            output_type="pil",
            return_dict=True,
            precision=self.args.precision,
            vae_precision=self.args.vae_precision,
            disable_autocast=getattr(self.args, "disable_autocast", False),
            vae_ver=self.args.vae,
            enable_tiling=getattr(self.args, "vae_tiling", False),
            enable_vae_sp=getattr(self.args, "vae_sp", False),
            data_type="video" if target_video_length > 1 else "image",
            n_tokens=n_tokens,
            embedded_guidance_scale=kwargs.get('embedded_guidance_scale', None),
            mask_strategy=kwargs.get('mask_strategy', None),
        )
        
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
