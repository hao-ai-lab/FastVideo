from fastvideo.inference_args import InferenceArgs
# from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TextData, LatentData, SchedulerData, GenerationParameters
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.modules import load_model
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.utils.data_utils import align_to
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.logger import init_logger
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

logger = init_logger(__name__)

T2V_PIPLINES = {
    "HunyuanVideo": ("hunyuan_video", "HunyuanVideoPipeline"),
}

FASTVIDEO_PIPLINES = {
    **T2V_PIPLINES,
}

def resolve_pipeline_cls(inference_args: InferenceArgs):
    """Resolve the pipeline class based on the inference args."""
    # TODO(will): resolve the pipeline class based on the model path
    # read from hf config file
    from fastvideo.pipelines.hunyuan_video import HunyuanVideoPipeline
    return HunyuanVideoPipeline


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


class PipelineLoader:
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

    def load_text_encoders(self, inference_args: InferenceArgs):
        """Load the text encoders based on the inference args."""
        text_encoders = []
        if inference_args.text_encoder is not None:
            text_encoders.append(TextEncoder(inference_args))
        if inference_args.text_encoder_2 is not None:
            text_encoders.append(TextEncoder(inference_args))
        return text_encoders


    @classmethod
    def load_pipeline(cls, inference_args: InferenceArgs):
        """Load the pipeline based on the inference args."""
        # Load configuration
        pipeline_cls = resolve_pipeline_cls(inference_args)
            
        logger.info(f"Pipeline class: {pipeline_cls}")

        # Load components
        model_components = cls.load_components(inference_args, inference_args.model_path)

        # Create pipeline
        pipeline = pipeline_cls(model_components=model_components)
    
    @classmethod
    def load_components(cls, args: InferenceArgs, model_path: Union[str, Path], device=None) -> ModelComponents:
        """
        Load all model components based on the args configuration.
        This method can be overridden by subclasses to load different model components.
        """
        logger.info(f"Loading components for {model_path}")
        # if device is None:
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # model_path = Path(model_path) if isinstance(model_path, str) else model_path

        text_encoders = self.load_text_encoders(args, device)
        if len(text_encoders) == 0:
            raise ValueError("No text encoder found")

        # Load primary text encoder
        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=args.text_len,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=args.prompt_template,
            prompt_template_video=args.prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        
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



def get_pipeline_loader(inference_args: InferenceArgs) -> PipelineLoader:
    """Get a pipeline loader based on the inference args."""

    return PipelineLoader(inference_args)
