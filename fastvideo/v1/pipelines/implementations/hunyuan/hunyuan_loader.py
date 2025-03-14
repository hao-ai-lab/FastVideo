import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.v1.pipelines.loader import PipelineLoader
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.implementations.hunyuan.constants import PRECISION_TO_TYPE, PROMPT_TEMPLATE
from fastvideo.v1.logger import init_logger
from typing import Tuple, Optional
from fastvideo.v1.models.dits import HunyuanVideoTransformer3DModel
import os
import glob
import json
from fastvideo.v1.loader.fsdp_load import load_fsdp_model
from fastvideo.v1.models.hunyuan.text_encoder import TextEncoder
from fastvideo.v1.models.vaes import AutoencoderKLHunyuanVideo
from safetensors.torch import load_file as safetensors_load_file
from fastvideo.v1.platforms import current_platform

logger = init_logger(__name__)

class HunyuanPipelineLoader(PipelineLoader):
    """Specific loader for Hunyuan video pipeline"""

    def __init__(self, inference_args: InferenceArgs, use_v0=False):
        super().__init__(inference_args, use_v0)

    def _load_transformer_state_dict(self, inference_args: InferenceArgs, model: nn.Module, model_path: Path):
        """Load the transformer state dict"""
        load_key = inference_args.load_key
        dit_weight = Path(inference_args.dit_weight)

        if dit_weight is None:
            model_dir = model_path / f"t2v_{inference_args.model_resolution}"
            files = list(model_dir.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {model_dir}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
                bare_model = False
            else:
                raise ValueError(f"Invalid model path: {dit_weight} with unrecognized weight format: "
                                 f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                                 f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                                 f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                                 f"specific weight file, please provide the full path to the file.")
        else:
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith("pytorch_model_"):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith("_model_states.pt") for f in files):
                    files = [f for f in files if str(f).endswith("_model_states.pt")]
                    model_path = files[0]
                    if len(files) > 1:
                        logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
                    bare_model = False
                else:
                    raise ValueError(f"Invalid model path: {dit_weight} with unrecognized weight format: "
                                     f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                                     f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                                     f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                                     f"specific weight file, please provide the full path to the file.")
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = "unknown"
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")

        print('model_path', model_path)
        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading torch model {model_path}...")
        if model_path.suffix == ".safetensors":
            # Use safetensors library for .safetensors files
            state_dict = safetensors_load_file(model_path)
        elif model_path.suffix == ".pt":
            # Use torch for .pt files
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        else:
            raise ValueError(f"Unsupported file format: {model_path}")

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                               f"are: {list(state_dict.keys())}.")
        model.load_state_dict(state_dict, strict=True)
        return model

    def load_transformer(self, inference_args: InferenceArgs):
        """Custom transformer loading for Hunyuan"""
        use_v1_loader = True
        if use_v1_loader:
            return self.load_transformer_v1(inference_args)
        else:
            return self.load_transformer_v0(inference_args)

    def load_transformer_v1(self, inference_args: InferenceArgs):
        """Custom transformer loading for Hunyuan"""
        
        # Path to model files
        # TODO(PY/Will): remove this hardcode
        path = "data/hunyuanvideo_community/transformer"
        path = Path(path)
        
        # Load config file
        config_path = path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Clean up config
        if "_class_name" in config:
            config.pop("_class_name")
        if "_diffusers_version" in config:
            config.pop("_diffusers_version")
            
        # Find all safetensors files
        safetensors_list = glob.glob(os.path.join(str(path), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {path}")
        
        logger.info(f"Loading model from {len(safetensors_list)} safetensors files in {path}")
        
        # Load the model using FSDP loader
        # TOOD(Will) new registry for DiT
        model = load_fsdp_model(
            model_cls=HunyuanVideoTransformer3DModel,
            init_params=config,
            weight_dir_list=safetensors_list,
            device=self.device,
            cpu_offload=inference_args.use_cpu_offload
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded HunyuanVideo model with {total_params / 1e9:.2f}B parameters")
        
        
        model.eval()
        return model

    def load_transformer_v0(self, inference_args: InferenceArgs):
        """Custom transformer loading for Hunyuan"""
        # TODO(will): replace this with abstracted model
        from fastvideo.v1.models.hunyuan.modules import load_model
        # Disable gradient
        torch.set_grad_enabled(False)

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {"device": self.device, "dtype": PRECISION_TO_TYPE[inference_args.precision]}
        in_channels = inference_args.latent_channels
        out_channels = inference_args.latent_channels

        model = load_model(
            inference_args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )
        model = model.to(self.device)
        model = self._load_transformer_state_dict(inference_args, model, inference_args.model_path)
        if inference_args.enable_torch_compile:
            model = torch.compile(model)
        model.eval()
        return model
    
    def load_vae(self, inference_args: InferenceArgs):
        """Custom VAE loading for Hunyuan"""
        # TODO(will): replace this with abstracted model
        use_v1_loader = True
        if use_v1_loader:
            return self.load_vae_v1(inference_args)
        else:
            return self.load_vae_v0(inference_args)
        
    def load_vae_v1(self, inference_args: InferenceArgs):
        """Custom VAE loading for Hunyuan"""
        path = "data/hunyuanvideo_community/vae"
        path = Path(path)
        # TODO(PY/Will): remove this hardcode
        # Load config file
        config_path = path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Clean up config

        class_name = config.pop("_class_name")
        if "_diffusers_version" in config:
            config.pop("_diffusers_version")
        vae = AutoencoderKLHunyuanVideo(**config).to(self.device)
        
        # Find all safetensors files
        safetensors_list = glob.glob(os.path.join(str(path), "*.safetensors"))
        # TODO(PY)
        assert len(safetensors_list) == 1, f"Found {len(safetensors_list)} safetensors files in {path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded)
        dtype = PRECISION_TO_TYPE[inference_args.vae_precision]
        vae = vae.eval().to(dtype)
        
        return vae, {"s_ratio": config["spatial_compression_ratio"], "t_ratio": config["temporal_compression_ratio"]}
        
        
        
        
    def load_vae_v0(self, inference_args: InferenceArgs):
        """Custom VAE loading for Hunyuan"""
        # TODO(will): replace this with abstracted model
        from fastvideo.v1.models.hunyuan.vae import load_vae
        vae, _, s_ratio, t_ratio = load_vae(
            inference_args.vae,
            inference_args.vae_precision,
            logger=logger,
            device=self.device
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
        return vae, vae_kwargs

    def load_text_encoders(self, inference_args: InferenceArgs) -> Tuple["TextEncoder", Optional["TextEncoder"]]:
        
        # Text encoder
        if inference_args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[inference_args.prompt_template_video].get("crop_start", 0)
        elif inference_args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[inference_args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = inference_args.text_len + crop_start

        # prompt_template
        prompt_template = (PROMPT_TEMPLATE[inference_args.prompt_template] if inference_args.prompt_template is not None else None)

        # prompt_template_video
        prompt_template_video = (PROMPT_TEMPLATE[inference_args.prompt_template_video]
                                 if inference_args.prompt_template_video is not None else None)

        text_encoder = TextEncoder(
            text_encoder_type=inference_args.text_encoder,
            max_length=max_length,
            text_encoder_precision=inference_args.text_encoder_precision,
            tokenizer_type=inference_args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=inference_args.hidden_state_skip_layer,
            apply_final_norm=inference_args.apply_final_norm,
            reproduce=inference_args.reproduce,
            logger=logger,
            device=self.device if not inference_args.use_cpu_offload else "cpu",
        )
        text_encoder_2 = None
        if inference_args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=inference_args.text_encoder_2,
                max_length=inference_args.text_len_2,
                text_encoder_precision=inference_args.text_encoder_precision_2,
                tokenizer_type=inference_args.tokenizer_2,
                reproduce=inference_args.reproduce,
                logger=logger,
                device=self.device if not inference_args.use_cpu_offload else "cpu",
            )
            
        return [text_encoder, text_encoder_2]
