# SPDX-License-Identifier: Apache-2.0
"""
TurboDiffusion Video Pipeline Implementation.

This module contains an implementation of the TurboDiffusion video diffusion pipeline
for 1-4 step video generation using rCM (recurrent Consistency Model) sampling
with SLA (Sparse-Linear Attention).
"""

import os
import re
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_rcm import RCMScheduler
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        DenoisingStage, InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)

# Weight mapping from TurboDiffusion -> FastVideo
TURBODIFFUSION_WEIGHT_MAPPING = {
    # Self attention
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
    # SLA proj_l weights
    r"^blocks\.(\d+)\.self_attn\.attn_op\.local_attn\.proj_l\.(.*)$": r"blocks.\1.attn1.attn_impl.proj_l.\2",
    # Cross attention
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    # Norms and FFN
    r"^blocks\.(\d+)\.norm1\.(.*)$": r"blocks.\1.norm1.\2",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
    r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
    # Embeddings
    r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
    # Head
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^head\.norm\.(.*)$": r"norm_out.\1",
    r"^head\.modulation$": r"scale_shift_table",
}


def load_turbodiffusion_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load TurboDiffusion checkpoint weights into a FastVideo model.
    
    Args:
        model: FastVideo WanTransformer3DModel
        checkpoint_path: Path to TurboDiffusion .pth checkpoint
        
    Returns:
        Model with TurboDiffusion weights loaded
    """
    logger.info(f"Loading TurboDiffusion weights from {checkpoint_path}")
    
    # Load checkpoint
    turbo_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Get model state dict for shape verification
    fv_model_state = model.state_dict()
    
    # Map weights
    loaded_weights = {}
    sla_count = 0
    
    for turbo_key, turbo_tensor in turbo_state_dict.items():
        fv_key = turbo_key
        
        # Apply mapping
        for pattern, replacement in TURBODIFFUSION_WEIGHT_MAPPING.items():
            if re.match(pattern, turbo_key):
                fv_key = re.sub(pattern, replacement, turbo_key)
                break
        
        # Handle patch embedding reshape: TurboDiff [1536, 64] -> FV [1536, 16, 1, 2, 2]
        if "patch_embedding" in fv_key and fv_key in fv_model_state:
            target_shape = fv_model_state[fv_key].shape
            if turbo_tensor.shape != target_shape:
                turbo_tensor = turbo_tensor.view(target_shape)
        
        # Check if key exists and shapes match
        if fv_key in fv_model_state:
            if turbo_tensor.shape == fv_model_state[fv_key].shape:
                loaded_weights[fv_key] = turbo_tensor
                if 'proj_l' in fv_key:
                    sla_count += 1
    
    logger.info(f"Mapped {len(loaded_weights)} weights (including {sla_count} SLA proj_l weights)")
    
    # Load weights with DTensor compatibility
    loaded_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in loaded_weights:
                target_dtype = param.dtype
                target_device = param.device
                new_weight = loaded_weights[name].to(target_device, dtype=target_dtype)
                
                # Handle DTensor by accessing the local tensor
                if hasattr(param, '_local_tensor'):
                    param._local_tensor.copy_(new_weight)
                else:
                    param.copy_(new_weight)
                loaded_count += 1
    
    logger.info(f"Successfully loaded {loaded_count} weights into FastVideo model")
    return model


class TurboDiffusionPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    TurboDiffusion video pipeline for 1-4 step generation.
    
    Uses RCM scheduler and SLA attention for fast, high-quality video generation.
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Use RCM scheduler for TurboDiffusion
        logger.info("Initializing RCM scheduler for TurboDiffusion")
        self.modules["scheduler"] = RCMScheduler(sigma_max=80.0)
        
        # Store checkpoint path for later loading
        self._turbodiffusion_checkpoint = getattr(
            fastvideo_args, 'turbodiffusion_checkpoint', None
        )

    def load_modules(self, fastvideo_args: FastVideoArgs, loaded_modules=None):
        """Override to load TurboDiffusion weights after transformer is loaded."""
        from fastvideo.utils import maybe_download_model
        
        # First, load all modules normally
        modules = super().load_modules(fastvideo_args, loaded_modules)
        
        # Auto-download TurboDiffusion checkpoint from HuggingFace
        turbo_model_id = "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P"
        logger.info(f"Downloading TurboDiffusion checkpoint from {turbo_model_id}...")
        turbo_path = maybe_download_model(turbo_model_id)
        checkpoint_path = os.path.join(turbo_path, "TurboWan2.1-T2V-1.3B-480P.pth")
        
        # Load TurboDiffusion weights into the transformer
        if os.path.exists(checkpoint_path):
            transformer = modules.get("transformer")
            if transformer is not None:
                load_turbodiffusion_weights(transformer, checkpoint_path)
        else:
            logger.error("TurboDiffusion checkpoint not found at '%s'", checkpoint_path)
        
        return modules

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae"),
                           pipeline=self))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae"),
                                           pipeline=self))


EntryClass = TurboDiffusionPipeline
