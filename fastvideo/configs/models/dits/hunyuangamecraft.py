# SPDX-License-Identifier: Apache-2.0
"""
Configuration for Hunyuan-GameCraft DiT model.

This extends the HunyuanVideo config with CameraNet support for 
camera pose conditioning via Plücker coordinates.
"""

from dataclasses import dataclass, field

import torch

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.hunyuanvideo import (
    HunyuanVideoArchConfig,
    is_double_block,
    is_single_block,
    is_refiner_block,
)


def is_camera_net(n: str, m) -> bool:
    """Check if module is part of CameraNet."""
    return "camera_net" in n


@dataclass
class HunyuanGameCraftArchConfig(HunyuanVideoArchConfig):
    """
    Architecture config for Hunyuan-GameCraft.
    
    Extends HunyuanVideo with CameraNet for camera pose conditioning.
    The model uses Plücker coordinate representation (6D) for camera poses.
    """
    
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block, is_refiner_block])
    
    _compile_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block])
    
    # Extend parent mapping with CameraNet-specific mappings
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # ==================== CameraNet mappings ====================
            # Unshuffle is a built-in module with no learned parameters
            
            # encode_first
            r"^camera_net\.encode_first\.0\.(.*)$": r"camera_net.encode_first.0.\1",
            r"^camera_net\.encode_first\.1\.(.*)$": r"camera_net.encode_first.1.\1",
            
            # encode_second  
            r"^camera_net\.encode_second\.0\.(.*)$": r"camera_net.encode_second.0.\1",
            r"^camera_net\.encode_second\.1\.(.*)$": r"camera_net.encode_second.1.\1",
            
            # final_proj
            r"^camera_net\.final_proj\.(.*)$": r"camera_net.final_proj.\1",
            
            # scale parameter
            r"^camera_net\.scale$": r"camera_net.scale",
            
            # camera_in (PatchEmbed)
            r"^camera_net\.camera_in\.proj\.(.*)$": r"camera_net.camera_in.proj.\1",
            
            # ==================== HunyuanVideo base mappings ====================
            # These are inherited from the base config but we define them here
            # to ensure proper ordering (CameraNet patterns should be tried first)
            
            # 1. context_embedder.time_text_embed submodules:
            r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_1\.(.*)$":
            r"txt_in.t_embedder.mlp.fc_in.\1",
            r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_2\.(.*)$":
            r"txt_in.t_embedder.mlp.fc_out.\1",
            r"^context_embedder\.proj_in\.(.*)$":
            r"txt_in.input_embedder.\1",
            r"^context_embedder\.time_text_embed\.text_embedder\.linear_1\.(.*)$":
            r"txt_in.c_embedder.fc_in.\1",
            r"^context_embedder\.time_text_embed\.text_embedder\.linear_2\.(.*)$":
            r"txt_in.c_embedder.fc_out.\1",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm1\.(.*)$":
            r"txt_in.refiner_blocks.\1.norm1.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm2\.(.*)$":
            r"txt_in.refiner_blocks.\1.norm2.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q\.(.*)$":
            (r"txt_in.refiner_blocks.\1.self_attn_qkv.\2", 0, 3),
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k\.(.*)$":
            (r"txt_in.refiner_blocks.\1.self_attn_qkv.\2", 1, 3),
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v\.(.*)$":
            (r"txt_in.refiner_blocks.\1.self_attn_qkv.\2", 2, 3),
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$":
            r"txt_in.refiner_blocks.\1.self_attn_proj.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$":
            r"txt_in.refiner_blocks.\1.mlp.fc_in.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$":
            r"txt_in.refiner_blocks.\1.mlp.fc_out.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm_out\.linear\.(.*)$":
            r"txt_in.refiner_blocks.\1.adaLN_modulation.linear.\2",

            # 3. x_embedder mapping:
            r"^x_embedder\.proj\.(.*)$":
            r"img_in.proj.\1",

            # 4. Top-level time_text_embed mappings:
            r"^time_text_embed\.timestep_embedder\.linear_1\.(.*)$":
            r"time_in.mlp.fc_in.\1",
            r"^time_text_embed\.timestep_embedder\.linear_2\.(.*)$":
            r"time_in.mlp.fc_out.\1",
            r"^time_text_embed\.guidance_embedder\.linear_1\.(.*)$":
            r"guidance_in.mlp.fc_in.\1",
            r"^time_text_embed\.guidance_embedder\.linear_2\.(.*)$":
            r"guidance_in.mlp.fc_out.\1",
            r"^time_text_embed\.text_embedder\.linear_1\.(.*)$":
            r"vector_in.fc_in.\1",
            r"^time_text_embed\.text_embedder\.linear_2\.(.*)$":
            r"vector_in.fc_out.\1",

            # 5. transformer_blocks (double stream) mapping:
            r"^transformer_blocks\.(\d+)\.norm1\.linear\.(.*)$":
            r"double_blocks.\1.img_mod.linear.\2",
            r"^transformer_blocks\.(\d+)\.norm1_context\.linear\.(.*)$":
            r"double_blocks.\1.txt_mod.linear.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$":
            r"double_blocks.\1.img_attn_q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$":
            r"double_blocks.\1.img_attn_k_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_q_proj\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_k_proj\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_v_proj\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$":
            r"double_blocks.\1.img_attn_proj.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.(.*)$":
            r"double_blocks.\1.txt_attn_proj.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_q\.(.*)$":
            r"double_blocks.\1.txt_attn_q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_k\.(.*)$":
            r"double_blocks.\1.txt_attn_k_norm.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.img_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.img_mlp.fc_out.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.0(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.txt_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.2(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.txt_mlp.fc_out.\2",

            # 6. single_transformer_blocks (single stream) mapping:
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$":
            r"single_blocks.\1.q_norm.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$":
            r"single_blocks.\1.k_norm.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 0, 4),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 1, 4),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 2, 4),
            r"^single_transformer_blocks\.(\d+)\.proj_mlp\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 3, 4),
            r"^single_transformer_blocks\.(\d+)\.proj_out\.(.*)$":
            r"single_blocks.\1.linear2.\2",
            r"^single_transformer_blocks\.(\d+)\.norm\.linear\.(.*)$":
            r"single_blocks.\1.modulation.linear.\2",

            # 7. Final layers mapping:
            r"^norm_out\.linear\.(.*)$":
            r"final_layer.adaLN_modulation.linear.\1",
            r"^proj_out\.(.*)$":
            r"final_layer.linear.\1",
        })
    
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    
    # GameCraft-specific parameters
    camera_in_channels: int = 6  # Plücker coordinates: 6D
    camera_downscale_coef: int = 8
    
    # HunyuanVideo architecture defaults (same as base HunyuanVideo)
    patch_size: int = 2
    patch_size_t: int = 1
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    mlp_ratio: float = 4.0
    num_layers: int = 20  # double stream blocks
    num_single_layers: int = 40  # single stream blocks
    num_refiner_layers: int = 2
    rope_axes_dim: tuple[int, int, int] = (16, 56, 56)
    guidance_embeds: bool = True  # GameCraft uses guidance embeddings
    dtype: torch.dtype | None = None
    text_embed_dim: int = 4096
    pooled_projection_dim: int = 768
    rope_theta: int = 256
    qk_norm: str = "rms_norm"
    
    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["img_in", "txt_in", "time_in", "vector_in", "camera_net"])

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size: int = self.attention_head_dim * self.num_attention_heads
        self.num_channels_latents: int = self.in_channels


@dataclass
class HunyuanGameCraftConfig(DiTConfig):
    """Top-level config for Hunyuan-GameCraft DiT."""
    arch_config: DiTArchConfig = field(default_factory=HunyuanGameCraftArchConfig)
    prefix: str = "HunyuanGameCraft"
