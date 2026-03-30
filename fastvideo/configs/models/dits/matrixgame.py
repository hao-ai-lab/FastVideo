from dataclasses import dataclass, field

import torch
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig


@dataclass
class MatrixGame2WanVideoArchConfig(WanVideoArchConfig):
    # Override param_names_mapping to remove patch_embedding transformation
    # because MatrixGame checkpoints already have patch_embedding.proj format
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(?!proj\.)(.*)$": r"patch_embedding.proj.\1",
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$":
            r"condition_embedder.image_embedder.ff.fc_in.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$":
            r"condition_embedder.image_embedder.ff.fc_out.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
        })

    action_config: dict = field(
        default_factory=lambda: {
            "blocks": list(range(15)),
            "enable_mouse": True,
            "enable_keyboard": True,
            "heads_num": 16,
            "hidden_size": 128,
            "img_hidden_size": 1536,
            "keyboard_dim_in": 4,
            "keyboard_hidden_dim": 1024,
            "mouse_dim_in": 2,
            "mouse_hidden_dim": 1024,
            "mouse_qk_dim_list": [8, 28, 28],
            "patch_size": [1, 2, 2],
            "qk_norm": True,
            "qkv_bias": False,
            "rope_dim_list": [8, 28, 28],
            "rope_theta": 256,
            "vae_time_compression_ratio": 4,
            "windows_size": 3,
        })

    local_attn_size: int = -1
    sink_size: int = 0
    num_frames_per_block: int = 3
    text_len: int = 512
    text_dim: int = 0
    image_dim: int = 1280


def _is_transformer_block(param_name: str, module: torch.nn.Module) -> bool:
    return bool("blocks" in param_name and param_name.split(".")[-1].isdigit())


@dataclass
class MatrixGame2WanVideoConfig(WanVideoConfig):
    arch_config: MatrixGame2WanVideoArchConfig = field(default_factory=MatrixGame2WanVideoArchConfig)
    prefix: str = "Wan"
    _compile_conditions: list = field(default_factory=lambda: [_is_transformer_block])


@dataclass
class MatrixGame3WanVideoArchConfig(WanVideoArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(weight|bias)$": r"patch_embedding.proj.\1",
            r"^patch_embedding_wancamctrl\.(.*)$": r"camera_patch_embedding.proj.\1",
            r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^head\.head\.(.*)$": r"proj_out.\1",
            r"^head\.modulation$": r"scale_shift_table",
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
            r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
            r"^patch_embedding\.(?!proj\.)(.*)$": r"patch_embedding.proj.\1",
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
        })
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 48
    out_channels: int = 48
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    ffn_dim: int = 13824
    num_layers: int = 40
    text_len: int = 512
    image_dim: int = 0
    use_text_crossattn: bool = True
    use_memory: bool = True
    sigma_theta: float = 0.8
    camera_embed_in_channels: int = 1536
    action_config: dict = field(
        default_factory=lambda: {
            "blocks": list(range(40)),
            "enable_mouse": True,
            "enable_keyboard": True,
            "heads_num": 16,
            "hidden_size": 128,
            "img_hidden_size": 5120,
            "keyboard_dim_in": 4,
            "keyboard_hidden_dim": 1024,
            "mouse_dim_in": 2,
            "mouse_hidden_dim": 1024,
            "mouse_qk_dim_list": [8, 28, 28],
            "patch_size": [1, 2, 2],
            "qk_norm": True,
            "qkv_bias": False,
            "rope_dim_list": [8, 28, 28],
            "rope_theta": 256,
            "vae_time_compression_ratio": 4,
            "windows_size": 3,
        })


@dataclass
class MatrixGame3WanVideoConfig(WanVideoConfig):
    arch_config: MatrixGame3WanVideoArchConfig = field(default_factory=MatrixGame3WanVideoArchConfig)
    prefix: str = "Wan"
    _compile_conditions: list = field(default_factory=lambda: [_is_transformer_block])
