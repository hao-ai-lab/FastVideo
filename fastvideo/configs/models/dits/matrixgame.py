from dataclasses import dataclass, field
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig


@dataclass
class MatrixGameWanVideoArchConfig(WanVideoArchConfig):
    # Override param_names_mapping to remove patch_embedding transformation
    # because MatrixGame checkpoints already have patch_embedding.proj format
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(?!proj\.)(.*)$":
            r"patch_embedding.proj.\1",
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$":
            r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$":
            r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$":
            r"condition_embedder.time_modulation.linear.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$":
            r"condition_embedder.image_embedder.ff.fc_in.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$":
            r"condition_embedder.image_embedder.ff.fc_out.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$":
            r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$":
            r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$":
            r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
            r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$":
            r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$":
            r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$":
            r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$":
            r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm2\.(.*)$":
            r"blocks.\1.self_attn_residual_norm.norm.\2",
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


@dataclass
class MatrixGameWanVideoConfig(WanVideoConfig):
    arch_config: MatrixGameWanVideoArchConfig = field(
        default_factory=MatrixGameWanVideoArchConfig)
    prefix: str = "Wan"
