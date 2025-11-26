from dataclasses import dataclass, field
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig

@dataclass
class MatrixGameWanVideoArchConfig(WanVideoArchConfig):
    action_config: dict = field(
        default_factory=lambda: {
            "blocks": list(range(15)),
            "enable_mouse": True,
            "enable_keyboard": True,
            "heads_num": 16,
            "hidden_size": 128,
            "img_hidden_size": 1536,
            "keyboard_dim_in": 6,
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
class MatrixGameWanVideoConfig(WanVideoConfig):
    arch_config: MatrixGameWanVideoArchConfig = field(
        default_factory=MatrixGameWanVideoArchConfig)
    prefix: str = "Wan"
