from dataclasses import dataclass, field
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig


@dataclass
class MatrixGameWanVideoArchConfig(WanVideoArchConfig):
    action_config: dict = field(
        default_factory=lambda: {
            "mouse_dim_in": 2,
            "keyboard_dim_in": 6,
            "hidden_size": 128,
            "img_hidden_size": 1536,
            "keyboard_hidden_dim": 1024,
            "mouse_hidden_dim": 1024,
            "vae_time_compression_ratio": 4,
            "windows_size": 3,
            "heads_num": 12,
            "patch_size": [1, 2, 2],
            "rope_dim_list": [8, 28, 28],
            "mouse_qk_dim_list": [8, 28, 28],
            "enable_mouse": True,
            "enable_keyboard": True,
            "blocks": [29]  # Action Module Index
        })


@dataclass
class MatrixGameWanVideoConfig(WanVideoConfig):
    arch_config: MatrixGameWanVideoArchConfig = field(
        default_factory=MatrixGameWanVideoArchConfig)
    prefix: str = "Wan"
