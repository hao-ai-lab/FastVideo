from fastvideo.v1.configs.models.vaes.base import VAEConfig, VAEArchConfig
from fastvideo.v1.configs.models.vaes.hunyuanvae import HunyuanVAEConfig, HunyuanVAEArchConfig
from fastvideo.v1.configs.models.vaes.wanvae import WanVAEConfig, WanVAEArchConfig

__all__ = [
    "VAEConfig", "VAEArchConfig",
    "HunyuanVAEConfig", "HunyuanVAEArchConfig", 
    "WanVAEConfig", "WanVAEArchConfig"
]