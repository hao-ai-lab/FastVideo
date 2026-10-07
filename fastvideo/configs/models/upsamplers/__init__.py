from fastvideo.configs.models.upsamplers.hunyuan15 import SRTo720pUpsamplerConfig, SRTo1080pUpsamplerConfig
from fastvideo.configs.models.upsamplers.kandinsky6_sr import (Kandinsky6SRLatentUpscalerConfig,
                                                               Kandinsky6SRLatentUpscalerEntryConfig)
from fastvideo.configs.models.upsamplers.base import UpsamplerConfig

__all__ = [
    "SRTo720pUpsamplerConfig", "SRTo1080pUpsamplerConfig", "Kandinsky6SRLatentUpscalerConfig",
    "Kandinsky6SRLatentUpscalerEntryConfig", "UpsamplerConfig"
]
