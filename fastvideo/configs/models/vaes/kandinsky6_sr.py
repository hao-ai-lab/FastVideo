# SPDX-License-Identifier: Apache-2.0
"""Config of the Kandinsky6 SR causal video KVAE (``Kandinsky6SRVAE``).

``vae/config.json`` comes in two layouts: an older one nesting the KVAE architecture under ``encoder_config`` /
``decoder_config`` dicts (the config stored the string ``"None"`` for their unused channel fields), and the current
one with those same knobs flattened to the top level instead -- ``decoder_ch`` / ``decoder_ch_mult`` are the only decoder-specific overrides, everything else
(``num_res_blocks``, ``z_channels``, the temporal-compression / norm / padding knobs) is shared between encoder and
decoder. ``update_model_arch`` copies whichever layout the checkpoint has; ``__post_init__`` synthesizes
``encoder_config`` / ``decoder_config`` from the flat fields when the nested ones weren't populated, so
``Kandinsky6SRVAE.__init__`` (and its ``_arch_kwargs`` validation) only ever has to handle the nested shape.
``scaling_factor`` multiplies the raw KVAE latent to get the latent space the SR DiT / latent upscaler work in.
"""
from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Kandinsky6SRVAEArchConfig(VAEArchConfig):
    vae_type: str = "video-kvae"
    encoder_config: dict[str, Any] = field(default_factory=dict)
    decoder_config: dict[str, Any] = field(default_factory=dict)
    scaling_factor: float = 1.0
    spatial_factor: int = 16
    temporal_factor: int = 4

    # Flat top-level KVAE architecture fields (current official vae/config.json layout -- see module
    # docstring). Left at their None default (and ignored) when the checkpoint instead uses the older
    # nested encoder_config/decoder_config layout.
    in_channels: int | None = None
    out_channels: int | None = None
    z_channels: int | None = None
    ch: int | None = None
    ch_mult: list[float] | None = None
    decoder_ch: int | None = None
    decoder_ch_mult: list[float] | None = None
    num_res_blocks: int | None = None
    padding_mode: str | None = None
    temporal_compress_times: int | None = None
    temporal_compress_start_level: int | None = None
    norm_type: str | None = None
    double_z: bool | None = None
    downsample_version: int | None = None
    fix_pxs: bool | None = None

    def __post_init__(self) -> None:
        if self.vae_type != "video-kvae":
            raise ValueError(f"Kandinsky6SRVAE supports only 'video-kvae', got {self.vae_type!r}")
        self.spatial_compression_ratio = int(self.spatial_factor)
        self.temporal_compression_ratio = int(self.temporal_factor)

        if not self.encoder_config and not self.decoder_config and self.ch is not None:
            shared = dict(num_res_blocks=self.num_res_blocks,
                          z_channels=self.z_channels,
                          temporal_compress_times=self.temporal_compress_times,
                          temporal_compress_start_level=self.temporal_compress_start_level,
                          norm_type=self.norm_type,
                          padding_mode=self.padding_mode)
            self.encoder_config = {
                **shared, "ch": self.ch,
                "ch_mult": self.ch_mult,
                "in_channels": self.in_channels,
                "double_z": self.double_z,
                "downsample_version": self.downsample_version,
                "fix_pxs": self.fix_pxs
            }
            self.decoder_config = {
                **shared, "ch": self.decoder_ch,
                "ch_mult": self.decoder_ch_mult,
                "out_ch": self.out_channels
            }


@dataclass
class Kandinsky6SRVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=Kandinsky6SRVAEArchConfig)
    # The KVAE has its own segment-cached encode / decode; the generic tiled-VAE knobs do not apply.
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
