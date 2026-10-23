# SPDX-License-Identifier: Apache-2.0
"""Pipeline config of the Kandinsky6 video super-resolution (VSR) pipeline.

Video -> video with no text encoder (the SR DiT is text-free), so the text-encoder lists are empty.  The bundle's
``scheduler`` component drives the denoising loop and the transformer config's ``sr_params`` hold the noise level and
RoPE scale; per-request knobs live in ``Kandinsky6SROptions`` and travel through ``ForwardBatch.extra``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from fastvideo.configs.models import DiTConfig, UpsamplerConfig, VAEConfig
from fastvideo.configs.models.dits.kandinsky6_sr import Kandinsky6SRConfig
from fastvideo.configs.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerConfig
from fastvideo.configs.models.vaes.kandinsky6_sr import Kandinsky6SRVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class Kandinsky6SRPipelineConfig(PipelineConfig):
    """Kandinsky6 SR: source video -> KVAE encode -> latent upscaler -> tiled SR DiT -> KVAE decode -> stitch."""

    # Read by ``VideoGenerator`` (plain class attributes, deliberately not dataclass fields):
    #   prompt_optional          -- a call without a text prompt is valid (V2V; the prompt is never used);
    #   output_shape_from_input  -- the output geometry comes from the input video, so ``VideoGenerator`` must not
    #                               pre-allocate the return buffer from the request's height / width / num_frames.
    prompt_optional: ClassVar[bool] = True
    output_shape_from_input: ClassVar[bool] = True

    dit_config: DiTConfig = field(default_factory=Kandinsky6SRConfig)
    vae_config: VAEConfig = field(default_factory=Kandinsky6SRVAEConfig)
    upsampler_config: UpsamplerConfig = field(default_factory=Kandinsky6SRLatentUpscalerConfig)

    # KVAE, latent upscaler and DiT run in bf16 (the latent state and the pi-Flow / Euler update maths stay fp32).
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    upsampler_precision: str = "bf16"

    # The KVAE tiles in time itself (segment cache) and the pipeline tiles in space: no generic tiled-VAE decode.
    vae_tiling: bool = False
    vae_sp: bool = False

    # No text encoders (see module docstring); ``check_pipeline_config`` wants the lists to have equal lengths.
    text_encoder_configs: tuple = field(default_factory=tuple)
    text_encoder_precisions: tuple = field(default_factory=tuple)
    preprocess_text_funcs: tuple = field(default_factory=tuple)
    postprocess_text_funcs: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # The SR pipeline always encodes (LU path) and decodes with the KVAE.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
