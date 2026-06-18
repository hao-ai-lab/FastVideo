"""FLUX.2 (Black Forest Labs) — dual-stream MMDiT text→image ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapters via
``ComponentSpec.adapter`` (``Flux2DiT``/``Flux2VAE``/``Flux2Mistral3Encoder``/``Flux2Qwen3Encoder`` in
``v2/platform/backends/torch_flux2.py``) and a NEW ``Flux2DenoiseLoop`` (the BFL empirical-mu flow-match
schedule + packed 2×2 latent geometry + single embedded-guidance forward), reusing ``stamp_wan21_checkpoints``
for the diffusers transformer/vae/text_encoder subfolder layout. ``build_flux2_card`` is FLUX.2-dev
(Mistral3, embedded guidance); ``build_flux2_klein_card`` is the distilled 4-step klein variants (Qwen3).
GATED weights → GPU is BRINGUP; the toy factories CPU-verify the recipe end-to-end.
"""
from __future__ import annotations

from v2.recipes.flux2.card import build_flux2_card, build_flux2_klein_card
from v2.recipes.flux2.loop import Flux2DenoiseLoop
from v2.recipes.flux2.program import build_flux2_program

__all__ = ["build_flux2_card", "build_flux2_klein_card", "build_flux2_program", "Flux2DenoiseLoop"]
