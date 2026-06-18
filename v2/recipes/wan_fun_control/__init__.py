"""Wan2.1-Fun-Control — control-video-conditioned Wan2.1 (V2V) ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): a thin recipe over the SHARED Wan architecture.
The control video is VAE-encoded and concatenated onto the noise latent
(``cat([noise(16), control(16), zero_pad(16)])`` → the 48ch Fun-Control DiT input), carried through the
existing i2v ``cond=`` thread of the shared ``WanDenoiseLoop`` / ``WanDiT`` adapter — so NO new DiT/VAE/T5
adapter and NO new sampler are needed. ``build_wan_fun_control_card`` + ``build_wan_fun_control_program``
are registered in ``v2/registry.py`` for ``IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers``.
"""
from __future__ import annotations

from v2.recipes.wan_fun_control.card import build_wan_fun_control_card
from v2.recipes.wan_fun_control.loop import WanDenoiseLoop, latent_shape
from v2.recipes.wan_fun_control.program import build_wan_fun_control_program

__all__ = ["build_wan_fun_control_card", "build_wan_fun_control_program", "WanDenoiseLoop", "latent_shape"]
