"""Wan2.1-Fun-Control reuses the canonical Wan denoise loop verbatim.

Fun-Control is architecturally a plain Wan2.1 flow-match DiT; the only delta from T2V is the DiT input
channel count (the control latent is concatenated onto the noise latent, see the program + the ``WanDiT``
adapter). The denoise control flow — flow-shift schedule, CFG, precision policy, the flow-match Euler step
— is identical to ``v2.recipes.wan21``, so we just re-export the shared ``WanDenoiseLoop``. It already
threads an optional ``i2v_cond`` slot through to the adapter's ``cond=`` kwarg; the Fun-Control program
reuses that exact slot to carry the control-video ``[control_latent | zero_pad]`` concat. Keeping a module
here (rather than importing ``WanDenoiseLoop`` in the card) keeps the recipe self-contained and gives one
place to fork the loop should Fun-Control ever need bespoke step math.
"""
from __future__ import annotations

# Re-export the canonical Wan loop + latent-geometry helper. ``WanDenoiseLoop`` is reused unchanged; the
# Fun-Control conditioning rides on its existing i2v ``cond=`` thread (program writes the control concat
# into the ``i2v_cond`` slot). ``latent_shape`` gives the Wan2.1 16/8/4 geometry.
from v2.recipes.wan21.loop import WanDenoiseLoop, latent_shape

# Wan2.1 VAE (AutoencoderKLWan): z_dim=16, 4x temporal, 8x spatial — the geometry the control latent and
# the zero-pad both inherit (so the concat is channel-aligned with the noise latent).
WAN_LATENT_CHANNELS = 16
WAN_TEMPORAL_RATIO = 4
WAN_SPATIAL_RATIO = 8

__all__ = ["WanDenoiseLoop", "latent_shape", "WAN_LATENT_CHANNELS", "WAN_TEMPORAL_RATIO", "WAN_SPATIAL_RATIO"]
