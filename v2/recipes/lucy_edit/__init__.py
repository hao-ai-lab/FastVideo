"""Lucy-Edit — Wan VIDEO-TO-VIDEO editor (decart-ai/Lucy-Edit-Dev / Lucy-Edit-1.1-Dev).

Self-contained recipe package (bucket-C). Lucy-Edit is the Wan2.2-5B network run as a prompt-driven
video editor: it reuses the Wan architecture (``WanTransformer3DModel`` / ``AutoencoderKLWan`` / UMT5)
with NO custom adapter, adding only (a) a ``video_vae_encode`` program node (the v2 analogue of
``VideoVAEEncodingStage``) that VAE-encodes the INPUT video into a conditioning latent and (b) v2v
conditioning in the loop — the conditioning latent is channel-concatenated with the noise latent
(96 = 48 + 48) via the shared ``WanDenoiseLoop`` ``i2v_cond`` hook (NO mask, NO CLIP image encoder).

Input-video plumbing is BRINGUP: the node reads a ``VideoPart`` when present and degrades to t2v
otherwise. Registered in ``v2/registry.py`` (both Lucy ids -> this card + program).
"""
from __future__ import annotations

from v2.recipes.lucy_edit.card import LUCY_NEG, build_lucy_edit_card
from v2.recipes.lucy_edit.loop import WanDenoiseLoop, latent_shape
from v2.recipes.lucy_edit.program import build_lucy_edit_program

__all__ = ["build_lucy_edit_card", "build_lucy_edit_program", "LUCY_NEG", "WanDenoiseLoop", "latent_shape"]
