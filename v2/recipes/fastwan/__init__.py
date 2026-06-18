"""FastWan (DMD few-step Wan) — Distribution-Matching-Distillation ported into the v2 substrate.

Self-contained recipe package (bucket-C pattern): the architecture is plain Wan
(``WanTransformer3DModel`` / ``AutoencoderKLWan`` / UMT5), so the Wan torch adapters are reused via
``load_id`` (no new adapter); the ONLY new piece is the DMD few-step schedule in ``FastWanDMDLoop``
(predict-x0-then-renoise over a fixed ``dmd_denoising_steps`` list). Reuses the Wan VAE/T5 adapters +
``stamp_wan21_checkpoints``.

Loadable: ``FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`` (full attention, no VSA).
BRINGUP (VSA kernel + non-strict ``to_gate_compress`` load): the other 3 FastWan ids — same cards.

``build_fastwan_card`` is an alias for the primary FullAttn card builder.
"""
from __future__ import annotations

from v2.recipes.fastwan.card import (
    build_fastwan_t2v_1_3b_card,
    build_fastwan_ti2v_5b_card,
)
from v2.recipes.fastwan.loop import FastWanDMDLoop
from v2.recipes.fastwan.program import build_fastwan_program

# Primary (loadable) card builder alias for the orchestrator/registry.
build_fastwan_card = build_fastwan_ti2v_5b_card

__all__ = [
    "build_fastwan_card",
    "build_fastwan_ti2v_5b_card",
    "build_fastwan_t2v_1_3b_card",
    "build_fastwan_program",
    "FastWanDMDLoop",
]
