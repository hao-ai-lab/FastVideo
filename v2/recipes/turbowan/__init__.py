"""TurboWan — rCM (Reparameterized Consistency Model) few-step distilled Wan, ported into v2.

Self-contained recipe package (bucket-C): the Wan architecture is reused unchanged (no new torch adapter —
the cards carry the same Wan ``load_id`` strings, built by the existing ``torch_backend`` dispatch). The
only new work is the rCM sampler/loop (``sampler.py`` + ``loop.py``), a faithful port of
``fastvideo/models/schedulers/scheduling_rcm.py:RCMScheduler`` + the TurboDiffusion denoise stage.

Builders:
  * ``build_turbowan_card`` / ``build_turbowan_program`` — T2V; serves TurboWan2.1-T2V-1.3B and -14B.
  * ``build_turbowan_i2v_a14b_card`` / ``build_turbowan_i2v_program`` — MoE i2v; serves TurboWan2.2-I2V-A14B.
"""
from __future__ import annotations

from v2.recipes.turbowan.card import build_turbowan_card, build_turbowan_i2v_a14b_card
from v2.recipes.turbowan.loop import TurboWanDenoiseLoop
from v2.recipes.turbowan.program import build_turbowan_i2v_program, build_turbowan_program
from v2.recipes.turbowan.sampler import build_rcm_sigmas, rcm_step

__all__ = [
    "build_turbowan_card",
    "build_turbowan_i2v_a14b_card",
    "build_turbowan_program",
    "build_turbowan_i2v_program",
    "TurboWanDenoiseLoop",
    "build_rcm_sigmas",
    "rcm_step",
]
