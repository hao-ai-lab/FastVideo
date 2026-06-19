"""GEN3C-Cosmos-7B — camera-controlled video diffusion (EDM denoiser) ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapters via
``ComponentSpec.adapter`` (``Gen3CDiT`` / ``Gen3CVAE`` / ``Gen3CT5Encoder`` in
``v2/recipes/gen3c/adapter.py``) and a new ``Gen3CDenoiseLoop`` (a TRUE EDM /
``EDMEulerScheduler`` denoiser: Karras ρ=7 σ schedule, ``c_in/c_skip/c_out`` preconditioning, model
timestep ``c_noise = 0.25·log σ``, frame-replace conditioning + pose-zeroed-uncond CFG). The
camera/MoGe-depth/3D-cache conditioning that fills the pose buffer is documented as BRINGUP (a pre-loop
stage needing a CUDA renderer + request-API extension); the registered card is the degenerate t2v path.
The orchestrator adds the registry entry (HF id ``FastVideo/GEN3C-Cosmos-7B-Diffusers``).
"""
from __future__ import annotations

from v2.recipes.gen3c.card import GEN3C_NEG, build_gen3c_card
from v2.recipes.gen3c.loop import Gen3CDenoiseLoop
from v2.recipes.gen3c.program import build_gen3c_program

__all__ = ["build_gen3c_card", "build_gen3c_program", "Gen3CDenoiseLoop", "GEN3C_NEG"]
