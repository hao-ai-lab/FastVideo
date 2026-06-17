"""Tiled-decode program: text_encode → diffusion_denoise → vae_tile (design_v3 §4, §17).

The VAE decode is a ``ModelLoopNode`` (the ``vae_tile`` loop), not a one-shot ``ComponentNode`` — so its
tiles become scheduled ``VAE_TILE`` WorkUnits that interleave with other requests' ``DIFFUSION_STEP``
units. (Contrast the plain Wan program, where decode is a single in-tick node.)
"""
from __future__ import annotations

from ...program import ComponentNode, ModelLoopNode, Program, ProgramKind
from ..common import text_encode_node_fn as _text_encode


def build_tiled_program() -> Program:
    return Program(
        program_id="wan.t2v.tiled", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise", loop_id="diffusion_denoise", output_slot="denoise_out",
                          reads=("text_embeds",), writes=("denoise_out",)),
            ModelLoopNode("vae_tile", loop_id="vae_tile", output_slot="tiled_out",
                          reads=("denoise_out",), writes=("tiled_out",)),
        ],
        output_artifacts={"video": "tiled_out", "latents": "denoise_out"},
    ).validate()
