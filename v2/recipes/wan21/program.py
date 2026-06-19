"""Wan2.1 T2V program: text_encode → diffusion_denoise → vae_decode.

The text-encode node uses the content-hash feature cache: the same prompt across requests (e.g. a
K=24 RL group) encodes once and reuses.
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    latents = slots["denoise_out"]["latents"]
    slots["video"] = instance.component("vae").decode(latents)


def build_wan_t2v_program() -> Program:
    return Program(
        program_id="wan.t2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
