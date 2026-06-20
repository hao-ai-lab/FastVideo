"""BAGEL/lance program: one resident MoT instance, two runtime-visible loops.

    tokenize → generate_text (ar_decode) → pack(text→cond) → generate_image (diffusion) → vae_decode

generate_text and generate_image hit the SAME resident weights; their steps are WorkUnits the
scheduler interleaves.
"""
from __future__ import annotations

from v2.core.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.omni import emit_text_node, pack_cond_from_tokens, tokenize_node, vae_decode_node


def build_bagel_program() -> Program:
    return Program(
        program_id="bagel.mot",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("tokenize", fn=tokenize_node, writes=("prompt_tokens", )),
            ModelLoopNode("generate_text",
                          loop_id="generate_text",
                          output_slot="gen_text_out",
                          reads=("prompt_tokens", ),
                          writes=("gen_text_out", )),
            ComponentNode("emit_text",
                          fn=emit_text_node("gen_text_out", "text"),
                          reads=("gen_text_out", ),
                          writes=("text", )),
            ComponentNode("pack",
                          fn=pack_cond_from_tokens("gen_text_out"),
                          reads=("gen_text_out", ),
                          writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("generate_image",
                          loop_id="generate_image",
                          output_slot="gen_image_out",
                          reads=("text_embeds", ),
                          writes=("gen_image_out", )),
            ComponentNode("vae_decode",
                          fn=vae_decode_node("gen_image_out", "image"),
                          reads=("gen_image_out", ),
                          writes=("image", )),
        ],
        output_artifacts={
            "text": "text",
            "image": "image"
        },
    ).validate()
