"""FastWan (DMD) t2v program: text_encode -> diffusion_denoise (DMD few-step) -> vae_decode.

Same inline node graph as the Wan t2v program — the DMD specifics live entirely in
``FastWanDMDLoop`` (the in-package loop), so the program is unchanged. (i2v conditioning for the TI2V
variant would add image-encode / first-frame-VAE nodes writing ``i2v_img_embeds`` / ``i2v_cond`` into
slots, exactly like ``v2/recipes/wan21/i2v.py``; the loop already reads them. The registered preset is
t2v, so the base program covers it.)
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_fastwan_program() -> Program:
    return Program(
        program_id="fastwan.dmd.t2v.inline",
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
