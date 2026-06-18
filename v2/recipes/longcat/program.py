"""LongCat-Video T2V program: text_encode -> diffusion_denoise (flow-match + CFG-zero) -> vae_decode.

Same inline shape as the Wan/Cosmos t2v programs — the LongCat specifics (the linspace sigma schedule, the
CFG-zero combine, and the negated-velocity sign convention) live entirely in ``LongCatDenoiseLoop`` and the
``LongCatDiT`` adapter, so the node graph is unchanged.

(i2v / video-continuation would add a VAE-encode node writing the first-frame conditioning latent into slots
and a loop that threads ``num_cond_latents`` + per-frame timestep masking — deferred; spec blocker #4/#5.)
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_longcat_program() -> Program:
    return Program(
        program_id="longcat.t2v.inline",
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
