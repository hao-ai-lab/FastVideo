"""Cosmos-Predict2.5 t2v program: text_encode -> diffusion_denoise (flow-match) -> vae_decode.

Same inline shape as the Wan / Cosmos2 t2v programs — the Cosmos2.5 specifics (per-frame plain-sigma
timestep, the mandatory masks + fps, the Reason1 100352-dim embedding) live entirely in
``Cosmos25DenoiseLoop`` and the ``torch_cosmos25`` adapters, so the node graph is unchanged. fps is read
by the loop from the request (``req.diffusion.fps``, default 24) into scratch and threaded to the DiT
adapter. (video2world/image2world conditioning would add a VAE-encode node writing
``conditioning_latents``/``cond_indicator`` into slots; the loop already reads them — BRINGUP.)
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_cosmos25_program() -> Program:
    return Program(
        program_id="cosmos25.t2v.inline",
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
