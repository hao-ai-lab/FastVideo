"""FLUX.2 t2i program: text_encode → diffusion_denoise → vae_decode → image.

Same inline shape as the Wan t2v program (text_encode → denoise → vae_decode); the FLUX.2 specifics live
entirely in ``Flux2DenoiseLoop`` (BFL empirical-mu schedule + packed geometry + single embedded-guidance
forward) and the ``Flux2DiT``/``Flux2VAE``/``Flux2TextEncoder`` adapters. The reads only ``text_embeds``
(no ``neg_text_embeds``): FLUX.2-dev uses embedded guidance and klein uses none — there is no CFG uncond
branch. The output artifact is an ``image`` (the loop produces a single-frame ``T==1`` packed latent which
the VAE unpacks + decodes to pixels).
"""
from __future__ import annotations

from v2.core.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import cached_text_encode


def _text_encode(instance, slots, request, ctx) -> None:
    """FLUX.2 conditions on a single chat-template multi-layer embedding — no negative prompt
    (embedded guidance / no CFG). Reuses the content-hash feature cache via ``cached_text_encode``."""
    slots["text_embeds"] = cached_text_encode(instance, request.prompt())


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["image"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_flux2_program() -> Program:
    return Program(
        program_id="flux2.t2i.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", )),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", ),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("image", )),
        ],
        output_artifacts={
            "image": "image",
            "latents": "denoise_out"
        },
    ).validate()
