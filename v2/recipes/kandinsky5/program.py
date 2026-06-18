"""Kandinsky-5 T2V program: encode(Qwen + CLIP) → diffusion_denoise → vae_decode.

Same inline shape as the Wan/Cosmos t2v programs, but the encode node fans out to BOTH text encoders
(the dual conditioning stream Kandinsky needs): Qwen2.5-VL token embeds for the cross-attended text and
the CLIP **pooled** vector for ``pooled_projections``. The denoise loop reads all four slots
(prompt/neg × qwen/clip). The vae-decode node hands the channels-last latent straight to the
``Kandinsky5VAE`` adapter, which does the ``[B,T,H,W,C] -> [B,C,T,H,W]`` reshape + scalar
``/scaling_factor`` un-normalization internally (faithful to the diffusers pipeline post-processing).
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import cached_text_encode


def _encode(instance, slots, request, ctx) -> None:
    # Qwen token embeds (cross-attended text) — cached by content hash like every text-encode node.
    slots["text_embeds"] = cached_text_encode(instance, request.prompt())
    slots["neg_text_embeds"] = cached_text_encode(instance, request.diffusion.negative_prompt)
    # CLIP pooled vectors (the mandatory pooled_projections). Encoded directly (the feature cache keys on
    # component_id "text_encoder"; the CLIP component is a distinct instance so we call it explicitly).
    clip = instance.component("text_encoder_2")
    slots["clip_pooled"] = clip.encode(request.prompt())
    slots["neg_clip_pooled"] = clip.encode(request.diffusion.negative_prompt)


def _vae_decode(instance, slots, request, ctx) -> None:
    # The channels-last latent [T,H,W,C] (loop output) -> the Kandinsky5VAE adapter reshapes to
    # channels-first and applies the scalar scaling_factor un-normalization before decoding.
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_kandinsky5_program() -> Program:
    return Program(
        program_id="kandinsky5.t2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode",
                          fn=_encode,
                          writes=("text_embeds", "neg_text_embeds", "clip_pooled", "neg_clip_pooled")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds", "clip_pooled", "neg_clip_pooled"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
