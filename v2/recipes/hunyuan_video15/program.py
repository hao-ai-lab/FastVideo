"""HunyuanVideo 1.5 t2v program: text_encode → diffusion_denoise → vae_decode.

Same inline shape as the Wan/Cosmos t2v programs — the arch specifics (two text embeds, the image-embed
list, the 33ch i2v concat) live entirely in the ``HunyuanVideo15DiT`` adapter, so the node graph is
unchanged except for the text-encode node, which must produce BOTH text embeddings.

HunyuanVideo 1.5 conditions on two encoders (faithful to ``HunyuanVideo15Pipeline``'s
``TextEncodingStage(text_encoders=[text_encoder, text_encoder_2])``):
  * ``text_encoder``   = Qwen2.5-VL  -> ``hidden_states[-3]`` cropped of the 108 template tokens (3584-d)
  * ``text_encoder_2`` = ByT5/Glyph  -> ``last_hidden_state`` of the quoted-glyph text (1472-d), or a
    zero-length embedding when the prompt carries no quoted text.

The text-encode node packs them as a ``(qwen, byt5)`` tuple in the slot; ``HunyuanVideo15DiT`` unpacks
that into the two-element ``encoder_hidden_states`` list. On the CPU **toy** backend there is a single
``text_encoder`` component (``ToyTextEncoder``) and no ``text_encoder_2``: the node then writes the plain
toy embedding (a single numpy array), which ``ToyDiT`` consumes directly — so the program CPU-verifies
unchanged. The feature cache (design_v3 §7.2) still applies to the Qwen path via ``cached_text_encode``.

BRINGUP (i2v): the i2v path adds a VAE-encode node that writes the 33-channel first-frame conditioning
latent into ``i2v_cond`` and zero image embeds into ``i2v_img_embeds`` (the slots the loop already reads);
``Hy15ImageEncodingStage`` builds those (image_embeds=zeros[1,729,1152]; image_latent=[expanded VAE
first-frame ⊕ mask] -> 33ch). No CLIP/SigLIP vision encoder is loaded (image_embeds are placeholder
zeros), so do NOT add an image_encoder component. Left as a documented hook for the t2v registration.
"""
from __future__ import annotations

from typing import Any

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import cached_text_encode


def _text_encode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    """Encode prompt + negative prompt. When a second text encoder (ByT5) exists, pack ``(qwen, byt5)``
    tuples; otherwise (toy backend) write the single embedding ``ToyDiT`` expects."""
    has_byt5 = "text_encoder_2" in instance.card.components

    def _encode(text: str) -> Any:
        qwen = cached_text_encode(instance, text)  # primary (Qwen2.5-VL) — feature-cached
        if not has_byt5:
            return qwen  # toy backend: single embedding
        byt5 = instance.component("text_encoder_2").encode(text)
        return (qwen, byt5)

    slots["text_embeds"] = _encode(request.prompt())
    slots["neg_text_embeds"] = _encode(request.diffusion.negative_prompt)


def _vae_decode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_hunyuan_video15_program() -> Program:
    return Program(
        program_id="hunyuan_video15.t2v.inline",
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
