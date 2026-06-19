"""TurboWan programs (node graphs identical to Wan — the rCM specifics live in the loop/sampler).

T2V: text_encode -> diffusion_denoise (rCM) -> vae_decode (same inline shape as the Wan/cosmos2 t2v
programs). I2V (Wan2.2-I2V-A14B): reuses the ``v2/recipes/wan21/i2v.py`` conditioning nodes verbatim (CLIP
image encode + first-frame [mask|cond] VAE encode), bound to TurboWan's ``i2v_denoise`` loop. No new
conditioning math — the only TurboWan delta is the loop's rCM sampler.
"""
from __future__ import annotations

from typing import Any

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode
from v2.recipes.wan21.i2v import _i2v_cond_encode, _i2v_image_encode


def _vae_decode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_turbowan_program() -> Program:
    """TurboWan2.1 T2V program (serves both the 1.3B and 14B ids)."""
    return Program(
        program_id="turbowan.t2v.inline",
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


def _i2v_vae_decode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    slots["video"] = instance.component("vae").decode(slots["i2v_out"]["latents"])


def build_turbowan_i2v_program() -> Program:
    """TurboWan2.2-I2V-A14B program — Wan i2v conditioning nodes + the rCM ``i2v_denoise`` loop."""
    return Program(
        program_id="turbowan.i2v",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("image_encode", fn=_i2v_image_encode, writes=("i2v_img_embeds", )),
            ComponentNode("cond_encode", fn=_i2v_cond_encode, writes=("i2v_cond", )),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="i2v_denoise",
                          output_slot="i2v_out",
                          reads=("text_embeds", "i2v_cond", "i2v_img_embeds"),
                          writes=("i2v_out", )),
            ComponentNode("vae_decode", fn=_i2v_vae_decode, reads=("i2v_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "i2v_out"
        },
    ).validate()
