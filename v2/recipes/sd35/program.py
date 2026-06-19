"""SD3.5 t2i program: sd3_text_encode -> diffusion_denoise (flow-match) -> vae_decode.

Single-stage INLINE program. The SD3-specific work lives in the custom text-encode node (the
triple-encoder joint-embed + dual-CLIP pooled assembly of ``SD35ConditioningStage``) and the VAE decode
node (the [-1,1] -> [0,1] remap of ``SD35DecodingStage``); the denoise node is the generic
``ModelLoopNode`` binding ``SD3DenoiseLoop``.

The text-encode node is backend-agnostic: on the GPU torch backend the CLIP adapters
(``SD3ClipEncoder``) return ``{"hidden": penultimate[-2], "pooled": pooler_output}`` and ``SD3T5Encoder``
returns ``last_hidden_state``, which ``assemble_sd3_conditioning`` combines into (joint_embed, pooled).
On the CPU toy backend the encoders return plain arrays (``ToyTextEncoder.encode``), so the node falls
back to a degenerate-but-finite assembly (joint = the array, pooled = its channel-mean broadcast) — enough
to exercise the dual-conditioning plumbing without any GPU/torch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.recipes.sd35.adapter import assemble_sd3_conditioning
from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind

_CLIP_L = "text_encoder"
_CLIP_G = "text_encoder_2"
_T5 = "text_encoder_3"


def _encode_one(instance: Any, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the three SD3.5 text encoders on ``prompt`` and assemble (joint_embed, pooled).

    Handles both backends: torch CLIP adapters return a ``{"hidden","pooled"}`` dict (-> the faithful
    ``assemble_sd3_conditioning``); the CPU toy encoders return a plain array (-> a degenerate, finite
    stand-in that still threads a distinct joint embed + pooled vector through the dual-conditioning path)."""
    clip_l = instance.component(_CLIP_L).encode(prompt)
    clip_g = instance.component(_CLIP_G).encode(prompt)
    t5 = instance.component(_T5).encode(prompt)
    if isinstance(clip_l, dict) and isinstance(clip_g, dict):
        return assemble_sd3_conditioning(clip_l, clip_g, t5)
    # CPU toy fallback: ToyTextEncoder.encode returns a [seq, dim] array (no pooled head). Concatenate
    # the three along seq for the joint embed; derive a small pooled vector from the CLIP means.
    a_l, a_g, a_t = np.asarray(clip_l), np.asarray(clip_g), np.asarray(t5)
    joint = np.concatenate([a_l, a_g, a_t], axis=-2).astype("float32")
    pooled = np.concatenate([a_l.mean(axis=-2), a_g.mean(axis=-2)], axis=-1).astype("float32")
    return joint, pooled


def _sd3_text_encode(instance, slots, request, ctx) -> None:
    joint, pooled = _encode_one(instance, request.prompt())
    neg_joint, neg_pooled = _encode_one(instance, request.diffusion.negative_prompt)
    slots["text_embeds"] = joint
    slots["pooled_projections"] = pooled
    slots["neg_text_embeds"] = neg_joint
    slots["neg_pooled_projections"] = neg_pooled


def _vae_decode(instance, slots, request, ctx) -> None:
    # SD3DiT denoises in normalized latent space; SD3VAE.decode inverts shift/scale and runs the
    # AutoencoderKL decode -> image in [-1, 1]. Remap to [0, 1] (SD35DecodingStage: (img/2+0.5).clamp).
    vae = instance.component("vae")
    latent = np.asarray(slots["denoise_out"]["latents"], dtype="float32")  # [C, h, w] image latent
    # The real SD3VAE adapter consumes the strict 4D image latent [C, h, w] directly. The CPU toy ToyVAE
    # is video-oriented ([C, T, h, w]); give it a singleton temporal axis so the decode CPU-verifies
    # (faithful to the fastvideo stage carrying a fake T dim around the strictly-4D DiT forward).
    if getattr(vae, "module", None) is None and latent.ndim == 3:
        latent = latent[:, None]  # -> [C, 1, h, w] for the toy decoder only
    image = vae.decode(latent)
    slots["image"] = np.clip(np.asarray(image, dtype="float32") / 2.0 + 0.5, 0.0, 1.0)


def build_sd35_program() -> Program:
    return Program(
        program_id="sd35.t2i.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("sd3_text_encode",
                          fn=_sd3_text_encode,
                          writes=("text_embeds", "neg_text_embeds", "pooled_projections", "neg_pooled_projections")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds", "pooled_projections", "neg_pooled_projections"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("image", )),
        ],
        output_artifacts={
            "image": "image",
            "latents": "denoise_out",
        },
    ).validate()
