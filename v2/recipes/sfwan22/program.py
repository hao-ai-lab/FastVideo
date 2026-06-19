"""Self-Forcing Wan2.2-A14B programs.

  * t2v: text_encode -> chunk_rollout -> vae_decode (the wan_causal streaming shape, with the SF Wan2.2
    boundary-routed DMD loop bound to ``chunk_rollout``).
  * i2v: image_encode (CLIP) + cond_encode (first-frame VAE [cond] latent) + text_encode -> chunk_rollout
    -> vae_decode. The conditioning encode reuses the wan21 i2v node helpers; the causal loop reads
    ``i2v_cond`` (the first-frame latent that becomes the held conditioning block) + ``i2v_img_embeds``.

The node graph is the only i2v/t2v difference — the boundary-routed DMD + causal block math is entirely
inside ``SFWan22ChunkRolloutLoop``.
"""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode

_WAN_TEMPORAL_RATIO = 4  # VAE temporal compression (i2v cond mask width, kept for parity with wan21 i2v)


def _vae_decode(instance, slots, request, ctx) -> None:
    latents = slots["rollout_out"]["latents"]
    slots["video"] = instance.component("vae").decode(latents) if latents is not None else None


def build_sfwan22_t2v_program() -> Program:
    return Program(
        program_id="sfwan22.t2v.stream",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("rollout",
                          loop_id="chunk_rollout",
                          output_slot="rollout_out",
                          reads=("text_embeds", "neg_text_embeds"),
                          writes=("rollout_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("rollout_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "rollout_out"
        },
    ).validate()


# --- i2v conditioning nodes (mirror fastvideo's causal first-frame replace + CLIP encode) ---------- #
def _i2v_image_encode(instance, slots, request, ctx) -> None:
    img = request.image()
    slots["i2v_img_embeds"] = (instance.component("image_encoder").encode_image(img.pixels)
                               if img is not None and getattr(img, "pixels", None) is not None else None)


def _i2v_cond_encode(instance, slots, request, ctx) -> None:
    """First-frame VAE conditioning. Causal Wan2.2-MoE i2v REPLACES the first latent frame with the
    image latent (it does NOT channel-concat a mask like the bidirectional Wan2.1 Fun-InP path), so the
    cond slot is just the encoded first-frame latent; the loop holds it fixed and primes both experts' KV
    (matching ``CausalDMDDenosingStage``'s first_frame_latent handling)."""
    img = request.image()
    if img is None or getattr(img, "pixels", None) is None:
        slots["i2v_cond"] = None
        return
    px = np.asarray(img.pixels, dtype="float32")  # [3, H, W] in [-1, 1]
    cond_video = px[:, None, :, :]  # [3, 1, H, W] (single conditioning frame)
    slots["i2v_cond"] = np.asarray(instance.component("vae").encode(cond_video), dtype="float32")  # [C,1,h,w]


def build_sfwan22_i2v_program() -> Program:
    return Program(
        program_id="sfwan22.i2v.stream",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("image_encode", fn=_i2v_image_encode, writes=("i2v_img_embeds", )),
            ComponentNode("cond_encode", fn=_i2v_cond_encode, writes=("i2v_cond", )),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("rollout",
                          loop_id="chunk_rollout",
                          output_slot="rollout_out",
                          reads=("text_embeds", "neg_text_embeds", "i2v_cond", "i2v_img_embeds"),
                          writes=("rollout_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("rollout_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "rollout_out"
        },
    ).validate()


# The package's primary program (the I2V Preview id is the package's primary HF id).
def build_sfwan22_program() -> Program:
    return build_sfwan22_i2v_program()
