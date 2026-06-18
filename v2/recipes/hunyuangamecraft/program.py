"""HunyuanGameCraft program: text_encode (LLaMA + CLIP) -> image_cond_encode (i2v) -> diffusion_denoise
-> vae_decode.

Same backbone as the Wan/Cosmos t2v programs, with two GameCraft-specific deltas:

  * **Dual text encode.** ``_text_encode_dual`` runs BOTH encoders: the LLaMA encoder (``text_encoder``)
    writes the DiT's ``text_states`` into ``text_embeds``/``neg_text_embeds`` (the usual slots the loop
    reads), and the CLIP encoder (``text_encoder_2``) writes the pooled ``text_states_2`` into
    ``clip_text_embeds``/``clip_neg_text_embeds`` (threaded to the adapter's ``context=`` per CFG branch).
    On the CPU toy both are ``ToyTextEncoder`` (cached via the content-hash feature cache).

  * **Image conditioning node (i2v).** ``_image_cond_encode`` mirrors ``GameCraftImageVAEEncodingStage``:
    VAE-encode the reference image -> scaled ``ref_latent`` (the ``GameCraftVAE`` adapter applies the
    scalar ``scaling_factor`` already), repeat to ``T_lat`` frames -> ``gt_latents``, build the binary
    ``conditioning_mask`` (latent_frames<=10: only frame 0; else first half), zero ``gt_latents`` on
    non-conditioned frames, and expose ``ref_latent_for_injection = gt_latents[:, 0:1]`` for the loop's
    per-step frame_replace. No request image -> a no-op (slots stay unset) -> the loop runs pure t2v
    (zero gt/mask), exactly the fastvideo fallback.

BRINGUP — the camera/action conditioning node is intentionally omitted: GameCraft's CameraNet needs a
Plücker-coordinate ``camera_states`` tensor derived from an action/camera sequence the v2 request API does
not carry yet, and the DiT's autoregressive latent-length special-casing (18/9/10) for multi-chunk
generation has no request surface. Adding a ``camera_encode`` node that writes the ``camera_states`` slot
(which the loop already reads) is the follow-up once the request API is extended. T_lat helper below
matches the GameCraft geometry (4x temporal compression).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import cached_text_encode

GAMECRAFT_TEMPORAL_RATIO = 4
GAMECRAFT_SPATIAL_RATIO = 8


def _stamp_text_encoder_2(instance: Any) -> None:
    """The shared ``from_config`` stamps the card via ``stamp_wan21_checkpoints``, whose subfolder map
    knows only ``transformer/vae/text_encoder`` — NOT GameCraft's second text encoder. So ``text_encoder_2``
    arrives at GPU build time with an empty ``checkpoint`` and ``_require_checkpoint`` fails. Derive the
    model root from the already-stamped ``text_encoder`` (LLaMA) checkpoint and point ``text_encoder_2`` at
    the sibling ``text_encoder_2/`` subfolder, here in the recipe's own node (no edit to the shared stamp).
    No-op once stamped or on the CPU toy (empty LLaMA checkpoint -> the toy factory needs no weights)."""
    import os
    comps = instance.card.components
    spec2 = comps.get("text_encoder_2")
    if spec2 is None or spec2.checkpoint:
        return
    llama_ckpt = getattr(comps.get("text_encoder"), "checkpoint", "") or ""
    if not llama_ckpt:
        return  # CPU toy / unstamped card: leave it for the toy factory
    root = os.path.dirname(os.path.normpath(llama_ckpt))
    spec2.checkpoint = os.path.join(root, "text_encoder_2")


def _text_encode_dual(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    """LLaMA (text_states) + CLIP (text_states_2) for prompt + negative prompt. The shared
    ``cached_text_encode`` keys on component_id, so the LLaMA and CLIP embeds never collide in the cache."""
    prompt = request.prompt()
    neg = request.diffusion.negative_prompt
    # LLaMA hidden states -> the loop's prompt-embed slots (text_states).
    slots["text_embeds"] = cached_text_encode(instance, prompt)
    slots["neg_text_embeds"] = cached_text_encode(instance, neg)
    # CLIP pooled -> dedicated slots (text_states_2), threaded through the adapter's context= per branch.
    _stamp_text_encoder_2(instance)  # stamp the CLIP checkpoint the Wan stamp skipped (GPU only)
    clip = instance.component("text_encoder_2")
    slots["clip_text_embeds"] = clip.encode(prompt)
    slots["clip_neg_text_embeds"] = clip.encode(neg)


def _image_cond_encode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    """i2v: VAE-encode the reference image -> gt_latents + conditioning_mask + ref_latent_for_injection.

    No-op for pure t2v (no request image) so the loop runs the degenerate standard denoise. Faithful to
    ``GameCraftImageVAEEncodingStage``: repeat the ref latent to all latent frames, then keep only the
    conditioned frames (first frame for short clips <=10 latent frames; first half for longer
    autoregressive clips) — both ``gt_latents`` and ``mask`` zeroed elsewhere."""
    part = request.image() if hasattr(request, "image") else None
    pixels = getattr(part, "pixels", None) if part is not None else None
    if pixels is None:
        return  # T2V mode: leave the i2v slots unset (loop falls back to zero gt/mask)

    d = request.diffusion
    # The GameCraftVAE adapter applies the scalar scaling_factor in ``encode`` (matches the fastvideo
    # stage's ``ref_latents.mul_(scaling_factor)``), so we use the returned latent directly. The real
    # adapter also runs the resize/center-crop/normalize preprocessing on the PIL image (BRINGUP); the
    # CPU toy VAE consumes the pixel array directly.
    ref_latent = np.asarray(instance.component("vae").encode(pixels), dtype="float32")  # [C,1,h,w] (or [C,h,w])
    if ref_latent.ndim == 3:  # [C, h, w] -> add the singleton temporal axis
        ref_latent = ref_latent[:, None]
    ref_latent = ref_latent[:, :1]  # keep the first (only) reference frame -> [C, 1, h, w]

    latent_frames = max(1, (max(1, d.num_frames) - 1) // GAMECRAFT_TEMPORAL_RATIO + 1)
    c, _t, h, w = ref_latent.shape
    gt_latents = np.repeat(ref_latent, latent_frames, axis=1)  # [C, T_lat, h, w]
    mask = np.ones((1, latent_frames, h, w), dtype="float32")  # [1, T_lat, h, w]
    if latent_frames <= 10:  # short i2v: only frame 0 is conditioned
        gt_latents[:, 1:] = 0.0
        mask[:, 1:] = 0.0
    else:  # autoregressive: first half conditioned
        half = latent_frames // 2
        gt_latents[:, half:] = 0.0
        mask[:, half:] = 0.0

    slots["gt_latents"] = gt_latents
    slots["conditioning_mask"] = mask
    slots["ref_latent_for_injection"] = gt_latents[:, 0:1].copy()  # the clean ref for frame_replace


def _vae_decode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_hunyuangamecraft_program() -> Program:
    return Program(
        program_id="gamecraft.i2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode",
                          fn=_text_encode_dual,
                          writes=("text_embeds", "neg_text_embeds", "clip_text_embeds", "clip_neg_text_embeds")),
            ComponentNode("image_cond_encode",
                          fn=_image_cond_encode,
                          writes=("gt_latents", "conditioning_mask", "ref_latent_for_injection")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds", "clip_text_embeds", "clip_neg_text_embeds",
                                 "gt_latents", "conditioning_mask", "ref_latent_for_injection"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
