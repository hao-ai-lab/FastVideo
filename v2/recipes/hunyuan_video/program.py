"""HunyuanVideo t2v program: hunyuan_text_encode -> diffusion_denoise (flow-match) -> vae_decode.

Same inline shape as the Wan/cosmos2 t2v programs, with ONE delta: HunyuanVideo has TWO text encoders
(a LLaMA per-token sequence + a CLIP-pooled global vector), so the text-encode node writes BOTH a
``text_embeds`` slot (the LLaMA sequence, the loop's primary ``text_embed``) and a ``text_pooled`` slot
(the CLIP vector, threaded to the DiT via the loop's ``context=`` channel). The dual-encode node uses the
content-hash feature cache (like ``recipes.common.text_encode_node_fn``) so a repeated prompt encodes once.

On the CPU toy backend both components are ``ToyTextEncoder`` (single arrays), so ``ToyDiT`` — which means
over both ``text_embed`` and ``context`` — runs end-to-end; on the GPU backend the real LLaMA/CLIP adapters
produce the sequence + pooled pair the ``HunyuanVideoDiT`` adapter reassembles into the 2-element
``encoder_hidden_states`` the Hunyuan forward expects.
"""
from __future__ import annotations

from typing import Any

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import cached_text_encode


def _ensure_text_encoder_2_stamped(instance: Any) -> None:
    """The shared ``stamp_wan21_checkpoints`` (called by the entrypoint after ``build_card``) only knows
    Wan's subfolder superset, which does NOT include ``text_encoder_2`` (HunyuanVideo's CLIP secondary
    encoder). Stamp it here — right before the component is materialized — from the already-stamped
    ``transformer`` checkpoint's parent (the model root). Idempotent / no-op once set; this keeps the fix
    inside the recipe (no edit to the shared stamp). Build-time stamping via ``build_*_card(checkpoint_root)``
    already covers the local-path path; this covers the registry+entrypoint path."""
    import os
    te2 = instance.card.components.get("text_encoder_2")
    if te2 is None or te2.checkpoint:
        return
    tr = instance.card.components.get("transformer")
    if tr is None or not tr.checkpoint:
        return
    root = os.path.dirname(os.path.normpath(tr.checkpoint))
    te2.checkpoint = os.path.join(root, "text_encoder_2")


def _encode_pooled(instance: Any, text: str) -> Any:
    """CLIP secondary encode (pooled global vector). Reuses the content-hash feature cache discipline of
    ``cached_text_encode`` but on the ``text_encoder_2`` component (a distinct cache partition by
    component_id). Falls back to the plain encode when the instance has no feature cache."""
    _ensure_text_encoder_2_stamped(instance)
    enc = instance.component("text_encoder_2")
    cache = instance.caches
    if cache is not None and cache.has("feature"):
        from v2.cache.keys import CacheKey, content_hash
        key = CacheKey(model_id=instance.card.model_id,
                       component_id="text_encoder_2",
                       weights_version=instance.version_of("text_encoder_2"),
                       adapter_versions=CacheKey.adapters(instance.adapter_versions),
                       precision=instance.card.precision.dtype_for("text_encoder_2"),
                       input_hashes=(("text", content_hash(text)), ))
        hit = cache.pool("feature").get(key)
        if hit is not None:
            return hit
        emb = enc.encode(text)
        cache.pool("feature").put(key, emb)
        return emb
    return enc.encode(text)


def _hunyuan_text_encode(instance, slots, request, ctx) -> None:
    """ComponentNode fn: prompt -> (LLaMA sequence in ``text_embeds`` + CLIP pooled in ``text_pooled``),
    plus the negative-prompt counterparts (used only when guidance_scale != 1.0; base HunyuanVideo runs
    guidance_scale=1.0 so the uncond branch collapses out in ClassicCFG.combine)."""
    prompt = request.prompt()
    neg = request.diffusion.negative_prompt
    slots["text_embeds"] = cached_text_encode(instance, prompt)  # LLaMA per-token sequence
    slots["text_pooled"] = _encode_pooled(instance, prompt)  # CLIP pooled global vector
    slots["neg_text_embeds"] = cached_text_encode(instance, neg)
    slots["neg_text_pooled"] = _encode_pooled(instance, neg)


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_hunyuan_video_program() -> Program:
    return Program(
        program_id="hunyuan_video.t2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode",
                          fn=_hunyuan_text_encode,
                          writes=("text_embeds", "text_pooled", "neg_text_embeds", "neg_text_pooled")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "text_pooled", "neg_text_embeds", "neg_text_pooled"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
