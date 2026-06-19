"""Shared component-node helpers (text encode with feature cache, etc.).

The content-hash feature cache lets a K-sample RL group encode its shared prompt once instead of
K times.
"""
from __future__ import annotations

from typing import Any

from v2.cache.keys import CacheKey, content_hash


def cached_text_encode(instance: Any, text: str) -> Any:
    enc = instance.component("text_encoder")
    cache = instance.caches
    if cache is not None and cache.has("feature"):
        # Every output-semantic field is in the key — adapter stack + precision partition it, and the
        # text-encoder's OWN version (not the instance's) so a transformer weight sync does not
        # invalidate the (frozen) text encoder's embeddings.
        key = CacheKey(model_id=instance.card.model_id,
                       component_id="text_encoder",
                       weights_version=instance.version_of("text_encoder"),
                       adapter_versions=CacheKey.adapters(instance.adapter_versions),
                       precision=instance.card.precision.dtype_for("text_encoder"),
                       input_hashes=(("text", content_hash(text)), ))
        hit = cache.pool("feature").get(key)
        if hit is not None:
            return hit
        emb = enc.encode(text)
        cache.pool("feature").put(key, emb)
        return emb
    return enc.encode(text)


def text_encode_node_fn(instance, slots, request, ctx) -> None:
    """ComponentNode fn: prompt + negative prompt → cached text embeddings in slots."""
    slots["text_embeds"] = cached_text_encode(instance, request.prompt())
    slots["neg_text_embeds"] = cached_text_encode(instance, request.diffusion.negative_prompt)
