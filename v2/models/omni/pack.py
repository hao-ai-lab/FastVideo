"""Omni program nodes shared by Cosmos3 and the bagel/lance MoT (design_v3 §4.1, §15b).

The ``pack`` node is the seam where the AR (understanding) pathway's tokens become the
conditioning the diffusion (generation) pathway consumes — "the AR reasoner ... used as prompt
upsampling before diffusion in the same request". Both pathways are the SAME resident MoT
module, so this is weight-sharing, not a hand-off between two engines.
"""
from __future__ import annotations

from v2.models.backend import ToyTokenizer


def tokenize_node(instance, slots, request, ctx) -> None:
    tok = instance.component("tokenizer") if instance.has_component("tokenizer") else ToyTokenizer()
    slots["prompt_tokens"] = tok.encode(request.prompt())


def _tokens_of(slot_value) -> list:
    if isinstance(slot_value, dict):
        return slot_value.get("tokens", [])
    return list(slot_value or [])


def pack_cond_from_tokens(token_slot: str):
    """Build a ComponentNode fn that packs the und-pathway tokens into the gen-pathway conditioning."""
    def fn(instance, slots, request, ctx) -> None:
        mot = instance.component("transformer")          # the SHARED MoT module (und + gen pathways)
        toks = _tokens_of(slots.get(token_slot))
        slots["text_embeds"] = mot.reasoner_embed(toks)
        slots["neg_text_embeds"] = mot.reasoner_embed([0])
    return fn


def emit_text_node(token_slot: str, out_slot: str = "text"):
    def fn(instance, slots, request, ctx) -> None:
        out = slots.get(token_slot, {})
        slots[out_slot] = out.get("text", "") if isinstance(out, dict) else str(out)
    return fn


def vae_decode_node(latent_slot: str, out_slot: str):
    def fn(instance, slots, request, ctx) -> None:
        latents = slots[latent_slot]["latents"] if isinstance(slots.get(latent_slot), dict) else None
        slots[out_slot] = instance.component("vae").decode(latents) if latents is not None else None
    return fn
