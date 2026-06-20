"""Wan-causal streaming program: text_encode → chunk_rollout → vae_decode."""
from __future__ import annotations

from v2.core.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    latents = slots["rollout_out"]["latents"]
    slots["video"] = instance.component("vae").decode(latents) if latents is not None else None


def build_wan_causal_program() -> Program:
    return Program(
        program_id="wan_causal.stream",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("rollout",
                          loop_id="chunk_rollout",
                          output_slot="rollout_out",
                          reads=("text_embeds", ),
                          writes=("rollout_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("rollout_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "rollout_out"
        },
    ).validate()
