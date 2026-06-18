"""Unified LM+generator serve program (design_v3 §15b; UniRL/PromptRL inference path):

    text_encode → refine (ar_decode on llm) → apply_refinement → denoise (diffusion_denoise) → vae_decode

The serve path is deterministic: the LM refines greedily (argmax action) and the generator runs the
ODE Euler sampler. The ``unified_rl`` *training* method drives the same two loops but in the ROLLOUT
profile — the LM *samples* an action and the generator uses the SDE sampler with per-step log-prob
capture (design_v3 §9.4: rollout sampler ≠ serve sampler, a controlled divergence). Same loops, two
profiles — the §10 "train ≡ serve through one loop" invariant, now spanning two loop *types*.
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.models.common import text_encode_node_fn
from v2.models.omni import vae_decode_node
from v2.models.unified.card import N_REFINE_ACTIONS


def apply_refinement_node(refine_slot: str, n_actions: int = N_REFINE_ACTIONS):
    """ComponentNode fn: the LM's chosen refinement action reshapes the diffusion conditioning."""
    def fn(instance, slots, request, ctx) -> None:
        llm = instance.component("llm")
        out = slots.get(refine_slot)
        toks = out.get("tokens", []) if isinstance(out, dict) else []
        action = (int(toks[0]) % n_actions) if toks else 0
        slots["text_embeds"] = llm.refined_embed(slots.get("text_embeds"), action)
        slots["refine_action"] = action
    return fn


def build_unified_program() -> Program:
    return Program(
        program_id="unified.unirl", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=text_encode_node_fn,
                          writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("refine", loop_id="ar_decode", output_slot="refine_out",
                          reads=("text_embeds",), writes=("refine_out",)),
            ComponentNode("apply_refine", fn=apply_refinement_node("refine_out"),
                          reads=("refine_out", "text_embeds"), writes=("text_embeds", "refine_action")),
            ModelLoopNode("denoise", loop_id="diffusion_denoise", output_slot="denoise_out",
                          reads=("text_embeds",), writes=("denoise_out",)),
            ComponentNode("vae_decode", fn=vae_decode_node("denoise_out", "video"),
                          reads=("denoise_out",), writes=("video",)),
        ],
        output_artifacts={"video": "video", "latents": "denoise_out"},
    ).validate()
