"""Cosmos3 omni program (design_v3 §15b):

    tokenize → reason (ar_decode) → pack(tokens→cond) → diffusion_denoise → vae_decode

The reasoner and the denoiser bind the SAME resident ``transformer``; the scheduler is the mode
multiplexer. This is the workload no DAG-of-engines can express (splitting them doubles the weights
and severs the shared state).
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.models.omni import emit_text_node, pack_cond_from_tokens, tokenize_node, vae_decode_node


def build_cosmos3_program() -> Program:
    return Program(
        program_id="cosmos3.omni", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("tokenize", fn=tokenize_node, writes=("prompt_tokens",)),
            ModelLoopNode("reason", loop_id="ar_decode", output_slot="reason_out",
                          reads=("prompt_tokens",), writes=("reason_out",)),
            ComponentNode("emit_text", fn=emit_text_node("reason_out", "reason_text"),
                          reads=("reason_out",), writes=("reason_text",)),
            ComponentNode("pack", fn=pack_cond_from_tokens("reason_out"),
                          reads=("reason_out",), writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise", loop_id="diffusion_denoise", output_slot="denoise_out",
                          reads=("text_embeds",), writes=("denoise_out",)),
            ComponentNode("vae_decode", fn=vae_decode_node("denoise_out", "video"),
                          reads=("denoise_out",), writes=("video",)),
        ],
        output_artifacts={"video": "video", "text": "reason_text", "latents": "denoise_out"},
    ).validate()
