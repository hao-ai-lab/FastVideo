"""Speculative-decoding program: tokenize → spec_decode (draft+verify) → emit_text (design_v3 §9.16)."""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.models.omni import emit_text_node, tokenize_node


def build_speculative_program() -> Program:
    return Program(
        program_id="speculative.decode", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("tokenize", fn=tokenize_node, writes=("prompt_tokens",)),
            ModelLoopNode("spec", loop_id="spec_decode", output_slot="spec_out",
                          reads=("prompt_tokens",), writes=("spec_out",)),
            ComponentNode("emit_text", fn=emit_text_node("spec_out", "text"),
                          reads=("spec_out",), writes=("text",)),
        ],
        output_artifacts={"text": "text", "tokens": "spec_out"},
    ).validate()
