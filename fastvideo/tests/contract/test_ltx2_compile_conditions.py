# SPDX-License-Identifier: Apache-2.0
"""Contract test: LTX2's per-block torch.compile wiring stays intact.

``ComposedPipelineBase._compile_with_conditions`` compiles the submodules
matched by ``model._compile_conditions`` and falls back to compiling the
whole model as one fullgraph when the list is empty. LTX2 shipped with the
conditions declared on its arch config but never exposed on the model
class, so it silently took the whole-model fallback (enormous unportable
graphs, 10+ minute cold compiles). These assertions pin the class-level
wiring and the block matcher so a refactor can't reintroduce the fallback
unnoticed. CPU-only; no model instantiation.
"""

from fastvideo.configs.models.dits.ltx2 import LTX2VideoConfig, is_ltx2_blocks


def test_ltx2_model_exposes_compile_conditions():
    from fastvideo.models.dits.ltx2 import LTX2Transformer3DModel

    conditions = LTX2Transformer3DModel._compile_conditions
    assert conditions, (
        "LTX2Transformer3DModel._compile_conditions is empty: "
        "torch.compile falls back to a whole-model fullgraph")
    assert is_ltx2_blocks in conditions
    assert LTX2VideoConfig()._compile_conditions == conditions


def test_is_ltx2_blocks_matches_only_top_level_blocks():
    assert is_ltx2_blocks("transformer_blocks.0", None)
    assert is_ltx2_blocks("model.transformer_blocks.47", None)
    # Submodules and lookalike names must not match — compiling them would
    # nest compiled regions inside the per-block graphs.
    assert not is_ltx2_blocks("transformer_blocks", None)
    assert not is_ltx2_blocks("transformer_blocks.0.attn1", None)
    assert not is_ltx2_blocks("audio_transformer_blocks.0", None)
