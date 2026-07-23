# SPDX-License-Identifier: Apache-2.0
"""Contract tests for LTX2 torch.compile integration.

``ComposedPipelineBase._compile_with_conditions`` compiles the submodules
matched by ``model._compile_conditions`` and falls back to compiling the
whole model as one fullgraph when the list is empty. LTX2 shipped with the
conditions declared on its arch config but never exposed on the model
class, so it silently took the whole-model fallback (enormous unportable
graphs, 10+ minute cold compiles). These assertions pin the class-level
wiring and the block matcher so a refactor can't reintroduce the fallback
unnoticed. CPU-only; no model instantiation.
"""

import torch

from fastvideo.configs.models.dits.ltx2 import LTX2VideoConfig, is_ltx2_blocks
from fastvideo.models.dits import ltx2


def test_ltx2_model_exposes_compile_conditions():
    conditions = ltx2.LTX2Transformer3DModel._compile_conditions
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


def test_fsdp_output_hook_registration_stays_outside_compiled_graph():
    register_hooks = ltx2.BasicAVTransformerBlock._register_fsdp_backward_hooks_on_output

    assert getattr(register_hooks, "_torchdynamo_disable", False) is True


def test_freq_grid_cache_is_bypassed_while_compiling():
    args = (10000.0, 3, 96)
    for generator, cached_generator in (
        (
            ltx2.generate_ltx_freq_grid_float64,
            ltx2._cached_generate_ltx_freq_grid_float64,
        ),
        (
            ltx2.generate_ltx_freq_grid_pytorch,
            ltx2._cached_generate_ltx_freq_grid_pytorch,
        ),
    ):
        cached_generator.cache_clear()
        eager = generator(*args)
        assert generator(*args) is eager

        cached_generator.cache_clear()
        compiled_generator = torch.compile(
            lambda zero: generator(*args) + zero,
            backend="eager",
            fullgraph=True,
        )
        torch.testing.assert_close(compiled_generator(torch.tensor(0.0)), eager)
        assert cached_generator.cache_info().currsize == 0
