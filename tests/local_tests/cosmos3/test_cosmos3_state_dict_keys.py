# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 state-dict-key contract for checkpoint conversion (Tier A scaffold).

Reference: ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:319-409``
(``Cosmos3OmniDiffusersPipeline._remap_ckpt_key``). The vllm-omni loader's
remap is the canonical source of truth: it defines which Diffusers
checkpoint keys become which target module-tree parameter names.

The FastVideo conversion script at
``scripts/checkpoint_conversion/cosmos3_convert.py`` (to be authored in
Phase 5) must produce a state-dict that loads cleanly into the FastVideo
Cosmos3 DiT module tree. This test pins the target side of that contract:

  * ``embed_tokens``/``norm`` live under ``language_model.``
  * ``norm_moe_gen`` is a top-level (not per-layer) module
  * Per-layer keys split into UND (``language_model.layers.{i}``) and
    GEN (``gen_layers.{i}``) sub-trees:
      - UND attention: ``self_attn.{q,k,v,o}_proj``, ``self_attn.{q,k}_norm``
      - GEN cross-attention: ``cross_attention.{q,k,v,o}_proj``,
        ``cross_attention.{q,k}_norm`` (from the ``*_moe_gen`` source keys)
      - UND/GEN norms: ``input_layernorm``, ``post_attention_layernorm``
      - UND/GEN MLPs: ``mlp.{gate,up,down}_proj``
  * Top-level adapters ``vae2llm``, ``llm2vae``, ``time_embedder`` live
    under ``transformer.`` (no remapping inside the transformer namespace).
  * ``lm_head.weight`` is skipped (mapped to ``None``).
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.local]


EXPECTED_REMAPS: dict[str, str | None] = {
    "model.embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
    "model.norm.weight": "transformer.language_model.norm.weight",
    "model.norm_moe_gen.weight": "transformer.norm_moe_gen.weight",
    "model.layers.3.self_attn.q_proj.weight": "transformer.language_model.layers.3.self_attn.q_proj.weight",
    "model.layers.3.self_attn.q_proj_moe_gen.weight": "transformer.gen_layers.3.cross_attention.q_proj.weight",
    "model.layers.3.self_attn.k_norm_moe_gen.weight": "transformer.gen_layers.3.cross_attention.k_norm.weight",
    "model.layers.3.input_layernorm.weight": "transformer.language_model.layers.3.input_layernorm.weight",
    "model.layers.3.input_layernorm_moe_gen.weight": "transformer.gen_layers.3.input_layernorm.weight",
    "model.layers.3.mlp.gate_proj.weight": "transformer.language_model.layers.3.mlp.gate_proj.weight",
    "model.layers.3.mlp_moe_gen.up_proj.weight": "transformer.gen_layers.3.mlp.up_proj.weight",
    "vae2llm.weight": "transformer.vae2llm.weight",
    "llm2vae.weight": "transformer.llm2vae.weight",
    "time_embedder.linear_1.weight": "transformer.time_embedder.linear_1.weight",
    "lm_head.weight": None,
}


def test_fastvideo_cosmos3_remap_matches_reference() -> None:
    """Asserts the FastVideo Cosmos3 checkpoint remap table matches the
    canonical vllm-omni reference at pipeline_cosmos3.py:319-409.

    Once the FastVideo Cosmos3 pipeline (or a standalone remap utility
    in ``scripts/checkpoint_conversion/cosmos3_convert.py``) exposes a
    ``_remap_ckpt_key`` callable, this test verifies every key in
    ``EXPECTED_REMAPS`` maps to the expected target name (or ``None``
    for skipped keys like ``lm_head.weight``).
    """
    try:
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
            Cosmos3OmniDiffusersPipeline,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 pipeline not yet implemented (Phase 2b)")

    remap = getattr(Cosmos3OmniDiffusersPipeline, "_remap_ckpt_key", None)
    if remap is None:
        pytest.skip("Cosmos3OmniDiffusersPipeline._remap_ckpt_key not yet defined")

    actual = {key: remap(key) for key in EXPECTED_REMAPS}
    assert actual == EXPECTED_REMAPS


def test_fastvideo_cosmos3_dit_module_tree_param_names() -> None:
    """Asserts the FastVideo Cosmos3 DiT module tree produces the param
    names that the checkpoint converter is expected to write.

    Once ``fastvideo.models.dits.cosmos3.Cosmos3VFMTransformer`` is
    constructible with a tiny config (see ``conftest._tiny_cosmos3_config``),
    this test instantiates a 1-layer model and verifies that the
    parameter-name set contains the expected UND/GEN/adapter prefixes:
      * ``language_model.layers.0.self_attn.q_proj.weight``
      * ``gen_layers.0.cross_attention.q_proj.weight``
      * ``norm_moe_gen.weight``
      * ``vae2llm.*`` and ``llm2vae.*``
    """
    try:
        from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer  # type: ignore
    except ImportError:
        pytest.skip("FastVideo Cosmos3 not yet implemented (Phase 2b)")

    from types import SimpleNamespace
    import torch

    from .conftest import _tiny_cosmos3_config

    config = _tiny_cosmos3_config(num_hidden_layers=1)
    model = Cosmos3VFMTransformer(SimpleNamespace(tf_model_config=config, dtype=torch.float32))
    names = set(name for name, _ in model.named_parameters())

    required_prefixes = [
        "language_model.layers.0.self_attn.q_proj",
        "gen_layers.0.cross_attention.q_proj",
        "norm_moe_gen",
        "vae2llm",
        "llm2vae",
    ]
    for prefix in required_prefixes:
        assert any(name.startswith(prefix) for name in names), (
            f"Cosmos3 DiT module tree missing expected prefix {prefix!r}; "
            f"converter at scripts/checkpoint_conversion/cosmos3_convert.py "
            f"will fail to load weights for this subtree."
        )
