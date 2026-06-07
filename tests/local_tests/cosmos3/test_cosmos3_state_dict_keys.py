# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 DiT param-name / checkpoint-key-surface contract.

The FastVideo native Cosmos3 DiT (``fastvideo.models.dits.cosmos3.
Cosmos3VFMTransformer``) deliberately mirrors the *published diffusers
checkpoint* transformer key surface so the converter at
``scripts/checkpoint_conversion/cosmos3_convert.py`` can strict-load with a
near-identity ``param_names_mapping``.

This pins the FastVideo side of that contract: a tiny 1-layer DiT must expose
exactly the published checkpoint's parameter names. The checkpoint key surface
(validated against ``nvidia/Cosmos3-Nano``; ``{i}`` ranges over the layers):

  Top level:
    embed_tokens.weight, norm.weight, norm_moe_gen.weight, lm_head.weight,
    proj_in.{weight,bias}, proj_out.{weight,bias},
    time_embedder.linear_1.{weight,bias}, time_embedder.linear_2.{weight,bias}
  Dormant heads (present for strict-load):
    action_proj_in.fc.weight, action_proj_in.bias.weight,
    action_proj_out.fc.weight, action_proj_out.bias.weight,
    action_modality_embed,
    audio_proj_in.{weight,bias}, audio_proj_out.{weight,bias},
    audio_modality_embed
  Per layer ``layers.{i}``:
    self_attn.{to_q,to_k,to_v,to_out,add_q_proj,add_k_proj,add_v_proj,
               to_add_out,norm_q,norm_k,norm_added_q,norm_added_k}.weight,
    mlp.{gate_proj,up_proj,down_proj}.weight,
    mlp_moe_gen.{gate_proj,up_proj,down_proj}.weight,
    {input_layernorm,input_layernorm_moe_gen,post_attention_layernorm,
     post_attention_layernorm_moe_gen}.weight

The earlier scaffold pinned the dead vllm-omni layout
(``language_model.layers`` / ``gen_layers`` / ``cross_attention`` /
``vae2llm`` / ``llm2vae``); that structure is gone — the native DiT is a single
dual-pathway ``layers`` ModuleList matching the diffusers checkpoint.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.local]


def _build_tiny_dit():
    """Construct a tiny 1-layer FastVideo Cosmos3 DiT, or skip if unavailable."""
    try:
        from fastvideo.configs.models.dits.cosmos3 import (
            Cosmos3ArchConfig,
            Cosmos3VideoConfig,
        )
        from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer
    except ImportError:  # pragma: no cover - import guard
        pytest.skip("FastVideo Cosmos3 DiT not importable in this environment")

    import torch

    arch = Cosmos3ArchConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        vocab_size=64,
        latent_patch_size=2,
        latent_channel=16,
        position_embedding_type="3d_rope",
        enable_fps_modulation=False,
        action_gen=True,
        action_dim=64,
        max_action_dim=64,
        num_embodiment_domains=32,
        sound_gen=True,
        sound_dim=64,
    )
    cfg = Cosmos3VideoConfig(arch_config=arch)
    return Cosmos3VFMTransformer(cfg, hf_config={}).to(torch.float32)


def test_fastvideo_cosmos3_dit_module_tree_param_names() -> None:
    """The native DiT param-name set must equal the published checkpoint surface."""
    model = _build_tiny_dit()
    names = {name for name, _ in model.named_parameters()}

    expected_top = {
        "embed_tokens.weight",
        "norm.weight",
        "norm_moe_gen.weight",
        "lm_head.weight",
        "proj_in.weight",
        "proj_in.bias",
        "proj_out.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    }
    expected_dormant = {
        "action_proj_in.fc.weight",
        "action_proj_in.bias.weight",
        "action_proj_out.fc.weight",
        "action_proj_out.bias.weight",
        "action_modality_embed",
        "audio_proj_in.weight",
        "audio_proj_in.bias",
        "audio_proj_out.weight",
        "audio_proj_out.bias",
        "audio_modality_embed",
    }
    layer_suffixes = {
        "self_attn.to_q.weight",
        "self_attn.to_k.weight",
        "self_attn.to_v.weight",
        "self_attn.to_out.weight",
        "self_attn.add_q_proj.weight",
        "self_attn.add_k_proj.weight",
        "self_attn.add_v_proj.weight",
        "self_attn.to_add_out.weight",
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "self_attn.norm_added_q.weight",
        "self_attn.norm_added_k.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp_moe_gen.gate_proj.weight",
        "mlp_moe_gen.up_proj.weight",
        "mlp_moe_gen.down_proj.weight",
        "input_layernorm.weight",
        "input_layernorm_moe_gen.weight",
        "post_attention_layernorm.weight",
        "post_attention_layernorm_moe_gen.weight",
    }
    expected_layer = {f"layers.0.{s}" for s in layer_suffixes}
    expected = expected_top | expected_dormant | expected_layer

    assert names == expected, (f"Cosmos3 DiT param surface mismatch.\n"
                               f"  missing: {sorted(expected - names)}\n"
                               f"  unexpected: {sorted(names - expected)}")


def test_fastvideo_cosmos3_dit_no_dead_vllm_omni_layout() -> None:
    """The dead vllm-omni layout (split language_model/gen_layers/cross_attention,
    vae2llm/llm2vae) must NOT appear in the native DiT param tree."""
    model = _build_tiny_dit()
    names = {name for name, _ in model.named_parameters()}
    dead_fragments = (
        "language_model.",
        "gen_layers.",
        "cross_attention.",
        "vae2llm",
        "llm2vae",
    )
    offenders = sorted(n for n in names if any(frag in n for frag in dead_fragments))
    assert not offenders, f"native DiT still exposes dead vllm-omni param names: {offenders}"


def test_fastvideo_cosmos3_dit_per_layer_counts() -> None:
    """Per-layer block count: 22 weights/layer; full 1-layer model = 44 params.

    (12 attention + 6 dual-MLP + 4 layernorm per layer; + 12 top-level
    + 10 dormant-head params.)
    """
    model = _build_tiny_dit()
    names = [name for name, _ in model.named_parameters()]
    per_layer = [n for n in names if n.startswith("layers.0.")]
    assert len(per_layer) == 22, f"expected 22 per-layer params, got {len(per_layer)}"
    assert len(names) == 44, f"expected 44 total params for tiny 1-layer DiT, got {len(names)}"
