# SPDX-License-Identifier: Apache-2.0
"""The Qwen3-VL stack is built only as far as MiniMax H3 reads.

H3 conditions on one intermediate hidden state. The layers above it were built,
weight-loaded and then discarded, which is 13.7 GB in bf16 and the difference
between fitting and not fitting on a 121 GB unified-memory device.

The dangerous part is not the truncation, it is getting the tuple index wrong.
`hidden_states` records each layer's *input*, so entry N is the output of layer
N-1, and the final entry comes from the norm that sits above the whole stack. A
truncated stack that still applies that norm puts a normalised tensor where the
raw one belongs: the length check in the conditioning stage still passes, and
conditioning silently changes. These tests pin the index, the content, and the
constant the two sides agree on.
"""
from __future__ import annotations

import os

import torch

# Matches the other encoder tests: the module registry these build against wants
# a process group, and a single-rank one needs a rendezvous address.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLLanguageModel
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER


def _small_arch(**overrides) -> MiniMaxH3Qwen3VLArchConfig:
    """A stack small enough to run on CPU but shaped like the real one.

    Everything goes through the constructor so ``__post_init__`` validates the
    small shape the same way it validates the real one.
    """
    kwargs: dict = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        # __post_init__ reads the sections out of rope_scaling, and they must
        # cover exactly half of each head.
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
        },
        vision_out_hidden_size=16,
    )
    kwargs.update(overrides)
    return MiniMaxH3Qwen3VLArchConfig(**kwargs)


def _small_config(**overrides) -> MiniMaxH3Qwen3VLConfig:
    """The outer config, which is what the modules take.

    ``ModelConfig.__getattr__`` forwards the architecture fields, so the modules
    read ``prefix`` off this object and everything else off ``arch_config``.
    """
    config = MiniMaxH3Qwen3VLConfig()
    config.arch_config = _small_arch(**overrides)
    return config


def test_default_matches_the_index_the_pipeline_reads() -> None:
    """The two sides cannot import each other, so pin them here instead.

    `fastvideo/models/` must not import from `fastvideo/pipelines/`, so the tap
    is written down twice. If they drift, conditioning reads a hidden state that
    was never built and the run dies with an index error at generation time,
    after a full model load.
    """
    assert MiniMaxH3Qwen3VLArchConfig().num_hidden_layers_override == MINIMAX_H3_TEXT_ENCODER_LAYER


def test_builds_only_up_to_the_override(distributed_setup) -> None:
    model = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=5))

    assert model.num_layers == 5
    assert len(model.layers) == 5
    # The norm sits above the tap, so a truncated stack must not keep it.
    assert model.norm is None


def test_override_none_keeps_the_full_stack(distributed_setup) -> None:
    model = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=None))

    assert model.num_layers == 8
    assert model.norm is not None


def test_override_above_the_stack_does_not_over_build(distributed_setup) -> None:
    # num_hidden_layers comes from the checkpoint's config.json via
    # update_model_arch, so a smaller variant must clamp rather than ask for
    # layers that do not exist.
    model = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=99))

    assert model.num_layers == 8
    assert model.norm is not None


def test_tapped_hidden_state_is_unchanged_by_truncation(distributed_setup) -> None:
    """The whole point: entry `tap` must be bit-identical either way."""
    tap = 5
    full = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=None))
    cut = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=tap))

    # These modules allocate uninitialised storage and expect a checkpoint, so
    # give them finite weights before running anything through them.
    torch.manual_seed(0)
    for parameter in full.parameters():
        parameter.data.normal_(std=0.02)
    # Then make the shared prefix identical, which is the only part the tapped
    # hidden state depends on.
    for (_, a), (_, b) in zip(full.layers[:tap].named_parameters(),
                              cut.layers[:tap].named_parameters(),
                              strict=True):
        b.data.copy_(a.data)
    torch.manual_seed(1)
    inputs_embeds = torch.randn(1, 6, 16)
    # mRoPE indexes three axes (t, h, w); text tokens share the same position on
    # all three.
    position_ids = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    with torch.no_grad():
        full_out = full(inputs_embeds, position_ids, None, True, None, None)
        cut_out = cut(inputs_embeds, position_ids, None, True, None, None)

    assert torch.equal(full_out.hidden_states[tap], cut_out.hidden_states[tap])
    # And the truncated model must not offer states it never computed.
    assert len(cut_out.hidden_states) == tap + 1


def test_truncated_model_drops_the_surplus_checkpoint_keys(distributed_setup) -> None:
    """The unexpected-key check is strict on purpose, so the surplus keys have
    to be filtered rather than the check relaxed."""
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    conditioner = MiniMaxH3Qwen3VLConditioner(_small_config(num_hidden_layers_override=5))

    assert conditioner._is_above_the_tap("language_model.layers.5.mlp.gate_proj.weight")
    assert conditioner._is_above_the_tap("language_model.layers.7.self_attn.q_proj.weight")
    assert conditioner._is_above_the_tap("language_model.norm.weight")
    # Kept: layers we built, the embeddings, and the vision tower.
    assert not conditioner._is_above_the_tap("language_model.layers.4.mlp.gate_proj.weight")
    assert not conditioner._is_above_the_tap("language_model.embed_tokens.weight")
    assert not conditioner._is_above_the_tap("visual.blocks.0.attn.qkv.weight")


def test_full_stack_filters_nothing(distributed_setup) -> None:
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    conditioner = MiniMaxH3Qwen3VLConditioner(_small_config(num_hidden_layers_override=None))

    assert not conditioner._is_above_the_tap("language_model.layers.7.mlp.gate_proj.weight")
    assert not conditioner._is_above_the_tap("language_model.norm.weight")
