# SPDX-License-Identifier: Apache-2.0
"""Focused CPU coverage for the native LingBot-Video sparse MoE path."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.testing import assert_close

from fastvideo.configs.models.dits.lingbot_video import LingBotVideoConfig
from fastvideo.models.dits.lingbot_video import (
    LingBotVideoBlock,
    LingBotVideoMLP,
    LingBotVideoRotaryEmbedding,
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
    LingBotVideoTransformer3DModel,
    _make_joint_position_ids,
)
from fastvideo.models.loader.fsdp_load import set_default_dtype


def test_router_selection_bias_does_not_change_gating_weight() -> None:
    """Select with the correction bias while gathering the bias-free sigmoid score."""
    router = LingBotVideoRouter(2, 3, 1, "sigmoid", True, None, None, 1.0)
    with torch.no_grad():
        router.weight.zero_()
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    indices, top_scores, logits, scores, scores_for_choice = router(torch.zeros(1, 2, dtype=torch.bfloat16))

    assert indices.item() == 2
    assert top_scores.dtype == torch.bfloat16
    assert top_scores.item() == 0.5
    assert logits.dtype == torch.float32
    assert scores.dtype == torch.float32
    assert scores_for_choice[0, 2].item() == 1.5


def test_router_group_limited_topk_uses_two_expert_group_score() -> None:
    """Keep choices inside the group whose two best expert scores have the largest sum."""
    router = LingBotVideoRouter(1, 8, 2, "sigmoid", False, 4, 1, 1.0)
    with torch.no_grad():
        router.weight[:, 0].copy_(torch.tensor([6.0, 5.0, 9.0, -9.0, 4.0, 4.0, 3.0, 3.0]))

    indices, _, _, _, _ = router(torch.ones(1, 1))

    assert set(indices[0].tolist()) == {0, 1}


def test_router_normalizes_then_applies_routed_scale() -> None:
    """Normalize selected bias-free scores to one before multiplying by route scale."""
    router = LingBotVideoRouter(1, 3, 2, "sigmoid", True, None, None, 2.5)
    with torch.no_grad():
        router.weight[:, 0].copy_(torch.tensor([2.0, 1.0, -1.0]))

    indices, top_scores, _, scores, _ = router(torch.ones(1, 1))
    selected_scores = scores.gather(1, indices)
    expected = selected_scores / selected_scores.sum(dim=-1, keepdim=True) * 2.5

    assert_close(top_scores, expected)
    assert_close(top_scores.sum(dim=-1), torch.tensor([2.5]))


def test_sparse_restore_uses_fp32_weighted_sum() -> None:
    """Restore expert-major rows to token-major order and weight each selected expert."""
    expert_output = torch.tensor(
        [[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0]],
        dtype=torch.bfloat16,
    )
    sorted_positions = torch.tensor([0, 2, 1, 3])
    sorted_scores = torch.tensor([0.25, 0.5, 0.75, 0.5])

    restored = LingBotVideoSparseMoeBlock._restore_tokens(expert_output,
                                                          sorted_positions,
                                                          sorted_scores,
                                                          num_tokens=2,
                                                          top_k=2)

    assert restored.dtype == torch.bfloat16
    assert_close(restored.float(), torch.tensor([[2.5, 3.5], [20.0, 30.0]]))


def test_grouped_expert_padding_preserves_segments() -> None:
    """Build aligned expert segments without per-expert device scalar reads."""
    tokens = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    counts = torch.tensor([2, 0, 1], dtype=torch.int64)

    input_shape, padded, indices, aligned_counts = LingBotVideoSparseMoeBlock._pad_grouped_tokens(tokens,
                                                                                                  counts,
                                                                                                  align=2)

    assert input_shape == torch.Size((4, 4))
    assert torch.equal(aligned_counts, torch.tensor([2, 2, 2], dtype=torch.int32))
    assert torch.equal(indices[:6], torch.tensor([0, 1, 3, 3, 2, 3]))
    assert_close(padded[:6], torch.vstack((tokens[:2], torch.zeros(2, 4), tokens[2:], torch.zeros(1, 4))))


def test_rotary_precomputed_maxima_matches_tensor_reduction() -> None:
    """Use known grid maxima without changing the gathered rotary frequencies."""
    positions = _make_joint_position_ids(3, 2, 2, 2, torch.device("cpu"))
    dynamic = LingBotVideoRotaryEmbedding((2, 2, 2), (2, 1, 1), 256.0)
    precomputed = LingBotVideoRotaryEmbedding((2, 2, 2), (2, 1, 1), 256.0)

    expected = dynamic(positions)
    actual = precomputed(positions, maxima=(5, 1, 1))

    assert torch.equal(actual, expected)
    assert precomputed.axes_lens == dynamic.axes_lens


def test_sparse_moe_cpu_fallback_matches_direct_expert_sum() -> None:
    """Exercise CPU dispatch and compare routed expert output with a direct calculation."""
    torch.manual_seed(7)
    block = LingBotVideoSparseMoeBlock(
        hidden_size=4,
        intermediate_size=8,
        num_experts=3,
        top_k=2,
        moe_intermediate_size=5,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.5,
        n_shared_experts=None,
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.25, 0.25)
    hidden_states = torch.randn(2, 3, 4)
    tokens = hidden_states.reshape(-1, 4)
    top_indices, top_scores, _, _, _ = block.router(tokens)
    expected = torch.zeros_like(tokens, dtype=torch.float32)
    for token_index in range(tokens.shape[0]):
        for choice_index in range(top_indices.shape[1]):
            expert_index = int(top_indices[token_index, choice_index])
            expert_input = tokens[token_index]
            expert_hidden = F.silu(expert_input @ block.experts.w1[expert_index].T)
            expert_hidden = expert_hidden * (expert_input @ block.experts.w3[expert_index].T)
            expert_output = expert_hidden @ block.experts.w2[expert_index].T
            expected[token_index] += expert_output.float() * top_scores[token_index, choice_index].float()

    actual = block(hidden_states)

    assert_close(actual.reshape(-1, 4).float(), expected, atol=1e-6, rtol=1e-6)


def test_block_selection_preserves_dense_layers() -> None:
    """Use Dense MLPs unless the configured sparse schedule selects the layer."""
    common = {
        "hidden_size": 12,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "norm_eps": 1e-6,
        "qkv_bias": False,
        "out_bias": True,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 6,
        "decoder_sparse_step": 2,
        "mlp_only_layers": (),
        "n_shared_experts": 1,
        "score_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
    }
    dense_variant = LingBotVideoBlock(**common, num_experts=0, layer_idx=1)
    scheduled_dense = LingBotVideoBlock(**common, num_experts=4, layer_idx=0)
    scheduled_sparse = LingBotVideoBlock(**common, num_experts=4, layer_idx=1)

    assert isinstance(dense_variant.ffn, LingBotVideoMLP)
    assert isinstance(scheduled_dense.ffn, LingBotVideoMLP)
    assert isinstance(scheduled_sparse.ffn, LingBotVideoSparseMoeBlock)
    assert set(dense_variant.ffn.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }


def test_released_moe_meta_state_names_shapes_and_dtype_policy() -> None:
    """Build the released 30B MoE state surface on meta without materializing weights."""
    released_config = {
        "patch_size": [1, 2, 2],
        "in_channels": 16,
        "out_channels": 16,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "depth": 48,
        "intermediate_size": 6144,
        "text_dim": 2560,
        "freq_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 256.0,
        "axes_dims": [32, 48, 48],
        "axes_lens": [4096, 512, 512],
        "qkv_bias": False,
        "out_bias": True,
        "patch_embed_bias": True,
        "timestep_mlp_bias": True,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "n_shared_experts": 1,
        "score_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
    }
    with set_default_dtype(torch.bfloat16), torch.device("meta"):
        model = LingBotVideoTransformer3DModel(LingBotVideoConfig(), released_config)
    state = model.state_dict()

    assert state["blocks.0.ffn.router.weight"].shape == (128, 2048)
    assert state["blocks.0.ffn.router.e_score_correction_bias"].shape == (128, )
    assert state["blocks.0.ffn.experts.w1"].shape == (128, 768, 2048)
    assert state["blocks.0.ffn.experts.w2"].shape == (128, 2048, 768)
    assert state["blocks.0.ffn.experts.w3"].shape == (128, 768, 2048)
    assert state["blocks.0.ffn.shared_experts.gate_proj.weight"].shape == (768, 2048)
    assert "blocks.47.ffn.router.weight" in state
    assert state["blocks.0.ffn.router.weight"].is_meta
    assert model._get_parameter_dtype("blocks.0.ffn.router.weight", torch.bfloat16) == torch.float32
    assert model._get_parameter_dtype("blocks.0.ffn.experts.w1", torch.bfloat16) == torch.bfloat16
    assert model._get_parameter_dtype("blocks.0.ffn.shared_experts.gate_proj.weight", torch.bfloat16) == torch.bfloat16
