# SPDX-License-Identifier: Apache-2.0
"""Native H3 vision batching against the original per-frame SDPA loop.

These use synthetic weights, not checkpoint or end-to-end quality evidence.
CUDA cases skip on CPU hosts; their collection does not establish GPU parity.
"""
import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import fastvideo.envs as envs
import fastvideo.models.encoders.minimax_h3_qwen3_vl as vision
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLArchConfig, MiniMaxH3Qwen3VLConfig


_SDPA = F.scaled_dot_product_attention


def _config():
    return MiniMaxH3Qwen3VLConfig(arch_config=MiniMaxH3Qwen3VLArchConfig(
        hidden_size=32, vision_out_hidden_size=32,
        vision_hidden_size=32, vision_num_heads=4, vision_intermediate_size=48,
        vision_depth=3, vision_in_channels=3, vision_patch_size=2,
        vision_temporal_patch_size=1, vision_spatial_merge_size=2,
        vision_num_position_embeddings=16, vision_deepstack_visual_indexes=(0, 2),
    ))


def _inputs(lengths, dtype=torch.float32, device="cpu"):
    torch.manual_seed(1901)
    hidden = torch.randn(sum(lengths), 32, dtype=dtype, device=device)
    angles = torch.randn(sum(lengths), 4, device=device).repeat(1, 2)
    return hidden, (angles.cos(), angles.sin())


def _legacy_attention(module, hidden, lengths, positions):
    """Independent original loop; never calls grouping or production forward."""
    tokens = hidden.shape[0]
    query, key, value = module.qkv(hidden).reshape(tokens, 3, module.num_heads, module.head_dim).unbind(1)
    cos, sin = (part.unsqueeze(1).float() for part in positions)

    def rotate(tensor):
        left, right = tensor.float().chunk(2, dim=-1)
        return (tensor.float() * cos + torch.cat((-right, left), dim=-1) * sin).to(tensor.dtype)

    query, key = rotate(query), rotate(key)
    outputs = []
    offset = 0
    for length in lengths:
        stop = offset + length
        output = _SDPA(query[offset:stop].transpose(0, 1).unsqueeze(0),
                       key[offset:stop].transpose(0, 1).unsqueeze(0),
                       value[offset:stop].transpose(0, 1).unsqueeze(0),
                       dropout_p=0.0, is_causal=False, scale=module.scaling)
        outputs.append(output.transpose(1, 2))
        offset = stop
    return module.proj(torch.cat(outputs, dim=1).reshape(tokens, -1))


def _assert_parity(dtype, device):
    torch.manual_seed(71)
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config()).to(device=device, dtype=dtype).eval()
    # Repeated frames, a single frame, mixed consecutive runs and alternating
    # sizes exercise ordering and prevent a global bucket/reorder shortcut.
    for lengths in ([8] * 5, [8], [4, 4, 8, 8, 8, 4], [4, 8, 4, 8]):
        hidden, positions = _inputs(lengths, dtype, device)
        if lengths == [4, 4, 8, 8, 8, 4]:
            hidden = torch.stack((hidden, hidden), dim=-1)[..., 0]
            assert not hidden.is_contiguous()
        with torch.no_grad():
            expected = _legacy_attention(attention, hidden, lengths, positions)
            actual = attention(hidden, lengths, positions, vision._vision_attention_groups(lengths, 3))
        tolerance = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=8e-3, rtol=2e-2)
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_attention_matches_original_segment_loop(dtype):
    _assert_parity(dtype, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native SDPA parity")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_attention_matches_original_segment_loop(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    _assert_parity(dtype, "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for production-head SDPA parity")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_attention_with_production_head_geometry(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    torch.manual_seed(1903)
    config = MiniMaxH3Qwen3VLConfig()
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(config).to(device="cuda", dtype=dtype).eval()
    assert (attention.num_heads, attention.head_dim) == (16, 72)
    lengths = [32, 32, 32]
    hidden = torch.randn(sum(lengths), 1152, device="cuda", dtype=dtype)
    angles = torch.randn(sum(lengths), 36, device="cuda").repeat(1, 2)
    positions = (angles.cos(), angles.sin())
    with torch.no_grad(), envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(8):
        expected = _legacy_attention(attention, hidden, lengths, positions)
        actual = attention(hidden, lengths, positions)
    tolerance = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=8e-3, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("cap", [1, 8])
def test_math_attention_keeps_output_views_until_final_concatenation(cap):
    """Head-major SDPA output must not be copied once per frame before cat."""
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config()).eval()
    lengths = [8, 8, 8, 4]
    hidden, positions = _inputs(lengths)
    output_storages = set()
    concatenated_frames = []
    original_cat = torch.cat

    def observe_sdpa(*args, **kwargs):
        output = _SDPA(*args, **kwargs)
        output_storages.add(output.untyped_storage().data_ptr())
        return output

    def observe_cat(tensors, dim=0, **kwargs):
        if dim == 0:
            concatenated_frames.extend(tensors)
            assert all(tensor.untyped_storage().data_ptr() in output_storages for tensor in tensors)
        return original_cat(tensors, dim=dim, **kwargs)

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(cap):
            with patch.object(vision.F, "scaled_dot_product_attention", side_effect=observe_sdpa):
                with patch.object(vision.torch, "cat", side_effect=observe_cat):
                    actual = attention(hidden, lengths, positions)
    assert len(concatenated_frames) == len(lengths)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_batch_cap_reduces_sdpa_calls_without_combining_different_sizes():
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config()).eval()
    for lengths, expected_shapes in (
        ([8] * 17, [(8, 4, 8, 8), (8, 4, 8, 8), (1, 4, 8, 8)]),
        ([4, 4, 8, 4, 4], [(2, 4, 4, 8), (1, 4, 8, 8), (2, 4, 4, 8)]),
    ):
        hidden, positions = _inputs(lengths)
        with torch.no_grad(), patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
            attention(hidden, lengths, positions, vision._vision_attention_groups(lengths, 8))
        assert [tuple(call.args[0].shape) for call in calls.call_args_list] == expected_shapes


def test_perturbing_one_frame_does_not_change_other_frames():
    lengths = [8] * 5
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config()).eval()
    hidden, positions = _inputs(lengths)
    changed = hidden.clone()
    changed[8:16] += torch.arange(32).float()
    groups = vision._vision_attention_groups(lengths, 8)
    with torch.no_grad():
        original = attention(hidden, lengths, positions, groups)
        perturbed = attention(changed, lengths, positions, groups)
    assert not torch.allclose(original[8:16], perturbed[8:16])
    torch.testing.assert_close(original[:8], perturbed[:8], atol=0, rtol=0)
    torch.testing.assert_close(original[16:], perturbed[16:], atol=0, rtol=0)


def test_grad_enabled_uses_singletons_and_matches_original_gradients():
    torch.manual_seed(79)
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config())
    reference = copy.deepcopy(attention)
    lengths = [8] * 4
    hidden, positions = _inputs(lengths)
    hidden.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    with patch(f"{__name__}._SDPA", wraps=_SDPA) as reference_calls:
        expected = _legacy_attention(reference, reference_hidden, lengths, positions)
    with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
        actual = attention(hidden, lengths, positions, vision._vision_attention_groups(lengths, 8))
    assert len(calls.call_args_list) == len(lengths)
    assert all(call.args[0].shape[0] == 1 for call in calls.call_args_list)
    assert [[tensor.stride() for tensor in call.args[:3]] for call in calls.call_args_list] == [
        [tensor.stride() for tensor in call.args[:3]] for call in reference_calls.call_args_list
    ]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=0, rtol=0)
    for parameter, reference_parameter in zip(attention.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0, rtol=0)


def _legacy_model(model, pixels, grid):
    # Keep patch/position/merger boundaries identical; replace every attention
    # forward with the independent pre-batching implementation above.
    with patch.object(vision.MiniMaxH3Qwen3VLVisionAttention, "forward",
                      lambda self, hidden, lengths, positions, *args: _legacy_attention(self, hidden, lengths, positions)):
        return model(pixels, grid)


def test_tiny_model_and_deepstack_match_original_loop_and_build_groups_once(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(8))
    torch.manual_seed(83)
    model = vision.MiniMaxH3Qwen3VLVisionModel(_config()).eval()
    # Temporal frames split into independent attention segments, including
    # adjacent entries with equal area and a later unequal area.
    grid = torch.tensor([[3, 2, 4], [2, 2, 4], [1, 2, 2]])
    pixels = torch.randn(44, 12)
    with torch.no_grad():
        expected, expected_deepstack = _legacy_model(model, pixels, grid)
        for cap, call_count in ((1, 18), (8, 6), (1, 18)):
            with envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(cap):
                with patch.object(vision, "_vision_attention_groups", wraps=vision._vision_attention_groups) as groups:
                    with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                        actual, actual_deepstack = model(pixels, grid)
            assert groups.call_count == 1
            assert calls.call_count == call_count
            assert actual.shape == (11, model.config.vision_out_hidden_size)
            assert len(actual_deepstack) == len(expected_deepstack) == 2
            for observed, reference in zip([actual, *actual_deepstack], [expected, *expected_deepstack], strict=True):
                torch.testing.assert_close(observed, reference, atol=2e-5, rtol=2e-5)


def test_three_argument_api_reads_batch_setting_each_forward():
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config()).eval()
    hidden, positions = _inputs([8] * 17)
    with torch.no_grad():
        with patch(f"{__name__}._SDPA", wraps=_SDPA) as reference_calls:
            expected_attention = _legacy_attention(attention, hidden, [8] * 17, positions)
        for cap, expected_batches in ((1, [1] * 17), (8, [8, 8, 1]), (1, [1] * 17)):
            with envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(cap):
                with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                    actual_attention = attention(hidden, [8] * 17, positions)
            assert [call.args[0].shape[0] for call in calls.call_args_list] == expected_batches
            torch.testing.assert_close(actual_attention, expected_attention, atol=2e-5, rtol=2e-5)
            if cap == 1:
                assert [[tensor.stride() for tensor in call.args[:3]] for call in calls.call_args_list] == [
                    [tensor.stride() for tensor in call.args[:3]] for call in reference_calls.call_args_list
                ]

        block = vision.MiniMaxH3Qwen3VLVisionBlock(_config()).eval()
        residual = hidden + _legacy_attention(block.attn, block.norm1(hidden), [8] * 17, positions)
        expected = residual + block.mlp(block.norm2(residual))
        with envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(8):
            actual = block(hidden, [8] * 17, positions)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("cap", [0, -1])
def test_invalid_batch_setting_fails_before_projection(env_overrides, cap):
    env_overrides.enter_context(envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(cap))
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(_config()).eval()
    hidden, positions = _inputs([8])
    with patch.object(attention.qkv, "forward", side_effect=AssertionError("projection ran before validation")):
        with pytest.raises(ValueError, match="positive|greater than|at least"):
            attention(hidden, [8], positions)
    model = vision.MiniMaxH3Qwen3VLVisionModel(_config()).eval()
    with patch.object(model.patch_embed, "forward", side_effect=AssertionError("patch embed ran before validation")):
        with pytest.raises(ValueError, match="positive|greater than|at least"):
            model(torch.randn(8, 12), torch.tensor([[1, 2, 4]]))
