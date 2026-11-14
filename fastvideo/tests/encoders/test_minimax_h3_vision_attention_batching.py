# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch varlen vision attention against the original SDPA loop.

CPU tests exercise the real SDPA fallback and substitute a semantic reference
for native varlen packing; they do not establish GPU parity or performance.
Real CUDA forward/backward tests skip only when required hardware is absent.
"""
import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

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


def _attention(dtype=torch.float32):
    return vision.MiniMaxH3Qwen3VLVisionAttention(_config()).to(dtype=dtype).eval()


def _inputs(lengths, dtype=torch.float32, device="cpu", hidden_size=32, head_dim=8):
    torch.manual_seed(1901)
    hidden = torch.randn(sum(lengths), hidden_size, dtype=dtype, device=device)
    angles = torch.randn(sum(lengths), head_dim // 2, device=device).repeat(1, 2)
    return hidden, (angles.cos(), angles.sin())


def _legacy_attention(module, hidden, lengths, positions):
    """Independent original loop; never calls production attention forward."""
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


def _semantic_varlen(query, key, value, cu_q, cu_k, max_q, max_k, *, scale, window_size):
    """CPU reference consuming only the native API's packed-frame boundaries."""
    assert cu_q is cu_k
    assert cu_q.dtype == torch.int32 and cu_q.is_contiguous()
    assert cu_q.device == query.device
    offsets = cu_q.tolist()
    assert offsets[0] == 0 and offsets[-1] == query.shape[0]
    assert max_q == max_k == max(b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True))
    assert isinstance(max_q, int)
    assert query.dtype == key.dtype == value.dtype
    assert all(tensor.stride(-1) == 1 for tensor in (query, key, value))
    assert scale == query.shape[-1] ** -0.5
    assert window_size == (-1, -1)
    outputs = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        output = _SDPA(query[start:end].transpose(0, 1).unsqueeze(0),
                       key[start:end].transpose(0, 1).unsqueeze(0),
                       value[start:end].transpose(0, 1).unsqueeze(0),
                       dropout_p=0.0, is_causal=False, scale=scale)
        outputs.append(output.transpose(1, 2).squeeze(0))
    return torch.cat(outputs, dim=0)


def test_cpu_semantic_packing_preserves_rope_dtype_and_original_order():
    dtype, lengths = torch.bfloat16, [4, 6, 4]
    attention = _attention(dtype)
    hidden, positions = _inputs(lengths, dtype)
    hidden = torch.stack((hidden, hidden), dim=-1)[..., 0]
    assert not hidden.is_contiguous()
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with patch.object(torch.backends.cuda, "can_use_flash_attention", return_value=True):
            with patch.object(vision, "varlen_attn", side_effect=_semantic_varlen) as calls:
                actual = attention(hidden, lengths, positions)
    assert calls.call_count == 1
    assert calls.call_args.args[0].shape == (sum(lengths), 4, 8)
    assert calls.call_args.args[0].dtype == dtype
    expected_offsets = [0]
    for length in lengths:
        expected_offsets.append(expected_offsets[-1] + length)
    assert calls.call_args.args[3].tolist() == expected_offsets
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_fallback_matches_original_without_native_varlen(dtype):
    attention = _attention(dtype)
    lengths = [4, 6, 4]
    hidden, positions = _inputs(lengths, dtype)
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with patch.object(vision, "varlen_attn", side_effect=AssertionError("CPU used native varlen")):
            with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                actual = attention(hidden, lengths, positions)
    assert calls.call_count == len(lengths)
    assert all(call.args[0].shape[0] == 1 for call in calls.call_args_list)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("supported", [False, True])
def test_varlen_eligibility_reuses_pytorch_with_view_only_probe(supported):
    query, key, value = torch.randn(11, 3, 4, 72, dtype=torch.bfloat16).unbind(1)
    with patch.object(torch.backends.cuda, "can_use_flash_attention", return_value=supported) as check:
        assert vision._can_use_vision_varlen(query, key, value) is supported
    params = check.call_args.args[0]
    for original, probed in zip((query, key, value), (params.query, params.key, params.value), strict=True):
        assert probed.shape == (1, 4, 11, 72)
        assert probed.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        assert probed.stride(-1) == original.stride(-1)
    assert params.attn_mask is None and params.dropout == 0.0 and not params.is_causal
    assert not params.enable_gqa


def test_varlen_rejects_head_alignment_that_dense_sdpa_can_pad():
    query = torch.randn(11, 4, 70, dtype=torch.bfloat16)
    with patch.object(torch.backends.cuda, "can_use_flash_attention", return_value=True):
        assert not vision._can_use_vision_varlen(query, query, query)


def test_tiny_model_deepstack_and_mixed_videos_build_metadata_once():
    torch.manual_seed(83)
    model = vision.MiniMaxH3Qwen3VLVisionModel(_config()).eval()
    grid = torch.tensor([[3, 2, 4], [2, 2, 4], [1, 2, 2]])
    pixels = torch.randn(44, 12)
    with torch.no_grad():
        with patch.object(vision.MiniMaxH3Qwen3VLVisionAttention, "forward",
                          lambda self, hidden, lengths, positions, *args: _legacy_attention(self, hidden, lengths, positions)):
            expected, expected_deepstack = model(pixels, grid)
        with patch.object(vision, "_vision_attention_metadata", wraps=vision._vision_attention_metadata) as metadata:
            with patch.object(vision, "_can_use_vision_varlen", return_value=True):
                with patch.object(vision, "varlen_attn", side_effect=_semantic_varlen) as calls:
                    actual, actual_deepstack = model(pixels, grid)
    assert metadata.call_count == 1
    assert calls.call_count == model.config.vision_depth
    offsets = [call.args[3] for call in calls.call_args_list]
    assert all(tensor is offsets[0] for tensor in offsets)
    assert offsets[0].tolist() == [0, 8, 16, 24, 32, 40, 44]
    assert actual.shape == (11, model.config.vision_out_hidden_size)
    assert len(actual_deepstack) == len(expected_deepstack) == 2
    for observed, reference in zip([actual, *actual_deepstack], [expected, *expected_deepstack], strict=True):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


def test_native_kernel_oom_propagates_without_sdpa_fallback():
    error = torch.OutOfMemoryError("execution OOM")
    attention = _attention()
    hidden, positions = _inputs([4, 6, 4])
    with patch.object(vision, "_can_use_vision_varlen", return_value=True), \
            patch.object(vision, "varlen_attn", side_effect=error) as calls:
        with patch.object(vision.F, "scaled_dot_product_attention", side_effect=AssertionError("fell back to SDPA")):
            with pytest.raises(type(error), match=str(error)):
                attention(hidden, [4, 6, 4], positions)
    assert calls.call_count == 1


@pytest.mark.parametrize("lengths", [[], [0]])
def test_invalid_lengths_fail_before_projection(lengths):
    attention = _attention()
    hidden, positions = _inputs([4])
    with patch.object(attention.qkv, "forward", side_effect=AssertionError("projected invalid input")):
        with pytest.raises(ValueError, match="sequence lengths"):
            attention(hidden, lengths, positions)


def test_metadata_rejects_int32_overflow_before_allocation():
    with pytest.raises(ValueError, match="int32"):
        vision._vision_attention_metadata([2**31], 2**31, torch.device("cpu"))


def test_model_rejects_grid_token_mismatch_before_position_processing():
    model = vision.MiniMaxH3Qwen3VLVisionModel(_config()).eval()
    grid = torch.tensor([[1, 2, 2]])
    pixels = torch.randn(3, 12)
    with patch.object(model, "_interpolate_position_embeddings",
                      side_effect=AssertionError("processed invalid pixel/grid geometry")):
        with pytest.raises(ValueError, match="sequence lengths must sum"):
            model(pixels, grid)


def _cuda_attention(dtype):
    if torch.version.hip is not None:
        pytest.skip("Native varlen path is enabled for NVIDIA CUDA")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    probe = torch.empty(1, 16, 1, 72, device="cuda", dtype=dtype)
    params = torch.backends.cuda.SDPAParams(probe, probe, probe, None, 0.0, False, False)
    if not torch.backends.cuda.can_use_flash_attention(params):
        pytest.skip("PyTorch Flash attention is unavailable for this device/build/settings")
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(MiniMaxH3Qwen3VLConfig()).to("cuda", dtype=dtype).eval()
    assert (attention.num_heads, attention.head_dim) == (16, 72)
    return attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP32 compatibility")
def test_cuda_fp32_fallback_matches_original():
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(MiniMaxH3Qwen3VLConfig()).to("cuda", dtype=torch.float32).eval()
    lengths = [16, 40, 24]
    hidden, positions = _inputs(lengths, torch.float32, "cuda", hidden_size=1152, head_dim=72)
    with torch.no_grad(), patch.object(vision, "varlen_attn", side_effect=AssertionError("FP32 used native varlen")):
        expected = _legacy_attention(attention, hidden, lengths, positions)
        actual = attention(hidden, lengths, positions)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native PyTorch varlen parity")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[32, 32, 32], [128, 260, 64]])
def test_cuda_native_varlen_forward_with_production_heads(dtype, lengths):
    attention = _cuda_attention(dtype)
    hidden, positions = _inputs(lengths, dtype, "cuda", hidden_size=1152, head_dim=72)
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with patch.object(vision, "varlen_attn", wraps=vision.varlen_attn) as calls:
            actual = attention(hidden, lengths, positions)
    assert calls.call_count == 1
    assert calls.call_args.args[3].device == hidden.device
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native PyTorch varlen backward parity")
def test_cuda_native_varlen_backward_with_production_heads():
    dtype = torch.bfloat16
    attention = _cuda_attention(dtype)
    reference = copy.deepcopy(attention)
    lengths = [128, 260, 64]
    hidden, positions = _inputs(lengths, dtype, "cuda", hidden_size=1152, head_dim=72)
    hidden.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    expected = _legacy_attention(reference, reference_hidden, lengths, positions)
    with patch.object(vision, "varlen_attn", wraps=vision.varlen_attn) as calls:
        actual = attention(hidden, lengths, positions)
    assert calls.call_count == 1
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=2e-2)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-2, rtol=5e-2)
    for parameter, reference_parameter in zip(attention.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=2e-2, rtol=5e-2)
