# SPDX-License-Identifier: Apache-2.0
"""Packed H3 vision attention against the original independent-frame loop.

CPU fake-kernel checks establish the packing/dispatch contract, not CUDA kernel
parity or speed. Real FA2 tests skip explicitly without the hardware/package.
"""
import builtins
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

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


def _attention(backend="sdpa", dtype=torch.float32):
    with envs.FASTVIDEO_H3_VISION_ATTN_BACKEND.override(backend):
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


def _fake_varlen(query, key, value, cu_q, cu_k, max_q, max_k, *, dropout_p, softmax_scale, causal):
    """CPU semantic substitute for FA2, intentionally using only its offsets."""
    assert cu_q is cu_k
    assert cu_q.dtype == torch.int32
    assert cu_q.device == query.device
    offsets = cu_q.tolist()
    assert offsets[0] == 0 and offsets[-1] == query.shape[0]
    assert max_q == max_k == max(b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True))
    assert isinstance(max_q, int)
    assert dropout_p == 0.0 and causal is False
    assert query.dtype == key.dtype == value.dtype
    assert softmax_scale == query.shape[-1] ** -0.5
    outputs = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        output = _SDPA(query[start:end].transpose(0, 1).unsqueeze(0),
                       key[start:end].transpose(0, 1).unsqueeze(0),
                       value[start:end].transpose(0, 1).unsqueeze(0),
                       dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        outputs.append(output.transpose(1, 2).squeeze(0))
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[8] * 5, [8], [4, 6, 4], [4, 8, 4, 8]])
def test_packed_contract_matches_original_rope_order_and_attention(dtype, lengths):
    attention = _attention("flash_attn_2", dtype)
    hidden, positions = _inputs(lengths, dtype)
    hidden = torch.stack((hidden, hidden), dim=-1)[..., 0]
    assert not hidden.is_contiguous()
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with patch.object(vision, "_can_use_vision_flash_attn_2", return_value=True):
            with patch.object(vision, "_vision_flash_attn_varlen", side_effect=_fake_varlen) as calls:
                actual = attention(hidden, lengths, positions)
    assert calls.call_count == 1
    assert calls.call_args.args[0].shape == (sum(lengths), 4, 8)
    assert calls.call_args.args[0].dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_packed_frames_remain_independent():
    lengths = [4, 6, 4]
    attention = _attention("flash_attn_2")
    hidden, positions = _inputs(lengths)
    changed = hidden.clone()
    changed[4:10] += torch.arange(32).float()
    with torch.no_grad(), patch.object(vision, "_can_use_vision_flash_attn_2", return_value=True):
        with patch.object(vision, "_vision_flash_attn_varlen", side_effect=_fake_varlen):
            original = attention(hidden, lengths, positions)
            perturbed = attention(changed, lengths, positions)
    assert not torch.allclose(original[4:10], perturbed[4:10])
    torch.testing.assert_close(original[:4], perturbed[:4], atol=0, rtol=0)
    torch.testing.assert_close(original[10:], perturbed[10:], atol=0, rtol=0)


def _legacy_model(model, pixels, grid):
    with patch.object(vision.MiniMaxH3Qwen3VLVisionAttention, "forward",
                      lambda self, hidden, lengths, positions, *args: _legacy_attention(self, hidden, lengths, positions)):
        return model(pixels, grid)


def test_tiny_model_deepstack_and_mixed_videos_build_metadata_once():
    torch.manual_seed(83)
    with envs.FASTVIDEO_H3_VISION_ATTN_BACKEND.override("flash_attn_2"):
        model = vision.MiniMaxH3Qwen3VLVisionModel(_config()).eval()
    # Multiple videos/images with different frame counts and spatial areas.
    grid = torch.tensor([[3, 2, 4], [2, 2, 4], [1, 2, 2]])
    pixels = torch.randn(44, 12)
    with torch.no_grad():
        expected, expected_deepstack = _legacy_model(model, pixels, grid)
        with patch.object(vision, "_can_use_vision_flash_attn_2", return_value=True):
            with patch.object(vision, "_vision_attention_metadata", wraps=vision._vision_attention_metadata) as metadata:
                with patch.object(vision, "_vision_flash_attn_varlen", side_effect=_fake_varlen) as calls:
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


@pytest.mark.parametrize("backend", ["sdpa", "flash_attn_2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_fallback_keeps_original_calls_and_never_imports_flash(backend, dtype):
    attention = _attention(backend, dtype)
    hidden, positions = _inputs([4, 6, 4], dtype)
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, [4, 6, 4], positions)
        with patch.object(vision, "_vision_flash_attn_varlen", side_effect=AssertionError("attempted CUDA import")):
            with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                actual = attention(hidden, [4, 6, 4], positions)
    assert [call.args[0].shape for call in calls.call_args_list] == [(1, 4, 4, 8), (1, 4, 6, 8), (1, 4, 4, 8)]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype,head_dim,capability,hip,eligible", [
    (torch.float16, 72, (8, 0), None, True),
    (torch.bfloat16, 72, (12, 1), None, True),  # Source eligibility, not GB10 kernel proof.
    (torch.float32, 72, (9, 0), None, False),
    (torch.float16, 72, (7, 5), None, False),
    (torch.float16, 70, (9, 0), None, False),
    (torch.float16, 264, (9, 0), None, False),
    (torch.float16, 72, (9, 0), "test-rocm", False),
])
def test_flash_eligibility_does_not_infer_support_from_sdpa_dispatch(dtype, head_dim, capability, hip, eligible):
    tensor = SimpleNamespace(device=torch.device("cuda"), dtype=dtype)
    with torch.no_grad(), patch.object(torch.version, "hip", hip):
        with patch.object(torch.cuda, "get_device_capability", return_value=capability):
            assert vision._can_use_vision_flash_attn_2(tensor, head_dim) is eligible


def test_backend_is_selected_at_construction():
    with envs.FASTVIDEO_H3_VISION_ATTN_BACKEND.override("flash_attn_2"):
        model = vision.MiniMaxH3Qwen3VLVisionModel(_config())
    with envs.FASTVIDEO_H3_VISION_ATTN_BACKEND.override("sdpa"):
        assert model.attn_implementation == "flash_attn_2"
        assert all(block.attn.attn_implementation == "flash_attn_2" for block in model.blocks)
        assert vision.MiniMaxH3Qwen3VLVisionAttention(_config()).attn_implementation == "sdpa"
    with envs.FASTVIDEO_H3_VISION_ATTN_BACKEND.override("invalid"):
        with pytest.raises(envs.EnvVarError):
            vision.MiniMaxH3Qwen3VLVisionModel(_config())


def test_grad_enabled_keeps_original_strides_and_gradients():
    attention = _attention("flash_attn_2")
    reference = copy.deepcopy(attention)
    lengths = [4, 6, 4]
    hidden, positions = _inputs(lengths)
    hidden.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    with patch(f"{__name__}._SDPA", wraps=_SDPA) as reference_calls:
        expected = _legacy_attention(reference, reference_hidden, lengths, positions)
    with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
        actual = attention(hidden, lengths, positions)
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
    fake_cuda = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
    assert not vision._can_use_vision_flash_attn_2(fake_cuda, 72)


def test_flash_dependency_is_lazy_and_import_failure_is_actionable():
    original_import = builtins.__import__

    def without_flash(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("missing test dependency")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=without_flash):
        attention = _attention("flash_attn_2")
        hidden, positions = _inputs([4])
        with torch.no_grad():
            attention(hidden, [4], positions)
        with pytest.raises(ImportError, match="compatible flash-attn installation"):
            vision._vision_flash_attn_varlen()


@pytest.mark.parametrize("error", [RuntimeError("kernel defect"), torch.OutOfMemoryError("test OOM")])
def test_eligible_flash_kernel_errors_are_not_silently_retried(error):
    attention = _attention("flash_attn_2")
    hidden, positions = _inputs([4, 6, 4])
    with torch.no_grad(), patch.object(vision, "_can_use_vision_flash_attn_2", return_value=True):
        with patch.object(vision, "_vision_flash_attn_varlen", side_effect=error):
            with patch.object(vision.F, "scaled_dot_product_attention", side_effect=AssertionError("retried SDPA")):
                with pytest.raises(type(error), match=str(error)):
                    attention(hidden, [4, 6, 4], positions)


@pytest.mark.parametrize("lengths", [[], [0], [-1, 5], [3]])
def test_invalid_lengths_fail_before_projection(lengths):
    attention = _attention()
    hidden, positions = _inputs([4])
    with patch.object(attention.qkv, "forward", side_effect=AssertionError("projected invalid input")):
        with pytest.raises(ValueError, match="sequence lengths"):
            attention(hidden, lengths, positions)


def test_bad_metadata_is_rejected_before_flash_call():
    attention = _attention("flash_attn_2")
    hidden, positions = _inputs([4, 6, 4])
    metadata = vision._vision_attention_metadata([6, 4, 4], hidden.device)
    with torch.no_grad(), patch.object(vision, "_can_use_vision_flash_attn_2", return_value=True):
        with patch.object(vision, "_vision_flash_attn_varlen", side_effect=AssertionError("bad metadata reached kernel")):
            with pytest.raises(ValueError, match="metadata must match"):
                attention(hidden, [4, 6, 4], positions, metadata)
    with pytest.raises(ValueError, match="int32"):
        vision._vision_attention_metadata([2**31], hidden.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual FA2 parity")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[32, 32, 32], [16, 40, 24]])
def test_real_cuda_flash_attn_2_with_production_head_geometry(dtype, lengths):
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FA2 route requires NVIDIA Ampere or newer")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    pytest.importorskip("flash_attn", reason="Compatible flash-attn installation required", exc_type=ImportError)
    with envs.FASTVIDEO_H3_VISION_ATTN_BACKEND.override("flash_attn_2"):
        attention = vision.MiniMaxH3Qwen3VLVisionAttention(MiniMaxH3Qwen3VLConfig()).to("cuda", dtype=dtype).eval()
    assert (attention.num_heads, attention.head_dim) == (16, 72)
    hidden, positions = _inputs(lengths, dtype, "cuda", hidden_size=1152, head_dim=72)
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with patch.object(vision, "_vision_flash_attn_varlen", wraps=vision._vision_flash_attn_varlen) as calls:
            actual = attention(hidden, lengths, positions)
    assert calls.call_count == 1
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=2e-2)
