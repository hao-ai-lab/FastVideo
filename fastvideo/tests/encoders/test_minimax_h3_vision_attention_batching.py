# SPDX-License-Identifier: Apache-2.0
"""H3 vision frame batching against the original per-frame SDPA loop.

CPU checks mock the selection decision, not GPU execution or speed. CUDA cases
exercise real selected backends and skip explicitly when unavailable.
"""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[8] * 5, [8], [4, 4, 6, 6, 4], [4, 6, 4, 6]])
def test_equal_length_batching_matches_original_rope_order_and_attention(dtype, lengths):
    attention = _attention(dtype)
    hidden, positions = _inputs(lengths, dtype)
    hidden = torch.stack((hidden, hidden), dim=-1)[..., 0]
    assert not hidden.is_contiguous()
    with torch.no_grad():
        expected = _legacy_attention(attention, hidden, lengths, positions)
        with patch.object(vision, "_vision_sdpa_uses_fused_backend", return_value=True):
            with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                actual = attention(hidden, lengths, positions)
    groups = vision._vision_attention_groups(lengths)
    assert [tuple(call.args[0].shape) for call in calls.call_args_list] == [
        (count, attention.num_heads, length, attention.head_dim) for length, count in groups
    ]
    tolerance = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=8e-3, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("backend,expected", [
    (SDPBackend.FLASH_ATTENTION, True),
    (SDPBackend.EFFICIENT_ATTENTION, True),
    (SDPBackend.CUDNN_ATTENTION, True),
    (SDPBackend.MATH, False),
    (SDPBackend.ERROR, False),
    (SDPBackend.OVERRIDEABLE, False),
])
def test_only_selected_fused_backends_allow_batching(backend, expected):
    tensors = [SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16) for _ in range(3)]
    with torch.no_grad(), patch.object(torch, "is_autocast_enabled", return_value=False):
        with patch.object(torch, "_fused_sdp_choice", return_value=backend.value) as choice:
            assert vision._vision_sdpa_uses_fused_backend(*tensors, 0.125) is expected
    assert choice.call_args.args == tuple(tensors)
    assert choice.call_args.kwargs == dict(attn_mask=None, dropout_p=0.0, is_causal=False, scale=0.125)


@pytest.mark.parametrize("dtypes,expected", [
    ((torch.bfloat16,) * 3, True),
    ((torch.float32,) * 3, False),
    ((torch.bfloat16, torch.bfloat16, torch.float32), False),
])
def test_cuda_autocast_requires_no_implicit_cast_before_probe(dtypes, expected):
    tensors = [SimpleNamespace(device=torch.device("cuda"), dtype=dtype) for dtype in dtypes]
    with torch.no_grad(), patch.object(torch, "is_autocast_enabled", return_value=True):
        with patch.object(torch, "get_autocast_dtype", return_value=torch.bfloat16):
            with patch.object(torch, "_fused_sdp_choice", return_value=SDPBackend.FLASH_ATTENTION.value) as choice:
                assert vision._vision_sdpa_uses_fused_backend(*tensors, 0.125) is expected
    assert choice.call_count == int(expected)


@pytest.mark.parametrize("error", [TypeError("unsupported signature"), NotImplementedError("missing dispatch"),
                                    RuntimeError("no viable kernel")])
def test_private_selector_compatibility_failures_keep_singletons(error):
    tensor = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
    with torch.no_grad(), patch.object(torch, "_fused_sdp_choice", side_effect=error):
        assert not vision._vision_sdpa_uses_fused_backend(tensor, tensor, tensor, 0.125)


def test_missing_private_selector_keeps_singletons(monkeypatch):
    monkeypatch.delattr(torch, "_fused_sdp_choice")
    tensor = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
    with torch.no_grad():
        assert not vision._vision_sdpa_uses_fused_backend(tensor, tensor, tensor, 0.125)


def test_private_selector_oom_is_not_swallowed():
    tensor = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
    with torch.no_grad(), patch.object(torch, "_fused_sdp_choice", side_effect=torch.OutOfMemoryError("probe OOM")):
        with pytest.raises(torch.OutOfMemoryError, match="probe OOM"):
            vision._vision_sdpa_uses_fused_backend(tensor, tensor, tensor, 0.125)


def test_exact_probed_views_and_settings_are_used_by_batched_sdpa():
    attention = _attention()
    lengths = [4, 4, 6, 6, 4]
    hidden, positions = _inputs(lengths)
    probed = []
    projected_storage = []
    original_qkv = attention.qkv.forward

    def project(*args):
        tensor = original_qkv(*args)
        projected_storage.append(tensor.untyped_storage().data_ptr())
        return tensor

    def probe(query, key, value, scale):
        assert value.untyped_storage().data_ptr() == projected_storage[-1]  # No value repacking/copy.
        probed.append((query, key, value, scale))
        return True

    def execute(*args, **kwargs):
        if args[0].shape[0] > 1:
            checked = probed.pop(0)
            assert all(actual is inspected for actual, inspected in zip(args, checked[:3], strict=True))
            assert kwargs == dict(dropout_p=0.0, is_causal=False, scale=checked[3])
        return _SDPA(*args, **kwargs)

    with torch.no_grad(), patch.object(attention.qkv, "forward", side_effect=project):
        with patch.object(vision, "_vision_sdpa_uses_fused_backend", side_effect=probe):
            with patch.object(vision.F, "scaled_dot_product_attention", side_effect=execute):
                attention(hidden, lengths, positions)
    assert not probed


def test_perturbing_one_frame_preserves_all_other_frames():
    lengths = [4, 4, 6, 6, 4]
    attention = _attention()
    hidden, positions = _inputs(lengths)
    changed = hidden.clone()
    changed[4:8] += torch.arange(32).float()
    with torch.no_grad(), patch.object(vision, "_vision_sdpa_uses_fused_backend", return_value=True):
        original = attention(hidden, lengths, positions)
        perturbed = attention(changed, lengths, positions)
    assert not torch.allclose(original[4:8], perturbed[4:8])
    torch.testing.assert_close(original[:4], perturbed[:4], atol=0, rtol=0)
    torch.testing.assert_close(original[8:], perturbed[8:], atol=0, rtol=0)


def test_tiny_model_deepstack_and_mixed_videos_build_groups_once():
    torch.manual_seed(83)
    model = vision.MiniMaxH3Qwen3VLVisionModel(_config()).eval()
    grid = torch.tensor([[3, 2, 4], [2, 2, 4], [1, 2, 2]])
    pixels = torch.randn(44, 12)
    with torch.no_grad():
        with patch.object(vision.MiniMaxH3Qwen3VLVisionAttention, "forward",
                          lambda self, hidden, lengths, positions, *args: _legacy_attention(self, hidden, lengths, positions)):
            expected, expected_deepstack = model(pixels, grid)
        with patch.object(vision, "_vision_sdpa_uses_fused_backend", return_value=True):
            with patch.object(vision, "_vision_attention_groups", wraps=vision._vision_attention_groups) as groups:
                with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                    actual, actual_deepstack = model(pixels, grid)
    assert groups.call_count == 1
    assert calls.call_count == model.config.vision_depth * 2
    assert [call.args[0].shape[0] for call in calls.call_args_list] == [5, 1] * model.config.vision_depth
    assert actual.shape == (11, model.config.vision_out_hidden_size)
    assert len(actual_deepstack) == len(expected_deepstack) == 2
    for observed, reference in zip([actual, *actual_deepstack], [expected, *expected_deepstack], strict=True):
        torch.testing.assert_close(observed, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("allow_batch", [False, True])
def test_attention_outputs_remain_views_until_final_cat(allow_batch):
    attention = _attention()
    hidden, positions = _inputs([8, 8, 8, 4])
    storages = set()
    frames = []
    original_cat = torch.cat

    def observe_sdpa(*args, **kwargs):
        output = _SDPA(*args, **kwargs)
        storages.add(output.untyped_storage().data_ptr())
        return output

    def observe_cat(tensors, dim=0, **kwargs):
        if dim == 0:
            frames.extend(tensors)
            assert all(tensor.untyped_storage().data_ptr() in storages for tensor in tensors)
        return original_cat(tensors, dim=dim, **kwargs)

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        with patch.object(vision, "_vision_sdpa_uses_fused_backend", return_value=allow_batch):
            with patch.object(vision.F, "scaled_dot_product_attention", side_effect=observe_sdpa):
                with patch.object(vision.torch, "cat", side_effect=observe_cat):
                    attention(hidden, [8, 8, 8, 4], positions)
    assert len(frames) == 4


def test_cpu_and_math_fallback_preserve_original_singleton_strides():
    attention = _attention()
    hidden, positions = _inputs([4, 4, 6, 6, 4])
    with torch.no_grad():
        with patch(f"{__name__}._SDPA", wraps=_SDPA) as reference_calls:
            expected = _legacy_attention(attention, hidden, [4, 4, 6, 6, 4], positions)
        with patch.object(torch, "_fused_sdp_choice", side_effect=AssertionError("CPU probed CUDA")):
            with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
                actual = attention(hidden, [4, 4, 6, 6, 4], positions)
    assert all(call.args[0].shape[0] == 1 for call in calls.call_args_list)
    assert [[tensor.stride() for tensor in call.args] for call in calls.call_args_list] == [
        [tensor.stride() for tensor in call.args] for call in reference_calls.call_args_list
    ]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_grad_enabled_preserves_original_gradients():
    attention = _attention()
    reference = copy.deepcopy(attention)
    lengths = [4, 4, 6, 6]
    hidden, positions = _inputs(lengths)
    hidden.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    expected = _legacy_attention(reference, reference_hidden, lengths, positions)
    with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
        actual = attention(hidden, lengths, positions)
    assert all(call.args[0].shape[0] == 1 for call in calls.call_args_list)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=0, rtol=0)
    for parameter, reference_parameter in zip(attention.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0, rtol=0)
    fake_cuda = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
    with patch.object(torch, "_fused_sdp_choice", side_effect=AssertionError("grad-enabled probe")):
        assert not vision._vision_sdpa_uses_fused_backend(fake_cuda, fake_cuda, fake_cuda, 0.125)


@pytest.mark.parametrize("error", [RuntimeError("kernel defect"), torch.OutOfMemoryError("execution OOM")])
def test_batched_sdpa_errors_are_never_retried(error):
    attention = _attention()
    hidden, positions = _inputs([4, 4])
    with torch.no_grad(), patch.object(vision, "_vision_sdpa_uses_fused_backend", return_value=True):
        with patch.object(vision.F, "scaled_dot_product_attention", side_effect=error) as calls:
            with pytest.raises(type(error), match=str(error)):
                attention(hidden, [4, 4], positions)
    assert calls.call_count == 1


@pytest.mark.parametrize("lengths", [[], [0], [-1, 5], [3]])
def test_invalid_lengths_fail_before_projection(lengths):
    attention = _attention()
    hidden, positions = _inputs([4])
    with patch.object(attention.qkv, "forward", side_effect=AssertionError("projected invalid input")):
        with pytest.raises(ValueError, match="sequence lengths"):
            attention(hidden, lengths, positions)


@pytest.mark.parametrize("lengths,groups", [([4, 6, 4], ((4, 2), (6, 1))), ([], ())])
def test_invalid_groups_do_not_combine_different_frames(lengths, groups):
    attention = _attention()
    hidden, positions = _inputs(lengths)
    with patch.object(attention.qkv, "forward", side_effect=AssertionError("projected invalid groups")):
        with pytest.raises(ValueError, match="groups must match"):
            attention(hidden, lengths, positions, groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real SDPA backend parity")
@pytest.mark.parametrize("backend", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                     SDPBackend.CUDNN_ATTENTION])
@pytest.mark.parametrize("lengths", [[32, 32, 32], [16, 16, 40, 40, 24]])
def test_cuda_selected_fused_sdpa_with_production_head_geometry(backend, lengths):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    attention = vision.MiniMaxH3Qwen3VLVisionAttention(MiniMaxH3Qwen3VLConfig()).to("cuda", dtype=torch.bfloat16).eval()
    assert (attention.num_heads, attention.head_dim) == (16, 72)
    hidden, positions = _inputs(lengths, torch.bfloat16, "cuda", hidden_size=1152, head_dim=72)
    select = vision._vision_sdpa_uses_fused_backend

    def require_supported_views(*args):
        # Inspect the actual post-RoPE production tensors. Skip only an
        # unsupported selection; never catch execution/OOM failures.
        if not select(*args):
            pytest.skip(f"{backend.name} does not support these production-head batched strides")
        return True

    with torch.no_grad(), sdpa_kernel(backend):
        with patch.object(vision.F, "scaled_dot_product_attention", wraps=_SDPA) as calls:
            with patch.object(vision, "_vision_sdpa_uses_fused_backend", side_effect=require_supported_views):
                actual = attention(hidden, lengths, positions)
        expected = _legacy_attention(attention, hidden, lengths, positions)
        assert any(call.args[0].shape[0] > 1 for call in calls.call_args_list)
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=2e-2)
