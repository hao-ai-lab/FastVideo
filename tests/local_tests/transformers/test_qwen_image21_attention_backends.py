# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch contracts; optional CUDA kernels are replaced by recording implementations."""

from __future__ import annotations

import builtins
import enum
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.testing import assert_close

ROOT = Path(__file__).resolve().parents[3]


class Backend(enum.Enum):
    TORCH_SDPA = enum.auto()
    FLASH_ATTN = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_3 = enum.auto()


SUPPORTED = (Backend.TORCH_SDPA, Backend.FLASH_ATTN, Backend.SAGE_ATTN)


@pytest.fixture(scope="module")
def native_module():
    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    class BaseDiT(nn.Module):
        def __init__(self, config, hf_config, **kwargs):
            super().__init__()
            self.config, self.hf_config = config, hf_config

    arch = SimpleNamespace(_fsdp_shard_conditions=[], _compile_conditions=[],
                           _supported_attention_backends=SUPPORTED, param_names_mapping={},
                           reverse_param_names_mapping={})
    dependencies = {}
    for name, attribute, value in (
        ("fastvideo.configs.models.dits.qwen_image21", "QwenImage21Config", lambda: SimpleNamespace(arch_config=arch)),
        ("fastvideo.layers.linear", "ReplicatedLinear", Linear),
        ("fastvideo.models.dits.base", "BaseDiT", BaseDiT),
        ("fastvideo.logger", "init_logger", logging.getLogger),
    ):
        dependency = ModuleType(name)
        setattr(dependency, attribute, value)
        dependencies[name] = dependency
    spec = importlib.util.spec_from_file_location("qwen_image21_backend_contracts",
                                                ROOT / "fastvideo/models/dits/qwen_image21.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        for name, dependency in dependencies.items():
            patch.setitem(sys.modules, name, dependency)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def recording_backend(native_module, monkeypatch):
    calls, resolutions = [], []
    sdpa = native_module._attention

    class LocalAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError("Qwen must resolve its explicit component request without an ambient scope")

        def forward(self, *args, **kwargs):
            raise AssertionError("Qwen segment attention must not inherit global forward metadata")

    def resolve(head_size, dtype, **kwargs):
        resolutions.append(dict(head_size=head_size, dtype=dtype, **kwargs))
        requested = kwargs["requested"]
        selected = requested if requested in kwargs["supported_attention_backends"] else kwargs["default_backend"]

        class Impl:
            def __init__(self, **impl_kwargs):
                assert impl_kwargs["causal"] is False

            def forward(self, q, k, v, attn_metadata):
                assert attn_metadata is None
                calls.append(dict(backend=selected, query_shape=q.shape, key_shape=k.shape,
                                  value_shape=v.shape, dtype=q.dtype))
                return sdpa(q, k, v)

        return SimpleNamespace(get_name=lambda: selected.name, get_impl_cls=lambda: Impl)

    dependencies = {
        "fastvideo.attention.layer": dict(LocalAttention=LocalAttention),
        "fastvideo.attention.selector": dict(get_attn_backend=resolve,
                                             backend_name_to_enum=lambda name: Backend[name]),
        "fastvideo.platforms": dict(AttentionBackendEnum=Backend),
        "fastvideo.utils": dict(get_compute_dtype=lambda: torch.float32),
    }
    for name, attributes in dependencies.items():
        dependency = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(dependency, attribute, value)
        monkeypatch.setitem(sys.modules, name, dependency)
    return SimpleNamespace(calls=calls, resolutions=resolutions, layer_class=LocalAttention)


def _model(native_module, requested=None):
    torch.manual_seed(17)
    config = SimpleNamespace(hidden_size=16, out_channels=4, num_attention_heads=2, num_channels_latents=4,
                             axes_dims_rope=(2, 2, 4), context_in_dim=12, eps=1e-6, in_channels=4,
                             patch_size=1, num_layers=2, attention_head_dim=8, mlp_ratio=3,
                             causal_condition=True, _resolved_attention_backend=requested)
    return native_module.QwenImage21Transformer2DModel(config, {}).eval()


def _inputs(padding=False):
    torch.manual_seed(19)
    return dict(hidden_states=torch.randn(1, 12, 4), encoder_hidden_states=torch.randn(1, 5, 12),
                timestep=torch.tensor([0.6]), img_shapes=[[(1, 2, 2)] * 3],
                img_mask=torch.tensor([[False, True, True, False, False, True]]),
                encoder_hidden_states_mask=torch.tensor([[True, True, True, True, False]]) if padding else None)


@pytest.mark.parametrize("requested", [None, Backend.TORCH_SDPA])
def test_sdpa_default_preserves_outputs_without_optional_imports(native_module, monkeypatch, requested):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("fastvideo.attention", "flash_attn", "sageattention")):
            raise AssertionError(f"SDPA must not import optional attention dispatch: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    baseline = _model(native_module)
    actual = _model(native_module, requested)
    assert all(block.attn.dense_attn is None for block in actual.transformer_blocks)
    with torch.no_grad():
        inputs = _inputs(padding=True)
        assert_close(actual(**inputs), baseline(**inputs), atol=0, rtol=0)


@pytest.mark.parametrize("requested", [Backend.FLASH_ATTN, Backend.SAGE_ATTN])
@pytest.mark.parametrize("padding", [False, True])
def test_only_mask_free_image_segments_use_selected_backend(native_module, recording_backend, monkeypatch,
                                                           requested, padding):
    torch.manual_seed(23)
    attn = native_module.QwenImage21Attention(16, 2, 8, 1e-6, requested_backend=requested)
    hidden = torch.randn(2, 11, 16)
    image_ids = torch.tensor([-1, -1] + [0] * 4 + [-1] + [1] * 4)
    valid = torch.ones(2, 11, dtype=torch.bool)
    if padding:
        valid[0, 0] = False
        valid[1, 6] = False
    index = torch.arange(11)
    same_image = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
    full_mask = ((index[:, None] >= index[None, :]) | same_image)[None, None] & valid[:, None, None]
    q = attn.norm_q(attn.to_q(hidden)[0].unflatten(-1, (2, 8)))
    k = attn.norm_k(attn.to_k(hidden)[0].unflatten(-1, (2, 8)))
    v = attn.to_v(hidden)[0].unflatten(-1, (2, 8))
    sdpa = native_module._attention
    expected = attn.to_out[0](sdpa(q, k, v, full_mask).flatten(2, 3))[0]
    masked_calls = []

    def record_mask(q, k, v, mask=None):
        assert mask is not None
        masked_calls.append((q.shape[1], k.shape[1], mask.clone()))
        return sdpa(q, k, v, mask)

    monkeypatch.setattr(native_module, "_attention", record_mask)
    actual = attn(hidden, segments=native_module._qwenimage21_prefix_segments(image_ids, 7),
                  key_valid=valid if padding else None)
    assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    expected_segments = [(0, 2), (2, 6), (6, 7), (7, 11)] if padding else [(0, 2), (6, 7)]
    for (query_len, key_len, mask), (start, end) in zip(masked_calls, expected_segments, strict=True):
        assert (query_len, key_len) == (end - start, end)
        expected_mask = full_mask[:, :, start:end, :end]
        assert torch.equal(mask.expand_as(expected_mask), expected_mask)
    expected_lengths = [] if padding else [(4, 6), (4, 11)]
    assert [(call["query_shape"][1], call["key_shape"][1]) for call in recording_backend.calls] == expected_lengths
    assert all(call["backend"] is requested for call in recording_backend.calls)


@pytest.mark.parametrize("requested", [Backend.FLASH_ATTN, Backend.SAGE_ATTN])
@pytest.mark.parametrize("padding", [False, True])
def test_cached_decode_preserves_prefix_masks_and_unequal_lengths(native_module, recording_backend, monkeypatch,
                                                               requested, padding):
    baseline = _model(native_module)
    model = _model(native_module, requested)
    assert model.state_dict().keys() == baseline.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, baseline.state_dict()[name])
    assert all(isinstance(block.attn.dense_attn, recording_backend.layer_class) for block in model.transformer_blocks)
    assert [resolution["requested"] for resolution in recording_backend.resolutions] == [requested, requested]
    assert all(resolution["supported_attention_backends"] == SUPPORTED for resolution in recording_backend.resolutions)
    assert all(resolution["default_backend"] is Backend.TORCH_SDPA for resolution in recording_backend.resolutions)
    inputs = _inputs(padding)
    cache = native_module.QwenImage21KVCache(2, "cpu")
    with torch.no_grad():
        assert_close(model(**inputs, kv_cache=cache, kv_cache_mode="extract"), baseline(**inputs), atol=0, rtol=0)
        prefix_keys = [layer.k.clone() for layer in cache.layer_caches]
        recording_backend.calls.clear()
        masked_calls = []
        sdpa = native_module._attention

        def record_mask(q, k, v, mask=None):
            assert mask is not None
            masked_calls.append(mask.clone())
            return sdpa(q, k, v, mask)

        monkeypatch.setattr(native_module, "_attention", record_mask)
        inputs["timestep"].fill_(0.25)
        inputs["hidden_states"][:, -4:] += 0.7
        actual = model(**inputs, kv_cache=cache, kv_cache_mode="cached")
        monkeypatch.setattr(native_module, "_attention", sdpa)
        expected = baseline(**inputs)[:, -4:]
    assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    if padding:
        assert recording_backend.calls == []
        assert len(masked_calls) == 2
        assert all(mask.shape == (1, 1, 1, 15) and not mask[0, 0, 0, 10] for mask in masked_calls)
    else:
        assert masked_calls == []
        assert len(recording_backend.calls) == 2
        assert all(call["query_shape"] == (1, 4, 2, 8) and call["key_shape"] == (1, 15, 2, 8)
                   and call["value_shape"] == call["key_shape"] for call in recording_backend.calls)
    for layer, prefix in zip(cache.layer_caches, prefix_keys, strict=True):
        assert torch.equal(layer.k, prefix)


def test_unsupported_backend_falls_back_to_original_sdpa(native_module, recording_backend):
    model = _model(native_module, Backend.SAGE_ATTN_3)
    assert all(block.attn.dense_attn is None for block in model.transformer_blocks)
    with torch.no_grad():
        inputs = _inputs()
        assert_close(model(**inputs), _model(native_module)(**inputs), atol=0, rtol=0)
    assert recording_backend.calls == []
