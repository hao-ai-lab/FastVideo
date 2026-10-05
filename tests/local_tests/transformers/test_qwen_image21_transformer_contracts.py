# SPDX-License-Identifier: Apache-2.0
"""CPU numerical contracts for Qwen-Image-2.1's interleaved stream.

These tests run the native architecture with an ordinary torch linear standing
in for FastVideo's GPU dispatch layer. They verify attention and cache maths;
production-loader and reference-model parity live in the companion parity test.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.testing import assert_close

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def native_module():
    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    class BaseDiT(nn.Module):
        def __init__(self, config, hf_config, **kwargs):
            super().__init__()
            self.config, self.hf_config = config, hf_config

    default_arch = SimpleNamespace(_fsdp_shard_conditions=[], _compile_conditions=[],
                                   _supported_attention_backends=(), param_names_mapping={},
                                   reverse_param_names_mapping={})
    config_module = ModuleType("fastvideo.configs.models.dits.qwen_image21")
    config_module.QwenImage21Config = lambda: SimpleNamespace(arch_config=default_arch)
    linear_module = ModuleType("fastvideo.layers.linear")
    linear_module.ReplicatedLinear = Linear
    base_module = ModuleType("fastvideo.models.dits.base")
    base_module.BaseDiT = BaseDiT
    logger_module = ModuleType("fastvideo.logger")
    logger_module.init_logger = logging.getLogger
    dependencies = {
        "fastvideo.configs.models.dits.qwen_image21": config_module,
        "fastvideo.layers.linear": linear_module,
        "fastvideo.models.dits.base": base_module,
        "fastvideo.logger": logger_module,
    }
    spec = importlib.util.spec_from_file_location("qwen_image21_cpu_contracts",
                                                ROOT / "fastvideo/models/dits/qwen_image21.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        for name, value in dependencies.items():
            patch.setitem(__import__("sys").modules, name, value)
        spec.loader.exec_module(module)
    return module


def _model(native_module, causal=True):
    torch.manual_seed(17)
    config = SimpleNamespace(hidden_size=16, out_channels=4, num_attention_heads=2, num_channels_latents=4,
                             axes_dims_rope=(2, 2, 4), context_in_dim=12, eps=1e-6, in_channels=4,
                             patch_size=1, num_layers=2, attention_head_dim=8, mlp_ratio=3,
                             causal_condition=causal)
    model = native_module.QwenImage21Transformer2DModel(config, {}).eval()
    return model


def _inputs(batch=1, adjacent=True):
    torch.manual_seed(19)
    # Two separate references, each represented by one VLM slot, then target slots.
    mask = [False, True, True, False, False, True] if adjacent else [False, True, False, True, False, True]
    return dict(hidden_states=torch.randn(batch, 12, 4),
                encoder_hidden_states=torch.randn(batch, 5, 12),
                timestep=torch.full((batch,), 0.6),
                img_shapes=[[(1, 2, 2), (1, 2, 2), (1, 2, 2)]] * batch,
                img_mask=torch.tensor([mask] * batch),
                encoder_hidden_states_mask=torch.tensor([[True, True, True, True, False]] * batch))


def test_adjacent_references_are_distinct_blocks(native_module):
    mask = torch.tensor([False] + [True] * 12)
    ids, target = native_module.QwenImage21Transformer2DModel.build_token_metadata(
        mask, [(1, 2, 2), (1, 2, 2), (1, 2, 2)])
    assert ids.tolist() == [-1] + [0] * 4 + [1] * 4 + [2] * 4
    assert target.tolist() == [False] * 9 + [True] * 4


def test_layerwise_offload_discovers_transformer_blocks(native_module):
    model = _model(native_module)
    lists = [name for name, module in model.named_children() if isinstance(module, nn.ModuleList)]
    assert lists[0] == "transformer_blocks"


@pytest.mark.parametrize("padding", [False, True])
def test_segmented_attention_matches_independent_dense_mask(native_module, padding):
    torch.manual_seed(23)
    attn = native_module.QwenImage21Attention(16, 2, 8, 1e-6)
    hidden = torch.randn(2, 15, 16)
    image_ids = torch.tensor([-1, -1] + [0] * 4 + [1] * 4 + [-1] + [2] * 4)
    valid = torch.ones(2, 15, dtype=torch.bool)
    if padding:
        valid[0, 0] = False
        valid[1, 1] = False
    q = attn.norm_q(attn.to_q(hidden)[0].unflatten(-1, (2, 8)))
    k = attn.norm_k(attn.to_k(hidden)[0].unflatten(-1, (2, 8)))
    v = attn.to_v(hidden)[0].unflatten(-1, (2, 8))
    index = torch.arange(15)
    same_image = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
    mask = ((index[:, None] >= index[None, :]) | same_image)[None, None] & valid[:, None, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                               v.transpose(1, 2), attn_mask=mask)
    expected = expected.masked_fill(~mask.any(-1, keepdim=True), 0).transpose(1, 2).flatten(2, 3)
    expected = attn.to_out[0](expected)[0]
    actual = attn(hidden, segments=native_module._qwenimage21_prefix_segments(image_ids, 11), key_valid=valid)
    assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("adjacent", [True, False])
@pytest.mark.parametrize("storage_device", [None, "cpu"])
def test_cached_decode_matches_full_recompute(native_module, batch, adjacent, storage_device):
    model = _model(native_module)
    inputs = _inputs(batch, adjacent)
    cache = native_module.QwenImage21KVCache(2, storage_device)
    with torch.no_grad():
        extract = model(**inputs, kv_cache=cache, kv_cache_mode="extract")
        plain = model(**inputs)
        assert_close(extract, plain, atol=0, rtol=0)
        inputs["timestep"] = torch.full((batch,), 0.25)
        inputs["hidden_states"] = inputs["hidden_states"].clone()
        inputs["hidden_states"][:, -4:] += 0.7
        cached = model(**inputs, kv_cache=cache, kv_cache_mode="cached")
        plain = model(**inputs)
    assert cached.shape == (batch, 4, 4)
    assert_close(cached, plain[:, -4:], atol=2e-6, rtol=2e-6)
    assert_close(extract[:, :-4], plain[:, :-4], atol=0, rtol=0)
    if storage_device == "cpu":
        assert all(layer.k.device.type == "cpu" for layer in cache.layer_caches)


def test_padding_embeddings_cannot_change_target(native_module):
    model = _model(native_module)
    inputs = _inputs()
    with torch.no_grad():
        expected = model(**inputs)[:, -4:]
        inputs["encoder_hidden_states"][:, -1] = 1000
        actual = model(**inputs)[:, -4:]
    assert_close(actual, expected, atol=0, rtol=0)


def test_text_only_generation_and_cache(native_module):
    model = _model(native_module)
    inputs = _inputs()
    inputs["hidden_states"] = inputs["hidden_states"][:, -4:]
    inputs["img_shapes"] = [[(1, 2, 2)]]
    inputs["img_mask"] = torch.tensor([[False, False, False, False, False, True]])
    cache = native_module.QwenImage21KVCache(2, "cpu")
    with torch.no_grad():
        extracted = model(**inputs, kv_cache=cache, kv_cache_mode="extract")
        inputs["timestep"].fill_(0.2)
        expected = model(**inputs)[:, -4:]
        actual = model(**inputs, kv_cache=cache, kv_cache_mode="cached")
    assert extracted.shape == (1, 9, 4)
    assert cache.prefix_len == 5
    assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_ten_reference_images_and_cache(native_module):
    model = _model(native_module)
    generator = torch.Generator().manual_seed(29)
    inputs = dict(hidden_states=torch.randn(1, 44, 4, generator=generator),
                  encoder_hidden_states=torch.randn(1, 13, 12, generator=generator),
                  timestep=torch.tensor([0.7]),
                  img_shapes=[[(1, 2, 2)] * 11],
                  img_mask=torch.tensor([[False] + [True] * 10 + [False, False, True]]))
    cache = native_module.QwenImage21KVCache(2, "cpu")
    with torch.no_grad():
        extracted = model(**inputs, kv_cache=cache, kv_cache_mode="extract")
        inputs["timestep"].fill_(0.2)
        actual = model(**inputs, kv_cache=cache, kv_cache_mode="cached")
        expected = model(**inputs)[:, -4:]
    assert extracted.shape == (1, 47, 4)
    assert cache.prefix_len == 43
    assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_cache_owns_only_prefix_storage(native_module):
    cache = native_module.QwenImage21KVLayerCache("cpu")
    tensor = torch.randn(1, 12, 2, 8)
    cache.store(tensor[:, :3], tensor[:, :3])
    assert cache.k.untyped_storage().nbytes() == cache.k.numel() * cache.k.element_size()
    assert cache.k.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
    tensor.zero_()
    assert bool(cache.k.any())
    cache.clear()
    with pytest.raises(RuntimeError, match="extract"):
        cache.get()


def test_host_cache_allocation_failure_has_specific_error(native_module, monkeypatch):
    cache = native_module.QwenImage21KVLayerCache("cpu")
    tensor = torch.randn(1, 3, 2, 8)

    def fail_copy(self, *args, **kwargs):
        raise RuntimeError("DefaultCPUAllocator: can't allocate memory")

    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    with pytest.raises(native_module.QwenImage21KVCacheAllocationError, match="CPU memory exhausted"):
        cache.store(tensor, tensor)
    assert cache.k is None and cache.v is None


@pytest.mark.parametrize("message", ["Invalid device configuration", "CUDA out of memory"])
def test_cache_transfer_errors_are_not_allocation_errors(native_module, monkeypatch, message):
    cache = native_module.QwenImage21KVLayerCache("cpu")
    tensor = torch.randn(1, 3, 2, 8)

    def fail_copy(self, *args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    with pytest.raises(RuntimeError, match=message) as error:
        cache.store(tensor, tensor)
    assert not isinstance(error.value, native_module.QwenImage21KVCacheAllocationError)


def test_cache_requires_causal_condition_and_extract(native_module):
    cache = native_module.QwenImage21KVCache(2, "cpu")
    with pytest.raises(ValueError, match="causal_condition"):
        _model(native_module, causal=False)(**_inputs(), kv_cache=cache, kv_cache_mode="extract")
    with pytest.raises(ValueError, match="not been extracted"):
        _model(native_module)(**_inputs(), kv_cache=cache, kv_cache_mode="cached")


def test_batch_requires_shared_reference_layout(native_module):
    inputs = _inputs(2)
    inputs["img_mask"][1, 1] = False
    with pytest.raises(ValueError, match="same image-slot layout"):
        _model(native_module)(**inputs)
