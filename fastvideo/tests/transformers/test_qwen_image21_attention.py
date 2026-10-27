# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 attention and prefix-cache regressions with tiny native weights.

RoPE checks run on CPU; attention/cache cases require one CUDA GPU and the
ordinary FastVideo runtime. No model download or optional FlashAttention
installation is needed. Independent checkpoint parity remains in
``tests/local_tests/transformers/``.
"""

from __future__ import annotations

from functools import cache

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing import assert_close

import fastvideo.envs as envs
import fastvideo.models.dits.qwen_image21 as qwen
from fastvideo.configs.models.dits.qwen_image21 import QwenImage21ArchConfig, QwenImage21Config
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.platforms import AttentionBackendEnum

@pytest.fixture
def qwen_setup(env_overrides, monkeypatch, request):
    if not torch.cuda.is_available():
        pytest.skip("Requires one CUDA GPU")
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    request.getfixturevalue("distributed_setup")
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        yield torch.device("cuda")


def _initialize_linears(module):
    # ReplicatedLinear allocates empty storage instead of initializing weights.
    for layer in module.modules():
        if isinstance(layer, ReplicatedLinear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    return module.eval()


def _model(device):
    config = QwenImage21Config(arch_config=QwenImage21ArchConfig(
        in_channels=4, out_channels=4, num_layers=2, num_attention_heads=2,
        attention_head_dim=8, context_in_dim=12, axes_dims_rope=(2, 2, 4), mlp_ratio=2))
    return _initialize_linears(qwen.QwenImage21Transformer2DModel(config).to(device=device, dtype=torch.float32))


def _inputs(device):
    # Adjacent reference slots must still form distinct bidirectional blocks.
    return dict(
        hidden_states=torch.randn(2, 12, 4, device=device),
        encoder_hidden_states=torch.randn(2, 5, 12, device=device),
        timestep=torch.tensor([0.6, 0.7], device=device),
        img_shapes=[[(1, 2, 2)] * 3 for _ in range(2)],
        img_mask=torch.tensor([[False, True, True, False, False, True]] * 2, device=device),
        encoder_hidden_states_mask=torch.tensor(
            [[True, True, True, True, False], [False, True, True, True, False]], device=device),
    )


@pytest.mark.parametrize("position", [8191, 8192, 9215, 9216])
def test_rope_positive_position_does_not_alias_negative_coordinates(position):
    rope = qwen.QwenImage21Rope(theta=10000, axes_dim=(2, 2, 2))
    mask = torch.tensor([True] + [False] * position)
    actual = rope([(1, 1, 1)], mask, torch.device("cpu"))

    # Each two-dimensional axis has frequency 1: its phase is exp(i * position).
    expected = torch.exp(1j * torch.tensor([position] * 3, dtype=torch.float32))
    assert_close(actual[-1], expected, atol=1e-6, rtol=1e-6)
    image_expected = torch.exp(1j * torch.tensor([0, -1, -1], dtype=torch.float32))
    assert_close(actual[0], image_expected, atol=1e-6, rtol=1e-6)
    assert actual.shape == (position + 1, 3)
    assert actual.dtype == torch.complex64 and actual.device.type == "cpu"


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_rope_long_layout_preserves_multi_image_coordinates_and_short_requests(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA RoPE placement requires one GPU")
    device = torch.device(device_name)
    rope = qwen.QwenImage21Rope(theta=10000, axes_dim=(2, 2, 2)).to(device=device, dtype=torch.bfloat16)
    shapes = [(1, 2049, 1), (1, 1, 2049)]

    def layout(text_tokens):
        mask = torch.tensor([False] * text_tokens + [True] * 2049 + [False] * 3 + [True] * 2049 + [False] * 2,
                            device=device)
        # Odd image axes span -1025..1023, beyond the old negative table.
        # Each image advances subsequent text by its largest spatial axis.
        positions = ([[p] * 3 for p in range(text_tokens)]
                     + [[text_tokens, h, -1] for h in range(-1025, 1024)]
                     + [[text_tokens + p] * 3 for p in (2049, 2050, 2051)]
                     + [[text_tokens + 2052, -1, w] for w in range(-1025, 1024)]
                     + [[text_tokens + p] * 3 for p in (4101, 4102)])
        expected = torch.exp(1j * torch.tensor(positions, dtype=torch.float32, device=device))
        return mask, expected

    short_mask, short_expected = layout(3)
    before = rope(shapes, short_mask, device).clone()
    assert_close(before, short_expected, atol=1e-6, rtol=1e-6)
    long_mask, long_expected = layout(9216)
    actual = rope(shapes, long_mask, device)
    assert_close(actual, long_expected, atol=1e-6, rtol=1e-6)
    assert actual.device == long_mask.device and actual.dtype == torch.complex64
    assert_close(rope(shapes, short_mask, device), before, atol=0, rtol=0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_rope_long_layout_keeps_nontrivial_short_frequencies_bitwise(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA RoPE placement requires one GPU")
    device = torch.device(device_name)
    rope = qwen.QwenImage21Rope(theta=10000, axes_dim=(2, 2, 4)).to(device=device, dtype=torch.bfloat16)
    shapes = [(1, 2, 2)]
    short_mask = torch.tensor([False] * 3 + [True] * 4, device=device)
    before = rope(shapes, short_mask, device).clone()
    rope(shapes, torch.tensor([False] * 9216 + [True] * 4, device=device), device)
    after = rope(shapes, short_mask, device)
    assert after.dtype == torch.complex64 and after.device == short_mask.device
    assert_close(after, before, atol=0, rtol=0)


def test_segmented_attention_matches_independent_dense_mask(qwen_setup):
    device = qwen_setup
    attention = _initialize_linears(qwen.QwenImage21Attention(16, 2, 8, 1e-6).to(device))
    hidden = torch.randn(2, 15, 16, device=device)
    image_ids = torch.tensor([-1, -1] + [0] * 4 + [1] * 4 + [-1] + [2] * 4, device=device)
    valid = torch.ones(2, 15, dtype=torch.bool, device=device)
    valid[0, 0] = False
    valid[1, 1] = False

    # Independently encode causality: preceding tokens, or any token in the
    # same image. In particular, adjacent reference images cannot see ahead.
    positions = torch.arange(15, device=device)
    causal = positions[:, None] >= positions[None, :]
    same_image = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
    mask = (causal | same_image)[None, None] & valid[:, None, None, :]
    query = attention.norm_q(attention.to_q(hidden)[0].unflatten(-1, (2, 8)))
    key = attention.norm_k(attention.to_k(hidden)[0].unflatten(-1, (2, 8)))
    value = attention.to_v(hidden)[0].unflatten(-1, (2, 8))
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask, dropout_p=0.0)
    expected = expected.masked_fill(~mask.any(-1, keepdim=True), 0).transpose(1, 2).flatten(2, 3)
    expected = attention.to_out[0](expected)[0]

    actual = attention(hidden, segments=qwen._qwenimage21_prefix_segments(image_ids, 11), key_valid=valid)

    assert torch.isfinite(actual).all()
    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_cached_decode_matches_recompute_and_owns_prefix_storage(qwen_setup):
    device = qwen_setup
    model = _model(device)
    inputs = _inputs(device)
    cache = qwen.QwenImage21KVCache(model.config.num_layers, storage_device="cpu")
    extracted = model(**inputs, kv_cache=cache, kv_cache_mode="extract")
    assert_close(extracted, model(**inputs), atol=0, rtol=0)
    assert cache.prefix_len == 11
    prefix_values = []
    for layer in cache.layer_caches:
        assert layer.k.shape == layer.v.shape == (2, 11, 2, 8)
        assert layer.k.device.type == layer.v.device.type == "cpu"
        for tensor in (layer.k, layer.v):
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        prefix_values.append((layer.k.clone(), layer.v.clone()))

    inputs["timestep"] = torch.tensor([0.25, 0.35], device=device)
    inputs["hidden_states"] = inputs["hidden_states"].clone()
    inputs["hidden_states"][:, -4:] += 0.7
    actual = model(**inputs, kv_cache=cache, kv_cache_mode="cached")
    expected = model(**inputs)

    assert actual.shape == (2, 4, 4)
    assert_close(actual, expected[:, -4:], atol=1e-5, rtol=1e-5)
    assert_close(extracted[:, :-4], expected[:, :-4], atol=0, rtol=0)
    for layer, (key, value) in zip(cache.layer_caches, prefix_values, strict=True):
        assert_close(layer.k, key, atol=0, rtol=0)
        assert_close(layer.v, value, atol=0, rtol=0)

    # A prefix view must own its bytes, even when source and cache are both on
    # the CPU. Otherwise retaining it keeps the entire prefill allocation alive.
    source = torch.randn(1, 15, 2, 8)
    owned = qwen.QwenImage21KVLayerCache("cpu")
    owned.store(source[:, :3], source[:, :3])
    original = source[:, :3].clone()
    assert owned.k.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert owned.k.untyped_storage().nbytes() == owned.k.numel() * owned.k.element_size()
    source.zero_()
    assert_close(owned.k, original, atol=0, rtol=0)
    assert_close(owned.v, original, atol=0, rtol=0)
    cache.clear()
    owned.clear()
    assert cache.prefix_len is None
    assert all(layer.k is None and layer.v is None for layer in cache.layer_caches)
    with pytest.raises(RuntimeError, match="extract"):
        owned.get()


def test_padding_embeddings_cannot_change_target_predictions(qwen_setup):
    device = qwen_setup
    model = _model(device)
    inputs = _inputs(device)
    expected = model(**inputs)[:, -4:]
    embeddings = inputs["encoder_hidden_states"].clone()
    embeddings[~inputs["encoder_hidden_states_mask"]] = 1000
    inputs["encoder_hidden_states"] = embeddings
    cache = qwen.QwenImage21KVCache(model.config.num_layers, storage_device="cpu")

    extracted = model(**inputs, kv_cache=cache, kv_cache_mode="extract")[:, -4:]
    cached = model(**inputs, kv_cache=cache, kv_cache_mode="cached")

    assert torch.isfinite(extracted).all()
    assert torch.isfinite(cached).all()
    assert_close(extracted, expected, atol=0, rtol=0)
    assert_close(cached, expected, atol=1e-5, rtol=1e-5)
    cache.clear()


def test_masked_calls_use_sdpa_and_dense_calls_use_selected_backend(qwen_setup, monkeypatch):
    import fastvideo.attention.selector as selector

    device = qwen_setup
    dense_calls, resolutions, masked_calls = [], [], []

    class RecordingImpl:

        def __init__(self, **kwargs):
            assert kwargs["causal"] is False
            assert kwargs["num_heads"] == kwargs["num_kv_heads"] == 2
            assert kwargs["head_size"] == 8

        def forward(self, query, key, value, attn_metadata):
            assert attn_metadata is None
            dense_calls.append((query.shape[1], key.shape[1]))
            return F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                dropout_p=0.0).transpose(1, 2)

    class RecordingBackend:

        @staticmethod
        def get_name():
            return "FLASH_ATTN"

        @staticmethod
        def get_impl_cls():
            return RecordingImpl

    def resolve_backend(head_size, dtype, **kwargs):
        assert head_size == 8
        resolutions.append(kwargs)
        return RecordingBackend

    monkeypatch.setattr(selector, "get_attn_backend", resolve_backend)
    baseline = _initialize_linears(qwen.QwenImage21Attention(
        16, 2, 8, 1e-6, requested_backend=AttentionBackendEnum.TORCH_SDPA).to(device))
    assert resolutions == []
    attention = qwen.QwenImage21Attention(16, 2, 8, 1e-6).to(device).eval()
    explicit_attention = qwen.QwenImage21Attention(
        16, 2, 8, 1e-6, requested_backend=AttentionBackendEnum.FLASH_ATTN).to(device).eval()
    attention.load_state_dict(baseline.state_dict())
    explicit_attention.load_state_dict(baseline.state_dict())
    assert [resolution.get("requested", selector.NO_REQUEST) for resolution in resolutions] == [
        selector.NO_REQUEST, AttentionBackendEnum.FLASH_ATTN]
    assert all("default_backend" not in resolution for resolution in resolutions)
    assert all(AttentionBackendEnum.FLASH_ATTN in resolution["supported_attention_backends"]
               for resolution in resolutions)

    hidden = torch.randn(1, 11, 16, device=device)
    image_ids = torch.tensor([-1, -1] + [0] * 4 + [-1] + [1] * 4, device=device)
    segments = qwen._qwenimage21_prefix_segments(image_ids, 7)
    valid = torch.ones(1, 11, dtype=torch.bool, device=device)
    valid[:, 0] = False
    expected = baseline(hidden, segments=segments)
    padded_expected = baseline(hidden, segments=segments, key_valid=valid)
    assert_close(explicit_attention(hidden, segments=segments), expected, atol=1e-5, rtol=1e-5)
    assert dense_calls == [(4, 6), (4, 11)]
    dense_calls.clear()
    sdpa = qwen._attention

    def record_masked_sdpa(query, key, value, mask=None):
        assert mask is not None
        masked_calls.append((query.shape[1], key.shape[1]))
        return sdpa(query, key, value, mask)

    monkeypatch.setattr(qwen, "_attention", record_masked_sdpa)
    cache = qwen.QwenImage21KVLayerCache()
    actual = attention(hidden, segments=segments, layer_cache=cache,
                       kv_cache_mode="extract", cache_write_slice=slice(0, 7))
    assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert masked_calls == [(2, 2), (1, 7)]
    assert dense_calls == [(4, 6), (4, 11)]

    dense_calls.clear()
    masked_calls.clear()
    cached = attention(hidden[:, 7:], layer_cache=cache, kv_cache_mode="cached")
    assert_close(cached, expected[:, 7:], atol=1e-5, rtol=1e-5)
    assert dense_calls == [(4, 11)]
    assert masked_calls == []

    dense_calls.clear()
    padded = attention(hidden, segments=segments, key_valid=valid)
    assert_close(padded, padded_expected, atol=1e-5, rtol=1e-5)
    assert dense_calls == []
    assert masked_calls == [(2, 2), (4, 6), (1, 7), (4, 11)]
    masked_calls.clear()
    padded_cached = attention(hidden[:, 7:], layer_cache=cache, kv_cache_mode="cached",
                              attention_mask=valid[:, None, None, :])
    assert_close(padded_cached, padded_expected[:, 7:], atol=1e-5, rtol=1e-5)
    assert dense_calls == []
    assert masked_calls == [(4, 11)]
    cache.clear()


@pytest.fixture
def selector_resolution(qwen_setup, monkeypatch):
    import fastvideo.attention.selector as selector
    import fastvideo.platforms as platforms
    from fastvideo.attention.backends.sdpa import SDPABackend

    class AlternativeImpl:

        def __init__(self, **kwargs):
            pass

    class AlternativeBackend:

        @staticmethod
        def get_name():
            return "FLASH_ATTN"

        @staticmethod
        def get_impl_cls():
            return AlternativeImpl

    def platform_backend(requested, head_size, dtype):
        # A dense alternative is available automatically, so losing an SDPA
        # override cannot silently pass by falling back to SDPA anyway.
        return "TORCH_SDPA" if requested is AttentionBackendEnum.TORCH_SDPA else "FLASH_ATTN"

    monkeypatch.setattr(platforms.current_platform, "get_attn_backend_cls", platform_backend)
    monkeypatch.setattr(selector, "resolve_obj_by_qualname",
                        lambda name: SDPABackend if name == "TORCH_SDPA" else AlternativeBackend)
    # Exercise the real selector and its cache without leaving fake platform
    # resolutions in the shared process cache or clearing unrelated entries.
    monkeypatch.setattr(selector, "_cached_get_attn_backend", cache(selector._cached_get_attn_backend.__wrapped__))


def _direct_attention(kind, recorded_backend=None):
    if kind == "block":
        return qwen.QwenImage21TransformerBlock(16, 2, 8, 2, 1e-6).attn
    config = QwenImage21Config(arch_config=QwenImage21ArchConfig(
        in_channels=4, out_channels=4, num_layers=1, num_attention_heads=2,
        attention_head_dim=8, context_in_dim=12, axes_dims_rope=(2, 2, 4), mlp_ratio=2))
    config._resolved_attention_backend = recorded_backend
    return qwen.QwenImage21Transformer2DModel(config).transformer_blocks[0].attn


def _selected_backend(attention):
    return (AttentionBackendEnum.TORCH_SDPA if attention.dense_attn is None else attention.dense_attn.backend)


@pytest.mark.parametrize("kind", ["transformer", "block"])
@pytest.mark.parametrize("environment", ["TORCH_SDPA", "FLASH_ATTN", None])
def test_direct_construction_honors_environment_backend(selector_resolution, kind, environment):
    with envs.FASTVIDEO_ATTENTION_BACKEND.override(environment):
        attention = _direct_attention(kind)
    expected = AttentionBackendEnum.TORCH_SDPA if environment == "TORCH_SDPA" else AttentionBackendEnum.FLASH_ATTN
    assert _selected_backend(attention) is expected


@pytest.mark.parametrize("recorded", [AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.FLASH_ATTN])
def test_direct_transformer_recorded_backend_overrides_environment(selector_resolution, recorded):
    from fastvideo.attention.selector import _component_attention_backend_scope

    environment = "FLASH_ATTN" if recorded is AttentionBackendEnum.TORCH_SDPA else "TORCH_SDPA"
    with envs.FASTVIDEO_ATTENTION_BACKEND.override(environment):
        with _component_attention_backend_scope(environment, component="qwen"):
            attention = _direct_attention("transformer", recorded)
    assert _selected_backend(attention) is recorded


@pytest.mark.parametrize("requested", [AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.FLASH_ATTN])
def test_direct_block_explicit_backend_overrides_environment(selector_resolution, requested):
    environment = "FLASH_ATTN" if requested is AttentionBackendEnum.TORCH_SDPA else "TORCH_SDPA"
    with envs.FASTVIDEO_ATTENTION_BACKEND.override(environment):
        block = qwen.QwenImage21TransformerBlock(16, 2, 8, 2, 1e-6, requested_backend=requested)
    assert _selected_backend(block.attn) is requested


@pytest.mark.parametrize("kind", ["transformer", "block"])
@pytest.mark.parametrize("scoped,consult_env,environment,expected", [
    (AttentionBackendEnum.TORCH_SDPA, False, "FLASH_ATTN", AttentionBackendEnum.TORCH_SDPA),
    (None, False, "TORCH_SDPA", AttentionBackendEnum.FLASH_ATTN),
    (None, True, "TORCH_SDPA", AttentionBackendEnum.TORCH_SDPA),
])
def test_direct_construction_honors_component_scope(selector_resolution, kind, scoped, consult_env, environment, expected):
    from fastvideo.attention.selector import _component_attention_backend_scope

    with envs.FASTVIDEO_ATTENTION_BACKEND.override(environment):
        with _component_attention_backend_scope(scoped, component="qwen", consult_env=consult_env):
            attention = _direct_attention(kind)
    assert _selected_backend(attention) is expected
