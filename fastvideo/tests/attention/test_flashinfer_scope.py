"""FlashInfer opt-in must reach real layers without leaking into Wan derivatives."""
import pytest
import torch

import fastvideo.attention.layer as attention_layer
import fastvideo.attention.selector as selector
import fastvideo.platforms as platforms
from fastvideo.attention.backends.flashinfer import FlashInferImpl
from fastvideo.attention.backends.sdpa import SDPAImpl
from fastvideo.configs.models.dits.matrixgame2 import MatrixGame2WanVideoArchConfig
from fastvideo.configs.models.dits.matrixgame3 import MatrixGame3WanVideoArchConfig
from fastvideo.configs.models.dits.dreamx_world import DreamXWorldArchConfig
from fastvideo.configs.models.dits.wan_animate import WanAnimateArchConfig
from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig
import fastvideo.models.wan.transformer as wan
from fastvideo.models.wan.causal_transformer import CausalWanTransformer3DModel
from fastvideo.models.wan.transformer import WanTransformerBlock, WanT2VCrossAttention, WanTransformer3DModel
from fastvideo.platforms.interface import AttentionBackendEnum, Platform as BasePlatform, PlatformEnum


def test_attention_backend_legacy_values():
    legacy = ["FLASH_ATTN", "TORCH_SDPA", "SAGE_ATTN", "SAGE_ATTN_THREE", "ATTN_QAT_INFER",
              "ATTN_QAT_TRAIN", "VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3", "BSA_ATTN",
              "VMOBA_ATTN", "SLA_ATTN", "SAGE_SLA_ATTN", "NABLA_ATTN", "NO_ATTENTION"]
    for value, name in enumerate(legacy, 1):
        assert AttentionBackendEnum(value) is AttentionBackendEnum[name]
    assert AttentionBackendEnum.FLASHINFER.value == 15


@pytest.fixture
def requested_flashinfer(monkeypatch):
    class Platform(BasePlatform):
        _enum = PlatformEnum.CPU
        device_name = "test"

        @classmethod
        def get_attn_backend_cls(cls, selected_backend, head_size, dtype):
            if selected_backend == AttentionBackendEnum.FLASHINFER:
                return "fastvideo.attention.backends.flashinfer.FlashInferBackend"
            assert selected_backend is None
            return "fastvideo.attention.backends.sdpa.SDPABackend"

    monkeypatch.setattr(platforms, "_current_platform", Platform())
    monkeypatch.setattr(attention_layer, "get_compute_dtype", lambda: torch.bfloat16)
    try:
        with selector._component_attention_backend_scope(AttentionBackendEnum.FLASHINFER, component="scope-test"):
            yield
    finally:
        selector._cached_get_attn_backend.cache_clear()


@pytest.mark.parametrize("config_cls,expected", [
    (WanVideoArchConfig, SDPAImpl),
    (MatrixGame2WanVideoArchConfig, SDPAImpl),
    (MatrixGame3WanVideoArchConfig, SDPAImpl),
    (DreamXWorldArchConfig, SDPAImpl),
    (WanAnimateArchConfig, SDPAImpl),
])
@pytest.mark.parametrize("image", [False, True])
def test_wan_block_instantiates_scoped_backend(requested_flashinfer, config_cls, expected, image):
    config = config_cls()
    block = WanTransformerBlock(dim=128, ffn_dim=256, num_heads=2, cross_attn_norm=True,
                                added_kv_proj_dim=128 if image else None,
                                supported_attention_backends=config._supported_attention_backends)
    assert isinstance(block.attn1.attn_impl, expected)
    assert isinstance(block.attn2.attn.attn_impl, expected)
    assert block.attn2.parallel_attention is False


def test_causal_wan_and_shared_cross_attention_do_not_opt_in(requested_flashinfer):
    assert AttentionBackendEnum.FLASHINFER not in CausalWanTransformer3DModel._supported_attention_backends
    cross = WanT2VCrossAttention(dim=128, num_heads=2)
    assert isinstance(cross.attn.attn_impl, SDPAImpl)


class UnvalidatedWanModel(WanTransformer3DModel):
    """Derived families declare their own supported backends, as existing models do."""
    _supported_attention_backends = WanVideoArchConfig()._supported_attention_backends


@pytest.mark.parametrize("model_cls,expected", [
    (WanTransformer3DModel, FlashInferImpl),
    (UnvalidatedWanModel, SDPAImpl),
])
@pytest.mark.parametrize("image", [False, True])
def test_concrete_wan_model_opt_in(requested_flashinfer, monkeypatch, model_cls, expected, image):
    monkeypatch.setattr(wan, "get_sp_world_size", lambda: 1)
    arch = WanVideoArchConfig(num_attention_heads=2, attention_head_dim=64, ffn_dim=256,
                              num_layers=1, text_dim=128, freq_dim=64,
                              added_kv_proj_dim=128 if image else None)
    config = WanVideoConfig(arch_config=arch)
    model = model_cls(config=config, hf_config={})
    assert isinstance(model.blocks[0].attn1.attn_impl, expected)
    assert isinstance(model.blocks[0].attn2.attn.attn_impl, expected)
    assert AttentionBackendEnum.FLASHINFER not in config._supported_attention_backends


def test_sparse_self_attention_backends_are_filtered_from_cross_attention(requested_flashinfer, monkeypatch):
    resolved_lists = []
    original = attention_layer.get_attn_backend

    def record(*args, **kwargs):
        resolved_lists.append(kwargs["supported_attention_backends"])
        return original(*args, **kwargs)

    monkeypatch.setattr(attention_layer, "get_attn_backend", record)
    block = WanTransformerBlock(dim=128, ffn_dim=256, num_heads=2, cross_attn_norm=True,
                                supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                                                              AttentionBackendEnum.TORCH_SDPA))
    assert isinstance(block.attn1.attn_impl, SDPAImpl)
    assert isinstance(block.attn2.attn.attn_impl, SDPAImpl)
    assert resolved_lists[-1] == (AttentionBackendEnum.TORCH_SDPA,)


def test_derived_families_preserve_inheritance_and_backend_scope(requested_flashinfer):
    from fastvideo.models.dits.dreamx_world import DreamXWorldTransformer3DModel
    from fastvideo.models.dits.wan_animate import WanAnimateTransformer3DModel

    for model_cls in (DreamXWorldTransformer3DModel, WanAnimateTransformer3DModel):
        assert issubclass(model_cls, WanTransformer3DModel)
        assert AttentionBackendEnum.FLASHINFER not in model_cls._supported_attention_backends
        block = WanTransformerBlock(dim=128, ffn_dim=256, num_heads=2, cross_attn_norm=True,
                                    supported_attention_backends=model_cls._supported_attention_backends)
        assert isinstance(block.attn1.attn_impl, SDPAImpl)
        assert isinstance(block.attn2.attn.attn_impl, SDPAImpl)


def test_standard_wan_preserves_existing_backends():
    legacy = WanVideoArchConfig()._supported_attention_backends
    assert set(WanTransformer3DModel._supported_attention_backends) == set(legacy) | {AttentionBackendEnum.FLASHINFER}
