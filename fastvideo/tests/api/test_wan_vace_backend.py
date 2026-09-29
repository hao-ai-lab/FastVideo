# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE constructor and metadata must share one dense backend request."""

import pytest
import torch

from fastvideo import envs

from fastvideo.attention.selector import _component_attention_backend_scope
from fastvideo.models.wan.vace_config import WanVACEArchConfig, WanVACEVideoConfig
from fastvideo.models.wan.vace_transformer import WanVACETransformer3DModel
from fastvideo.pipelines.basic.wan.stages.vace_denoising import WanVACEDenoisingStage
from fastvideo.platforms import AttentionBackendEnum


@pytest.fixture(autouse=True)
def single_process_model_parallel(monkeypatch):
    monkeypatch.setattr("fastvideo.models.wan.vace_transformer.get_sp_world_size", lambda: 1)


def _tiny_config(backend):
    arch = WanVACEArchConfig(num_attention_heads=2,
                             attention_head_dim=8,
                             text_dim=16,
                             freq_dim=8,
                             ffn_dim=32,
                             num_layers=1,
                             vace_layers=[0])
    config = WanVACEVideoConfig(arch_config=arch)
    config._resolved_attention_backend = backend
    return config


def test_vace_explicit_dense_overrides_sparse_environment():
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("VIDEO_SPARSE_ATTN"):
        config = _tiny_config(AttentionBackendEnum.TORCH_SDPA)
        with _component_attention_backend_scope(AttentionBackendEnum.TORCH_SDPA, component="transformer"):
            model = WanVACETransformer3DModel(config, hf_config={})

        stage = WanVACEDenoisingStage(transformer=model, scheduler=object())
        assert AttentionBackendEnum.VIDEO_SPARSE_ATTN not in model.supported_attention_backends
        assert AttentionBackendEnum.TORCH_SDPA in model.supported_attention_backends
        assert type(model.blocks[0]).__name__ == "WanVACEMainBlock"
        assert not hasattr(model.blocks[0], "to_gate_compress")
        assert model.blocks[0].attn1.backend is AttentionBackendEnum.TORCH_SDPA
        assert model.blocks[0].attn2.attn.backend is AttentionBackendEnum.TORCH_SDPA
        assert model.vace_blocks[0].attn1.backend is AttentionBackendEnum.TORCH_SDPA
        assert model.vace_blocks[0].attn2.attn.backend is AttentionBackendEnum.TORCH_SDPA
        assert stage.attn_backend.get_name() == "TORCH_SDPA"


def test_vace_sparse_request_fails_before_layer_construction(monkeypatch):
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"):
        monkeypatch.setattr("fastvideo.models.wan.vace_transformer.PatchEmbed",
                            lambda *args, **kwargs: pytest.fail("layer construction reached"))
        config = _tiny_config(AttentionBackendEnum.VIDEO_SPARSE_ATTN)

        with _component_attention_backend_scope(AttentionBackendEnum.VIDEO_SPARSE_ATTN, component="transformer"):
            with pytest.raises(ValueError, match="Wan-VACE.*VIDEO_SPARSE_ATTN.*TORCH_SDPA"):
                WanVACETransformer3DModel(config, hf_config={})


def test_vace_does_not_change_regular_wan_backend_support():
    from fastvideo.models.wan.config import WanVideoConfig

    assert AttentionBackendEnum.VIDEO_SPARSE_ATTN in WanVideoConfig()._supported_attention_backends
    assert AttentionBackendEnum.VIDEO_SPARSE_ATTN not in WanVACEVideoConfig()._supported_attention_backends


def test_vace_modulation_tables_keep_diffusers_fp32_loading_precision():
    config = _tiny_config(AttentionBackendEnum.TORCH_SDPA)
    with _component_attention_backend_scope(AttentionBackendEnum.TORCH_SDPA, component="transformer"):
        model = WanVACETransformer3DModel(config, hf_config={})

    assert model._get_parameter_dtype("scale_shift_table", torch.bfloat16) == torch.float32
    assert model._get_parameter_dtype("vace_blocks.0.scale_shift_table", torch.bfloat16) == torch.float32
    assert model._get_parameter_dtype("condition_embedder.time_embedder.mlp.fc_in.weight",
                                      torch.bfloat16) == torch.float32
    assert model._get_parameter_dtype("vace_blocks.0.to_q.weight", torch.bfloat16) == torch.bfloat16
    assert model._get_parameter_dtype("blocks.0.self_attn_residual_norm.norm.weight",
                                      torch.bfloat16) == torch.float32
    assert model._get_parameter_dtype("vace_blocks.0.self_attn_residual_norm.norm.weight",
                                      torch.bfloat16) == torch.float32
    assert model.condition_embedder.cast_temb_to_context_dtype


def test_vace_reference_norms_and_residuals_are_instance_local():
    from fastvideo.layers.layernorm import RMSNorm
    from fastvideo.models.wan.transformer import WanTransformerBlock

    config = _tiny_config(AttentionBackendEnum.TORCH_SDPA)
    with _component_attention_backend_scope(AttentionBackendEnum.TORCH_SDPA, component="transformer"):
        vace = WanVACETransformer3DModel(config, hf_config={})
        regular = WanTransformerBlock(16,
                                      32,
                                      2,
                                      qk_norm="rms_norm_across_heads",
                                      cross_attn_norm=True,
                                      supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))

    assert isinstance(vace.vace_blocks[0].norm_q, torch.nn.RMSNorm)
    assert isinstance(vace.vace_blocks[0].attn2.norm_q, torch.nn.RMSNorm)
    assert isinstance(vace.blocks[0].norm_q, torch.nn.RMSNorm)
    assert isinstance(vace.blocks[0].attn2.norm_q, torch.nn.RMSNorm)
    assert type(vace.blocks[0]).__name__ == "WanVACEMainBlock"
    assert isinstance(regular.norm_q, RMSNorm)
    assert isinstance(regular.attn2.norm_q, RMSNorm)
    assert type(regular).__name__ == "WanTransformerBlock"


def test_vace_fp32_time_embedding_casts_to_bf16_context():
    from fastvideo.models.wan.transformer import WanTimeTextImageEmbedding

    embedding = WanTimeTextImageEmbedding(dim=16,
                                          time_freq_dim=8,
                                          text_embed_dim=16,
                                          cast_temb_to_context_dtype=True).bfloat16()
    embedding.time_embedder.float()
    temb, timestep_proj, _, _ = embedding(torch.tensor([500]), torch.randn(1, 4, 16, dtype=torch.bfloat16))
    assert temb.dtype == torch.bfloat16
    assert timestep_proj.dtype == torch.bfloat16

    ordinary = WanTimeTextImageEmbedding(dim=16, time_freq_dim=8, text_embed_dim=16)
    assert not ordinary.cast_temb_to_context_dtype


def test_vace_computes_timestep_frequencies_on_input_device():
    from fastvideo.layers.visual_embedding import timestep_embedding
    from fastvideo.models.wan.transformer import WanTimeTextImageEmbedding

    config = _tiny_config(AttentionBackendEnum.TORCH_SDPA)
    with _component_attention_backend_scope(AttentionBackendEnum.TORCH_SDPA, component="transformer"):
        model = WanVACETransformer3DModel(config, hf_config={})
    assert model.condition_embedder.time_embedder.freqs_on_input_device
    assert not WanTimeTextImageEmbedding(dim=16, time_freq_dim=8, text_embed_dim=16).time_embedder.freqs_on_input_device

    t = torch.tensor([968.73])
    assert torch.equal(timestep_embedding(t, 256), timestep_embedding(t, 256, freqs_on_input_device=True))
