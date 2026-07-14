# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn

import pytest

from fastvideo.attention.selector import global_force_attn_backend_context_manager
from fastvideo.configs.pipelines.wan import (
    FastWan2_2_TI2V_5B_Config,
    FastWan2_2_TI2V_5B_FullAttn_Config,
)
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.models.dits import wanvideo as wanvideo_module
from fastvideo.models.dits.wanvideo import (
    WanTransformerBlock,
    WanTransformerBlock_VSA,
    WanTransformer3DModel,
    _select_wan_transformer_block,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.registry import (
    get_pipeline_config_cls_from_name,
    get_registered_models_with_workloads,
)

FULLATTN_MODEL_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
FULLATTN_SHORT_NAME = "FastWan2.2-TI2V-5B-FullAttn-Diffusers"
FASTWAN_TI2V_MODEL_ID = "FastVideo/FastWan2.2-TI2V-5B-Diffusers"


def _workloads_for(model_id: str) -> list[str]:
    for model in get_registered_models_with_workloads():
        if model["id"] == model_id:
            return model["workload_types"]
    raise AssertionError(f"{model_id} was not registered")


def test_fastwan_fullattn_registry_is_t2v_only() -> None:
    assert get_pipeline_config_cls_from_name(FULLATTN_MODEL_ID) is FastWan2_2_TI2V_5B_FullAttn_Config
    assert get_pipeline_config_cls_from_name(FULLATTN_SHORT_NAME) is FastWan2_2_TI2V_5B_FullAttn_Config
    assert get_pipeline_config_cls_from_name(FASTWAN_TI2V_MODEL_ID) is FastWan2_2_TI2V_5B_Config

    assert _workloads_for(FULLATTN_MODEL_ID) == ["t2v"]
    i2v_model_ids = {model["id"] for model in get_registered_models_with_workloads("i2v")}
    assert FULLATTN_MODEL_ID not in i2v_model_ids
    assert FASTWAN_TI2V_MODEL_ID in i2v_model_ids


def test_fastwan_fullattn_config_is_dense_t2v_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    config = FastWan2_2_TI2V_5B_FullAttn_Config()

    assert config.ti2v_task is False
    assert config.vae_config.load_encoder is False
    assert config.vae_config.load_decoder is True
    assert config.dit_config.expand_timesteps is True

    supported_backends = config.dit_config._supported_attention_backends
    assert AttentionBackendEnum.VIDEO_SPARSE_ATTN not in supported_backends
    assert AttentionBackendEnum.FLASH_ATTN in supported_backends
    assert AttentionBackendEnum.TORCH_SDPA in supported_backends

    config.check_pipeline_config()


def test_fastwan_fullattn_fastvideo_args_accepts_t2v_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    args = FastVideoArgs.from_kwargs(model_path=FULLATTN_MODEL_ID, workload_type="t2v")

    assert args.workload_type is WorkloadType.T2V
    assert isinstance(args.pipeline_config, FastWan2_2_TI2V_5B_FullAttn_Config)


def test_fastwan_fullattn_fastvideo_args_prefers_global_dense_over_env_vsa(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        args = FastVideoArgs.from_kwargs(model_path=FULLATTN_MODEL_ID, workload_type="t2v")

    assert args.workload_type is WorkloadType.T2V
    assert isinstance(args.pipeline_config, FastWan2_2_TI2V_5B_FullAttn_Config)


def test_fastwan_fullattn_fastvideo_args_rejects_global_vsa_when_env_unset(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    with global_force_attn_backend_context_manager(AttentionBackendEnum.VIDEO_SPARSE_ATTN):
        with pytest.raises(ValueError, match="FullAttn.*VIDEO_SPARSE_ATTN"):
            FastVideoArgs.from_kwargs(model_path=FULLATTN_MODEL_ID, workload_type="t2v")


def test_fastwan_fullattn_fastvideo_args_rejects_i2v_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    with pytest.raises(ValueError, match="does not support workload type 'i2v'.*t2v"):
        FastVideoArgs.from_kwargs(model_path=FULLATTN_MODEL_ID, workload_type="i2v")


def test_fastwan_fullattn_config_rejects_vsa_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    with pytest.raises(ValueError, match="FullAttn.*VIDEO_SPARSE_ATTN"):
        FastWan2_2_TI2V_5B_FullAttn_Config().check_pipeline_config()


def test_wan_block_selection_rejects_vsa_for_fullattn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    with pytest.raises(ValueError, match="VIDEO_SPARSE_ATTN.*FullAttn"):
        _select_wan_transformer_block(FastWan2_2_TI2V_5B_FullAttn_Config().dit_config)


def test_wan_block_selection_preserves_existing_vsa_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    assert _select_wan_transformer_block(FastWan2_2_TI2V_5B_Config().dit_config) is WanTransformerBlock_VSA

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    assert _select_wan_transformer_block(FastWan2_2_TI2V_5B_Config().dit_config) is WanTransformerBlock


def test_wan_block_selection_honors_global_vsa_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    with global_force_attn_backend_context_manager(AttentionBackendEnum.VIDEO_SPARSE_ATTN):
        assert _select_wan_transformer_block(FastWan2_2_TI2V_5B_Config().dit_config) is WanTransformerBlock_VSA


def test_wan_block_selection_prefers_global_force_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        assert _select_wan_transformer_block(FastWan2_2_TI2V_5B_Config().dit_config) is WanTransformerBlock
        assert _select_wan_transformer_block(FastWan2_2_TI2V_5B_FullAttn_Config().dit_config) is WanTransformerBlock


def test_wan_transformer_uses_config_supported_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    captured_backends = []

    class CapturingWanTransformerBlock(nn.Module):

        def __init__(self,
                     dim: int,
                     ffn_dim: int,
                     num_heads: int,
                     qk_norm: str = "rms_norm_across_heads",
                     cross_attn_norm: bool = False,
                     eps: float = 1e-6,
                     added_kv_proj_dim: int | None = None,
                     supported_attention_backends: tuple[
                         AttentionBackendEnum, ...] | None = None,
                     **kwargs) -> None:
            super().__init__()
            captured_backends.append(supported_attention_backends)

    monkeypatch.setattr(wanvideo_module, "WanTransformerBlock", CapturingWanTransformerBlock)
    monkeypatch.setattr(wanvideo_module, "get_sp_world_size", lambda: 1)

    config = FastWan2_2_TI2V_5B_FullAttn_Config().dit_config
    config.arch_config.num_attention_heads = 1
    config.arch_config.attention_head_dim = 4
    config.arch_config.in_channels = 4
    config.arch_config.out_channels = 4
    config.arch_config.num_layers = 2
    config.arch_config.ffn_dim = 8
    config.arch_config.text_dim = 4
    config.arch_config.freq_dim = 4
    config.arch_config.patch_size = (1, 1, 1)
    config.arch_config.__post_init__()

    WanTransformer3DModel(config=config, hf_config={})

    assert captured_backends == [
        config._supported_attention_backends,
        config._supported_attention_backends,
    ]
    assert AttentionBackendEnum.VIDEO_SPARSE_ATTN not in captured_backends[0]
