# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.configs.pipelines.wan import (
    FastWan2_2_TI2V_5B_Config,
    FastWan2_2_TI2V_5B_FullAttn_Config,
)
from fastvideo.models.dits.wanvideo import (
    WanTransformerBlock,
    WanTransformerBlock_VSA,
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
