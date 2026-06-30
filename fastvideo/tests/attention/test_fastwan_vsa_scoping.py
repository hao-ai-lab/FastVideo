import json
from pathlib import Path

import pytest

from fastvideo.attention.selector import (
    check_attn_backend_requirement,
    get_global_forced_attn_backend,
    global_force_attn_backend,
)
from fastvideo.configs.models.dits.base import DiTConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config,
    FastWan2_2_TI2V_5B_Config,
    FastWan2_2_TI2V_5B_FullAttn_Config,
    WanT2V480PConfig,
)
from fastvideo.models.dits.wanvideo import (
    WanTransformerBlock,
    WanTransformerBlock_VSA,
    _select_wan_transformer_block,
)
from fastvideo.platforms.interface import AttentionBackendEnum
from fastvideo.registry import get_pipeline_config_cls_from_name

VSA = AttentionBackendEnum.VIDEO_SPARSE_ATTN


@pytest.fixture(autouse=True)
def reset_forced_attn_backend():
    global_force_attn_backend(None)
    yield
    global_force_attn_backend(None)


def test_fastwan_required_vsa_does_not_leak_to_base_wan_config():
    # Given: no process-global attention backend force is active.
    assert get_global_forced_attn_backend() is None

    # When: a FastWan config is instantiated before a base Wan config.
    fastwan_config = FastWan2_1_T2V_480P_Config()
    base_wan_config = WanT2V480PConfig()

    # Then: FastWan carries the VSA requirement on its own DiT config only.
    assert (
        fastwan_config.dit_config.required_attention_backend
        == AttentionBackendEnum.VIDEO_SPARSE_ATTN
    )
    assert base_wan_config.dit_config.required_attention_backend is None
    assert get_global_forced_attn_backend() is None


def test_fastwan_2_2_required_vsa_does_not_mutate_global_backend():
    # Given: no process-global attention backend force is active.
    assert get_global_forced_attn_backend() is None

    # When: the FastWan 2.2 TI2V config is instantiated.
    fastwan_config = FastWan2_2_TI2V_5B_Config()

    # Then: the VSA requirement stays scoped to that DiT config.
    assert (
        fastwan_config.dit_config.required_attention_backend
        == AttentionBackendEnum.VIDEO_SPARSE_ATTN
    )
    assert get_global_forced_attn_backend() is None


def test_fastwan_2_2_fullattn_config_does_not_require_vsa():
    config = FastWan2_2_TI2V_5B_FullAttn_Config()

    assert config.dit_config.required_attention_backend is None
    assert config.dit_config.incompatible_attention_backends == (VSA, )


@pytest.mark.parametrize(
    "model_id",
    [
        "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
    ],
)
def test_fastwan_2_2_fullattn_hf_ids_resolve_dense_config(model_id: str) -> None:
    assert get_pipeline_config_cls_from_name(model_id) is FastWan2_2_TI2V_5B_FullAttn_Config


def _write_minimal_wan_dmd_repo(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "transformer").mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": "WanDMDPipeline",
            "_diffusers_version": "0.35.0.dev0",
            "transformer": ["diffusers", "WanTransformer3DModel"],
        }),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "relative_model_path",
    [
        Path("models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers") / "snapshots" / "deadbeef",
        Path("models--FastVideo--FastWan2.2-TI2V-5B-Diffusers") / "snapshots" / "deadbeef",
    ],
)
def test_fastwan_2_2_fullattn_local_path_resolves_dense_config(tmp_path: Path,
                                                               relative_model_path: Path) -> None:
    model_dir = tmp_path / relative_model_path
    _write_minimal_wan_dmd_repo(model_dir)

    resolved_cls = get_pipeline_config_cls_from_name(str(model_dir))

    assert resolved_cls is FastWan2_2_TI2V_5B_FullAttn_Config


def test_fastwan_2_2_fullattn_rejects_vsa_before_block_construction():
    config = FastWan2_2_TI2V_5B_FullAttn_Config().dit_config
    global_force_attn_backend(VSA)

    with pytest.raises(ValueError, match=r"incompatible with the VIDEO_SPARSE_ATTN attention backend"):
        _select_wan_transformer_block(config, model_name="FullAttn")


@pytest.mark.parametrize("backend", [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA])
def test_fastwan_2_2_fullattn_accepts_dense_backends(backend):
    config = FastWan2_2_TI2V_5B_FullAttn_Config().dit_config
    global_force_attn_backend(backend)

    assert _select_wan_transformer_block(config, model_name="FullAttn") is WanTransformerBlock


def test_sparse_fastwan_selects_vsa_block():
    config = FastWan2_2_TI2V_5B_Config().dit_config
    global_force_attn_backend(VSA)

    assert _select_wan_transformer_block(config, model_name="FastWan") is WanTransformerBlock_VSA


@pytest.mark.parametrize("config_cls", [FastWan2_1_T2V_480P_Config, FastWan2_2_TI2V_5B_Config])
def test_sparse_fastwan_pipeline_config_json_roundtrip(tmp_path, config_cls):
    config_path = tmp_path / "pipeline_config.json"
    config = config_cls()

    config.dump_to_json(str(config_path))

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["dit_config"]["required_attention_backend"] == VSA.name

    restored = config_cls()
    restored.load_from_json(str(config_path))
    assert restored.dit_config.required_attention_backend is VSA


def test_fullattn_pipeline_config_json_roundtrip(tmp_path):
    config_path = tmp_path / "pipeline_config.json"
    config = FastWan2_2_TI2V_5B_FullAttn_Config()

    config.dump_to_json(str(config_path))

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["dit_config"]["required_attention_backend"] is None
    assert payload["dit_config"]["incompatible_attention_backends"] == [VSA.name]

    restored = FastWan2_2_TI2V_5B_FullAttn_Config()
    restored.load_from_json(str(config_path))
    assert restored.dit_config.required_attention_backend is None
    assert restored.dit_config.incompatible_attention_backends == (VSA, )


def test_fullattn_config_cannot_be_overridden_to_require_vsa():
    config = FastWan2_2_TI2V_5B_FullAttn_Config()

    config.update_pipeline_config({"dit_config": {"required_attention_backend": VSA.name}})

    assert config.dit_config.required_attention_backend is None
    assert config.dit_config.incompatible_attention_backends == (VSA, )


def test_pipeline_config_load_restores_attention_backend_constraints(tmp_path):
    config_path = tmp_path / "pipeline_config.json"
    config = PipelineConfig(
        dit_config=DiTConfig(
            required_attention_backend=VSA,
            incompatible_attention_backends=(AttentionBackendEnum.FLASH_ATTN, ),
        ))
    config.dump_to_json(str(config_path))

    restored = PipelineConfig()
    restored.load_from_json(str(config_path))

    assert restored.dit_config.required_attention_backend is VSA
    assert restored.dit_config.incompatible_attention_backends == (AttentionBackendEnum.FLASH_ATTN, )


def test_check_requirement_returns_none_when_unrequired_and_unset(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    assert check_attn_backend_requirement(None) is None


def test_check_requirement_passes_when_env_matches(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", VSA.name)
    assert check_attn_backend_requirement(VSA) == VSA


def test_check_requirement_passes_when_global_force_matches(monkeypatch):
    # Env unset, but a global force satisfies the requirement (force > env).
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    global_force_attn_backend(VSA)
    assert check_attn_backend_requirement(VSA) == VSA


def test_check_requirement_raises_when_env_unset(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    with pytest.raises(ValueError) as excinfo:
        check_attn_backend_requirement(VSA, model_name="FastWan")
    message = str(excinfo.value)
    assert VSA.name in message
    assert "FastWan" in message


def test_check_requirement_raises_when_env_mismatches(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    with pytest.raises(ValueError):
        check_attn_backend_requirement(VSA)
