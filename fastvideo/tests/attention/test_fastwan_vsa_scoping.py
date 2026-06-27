import pytest

from fastvideo.attention.selector import (
    check_attn_backend_requirement,
    get_global_forced_attn_backend,
    global_force_attn_backend,
)
from fastvideo.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config,
    FastWan2_2_TI2V_5B_Config,
    FastWan2_2_TI2V_5B_FullAttn_Config,
    WanT2V480PConfig,
)
from fastvideo.platforms.interface import AttentionBackendEnum

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
    # The dense FullAttn checkpoint shares FastWan's DMD schedule but must NOT be
    # forced onto VSA -- it runs dense attention.
    assert FastWan2_2_TI2V_5B_FullAttn_Config().dit_config.required_attention_backend is None


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
