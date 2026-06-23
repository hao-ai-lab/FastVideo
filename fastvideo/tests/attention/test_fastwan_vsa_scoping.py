import pytest

from fastvideo.attention.selector import (
    get_global_forced_attn_backend,
    global_force_attn_backend,
)
from fastvideo.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config,
    FastWan2_2_TI2V_5B_Config,
    WanT2V480PConfig,
)
from fastvideo.platforms.interface import AttentionBackendEnum


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
