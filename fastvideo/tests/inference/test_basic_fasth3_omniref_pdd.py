# SPDX-License-Identifier: Apache-2.0
"""CPU checks of examples/inference/basic/basic_fasth3_omniref_pdd.py (no weights, no GPU)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fastvideo.platforms.interface import DeviceCapability

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "examples" / "inference" / "basic" / "basic_fasth3_omniref_pdd.py"
BASE_PIN = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
CONTRACT = {
    "schema_version": "fasth3-inference-contract-v1",
    "base_model_revision": f"hf://MiniMaxAI/MiniMax-H3@{BASE_PIN}",
    "pdd_steps": 32,
    "pdd_step_indices": list(range(0, 33, 4)),
    "num_inference_steps": 8,
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "audio_scheduler_shift": 3.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_ref_keep_rate": 0.1,
}
MINIMAL_CONTRACT = {
    key: CONTRACT[key]
    for key in ("schema_version", "pdd_steps", "pdd_step_indices", "num_inference_steps", "transformer_forwards",
                "video_scheduler_shift", "audio_scheduler_shift")
}


def _load_example():
    spec = importlib.util.spec_from_file_location("basic_fasth3_omniref_pdd", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()


def _args(*overrides: str):
    return example.parse_args(["--model-path", "export", "--image", "a.png", "--prompt", "p", *overrides])


@pytest.mark.parametrize("overrides,expected", [
    ((), ("MiniMaxAI/MiniMax-H3", BASE_PIN)),
    (("--base-revision", "abc"), ("MiniMaxAI/MiniMax-H3", "abc")),
    # The export's pin names a revision of its own base repo; it never applies to another one.
    (("--base-model-path", "someone/mirror"), ("someone/mirror", None)),
    (("--base-model-path", "someone/mirror", "--base-revision", "abc"), ("someone/mirror", "abc")),
])
def test_base_model_source(overrides, expected):
    assert example.base_model_source(_args(*overrides), CONTRACT) == expected


def test_minimal_contract_keeps_fastvideo_attention_defaults(tmp_path):
    assert example.attention_settings(CONTRACT) == {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "VSA_sparsity": 0.9,
        "VSA_tile_size": 128,
    }
    assert example.attention_settings(MINIMAL_CONTRACT) == {}
    config = example.build_generator_config(tmp_path, MINIMAL_CONTRACT, 1)
    assert config.pipeline.experimental == {}
    example.validate_attention_runtime(MINIMAL_CONTRACT, 1)


@pytest.mark.parametrize("num_gpus,sharded", [(1, False), (4, True)])
def test_multi_gpu_runs_shard_the_dit(tmp_path, num_gpus, sharded):
    config = example.build_generator_config(tmp_path, CONTRACT, num_gpus)
    assert config.engine.use_fsdp_inference is sharded
    assert config.engine.parallelism.sp_size == num_gpus
    assert config.pipeline.components.override_pipeline_cls_name == "MiniMaxH3Ref2VAModularPipeline"


@pytest.fixture
def capabilities(monkeypatch):
    from fastvideo.platforms import current_platform

    found: list[DeviceCapability | None] = []
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: found[device_id])
    return found


def test_tile128_needs_sm100a_devices(capabilities, monkeypatch):
    monkeypatch.setattr(example, "_tile128_kernel_is_installed", lambda: True)
    capabilities[:] = [DeviceCapability(10, 0), DeviceCapability(9, 0)]
    with pytest.raises(RuntimeError, match="sm_100a/sm_103a GPUs .* are: sm_100, sm_90"):
        example.validate_attention_runtime(CONTRACT, 2)
    capabilities[:] = [None]
    with pytest.raises(RuntimeError, match="are: none"):
        example.validate_attention_runtime(CONTRACT, 1)
    capabilities[:] = [DeviceCapability(10, 0), DeviceCapability(10, 3)]
    example.validate_attention_runtime(CONTRACT, 2)


def test_tile128_needs_the_128_token_kernel(capabilities, monkeypatch):
    capabilities[:] = [DeviceCapability(10, 0)]
    monkeypatch.setattr(example, "_tile128_kernel_is_installed", lambda: False)
    with pytest.raises(RuntimeError, match="128-token block_sparse_attn_sm100a forward"):
        example.validate_attention_runtime(CONTRACT, 1)
    # Other tile sizes do not need the sm_100a kernel.
    example.validate_attention_runtime({**CONTRACT, "vsa_tile_size": 64}, 1)
