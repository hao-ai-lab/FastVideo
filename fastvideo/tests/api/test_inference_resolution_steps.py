# SPDX-License-Identifier: Apache-2.0
"""Each inference resolution step decides its fields with the documented precedence and records its name."""
import pytest

from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.schema import AttentionConfig, EngineConfig, GeneratorConfig, ParallelismConfig
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _resolve(raw, env_values=None):
    with isolated_environment(env_values):
        return resolve_inference_config(raw)


def test_environment_fills_unset_attention_backend():
    resolved = _resolve({"model_path": WAN_T2V}, {"FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA"})

    provenance = resolved.provenance("engine.attention.backend")
    assert provenance.value == "TORCH_SDPA"
    assert provenance.source == "fill_attention_backend_from_env"
    assert provenance.explicit is False


def test_explicit_attention_backend_wins_over_environment():
    raw = {"model_path": WAN_T2V, "engine": {"attention": {"backend": "FLASH_ATTN"}}}
    resolved = _resolve(raw, {"FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA"})

    provenance = resolved.provenance("engine.attention.backend")
    assert (provenance.value, provenance.source, provenance.explicit) == ("FLASH_ATTN", "input", True)


def test_environment_turns_on_regional_compile_and_parallel_vae():
    resolved = _resolve(
        {"model_path": WAN_T2V},
        {
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": True,
            "FASTVIDEO_VAE_PARALLEL_DECODE": True,
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "all_gather",
        },
    )

    assert resolved.engine.compile.regional is True
    assert resolved.pipeline.minimax_h3.vae_parallel_decode is True
    assert resolved.pipeline.minimax_h3.vae_parallel_encode is None
    assert resolved.pipeline.minimax_h3.vae_parallel_decode_strategy == "all_gather"
    assert resolved.provenance("engine.compile.regional").source == "fill_regional_compile_from_env"


def test_decode_strategy_defaults_to_gather():
    resolved = _resolve({"model_path": WAN_T2V})

    provenance = resolved.provenance("pipeline.minimax_h3.vae_parallel_decode_strategy")
    assert (provenance.value, provenance.source) == ("gather", "fill_vae_parallel_from_env")


@pytest.mark.parametrize(("parallelism", "expected"), [
    ({}, {"tp_size": 1, "sp_size": 4, "hsdp_shard_dim": 4}),
    ({"tp_size": 2, "sp_size": 2}, {"tp_size": 2, "sp_size": 2, "hsdp_shard_dim": 4}),
])
def test_parallel_placeholders_take_num_gpus(parallelism, expected):
    resolved = _resolve({"model_path": WAN_T2V, "engine": {"num_gpus": 4, "parallelism": parallelism}})

    for name, value in expected.items():
        provenance = resolved.provenance(f"engine.parallelism.{name}")
        assert provenance.value == value
        assert provenance.source == ("input" if name in parallelism else "derive_parallel_sizes")


def test_object_fields_that_differ_from_defaults_are_explicit():
    config = GeneratorConfig(
        model_path=WAN_T2V,
        engine=EngineConfig(parallelism=ParallelismConfig(sp_size=1), attention=AttentionConfig(backend="FLASH_ATTN")),
    )
    resolved = _resolve(config, {"FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA"})

    assert resolved.is_explicit("engine.parallelism.sp_size")
    assert resolved.is_explicit("engine.attention.backend")
    assert not resolved.is_explicit("engine.parallelism.tp_size")
    assert resolved.engine.attention.backend == "FLASH_ATTN"
