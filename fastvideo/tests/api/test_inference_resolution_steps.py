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


def test_unsupported_environment_attention_backend_raises():
    with pytest.raises(ValueError, match="FASTVIDEO_ATTENTION_BACKEND='NOT_A_BACKEND' is not a supported"):
        _resolve({"model_path": WAN_T2V}, {"FASTVIDEO_ATTENTION_BACKEND": "NOT_A_BACKEND"})


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
    assert resolved.pipeline.minimax_h3.vae_parallel_encode is False
    assert resolved.provenance("pipeline.minimax_h3.vae_parallel_encode").source == "fill_runtime_defaults"
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


def test_model_defaults_fill_unset_fields_and_keep_explicit_ones():
    raw = {"model_path": WAN_T2V, "engine": {"precision": {"dit": "fp32"}}}
    resolved = _resolve(raw)

    dit = resolved.provenance("engine.precision.dit")
    assert (dit.value, dit.source) == ("fp32", "input")
    vae = resolved.provenance("engine.precision.vae")
    assert vae.value == "fp32"
    assert vae.source == "fill_pipeline_config_defaults[WanT2V480PConfig]"
    flow_shift = resolved.provenance("pipeline.flow_shift")
    assert (flow_shift.raw_value, flow_shift.value) == (None, 3.0)


def test_environment_takes_precedence_over_model_defaults():
    from fastvideo.api.inference_resolution import inference_resolution_steps

    names = [step.__qualname__ for step in inference_resolution_steps(GeneratorConfig(model_path=WAN_T2V))]
    assert names == [
        "fill_attention_backend_from_env",
        "fill_regional_compile_from_env",
        "fill_vae_parallel_from_env",
        "fill_pipeline_config_defaults[WanT2V480PConfig]",
        "copy_refine_preset_overrides",
        "route_flat_override_keys",
        "derive_parallel_sizes",
        "derive_vae_tiling_from_ltx2_tile_sizes",
        "fill_vae_tiling_default[WanT2V480PConfig]",
        "load_moba_config",
        "validate_lora_strength",
        "validate_attention_backend",
        "validate_vae_parallel_decode_strategy",
        "warn_deprecated_environment_variables",
        "validate_parallel_sizes",
        "derive_num_gpus_from_parallel_sizes",
        "fill_runtime_defaults",
    ]


def test_video_generator_keeps_the_resolved_config(monkeypatch):
    from fastvideo.entrypoints import video_generator

    class _NoExecutor:
        """Executor stand-in so that VideoGenerator.__init__ runs without starting workers."""

        def __init__(self, fastvideo_args, log_queue=None):
            pass

    monkeypatch.setattr(video_generator.Executor, "get_class", staticmethod(lambda fastvideo_args: _NoExecutor))
    with isolated_environment():
        generator = video_generator.VideoGenerator.from_config({"model_path": WAN_T2V, "engine": {"num_gpus": 2}})

    assert generator.fastvideo_args.sp_size == 2
    provenance = generator.resolved_config.provenance("engine.parallelism.sp_size")
    assert (provenance.value, provenance.source) == (2, "derive_parallel_sizes")


@pytest.mark.parametrize(("pipeline", "expected"), [
    ({"ltx2": {"vae_spatial_tile_size_in_pixels": 512}}, (True, "derive_vae_tiling_from_ltx2_tile_sizes")),
    ({"ltx2": {"vae_spatial_tile_size_in_pixels": 512}, "vae_tiling": False}, (False, "input")),
    ({}, (True, "fill_vae_tiling_default[LTX2T2VConfig]")),
])
def test_ltx2_tile_sizes_turn_on_unset_vae_tiling(pipeline, expected):
    resolved = _resolve({"model_path": "FastVideo/LTX2-Distilled-Diffusers", "pipeline": pipeline})

    provenance = resolved.provenance("pipeline.vae_tiling")
    assert (provenance.value, provenance.source) == expected
