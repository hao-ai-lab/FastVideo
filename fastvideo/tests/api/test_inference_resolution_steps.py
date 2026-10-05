# SPDX-License-Identifier: Apache-2.0
"""Each inference resolution step decides its fields with the documented precedence and records its name."""
import pytest

from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.schema import AttentionConfig, EngineConfig, GeneratorConfig, ParallelismConfig
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LTX2 = "FastVideo/LTX2-Distilled-Diffusers"
MINIMAX_H3 = "FastVideo/FastVideo-FastH3-8-Step-V2"
LONGCAT = "FastVideo/LongCat-Video-T2V-Diffusers"


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
        {"model_path": MINIMAX_H3},
        {
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": True,
            "FASTVIDEO_VAE_PARALLEL_DECODE": True,
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "all_gather",
        },
    )

    assert resolved.engine.compile.regional is True
    assert resolved.pipeline.model.vae_parallel_decode is True
    assert resolved.pipeline.model.vae_parallel_encode is False
    assert resolved.provenance("pipeline.model.vae_parallel_encode").source == "fill_runtime_defaults"
    assert resolved.pipeline.model.vae_parallel_decode_strategy == "all_gather"
    assert resolved.provenance("engine.compile.regional").source == "fill_regional_compile_from_env"


def test_decode_strategy_defaults_to_gather():
    resolved = _resolve({"model_path": MINIMAX_H3})

    provenance = resolved.provenance("pipeline.model.vae_parallel_decode_strategy")
    assert (provenance.value, provenance.source) == ("gather", "fill_vae_parallel_from_env")


def test_parallel_vae_environment_decides_nothing_for_another_family():
    resolved = _resolve({"model_path": WAN_T2V}, {"FASTVIDEO_VAE_PARALLEL_DECODE": True})

    assert resolved.pipeline.model.family == "generic"
    assert [source for source, _ in resolved.decisions if source == "fill_vae_parallel_from_env"] == []


@pytest.mark.parametrize(("model_path", "family", "options_type"), [
    (LTX2, "ltx2", "LTX2Options"),
    (MINIMAX_H3, "minimax_h3", "MiniMaxH3Options"),
    (LONGCAT, "longcat", "LongCatOptions"),
    (WAN_T2V, "generic", "GenericModelOptions"),
])
def test_fill_model_family_fills_the_family_block_of_the_model(model_path, family, options_type):
    resolved = _resolve({"model_path": model_path})

    model = resolved.pipeline.model
    assert model.family == family
    assert type(resolved.to_config().pipeline.model).__name__ == options_type
    assert (model.dit, model.vae) == ({}, {})
    assert resolved.provenance("pipeline.model").source == f"fill_model_family[{family}]"
    assert resolved.provenance("pipeline.model").explicit is False


def test_generic_block_on_a_family_model_becomes_the_family_block_with_its_overrides():
    resolved = _resolve({"model_path": LTX2, "pipeline": {"model": {"generic": {"dit": {"prefix": "dit"}}}}})

    assert resolved.pipeline.model.family == "ltx2"
    assert resolved.pipeline.model.dit == {"prefix": "dit"}
    assert resolved.pipeline.model.refine.enabled is False
    assert resolved.provenance("pipeline.model.dit.prefix").explicit is True
    assert resolved.pipeline_config.dit_config.prefix == "dit"


def test_validate_model_family_names_the_registry_family():
    with pytest.raises(ValueError, match=r"pipeline\.model\.ltx2 does not apply to .*WanT2V480PConfig.*pipeline\.model\.generic"):
        _resolve({"model_path": WAN_T2V, "pipeline": {"model": {"ltx2": {"refine": {"enabled": True}}}}})
    with pytest.raises(ValueError, match=r"pipeline\.model\.longcat does not apply to .*pipeline\.model\.minimax_h3"):
        _resolve({"model_path": MINIMAX_H3, "pipeline": {"model": {"longcat": {"enable_bsa": True}}}})


def test_written_family_block_is_explicit_without_its_tag():
    resolved = _resolve({"model_path": LTX2, "pipeline": {"model": {"ltx2": {"refine": {"lora_path": ""}}}}})

    assert resolved.is_explicit("pipeline.model.refine.lora_path") is True
    assert resolved.is_explicit("pipeline.model.refine.enabled") is False
    assert resolved.provenance("pipeline.model.refine.lora_path").source == "input"


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
        "validate_model_family",
        "fill_model_family[generic]",
        "fill_attention_backend_from_env",
        "fill_regional_compile_from_env",
        "fill_vae_parallel_from_env",
        "fill_pipeline_config_defaults[WanT2V480PConfig]",
        "copy_refine_preset_overrides",
        "fill_ltx2_refine_from_checkpoint",
        "fill_dmd_schedule_from_checkpoint",
        "derive_parallel_sizes",
        "derive_vae_tiling_from_ltx2_tile_sizes",
        "fill_vae_tiling_default[WanT2V480PConfig]",
        "load_moba_config",
        "apply_unified_memory_offload_policy",
        "fill_lazy_module_load",
        "apply_mps_offload_policy",
        "apply_layerwise_offload_conflicts",
        "validate_unsupported_fields",
        "validate_experimental_keys",
        "validate_lora_strength",
        "validate_attention_backend",
        "validate_vae_parallel_decode_strategy",
        "warn_deprecated_environment_variables",
        "apply_nvfp4_fa4_env",
        "validate_parallel_sizes",
        "derive_num_gpus_from_parallel_sizes",
        "fill_runtime_defaults",
    ]


def test_video_generator_keeps_the_resolved_config(monkeypatch):
    from fastvideo.entrypoints import video_generator

    class _NoExecutor:
        """Executor stand-in so that VideoGenerator.__init__ runs without starting workers."""

        def __init__(self, resolved_config, log_queue=None):
            pass

    monkeypatch.setattr(video_generator.Executor, "get_class", staticmethod(lambda resolved_config: _NoExecutor))
    with isolated_environment():
        generator = video_generator.VideoGenerator.from_config({"model_path": WAN_T2V, "engine": {"num_gpus": 2}})

    assert generator.resolved_config.engine.parallelism.sp_size == 2
    provenance = generator.resolved_config.provenance("engine.parallelism.sp_size")
    assert (provenance.value, provenance.source) == (2, "derive_parallel_sizes")


@pytest.mark.parametrize(("pipeline", "expected"), [
    ({"model": {"ltx2": {"vae_spatial_tile_size_in_pixels": 512}}}, (True, "derive_vae_tiling_from_ltx2_tile_sizes")),
    ({"model": {"ltx2": {"vae_spatial_tile_size_in_pixels": 512}}, "vae_tiling": False}, (False, "input")),
    ({}, (True, "fill_vae_tiling_default[LTX2T2VConfig]")),
])
def test_ltx2_tile_sizes_turn_on_unset_vae_tiling(pipeline, expected):
    resolved = _resolve({"model_path": "FastVideo/LTX2-Distilled-Diffusers", "pipeline": pipeline})

    provenance = resolved.provenance("pipeline.vae_tiling")
    assert (provenance.value, provenance.source) == expected
