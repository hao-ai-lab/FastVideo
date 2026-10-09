# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.api import (
    GenerationRequest,
    GeneratorConfig,
    RunConfig,
    SamplingConfig,
    ServeConfig,
    config_to_dict,
)


def test_run_config_roundtrip_preserves_nested_defaults() -> None:
    config = RunConfig(
        generator=GeneratorConfig(model_path="hf://model"),
        request=GenerationRequest(
            prompt="hello",
            sampling=SamplingConfig(num_frames=48, width=832, height=480),
        ),
    )

    dumped = config_to_dict(config)

    assert dumped["generator"]["model_path"] == "hf://model"
    assert dumped["generator"]["engine"]["execution_backend"] == "mp"
    assert dumped["request"]["sampling"]["num_frames"] == 48
    assert dumped["request"]["sampling"]["guidance_scale_2"] is None
    assert dumped["request"]["output"]["save_video"] is True


def test_serve_config_includes_server_and_default_request_defaults() -> None:
    config = ServeConfig(generator=GeneratorConfig(model_path="/models/ltx2"))

    dumped = config_to_dict(config)

    assert dumped["server"] == {
        "host": "0.0.0.0",
        "port": 8000,
        "output_dir": "outputs/",
        "served_model_name": None,
    }
    assert dumped["default_request"]["sampling"]["fps"] == 24
    assert dumped["default_request"]["runtime"]["enable_teacache"] is False


def test_causal_cuda_graph_config_roundtrip_and_compatibility() -> None:
    from fastvideo.api.compat import legacy_from_pretrained_to_config
    from fastvideo.api.parser import parse_config

    config = legacy_from_pretrained_to_config("hf://model", {"enable_causal_cuda_graph": True})
    assert config.engine.enable_causal_cuda_graph is True
    restored = parse_config(GeneratorConfig, config_to_dict(config))
    assert restored.engine.enable_causal_cuda_graph is True
    assert GeneratorConfig(model_path="hf://model").engine.enable_causal_cuda_graph is False


@pytest.mark.parametrize("enabled", [False, True])
def test_from_pretrained_accepts_graph_flag_without_legacy_warning(monkeypatch, enabled):
    import warnings
    from fastvideo import VideoGenerator

    received = []
    result = object()

    def from_config(cls, config, **kwargs):
        received.append(config)
        return result

    monkeypatch.setattr(VideoGenerator, "from_config", classmethod(from_config))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = VideoGenerator.from_pretrained("hf://model", enable_causal_cuda_graph=enabled)
    assert actual is result
    assert received[0].engine.enable_causal_cuda_graph is enabled
    assert not any(issubclass(warning.category, DeprecationWarning) for warning in recorded)


@pytest.mark.parametrize("flags,expected", [
    ([], False),
    (["--enable-causal-cuda-graph"], True),
    (["--enable-causal-cuda-graph", "true"], True),
    (["--enable-causal-cuda-graph", "false"], False),
])
def test_legacy_parser_can_enable_and_disable_causal_cuda_graph(flags, expected):
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FastVideoArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--model-path", "hf://model", *flags])
    assert args.enable_causal_cuda_graph is expected


def test_causal_cuda_graph_yaml_reaches_runtime_and_model_config() -> None:
    from pathlib import Path
    from fastvideo.api import load_config
    from fastvideo.api.compat import generator_config_to_fastvideo_args

    path = Path(__file__).resolve().parents[3] / "examples/inference/optimizations/causal_cuda_graph.yaml"
    config = load_config(RunConfig, path)
    args = generator_config_to_fastvideo_args(config.generator)
    assert args.enable_causal_cuda_graph is True
    assert args.dit_cpu_offload is False and args.dit_layerwise_offload is False
    assert args.pipeline_config.dit_config.arch_config.local_attn_size == 9
    assert args.pipeline_config.dit_config.arch_config.rope_cache_policy == "relativistic"


def test_causal_pipeline_override_defaults_preserve_architecture():
    from fastvideo.models.wan.pipeline_config import SelfForcingWanT2V480PConfig
    pipeline = SelfForcingWanT2V480PConfig()
    assert pipeline.dit_config.arch_config.local_attn_size == -1
    assert pipeline.dit_config.arch_config.rope_cache_policy == "absolute"


@pytest.mark.parametrize("overrides,message", [
    ({"causal_local_attn_size": 0}, "positive latent-frame"),
    ({"causal_local_attn_size": -2}, "positive latent-frame"),
    ({"causal_rope_cache_policy": "invalid"}, "absolute or relativistic"),
])
def test_invalid_causal_pipeline_overrides_are_rejected(overrides, message):
    from fastvideo.models.wan.pipeline_config import SelfForcingWanT2V480PConfig

    with pytest.raises(ValueError, match=message):
        SelfForcingWanT2V480PConfig(**overrides)


def test_engine_graph_flag_preserves_existing_positional_quantization_argument():
    from fastvideo.api import CompileConfig, EngineConfig, OffloadConfig, ParallelismConfig, QuantizationConfig

    quantization = QuantizationConfig()
    config = EngineConfig(1, "mp", ParallelismConfig(), OffloadConfig(), CompileConfig(), True, False, False, quantization)
    assert config.quantization is quantization
    assert config.enable_causal_cuda_graph is False
