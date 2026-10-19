# SPDX-License-Identifier: Apache-2.0
"""CPU checks of examples/inference/basic/basic_fasth3_omniref_pdd.py (no weights, no GPU)."""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _load_example():
    spec = importlib.util.spec_from_file_location("basic_fasth3_omniref_pdd", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves the example's postponed annotations through the
    # defining module's __dict__, so it has to be registered before it runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
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
def test_base_model_source_command_line_overrides(overrides, expected):
    assert example.base_model_source(_args(*overrides), CONTRACT) == expected


@pytest.mark.parametrize("pin", [
    BASE_PIN, f"MiniMaxAI/MiniMax-H3@{BASE_PIN}", "hf://MiniMaxAI/MiniMax-H3", "hf://@abc", "", None,
])
@pytest.mark.parametrize("overrides", [(), ("--base-model-path", "someone/mirror")])
def test_base_model_source_unparsable_base_pin(pin, overrides):
    """Never a silent fall back to the base repo's latest revision; the pipeline rejects the same values."""
    with pytest.raises(ValueError, match="must be hf://<repo id>@<revision>"):
        example.base_model_source(_args(*overrides), {**CONTRACT, "base_model_revision": pin})


_COMPONENT = ["diffusers", "Component"]
_FULL_MANIFEST = {
    "_class_name": "MiniMaxH3ModularPipeline",
    **{name: _COMPONENT for name in ("transformer", "transformer_ref", "scheduler", "audio_scheduler",
                                     "text_encoder", "tokenizer", "processor", "vae", "audio_vae")},
}
_EXPORT_ONLY_MANIFEST = {
    "_class_name": "MiniMaxH3ModularPipeline",
    **{name: _COMPONENT for name in ("transformer_ref", "scheduler", "audio_scheduler")},
}


def _snapshot_dir(root: Path, components, manifest):
    root.mkdir()
    for name in components:
        (root / name).mkdir()
    if manifest is not None:
        (root / "modular_model_index.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize("export_manifest,chosen", [
    (_FULL_MANIFEST, "export"),
    (_EXPORT_ONLY_MANIFEST, "base"),
    (None, "base"),
])
def test_compose_model_dir_selects_complete_manifest(tmp_path, export_manifest, chosen):
    export = _snapshot_dir(tmp_path / "export", example.EXPORT_COMPONENTS, export_manifest)
    (export / example.CONTRACT).write_text(json.dumps(CONTRACT))
    base = _snapshot_dir(tmp_path / "base", example.BASE_COMPONENTS, _FULL_MANIFEST)
    composed = example.compose_model_dir(export, base, tmp_path / "composed")
    manifest = composed / "modular_model_index.json"
    assert manifest.resolve() == ((export if chosen == "export" else base) / "modular_model_index.json").resolve()
    for name in (*example.EXPORT_COMPONENTS, *example.BASE_COMPONENTS, example.CONTRACT):
        assert (composed / name).exists()


def test_compose_model_dir_no_complete_manifest(tmp_path):
    export = _snapshot_dir(tmp_path / "export", example.EXPORT_COMPONENTS, _EXPORT_ONLY_MANIFEST)
    (export / example.CONTRACT).write_text(json.dumps(CONTRACT))
    base = _snapshot_dir(tmp_path / "base", example.BASE_COMPONENTS, _EXPORT_ONLY_MANIFEST)
    with pytest.raises(FileNotFoundError, match="manifest that declares"):
        example.compose_model_dir(export, base, tmp_path / "composed")


def test_build_generator_config_leaves_trained_settings_to_checkpoint(tmp_path):
    # FastVideo reads the trained attention settings from the composed directory's fastvideo_inference.json.
    assert example.build_generator_config(tmp_path, 1).pipeline.experimental == {}


@pytest.mark.parametrize("num_gpus,sharded", [(1, False), (4, True)])
def test_build_generator_config_multi_gpu_shards_dit(tmp_path, num_gpus, sharded):
    config = example.build_generator_config(tmp_path, num_gpus)
    assert config.engine.use_fsdp_inference is sharded
    assert config.engine.parallelism.sp_size == num_gpus
    assert config.pipeline.components.override_pipeline_cls_name == "MiniMaxH3Ref2VAModularPipeline"


def test_default_config_offloads_the_encoders(tmp_path):
    offload = example.build_generator_config(tmp_path, 4).engine.offload
    assert (offload.text_encoder, offload.vae, offload.pin_cpu_memory) == (True, True, False)


@pytest.mark.parametrize("num_gpus", [1, 4])
def test_lossless_accel_config(tmp_path, num_gpus):
    config = example.build_generator_config(tmp_path, num_gpus, lossless_accel=True)
    offload = config.engine.offload
    assert (offload.text_encoder, offload.vae, offload.pin_cpu_memory) == (False, False, True)
    assert config.pipeline.experimental == {
        "vae_parallel_decode": num_gpus > 1,
        "vae_parallel_encode": num_gpus > 1,
    }


def test_lossless_accel_env_names_registered_flags_and_keeps_user_values() -> None:
    from fastvideo import envs

    with contextlib.ExitStack() as stack:
        for name in example.LOSSLESS_ACCEL_ENV:
            assert name in envs.environment_variables
            stack.enter_context(envs.environment_variables[name].override(None))
        stack.enter_context(envs.FASTVIDEO_ULYSSES_A2A.override("off"))
        example.apply_lossless_accel_env()
        assert envs.FASTVIDEO_ULYSSES_A2A.get() == "off"
        assert envs.FASTVIDEO_MINIMAX_H3_EXACT_KERNELS.get() == "all"
        assert envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.get() is True
        assert envs.FASTVIDEO_H3_VAE_TILE_PARALLEL.get() is True
        assert envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.get() == 16


def test_lossless_accel_flag_parses():
    argv = ["--model-path", "m", "--prompt", "p", "--image", "a.png"]
    assert example.parse_args(argv).lossless_accel is False
    assert example.parse_args([*argv, "--lossless-accel"]).lossless_accel is True


@pytest.mark.parametrize("choice", ["default", "fp8", "bf16"])
def test_attention_prefix_flag_parses(choice):
    argv = ["--model-path", "m", "--prompt", "p", "--image", "a.png", "--attention-prefix", choice]
    assert example.parse_args(argv).attention_prefix == choice


@pytest.mark.parametrize(("choice", "expected"), [
    ("default", "fp8"),
    ("fp8", "fp8"),
    ("bf16", "bf16"),
])
def test_attention_prefix_keeps_the_profile_default_or_overrides_it(choice, expected):
    from fastvideo import envs

    with envs.FASTVIDEO_H3_VSA_FP4_PREFIX.override("fp8"):
        example.apply_attention_prefix(choice)
        assert envs.FASTVIDEO_H3_VSA_FP4_PREFIX.get() == expected


def test_4090_rejects_packed_export_before_downloading_base(tmp_path, monkeypatch):
    export = tmp_path / "packed"
    (export / "transformer_ref").mkdir(parents=True)
    (export / "transformer_ref" / "nvfp4_weights.safetensors").touch()
    (export / example.CONTRACT).write_text(json.dumps({**CONTRACT, "model_type": "ref2va"}))
    calls = []

    def snapshot(*args):
        calls.append(args)
        return export

    monkeypatch.setattr(example, "_snapshot", snapshot)
    with pytest.raises(ValueError, match="requires a dense or FP8"):
        example.resolve_model(_args("--profile", "rtx-4090"))
    assert len(calls) == 1


def test_4090_profile_uses_fp8_and_tile128_bf16_fallback(tmp_path):
    import yaml

    profile = example.DEVICE_PROFILES["rtx-4090"]
    config = example.build_generator_config(tmp_path, 1, profile=profile)
    assert config.engine.quantization.transformer_quant == "FP8"
    assert profile.env["FASTVIDEO_H3_VSA_FP4"] == "0"
    assert profile.env["FASTVIDEO_H3_VSA_FINE_ATTENTION"] == "emulator-bf16"
    recipe = yaml.safe_load(EXAMPLE_PATH.with_name("basic_fasth3_omniref_rtx4090.yaml").read_text())
    assert "NVFP4" not in recipe["generator"]["model_path"]
    assert recipe["generator"]["engine"]["quantization"]["transformer_quant"] == "FP8"


def test_4090_accepts_dense_export(tmp_path, monkeypatch):
    export = _snapshot_dir(tmp_path / "export", example.EXPORT_COMPONENTS, _EXPORT_ONLY_MANIFEST)
    (export / example.CONTRACT).write_text(json.dumps({**CONTRACT, "model_type": "ref2va"}))
    base = _snapshot_dir(tmp_path / "base", example.BASE_COMPONENTS, _FULL_MANIFEST)
    snapshots = iter((export, base))
    monkeypatch.setattr(example, "_snapshot", lambda *args: next(snapshots))
    args = _args("--profile", "rtx-4090", "--composed-dir", str(tmp_path / "composed"))
    composed, _ = example.resolve_model(args)
    assert (composed / "transformer_ref").resolve() == (export / "transformer_ref").resolve()


@pytest.mark.parametrize("profile,dimensions,expected", [
    (None, (), (832, 480)),
    ("rtx-pro-6000", (), (1344, 768)),
    ("rtx-pro-6000", ("--width", "832", "--height", "480"), (832, 480)),
    ("rtx-pro-6000", ("--width", "832"), (832, 768)),
    ("rtx-pro-6000", ("--height", "480"), (1344, 480)),
    ("rtx-pro-6000", ("--width", "1024", "--height", "640"), (1024, 640)),
    (None, ("--width", "1024"), (1024, 480)),
    (None, ("--height", "640"), (832, 640)),
])
def test_main_preserves_explicit_dimensions_before_profile_defaults(tmp_path, monkeypatch, profile, dimensions, expected):
    options = ["--output", str(tmp_path), *dimensions]
    if profile:
        options += ["--profile", profile]
    args = _args(*options)
    if "--width" not in dimensions:
        assert args.width is None
    if "--height" not in dimensions:
        assert args.height is None
    requests = []
    shutdown = []

    def generate(request):
        requests.append(request)
        return SimpleNamespace(video_path="out.mp4", generation_time=None, peak_memory_mb=None)

    generator = SimpleNamespace(generate=generate, shutdown=lambda: shutdown.append(True))
    monkeypatch.setattr(example, "parse_args", lambda: args)
    monkeypatch.setattr(example, "resolve_model", lambda args: (tmp_path, CONTRACT))
    monkeypatch.setattr(example, "apply_profile", lambda *args: None)
    monkeypatch.setattr(example.VideoGenerator, "from_config", lambda config: generator)
    example.main()
    assert (requests[0].sampling.width, requests[0].sampling.height) == expected
    assert shutdown == [True]


@pytest.mark.parametrize("profile", ["spark", "spark_pair"])
@pytest.mark.parametrize("version", ["v2", "v21"])
def test_spark_recipe_selects_its_checkpoint_and_fp4(profile, version, monkeypatch, tmp_path):
    import yaml
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import GeneratorConfig
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs

    path = EXAMPLE_PATH.with_name(f"basic_fasth3_{profile}_{version}_nvfp4.yaml")
    raw = yaml.safe_load(path.read_text())["generator"]
    expected_version = "V2.1" if version == "v21" else "V2"
    assert raw["model_path"] == f"FastVideo/FastVideo-FastH3-8-Step-{expected_version}-NVFP4-Consumer"
    monkeypatch.setattr("fastvideo.registry.get_pipeline_config_cls_from_name", lambda path: MiniMaxH3PipelineConfig)
    original_from_kwargs = FastVideoArgs.from_kwargs

    def from_kwargs(**kwargs):
        kwargs["model_path"] = str(tmp_path)
        kwargs["pipeline_config"] = MiniMaxH3PipelineConfig()
        return original_from_kwargs(**kwargs)

    # Supply the pipeline locally so this configuration test never downloads a checkpoint.
    monkeypatch.setattr(FastVideoArgs, "from_kwargs", from_kwargs)
    args = generator_config_to_fastvideo_args(parse_config(GeneratorConfig, raw))
    assert args.h3_vsa_fp4["enabled"] is True
    assert args.VSA_tile_size == 64 and args.VSA_sparsity == 0.8
    assert args.sp_size == (2 if profile == "spark_pair" else 1)
    if version == "v21":
        assert args.h3_vsa_fp4 == {"enabled": True, "prefix_precision": "fp8", "two_level_p": True, "smooth_k": True}


@pytest.mark.parametrize("device", ["spark", "spark_pair", "rtx5090", "rtx_pro6000"])
def test_blackwell_omniref_yaml_enables_protected_fp4(device):
    import yaml

    raw = yaml.safe_load(EXAMPLE_PATH.with_name(f"basic_fasth3_omniref_{device}.yaml").read_text())
    assert raw["generator"]["pipeline"]["experimental"]["h3_vsa_fp4"] == {
        "enabled": True, "prefix_precision": "fp8", "two_level_p": True, "smooth_k": True}


@pytest.mark.parametrize("options", [None, {"unknown": True}, {"enabled": "true"}, {"smooth_k": 1},
                                    {"prefix_precision": "fp4"}])
def test_invalid_fp4_config_is_rejected_before_loading(options):
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs

    with pytest.raises(ValueError, match="h3_vsa_fp4"):
        FastVideoArgs(model_path="unused", pipeline_config=MiniMaxH3PipelineConfig(), h3_vsa_fp4=options)


@pytest.mark.parametrize("compile_flag", ["inference_torch_compile", "enable_torch_compile"])
def test_explicit_fp4_cannot_silently_compile_to_bf16(compile_flag):
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs

    with pytest.raises(ValueError, match="eager DiT inference"):
        FastVideoArgs(model_path="unused", pipeline_config=MiniMaxH3PipelineConfig(), h3_vsa_fp4={"enabled": True},
                      **{compile_flag: True})


def test_row_split_is_an_explicit_spark_opt_in():
    from fastvideo import envs

    assert envs.environment_variables["FASTVIDEO_NVFP4_MM_ROW_SPLIT"].default == "off"
    assert example.DEVICE_PROFILES["spark"].env["FASTVIDEO_NVFP4_MM_ROW_SPLIT"] == "table"
    assert "FASTVIDEO_NVFP4_MM_ROW_SPLIT" not in example.DEVICE_PROFILES["rtx-pro-6000"].env
