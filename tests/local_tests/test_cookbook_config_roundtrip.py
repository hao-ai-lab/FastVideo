"""Check recipe YAML downloads with the real serving CLI and request adapter.

No server, model weights, or GPU generation is started. The Node resolver is
used directly, so these tests cover the browser's serializer and launch output.
"""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from fastvideo.api.compat import explicit_request_updates, generator_config_to_fastvideo_args
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.cli.serve import ServeSubcommand
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import build_generation_request
from fastvideo.utils import FlexibleArgumentParser

ROOT = Path(__file__).resolve().parents[2]


def export_catalogs(output_dir, recipes_dir=None):
    """Generate authored catalogs independently, then check them with native Python."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for cookbook metadata generation")
    command = [node, str(ROOT / "docs/build-cookbook-config.mjs"), "--output-dir", str(output_dir)]
    if recipes_dir is not None:
        command += ["--recipes-dir", str(recipes_dir)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads((output_dir / "index.json").read_text())


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    output = tmp_path_factory.mktemp("cookbook-catalog")
    index = export_catalogs(output_dir=output)
    model = next(model for model in index["models"] if model["id"] == "fastwan21")
    return json.loads((output / model["deployments"][0]["catalog_url"]).read_text())


def resolve_in_browser(catalog, selections):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser YAML serializer")
    script = """
const {readFileSync} = require('node:fs');
const {resolveConfig} = require('./docs/assets/cookbook-config.js');
const input = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(resolveConfig(input.catalog, input.selections)));
"""
    result = subprocess.run([node, "-e", script], input=json.dumps({"catalog": catalog, "selections": selections}),
                            capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(result.stdout)


def parse_download(result, catalog, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(result["yaml"], encoding="utf-8")
    assert yaml.safe_load(result["yaml"]) == result["config"]
    tokens = shlex.split(result["command"])
    expected_env = [f"{name}={value}" for name, value in catalog["env"].items()]
    assert tokens == expected_env + ["fastvideo", "serve", "--config", "config.yaml"]
    parser = FlexibleArgumentParser()
    subcommand = ServeSubcommand()
    subcommand.subparser_init(parser.add_subparsers(dest="subparser"))
    args, unknown = parser.parse_known_args(["serve", "--config", str(config_path)])
    assert unknown == []
    args._unknown = unknown
    subcommand.validate(args)
    parsed = build_serve_config(args, unknown)
    translated = generator_config_to_fastvideo_args(parsed.generator)
    assert translated.pipeline_config.dmd_denoising_steps == [1000, 757, 522]
    assert translated.VSA_sparsity == 0.8
    return parsed


@pytest.mark.parametrize("selections", [
    {}, {"server.port": 9000}, {"generator.engine.compile.enabled": False},
    {"generator.engine.compile.enabled": True}, {"default_request.sampling.num_frames": 121},
    {"default_request.sampling.num_frames": 82, "server.port": 8080},
])
def test_every_visible_choice_and_numeric_edits_round_trip(catalog, selections, tmp_path):
    result = resolve_in_browser(catalog, selections)
    parsed = parse_download(result, catalog, tmp_path)
    assert parsed.generator.model_path == catalog["model"]["id"]
    assert parsed.default_request.sampling.num_frames == selections.get("default_request.sampling.num_frames", 81)
    assert parsed.generator.engine.compile.enabled is selections.get("generator.engine.compile.enabled", True)
    assert parsed.default_request.output.return_frames is False
    assert parsed.default_request.sampling.num_inference_steps == 3
    assert parsed.server.port == selections.get("server.port", 8000)
    assert parsed.default_request.negative_prompt == catalog["base_config"]["default_request"]["negative_prompt"]
    updates = explicit_request_updates(parsed.default_request)
    assert updates["num_frames"] == selections.get("default_request.sampling.num_frames", 81)
    assert updates["num_inference_steps"] == 3


def test_reset_and_download_without_edits_restore_exact_native_baseline(catalog, tmp_path):
    result = resolve_in_browser(catalog, {})
    native = yaml.safe_load((ROOT / "examples/serving/openai_fastwan21_1_3b.yaml").read_text())
    assert result["config"] == native
    resolve_in_browser(catalog, {"server.port": 9999})
    assert resolve_in_browser(catalog, {})["config"] == native
    parse_download(result, catalog, tmp_path)


def test_sample_request_uses_alias_and_inherits_baseline_but_client_values_win(catalog, tmp_path):
    result = resolve_in_browser(catalog, {"default_request.sampling.num_frames": 121, "server.port": 9000})
    config = parse_download(result, catalog, tmp_path)
    assert "http://127.0.0.1:9000/v1/videos/sync" in result["clientCommand"]
    assert result["clientRequest"]["model"] == "fastwan21-1.3b"
    args = SimpleNamespace(model_path=config.generator.model_path, lora_path=None, lora_nickname="default",
                           override_pipeline_cls_name=None)
    for body, frames in [(result["clientRequest"], 121), ({**result["clientRequest"], "num_frames": 41}, 41)]:
        request = build_generation_request("cookbook-recipe-test", VideoGenerationRequest(**body), args,
                                           output_dir=str(tmp_path), served_model_name=config.server.served_model_name,
                                           default_request=config.default_request)
        assert request.sampling.num_frames == frames
        assert request.sampling.num_inference_steps == 3


def test_nullable_and_zero_edit_remain_explicit_through_native_parser(catalog, tmp_path):
    # Synthetic presentation extension uses real public field definitions, not a model-specific renderer.
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    manifest = yaml.safe_load((ROOT / "docs/cookbook/recipes/fastwan21.yaml").read_text())
    manifest["deployments"]["cuda-rest"]["controls"] += ["generator.pipeline.vae_tiling", "default_request.sampling.guidance_scale"]
    manifest["deployments"]["cuda-rest"]["overrides"] = {
        "generator.pipeline.vae_tiling": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": None},
        "default_request.sampling.guidance_scale": {"type": "number"},
    }
    (recipes / "nullable-example.yaml").write_text(yaml.safe_dump(manifest))
    output = tmp_path / "catalogs"
    index = export_catalogs(output, recipes)
    extended = json.loads((output / index["models"][0]["deployments"][0]["catalog_url"]).read_text())
    result = resolve_in_browser(extended, {"generator.pipeline.vae_tiling": None,
                                            "default_request.sampling.guidance_scale": 0.0})
    config = parse_download(result, extended, tmp_path)
    assert config.generator.pipeline.vae_tiling is None
    assert explicit_request_updates(config.default_request)["guidance_scale"] == 0.0


def test_mlx_yaml_download_round_trips_native_config_without_cuda_settings(tmp_path, monkeypatch):
    from fastvideo.entrypoints.openai import mlx_server

    def forbidden(*args, **kwargs):
        pytest.fail("A metadata round trip must not construct the MLX generator")

    monkeypatch.setattr(mlx_server, "MLXH3Generator", forbidden)
    output = tmp_path / "catalogs"
    index = export_catalogs(output_dir=output)
    model = next(model for model in index["models"] if model["id"] == "fasth3-8step")
    deployment = next(item for item in model["deployments"] if item["runtime"] == "fastvideo-mlx-rest")
    catalog = json.loads((output / deployment["catalog_url"]).read_text())
    selections = {"server.port": 9001, "generator.model_root": "./my weights", "generator.vae_dtype": "bf16",
                  "generator.vsa_sparsity": 0.0}
    result = resolve_in_browser(catalog, selections)
    assert shlex.split(result["command"]) == [
        "python", "-m", "fastvideo.entrypoints.openai.mlx_server", "--config", "config.yaml"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(result["yaml"])
    parsed = mlx_server.load_config(str(config_path))
    assert parsed.generator.model_root == "./my weights"
    assert parsed.generator.vae_dtype == "bf16"
    assert parsed.generator.vsa_sparsity == 0.0
    assert parsed.server.port == 9001
    assert parsed.default_request == catalog["base_config"]["default_request"]
    assert "engine" not in yaml.safe_load(result["yaml"])["generator"]
    assert result["clientRequest"]["model"] == "fasth3"
    assert "http://127.0.0.1:9001/v1/videos/sync" in result["clientCommand"]
    validation = parsed.model_copy(deep=True)
    validation.server.output_dir = str(tmp_path / "validation-output")
    mlx_server.create_mlx_app(validation)
    assert resolve_in_browser(catalog, {})["config"] == catalog["base_config"]


    # An omitted alias uses the native MLX server default in the request only;
    # serialization must not silently insert that inherited field into YAML.
    del catalog["base_config"]["server"]["served_model_name"]
    inherited = resolve_in_browser(catalog, {})
    assert inherited["clientRequest"]["model"] == "fasth3"
    assert "served_model_name" not in yaml.safe_load(inherited["yaml"])["server"]
    config_path.write_text(inherited["yaml"])
    assert mlx_server.load_config(str(config_path)).server.served_model_name == "fasth3"


def test_streaming_yaml_and_launch_preserve_native_session_settings(tmp_path):
    output = tmp_path / "catalogs"
    index = export_catalogs(output_dir=output)
    model = next(model for model in index["models"] if model["id"] == "ltx2-distilled")
    catalog = json.loads((output / model["deployments"][0]["catalog_url"]).read_text())
    result = resolve_in_browser(catalog, {"server.host": "0.0.0.0", "server.port": 8010,
                                        "default_request.sampling.num_frames": 129})
    assert shlex.split(result["command"]) == ["fastvideo", "serve", "--config", "config.yaml"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(result["yaml"])
    parser = FlexibleArgumentParser()
    subcommand = ServeSubcommand()
    subcommand.subparser_init(parser.add_subparsers(dest="subparser"))
    args, unknown = parser.parse_known_args(["serve", "--config", str(config_path)])
    args._unknown = unknown
    subcommand.validate(args)
    native = build_serve_config(args, unknown)
    generator_config_to_fastvideo_args(native.generator)
    assert native.streaming is not None
    assert native.server.port == 8010
    assert native.default_request.sampling.num_frames == 129
    assert yaml.safe_load(result["yaml"])["streaming"] == catalog["base_config"]["streaming"]
    assert result["clientRequest"] is None
    assert result["websocketUrl"] == "ws://127.0.0.1:8010/v1/stream"
    assert result["clientCommand"] == "curl --fail-with-body http://127.0.0.1:8010/health"
    assert "/v1/videos" not in result["clientCommand"]
    assert resolve_in_browser(catalog, {})["config"] == catalog["base_config"]


def test_i2v_native_baseline_and_sample_require_an_image(tmp_path):
    output = tmp_path / "catalogs"
    index = export_catalogs(output_dir=output)
    model = next(model for model in index["models"] if model["id"] == "wan21-i2v")
    catalog = json.loads((output / model["deployments"][0]["catalog_url"]).read_text())
    result = resolve_in_browser(catalog, {"server.port": 9002, "generator.pipeline.experimental.flow_shift": 4.0})
    native_yaml = yaml.safe_load((ROOT / "examples/serving/openai_wan21_i2v_14b.yaml").read_text())
    assert resolve_in_browser(catalog, {})["config"] == native_yaml
    assert result["config"]["generator"]["engine"] == native_yaml["generator"]["engine"]
    assert result["config"]["generator"]["pipeline"]["workload_type"] == "i2v"
    body = result["clientRequest"]
    assert body["input_reference"] == "/absolute/path/to/first-frame.png"
    assert body["model"] == "wan21-i2v-14b"
    assert "http://127.0.0.1:9002/v1/videos/sync" in result["clientCommand"]
    VideoGenerationRequest(**body)
    # The model-local authored option maps to a real pipeline setting.
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import ServeConfig
    parsed = parse_config(ServeConfig, yaml.safe_load(result["yaml"]))
    translated = generator_config_to_fastvideo_args(parsed.generator)
    assert translated.pipeline_config.flow_shift == 4.0


@pytest.mark.parametrize("parallelism", [2, -1])
def test_fasth3_gpu_and_parallelism_edits_round_trip_native_config(tmp_path, parallelism):
    output = tmp_path / "catalogs"
    index = export_catalogs(output_dir=output)
    model = next(model for model in index["models"] if model["id"] == "fasth3-8step")
    deployment = next(item for item in model["deployments"] if item["runtime"] == "fastvideo-cuda-rest")
    catalog = json.loads((output / deployment["catalog_url"]).read_text())
    result = resolve_in_browser(catalog, {
        "generator.engine.num_gpus": 2,
        "generator.engine.parallelism.sp_size": parallelism,
    })
    config_path = tmp_path / "config.yaml"
    config_path.write_text(result["yaml"])
    parser = FlexibleArgumentParser()
    subcommand = ServeSubcommand()
    subcommand.subparser_init(parser.add_subparsers(dest="subparser"))
    args, unknown = parser.parse_known_args(["serve", "--config", str(config_path)])
    args._unknown = unknown
    subcommand.validate(args)
    native = build_serve_config(args, unknown)
    translated = generator_config_to_fastvideo_args(native.generator)
    assert native.generator.engine.num_gpus == 2
    assert native.generator.engine.parallelism.sp_size == parallelism
    assert translated.num_gpus == 2
    assert translated.sp_size == 2
    assert translated.tp_size == 1
    assert native.generator.engine.use_fsdp_inference is False
    assert result["config"]["generator"]["pipeline"] == catalog["base_config"]["generator"]["pipeline"]
    assert resolve_in_browser(catalog, {})["config"] == catalog["base_config"]
