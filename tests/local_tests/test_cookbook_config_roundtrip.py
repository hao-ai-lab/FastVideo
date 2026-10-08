"""Parse the browser's actual YAML download and verify server request defaults.

These checks execute the shared JavaScript resolver, load its YAML with the real
serving CLI, and exercise the existing HTTP request adapter. No server, model
weights, or GPU generation is started.
"""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from docs.cookbook_config import build_catalog, export_catalogs
from fastvideo.api.compat import explicit_request_updates
from fastvideo.api.schema import ServeConfig, GeneratorConfig
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.cli.serve import ServeSubcommand
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import build_generation_request
from fastvideo.utils import FlexibleArgumentParser

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cookbook-catalog")
    selection = directory / "models.yaml"
    selection.write_text(yaml.safe_dump({"models": ["Wan-AI/Wan2.2-TI2V-5B-Diffusers"]}), encoding="utf-8")
    output = directory / "assets"
    index = export_catalogs(output_dir=output, models_file=selection)
    return json.loads((output / index["models"][0]["catalog_url"]).read_text())


def resolve_in_browser(catalog, selections):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to test the browser's YAML serializer")
    script = """
const {readFileSync} = require('node:fs');
const {resolveConfig} = require('./docs/assets/cookbook-config.js');
const input = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(resolveConfig(input.catalog, input.selections)));
"""
    completed = subprocess.run(
        [node, "-e", script],
        input=json.dumps({"catalog": catalog, "selections": selections}),
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(completed.stdout)


def parse_download(result, tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(result["yaml"], encoding="utf-8")
    assert yaml.safe_load(result["yaml"]) == result["config"]
    assert shlex.split(result["command"]) == ["fastvideo", "serve", "--config", "config.yaml"]

    monkeypatch.chdir(tmp_path)
    parser = FlexibleArgumentParser()
    subcommand = ServeSubcommand()
    subcommand.subparser_init(parser.add_subparsers(dest="subparser"))
    args, unknown = parser.parse_known_args(shlex.split(result["command"])[1:])
    assert unknown == []
    args._unknown = unknown
    subcommand.validate(args)
    return build_serve_config(args, unknown)


@pytest.mark.parametrize("selections", [
    {},
    {
        "default_request.sampling.num_frames": 81,
        "generator.engine.offload.vae": False,
        "server.port": 9000,
    },
    {
        "generator.engine.num_gpus": 2,
        "server.host": "localhost",
        "default_request.sampling.seed": -42,
        "default_request.sampling.guidance_scale": 3.5,
    },
    {"default_request.sampling.guidance_scale": 0.0000001},
    {"default_request.sampling.num_frames": 82},
    {"generator.pipeline.vae_tiling": None, "generator.engine.offload.lazy_module_load": None},
    {"/generator/pipeline/experimental/vae_config.load_encoder": True},
])
def test_download_round_trips_through_real_cli(catalog, selections, tmp_path, monkeypatch):
    result = resolve_in_browser(catalog, selections)
    config = parse_download(result, tmp_path, monkeypatch)

    assert config.generator.model_path == catalog["model"]["id"]
    frame_path = "default_request.sampling.num_frames"
    native = ServeConfig(generator=GeneratorConfig(model_path=catalog["model"]["id"]))
    expected_frames = selections.get(frame_path, native.default_request.sampling.num_frames)
    assert config.default_request.sampling.num_frames == expected_frames
    assert config.generator.engine.offload.vae is selections.get("generator.engine.offload.vae", True)
    assert config.default_request.output.return_frames is native.default_request.output.return_frames
    if "default_request.sampling.seed" in selections:
        assert config.default_request.sampling.seed == selections["default_request.sampling.seed"]
    if "default_request.sampling.guidance_scale" in selections:
        assert config.default_request.sampling.guidance_scale == selections["default_request.sampling.guidance_scale"]
    updates = explicit_request_updates(config.default_request)
    expected_updates = {
        path.rsplit(".", 1)[-1]: value
        for path, value in selections.items() if path.startswith("default_request.sampling.")
    }
    assert updates == expected_updates
    if "/generator/pipeline/experimental/vae_config.load_encoder" in selections:
        assert config.generator.pipeline.experimental == {"vae_config.load_encoder": True}
    if "generator.pipeline.vae_tiling" in selections:
        assert config.generator.pipeline.vae_tiling is None
        assert config.generator.engine.offload.lazy_module_load is None


@pytest.mark.parametrize("model_id, frames", [
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 121),
    ("Tongyi-MAI/Z-Image-Turbo", 1),
])
def test_model_defaults_are_displayed_without_pinning_request_values(model_id, frames, tmp_path, monkeypatch):
    result = resolve_in_browser(build_catalog(model_id), {})
    config = parse_download(result, tmp_path, monkeypatch)
    assert result["values"]["default_request"]["sampling"]["num_frames"] == frames
    assert "default_request" not in result["config"]
    assert explicit_request_updates(config.default_request) == {}
    assert config.generator.model_path == model_id


def test_prompt_only_client_inherits_yaml_and_explicit_request_wins(catalog, tmp_path, monkeypatch):
    result = resolve_in_browser(catalog, {"default_request.sampling.num_frames": 81, "server.port": 9000})
    config = parse_download(result, tmp_path, monkeypatch)
    body = {"model": catalog["model"]["id"], "prompt": "A calm river"}
    args = SimpleNamespace(model_path=config.generator.model_path,
                           lora_path=None,
                           lora_nickname="default",
                           override_pipeline_cls_name=None)

    for request_body, expected_frames in [(body, 81), ({**body, "num_frames": 41}, 41)]:
        request = build_generation_request(
            "cookbook-config-test",
            VideoGenerationRequest(**request_body),
            args,
            output_dir=str(tmp_path),
            served_model_name=config.server.served_model_name or config.generator.model_path,
            default_request=config.default_request,
        )
        assert request.sampling.num_frames == expected_frames
