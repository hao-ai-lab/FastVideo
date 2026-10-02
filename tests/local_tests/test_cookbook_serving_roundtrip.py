"""Exercise the actual browser serializer against the serving CLI parser, without inference."""

import base64
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docs.cookbook_serving import build_serving_example
from fastvideo.api.compat import explicit_request_updates
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.cli.serve import ServeSubcommand
from fastvideo.entrypoints.openai import image_api
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import build_generation_request
from fastvideo.entrypoints.openai.stores import AsyncDictStore
from fastvideo.utils import FlexibleArgumentParser

ROOT = Path(__file__).resolve().parents[2]


def resolve_example(recipe_id, selections=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the browser serializer")
    bundle = build_serving_example()
    # Exercise shell quoting as well as nested numeric/boolean options.
    recipe = next(item for item in bundle["recipes"] if item["id"] == recipe_id)
    recipe["settings"]["server"]["output_dir"] = "outputs/a user's run"
    script = """
const fs = require('node:fs');
const { resolveRecipe } = require('./docs/assets/cookbook-serving.js');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(resolveRecipe(input.bundle, input.recipeId, input.selections)));
"""
    result = json.loads(subprocess.run(
        [node, "-e", script],
        input=json.dumps({"bundle": bundle, "recipeId": recipe_id, "selections": selections or {}}),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    ).stdout)
    return result


def parse_launch(result):
    argv = shlex.split(result["command"].replace("\\\n", ""))
    assert argv == result["argv"]
    assert argv[:2] == ["fastvideo", "serve"]
    assert "--config" not in argv
    assert not any(arg.startswith("--default_request.") for arg in argv)

    parser = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    args, unknown = parser.parse_known_args(argv[1:])
    args._unknown = unknown
    ServeSubcommand().validate(args)
    return build_serve_config(args, unknown)


@pytest.mark.parametrize("values", [
    {},
    {"port": 9000, "vae_offload": False},
    {"host": "0.0.0.0"},
])
def test_generated_command_round_trips_through_real_serve_parser(values):
    result = resolve_example("fasth3-v2", {"values": values})
    config = parse_launch(result)

    assert config.generator.model_path == "FastVideo/FastVideo-FastH3-8-Step-V2"
    assert config.generator.engine.num_gpus == config.generator.engine.parallelism.sp_size == 4
    assert config.generator.engine.offload.vae is values.get("vae_offload", True)
    assert config.generator.pipeline.experimental["VSA_sparsity"] == 0.8
    assert config.server.port == values.get("port", 8000)
    assert config.server.host == values.get("host", "127.0.0.1")
    assert config.server.output_dir == "outputs/a user's run"
    assert explicit_request_updates(config.default_request) == {}
    assert "default_request" not in result["config"]
    endpoint = f"http://127.0.0.1:{config.server.port}"
    assert endpoint in result["healthCommand"]
    assert endpoint in result["clientCommand"]


@pytest.mark.parametrize("recipe_id, model, gpus, steps, workload", [
    ("zimage-turbo", "Tongyi-MAI/Z-Image-Turbo", 1, 8, "t2i"),
    ("wan21-i2v", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", 2, 40, "i2v"),
])
def test_additional_models_generate_valid_cli_and_request_payloads(recipe_id, model, gpus, steps, workload):
    result = resolve_example(recipe_id)
    config = parse_launch(result)
    assert config.generator.model_path == model
    assert config.generator.engine.num_gpus == config.generator.engine.parallelism.sp_size == gpus
    assert config.generator.pipeline.workload_type == workload
    assert explicit_request_updates(config.default_request) == {}
    assert result["clientRequest"]["body"]["num_inference_steps"] == steps
    assert "VSA_sparsity" not in config.generator.pipeline.experimental
    # The displayed shell command must carry the same JSON tested by the API checks below.
    argv = shlex.split(result["clientCommand"].replace("\\\n", ""))
    assert json.loads(argv[argv.index("--data") + 1]) == result["clientRequest"]["body"]


def test_image_client_matches_real_image_endpoint(monkeypatch, tmp_path):
    result = resolve_example("zimage-turbo")
    config = parse_launch(result)
    image_bytes = b"mock image output"
    generate = Mock(side_effect=lambda **kwargs: Path(kwargs["output_path"]).write_bytes(image_bytes))

    async def run_serialized(function, **kwargs):
        return function(**kwargs)

    engine = SimpleNamespace(generator=SimpleNamespace(generate_video=generate), run_serialized=run_serialized)
    monkeypatch.setattr(image_api, "get_serving_engine", lambda: engine)
    monkeypatch.setattr(image_api, "get_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(image_api, "get_server_args", lambda: SimpleNamespace(lora_path=None))
    monkeypatch.setattr(image_api, "get_served_model_name", lambda: config.server.served_model_name)
    monkeypatch.setattr(image_api, "IMAGE_STORE", AsyncDictStore())
    app = FastAPI()
    app.include_router(image_api.router)
    request = result["clientRequest"]
    with TestClient(app) as client:
        response = client.post(urlsplit(request["endpoint"]).path, json=request["body"])
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["data"][0]["b64_json"]) == image_bytes
    kwargs = generate.call_args.kwargs
    assert kwargs["seed"] == 42
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 0
    assert kwargs["height"] == kwargs["width"] == 1024
    assert kwargs["num_frames"] == 1
    assert kwargs["output_path"].endswith(".png")


def test_adjusted_h3_options_pass_the_real_request_adapter(tmp_path):
    result = resolve_example("fasth3-v2", {"values": {
        "num_gpus": 2,
        "vsa_sparsity": 0.5,
    }})
    config = parse_launch(result)
    assert config.generator.engine.num_gpus == config.generator.engine.parallelism.sp_size == 2
    assert config.generator.pipeline.experimental["VSA_sparsity"] == 0.5
    args = SimpleNamespace(model_path=config.generator.model_path, lora_path=None, override_pipeline_cls_name=None)
    request = VideoGenerationRequest.model_validate(result["clientRequest"]["body"])
    resolved = build_generation_request(
        "cookbook-h3-test", request, args,
        served_model_name=config.server.served_model_name,
        output_dir=str(tmp_path),
        default_request=config.default_request,
    )
    assert explicit_request_updates(config.default_request) == {}
    assert resolved.sampling.num_frames == 124
    assert resolved.sampling.num_inference_steps == 9


def test_i2v_client_matches_video_adapter(tmp_path):
    image_url = "https://example.com/first-frame.png"
    result = resolve_example("wan21-i2v")
    config = parse_launch(result)
    # Simulate replacing the placeholder in the copied static test request.
    body = dict(result["clientRequest"]["body"])
    body["input_reference"] = image_url
    request = VideoGenerationRequest.model_validate(body)
    assert request.task is None
    args = SimpleNamespace(model_path=config.generator.model_path, lora_path=None, override_pipeline_cls_name=None)
    resolved = build_generation_request(
        "cookbook-test", request, args,
        served_model_name=config.server.served_model_name,
        output_dir=str(tmp_path),
        default_request=config.default_request,
    )
    assert resolved.inputs.image_path == image_url
    assert resolved.sampling.num_frames == 77
    assert resolved.sampling.num_inference_steps == 40
    assert resolved.sampling.guidance_scale == 5
    assert (resolved.sampling.width, resolved.sampling.height, resolved.sampling.fps) == (832, 480, 16)
    assert config.generator.pipeline.experimental["flow_shift"] == 3
