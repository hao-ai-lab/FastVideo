from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.entrypoints.cli.generate import GenerateSubcommand
from fastvideo.entrypoints.cli.inference_config import (
    build_generate_run_config,
    build_serve_config,
)
from fastvideo.entrypoints.cli.serve import ServeSubcommand
from fastvideo.entrypoints.openai import api_server
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.utils import FlexibleArgumentParser


def _parse_generate_args(argv: list[str]):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    GenerateSubcommand().subparser_init(subparsers)
    args, unknown = parser.parse_known_args(["generate", *argv])
    args._unknown = unknown
    return args, unknown


def _parse_serve_args(argv: list[str]):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    ServeSubcommand().subparser_init(subparsers)
    args, unknown = parser.parse_known_args(["serve", *argv])
    args._unknown = unknown
    return args, unknown


def test_generate_parser_preserves_unknown_dotted_overrides(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "request:\n"
        "  prompt: hello\n",
        encoding="utf-8",
    )

    args, unknown = _parse_generate_args([
        "--config",
        str(config_path),
        "--request.sampling.seed",
        "42",
    ])

    assert args.config == str(config_path)
    assert unknown == ["--request.sampling.seed", "42"]


def test_build_generate_run_config_loads_nested_config_and_cli_overrides(
    tmp_path,
):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "  engine:\n"
        "    num_gpus: 1\n"
        "request:\n"
        "  prompt: hello\n"
        "  output:\n"
        "    return_frames: true\n",
        encoding="utf-8",
    )

    args, unknown = _parse_generate_args([
        "--config",
        str(config_path),
        "--num-gpus",
        "2",
        "--request.sampling.seed",
        "7",
    ])

    config = build_generate_run_config(args, unknown)

    assert config.generator.model_path == "test-model"
    assert config.generator.engine.num_gpus == 2
    assert config.request.prompt == "hello"
    assert config.request.sampling.seed == 7
    assert config.request.output.return_frames is True


def test_build_generate_run_config_loads_nested_json_config(tmp_path):
    config_path = tmp_path / "run.json"
    config_path.write_text(
        '{"generator":{"model_path":"json-model"},'
        '"request":{"prompt":"hello"}}',
        encoding="utf-8",
    )

    args, unknown = _parse_generate_args(["--config", str(config_path)])
    config = build_generate_run_config(args, unknown)

    assert config.generator.model_path == "json-model"
    assert config.request.prompt == "hello"
    assert config.request.output.return_frames is False


def test_build_generate_run_config_translates_flat_legacy_config(tmp_path):
    config_path = tmp_path / "run-flat.yaml"
    config_path.write_text(
        "model_path: flat-model\n"
        "workload_type: i2v\n"
        "prompt: hello\n"
        "num_frames: 81\n"
        "output_path: out/\n",
        encoding="utf-8",
    )

    args, unknown = _parse_generate_args(["--config", str(config_path)])
    config = build_generate_run_config(args, unknown)

    assert config.generator.model_path == "flat-model"
    assert config.generator.pipeline.workload_type == "i2v"
    assert config.request.prompt == "hello"
    assert config.request.sampling.num_frames == 81
    assert config.request.output.output_path == "out/"
    assert config.request.output.return_frames is False


def test_build_generate_run_config_requires_single_prompt_source(tmp_path):
    missing_prompt_path = tmp_path / "missing.yaml"
    missing_prompt_path.write_text(
        "generator:\n"
        "  model_path: test-model\n",
        encoding="utf-8",
    )
    args, unknown = _parse_generate_args(["--config", str(missing_prompt_path)])
    with pytest.raises(
        ValueError,
        match="Either request.prompt or request.inputs.prompt_path must be provided",
    ):
        build_generate_run_config(args, unknown)

    conflicting_prompt_path = tmp_path / "conflict.yaml"
    conflicting_prompt_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "request:\n"
        "  prompt: hello\n"
        "  inputs:\n"
        "    prompt_path: prompts.txt\n",
        encoding="utf-8",
    )
    args, unknown = _parse_generate_args(["--config", str(conflicting_prompt_path)])
    with pytest.raises(
        ValueError,
        match="Cannot provide both request.prompt and request.inputs.prompt_path",
    ):
        build_generate_run_config(args, unknown)


def test_build_serve_config_translates_flat_legacy_config(tmp_path):
    config_path = tmp_path / "serve-flat.yaml"
    config_path.write_text(
        "model_path: serve-model\n"
        "num_gpus: 4\n"
        "workload_type: i2v\n"
        "host: 127.0.0.1\n"
        "port: 9001\n"
        "output_dir: served/\n",
        encoding="utf-8",
    )

    args, unknown = _parse_serve_args(["--config", str(config_path)])
    config = build_serve_config(args, unknown)

    assert config.generator.model_path == "serve-model"
    assert config.generator.engine.num_gpus == 4
    assert config.generator.pipeline.workload_type == "i2v"
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 9001
    assert config.server.output_dir == "served/"
    assert "host" not in config.generator.pipeline.experimental
    assert "port" not in config.generator.pipeline.experimental
    assert "output_dir" not in config.generator.pipeline.experimental


def test_build_serve_config_loads_nested_config_and_overrides(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: serve-model\n"
        "server:\n"
        "  host: 0.0.0.0\n"
        "  port: 8000\n",
        encoding="utf-8",
    )

    args, unknown = _parse_serve_args([
        "--config",
        str(config_path),
        "--num-gpus",
        "3",
        "--server.port",
        "9100",
    ])

    config = build_serve_config(args, unknown)

    assert config.generator.model_path == "serve-model"
    assert config.generator.engine.num_gpus == 3
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9100


def test_serve_subcommand_requires_config():
    args, _ = _parse_serve_args([
        "--model-path",
        "serve-model",
    ])

    with pytest.raises(
        ValueError,
        match="fastvideo serve requires --config PATH",
    ):
        ServeSubcommand().validate(args)


def test_generate_subcommand_dispatches_via_typed_config(monkeypatch):
    args, _ = _parse_generate_args([
        "--model-path",
        "test-model",
        "--prompt",
        "hello world",
        "--num-frames",
        "81",
    ])
    captured: dict[str, object] = {}

    class FakeGenerator:

        def generate(self, request):
            captured["request"] = request
            return None

    def fake_from_config(cls, config):
        captured["config"] = config
        return FakeGenerator()

    monkeypatch.setattr(
        VideoGenerator,
        "from_config",
        classmethod(fake_from_config),
    )

    GenerateSubcommand().cmd(args)

    request = captured["request"]
    assert captured["config"].model_path == "test-model"
    assert request.prompt == "hello world"
    assert request.sampling.num_frames == 81
    assert request.output.return_frames is False


def test_serve_subcommand_dispatches_via_typed_config(tmp_path, monkeypatch):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: serve-model\n",
        encoding="utf-8",
    )
    args, _ = _parse_serve_args([
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--output-dir",
        "serve-outputs/",
        "--num-gpus",
        "2",
    ])
    captured: dict[str, object] = {}

    def fake_generator_config_to_fastvideo_args(config):
        captured["config"] = config
        return SimpleNamespace(model_path=config.model_path)

    def fake_run_server(fastvideo_args, host, port, output_dir):
        captured["fastvideo_args"] = fastvideo_args
        captured["host"] = host
        captured["port"] = port
        captured["output_dir"] = output_dir

    monkeypatch.setattr(
        "fastvideo.entrypoints.cli.serve.generator_config_to_fastvideo_args",
        fake_generator_config_to_fastvideo_args,
    )
    monkeypatch.setattr(api_server, "run_server", fake_run_server)

    ServeSubcommand().cmd(args)

    assert captured["config"].model_path == "serve-model"
    assert captured["config"].engine.num_gpus == 2
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000
    assert captured["output_dir"] == "serve-outputs/"


def test_serve_subcommand_rejects_non_default_default_request(tmp_path):
    config_path = tmp_path / "serve-default-request.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: serve-model\n"
        "default_request:\n"
        "  prompt: hello\n",
        encoding="utf-8",
    )
    args, _ = _parse_serve_args(["--config", str(config_path)])

    with pytest.raises(
        NotImplementedError,
        match="ServeConfig.default_request is not wired",
    ):
        ServeSubcommand().cmd(args)
