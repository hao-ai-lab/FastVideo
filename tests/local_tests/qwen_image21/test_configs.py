# SPDX-License-Identifier: Apache-2.0
"""Exercise the typed CLI parser without importing the GPU runtime."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(monkeypatch, name, path, **namespace):
    class StripRuntime(ast.NodeTransformer):

        def visit_ImportFrom(self, node):
            return ast.copy_location(ast.Pass(), node) if node.module and node.module.startswith("fastvideo") else node

    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    module.__dict__.update(namespace)
    source = ROOT / path
    tree = StripRuntime().visit(ast.parse(source.read_text()))
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    return module


@pytest.fixture
def cli(monkeypatch):
    schema = _load(monkeypatch, "qwen21_schema", "fastvideo/api/schema.py")
    error = _load(monkeypatch, "qwen21_errors", "fastvideo/api/errors.py")
    overrides = _load(monkeypatch, "qwen21_overrides", "fastvideo/api/overrides.py",
                      ConfigValidationError=error.ConfigValidationError)
    parser = _load(monkeypatch, "qwen21_parser", "fastvideo/api/parser.py",
                   ConfigValidationError=error.ConfigValidationError,
                   apply_overrides=overrides.apply_overrides, normalize_overrides=overrides.normalize_overrides,
                   GenerationRequest=schema.GenerationRequest, RunConfig=schema.RunConfig, ServeConfig=schema.ServeConfig,
                   bind_generation_request_raw=lambda config, _raw: config,
                   bind_run_config_raw=lambda config, _raw: config, bind_serve_config_raw=lambda config, _raw: config)
    return _load(monkeypatch, "qwen21_cli", "fastvideo/entrypoints/cli/inference_config.py",
                 apply_overrides=overrides.apply_overrides, parse_cli_overrides=overrides.parse_cli_overrides,
                 load_raw_config=parser.load_raw_config, parse_config=parser.parse_config,
                 RunConfig=schema.RunConfig, ServeConfig=schema.ServeConfig)


@pytest.mark.parametrize("file,workload,refs", [("qwen_image21_t2i.yaml", "t2i", None),
                                             ("qwen_image21_edit.yaml", "i2i", ["source.png", "mask.png"])])
def test_qwen_image21_cli_examples_parse(cli, file, workload, refs):
    config = cli.build_generate_run_config(SimpleNamespace(config=str(ROOT / "examples/inference/basic" / file)))
    assert config.generator.pipeline.preset is None
    assert config.generator.pipeline.workload_type == workload
    assert config.generator.engine.offload.lazy_module_load
    assert config.request.inputs.references == refs
    assert config.request.sampling.reference_resolution == 1024
    assert config.request.sampling.num_frames == 1
    assert config.request.sampling.use_kv_cache


def test_qwen_image21_cli_dotted_reference_and_sampling_overrides(cli):
    config = cli.build_generate_run_config(
        SimpleNamespace(config=str(ROOT / "examples/inference/basic/qwen_image21_edit.yaml")),
        ["--request.inputs.references", '["a.png", "b.png"]', "--request.sampling.use_kv_cache", "false",
         "--request.sampling.true_cfg_scale", "2", "--request.negative_prompt", ""])
    assert config.request.inputs.references == ["a.png", "b.png"]
    assert config.request.sampling.use_kv_cache is False
    assert config.request.sampling.true_cfg_scale == 2
    assert config.request.negative_prompt == ""


def test_clip_tokenizer_supported_dependency_pair():
    transformers = pytest.importorskip("transformers", reason="run in the installed GPU environment for tokenizer regression")
    tokenizer = transformers.CLIPTokenizer(vocab={"<|startoftext|>": 0, "<|endoftext|>": 1, "h": 2, "i</w>": 3}, merges=[])
    assert tokenizer("hi")["input_ids"] == [0, 2, 3, 1]
