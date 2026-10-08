"""Serving schema fidelity and registry-driven metadata export; no model weights."""

import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
from enum import Enum
from unittest.mock import patch

import pytest
import yaml
from pydantic import BaseModel, Field, TypeAdapter

from docs import cookbook_config
from fastvideo.api.schema import ServeConfig

WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
IMAGE = "Tongyi-MAI/Z-Image-Turbo"


def leaf(schema, path):
    for key in path.split("."):
        schema = schema["properties"][key]
    return schema


@pytest.fixture(scope="module")
def catalog():
    return cookbook_config.build_catalog(WAN)


def selection_file(tmp_path, model_ids):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": model_ids}), encoding="utf-8")
    return path


@pytest.mark.parametrize("model_id,frames,width", [(WAN, 121, 1280), (IMAGE, 1, 1024)])
def test_model_id_selects_registered_defaults(model_id, frames, width):
    catalog = cookbook_config.build_catalog(model_id)
    schema = catalog["schema"]
    assert catalog["model"]["id"] == model_id
    assert leaf(schema, "generator.model_path")["const"] == model_id
    assert leaf(schema, "default_request.sampling.num_frames")["default"] == frames
    assert leaf(schema, "default_request.sampling.width")["default"] == width


def test_catalog_keeps_every_serving_property_type_and_null(catalog):
    native = TypeAdapter(ServeConfig).json_schema()
    expanded = cookbook_config._inline_references(native, native)
    actual = catalog["schema"]

    def compare(expected, exported):
        if not isinstance(expected, dict):
            return
        # Model presets and pipeline declarations may override defaults, while
        # the serving field types and nullable branches remain intact.
        for key in ("type", "required", "enum", "const"):
            if key in expected:
                assert exported[key] == expected[key]
        if "anyOf" in expected:
            assert len(expected["anyOf"]) == len(exported["anyOf"])
            for left, right in zip(expected["anyOf"], exported["anyOf"]):
                compare(left, right)
        if "properties" in expected:
            assert exported["properties"].keys() == expected["properties"].keys()
            for key, value in expected["properties"].items():
                compare(value, exported["properties"][key])

    compare(expanded, actual)
    assert leaf(actual, "generator.engine.offload.lazy_module_load")["default"] is None
    assert leaf(actual, "generator.pipeline.vae_tiling")["anyOf"] == [{"type": "boolean"}, {"type": "null"}]
    assert leaf(actual, "server.served_model_name")["default"] is None
    assert leaf(actual, "default_request.output.return_frames")["default"] is True
    assert "requests" not in catalog


def test_http_bounds_are_not_serving_constraints(catalog):
    schema = catalog["schema"]
    guidance = leaf(schema, "default_request.sampling.guidance_scale")
    assert "minimum" not in guidance and "maximum" not in guidance
    steps = leaf(schema, "default_request.sampling.num_inference_steps")
    assert steps["maximum"] == cookbook_config.JS_SAFE_INTEGER
    frames = leaf(schema, "default_request.sampling.num_frames")
    assert frames["minimum"] == -cookbook_config.JS_SAFE_INTEGER
    assert "multipleOf" not in frames
    assert "pattern" not in leaf(schema, "server.host")


def test_pipeline_options_come_from_public_declarations():
    schema = cookbook_config.build_catalog(WAN)["schema"]
    assert leaf(schema, "generator.pipeline.vae_tiling")["default"] is False
    options = leaf(schema, "generator.pipeline.experimental")["properties"]
    assert options["flow_shift"]["default"] == 5.0
    assert {part["type"] for part in options["flow_shift"]["anyOf"]} == {"number", "null"}
    assert options["dit_precision"]["enum"] == ["fp32", "fp16", "bf16"]
    assert options["text_encoder_precisions"]["default"] == ["fp32"]
    assert "vae_config.load_encoder" in options
    assert "postprocess_text_funcs" not in options
    assert "precision" not in options  # Not a declared public pipeline argument.


def test_selection_file_keeps_requested_order(tmp_path):
    path = selection_file(tmp_path, [IMAGE, WAN])
    assert cookbook_config.load_model_ids(path) == [IMAGE, WAN]


def test_default_selection_is_explicit_and_registered():
    from fastvideo.registry import get_registered_models_with_workloads

    selected = cookbook_config.load_model_ids()
    registered = {row["id"] for row in get_registered_models_with_workloads() if row["workload_types"]}
    assert selected
    assert set(selected) <= registered
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize("source", [
    "",
    "[]",
    "{}",
    "models: []",
    "models: one-model",
    "models: [null]",
    "models: [42]",
    'models: [""]',
    'models: ["   "]',
    "models: [Test/Model, Test/Model]",
])
def test_selection_file_rejects_invalid_structure_and_ids(tmp_path, source):
    path = tmp_path / "invalid.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        cookbook_config.load_model_ids(path)


def test_export_contains_only_selected_complete_catalogs_in_order(tmp_path):
    path = selection_file(tmp_path, [IMAGE, WAN])
    output = tmp_path / "assets"
    index = cookbook_config.export_catalogs(output_dir=output, models_file=path)

    assert set(index) == {"models"}
    assert [row["id"] for row in index["models"]] == [IMAGE, WAN]
    assert json.loads((output / "index.json").read_text()) == index
    expected_files = {"index.json"}
    for row, frames in zip(index["models"], [1, 121]):
        assert set(row) == {"id", "label", "workload_types", "catalog_url"}
        assert row["label"] and row["workload_types"]
        digest = hashlib.sha256(row["id"].encode("utf-8")).hexdigest()
        assert row["catalog_url"] == f"models/{digest}.json"
        expected_files.add(row["catalog_url"])
        catalog = json.loads((output / row["catalog_url"]).read_text())
        assert catalog == cookbook_config.build_catalog(row["id"])
        assert set(catalog) == {"model", "runtime", "schema"}
        assert leaf(catalog["schema"], "default_request.sampling.num_frames")["default"] == frames
    assert {str(path.relative_to(output)) for path in output.rglob("*.json")} == expected_files


def test_selected_catalogs_share_discovery_and_pipeline_schema(monkeypatch):
    import fastvideo.registry as registry

    aliases = ["FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", "FastVideo/FastWan2.2-TI2V-5B-Diffusers"]
    assert registry.get_pipeline_config_cls_from_name(aliases[0]) is registry.get_pipeline_config_cls_from_name(aliases[1])
    original_schema = TypeAdapter.json_schema
    serving_schemas = []

    def record_schema(adapter, *args, **kwargs):
        if adapter._type is ServeConfig:
            serving_schemas.append(adapter)
        return original_schema(adapter, *args, **kwargs)

    monkeypatch.setattr(TypeAdapter, "json_schema", record_schema)
    with patch.object(registry, "get_registered_models_with_workloads",
                      wraps=registry.get_registered_models_with_workloads) as discover:
        with patch.object(cookbook_config, "_pipeline_overlay", wraps=cookbook_config._pipeline_overlay) as pipeline:
            catalogs = cookbook_config.build_catalogs(aliases)

    assert [catalog["model"]["id"] for catalog in catalogs] == aliases
    assert discover.call_count == 1
    assert len(serving_schemas) == 1
    assert pipeline.call_count == 1


def test_export_removes_only_stale_owned_model_files(tmp_path):
    path = selection_file(tmp_path, [WAN, IMAGE])
    output = tmp_path / "assets"
    original = cookbook_config.export_catalogs(output_dir=output, models_file=path)
    stale = output / original["models"][1]["catalog_url"]
    unrelated = [output / "notes.json", output / "models" / "custom.json", output / "models" / "notes.txt",
                 output / "models" / "nested" / stale.name]
    for file in unrelated:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("keep me", encoding="utf-8")

    selection_file(tmp_path, [WAN])
    current = cookbook_config.export_catalogs(output_dir=output, models_file=path)
    assert [row["id"] for row in current["models"]] == [WAN]
    assert not stale.exists()
    assert all(file.read_text() == "keep me" for file in unrelated)


@pytest.mark.parametrize("model_ids", [[WAN, WAN], [WAN, "Unknown/Model"]])
def test_bad_selection_preserves_existing_assets(tmp_path, model_ids):
    path = selection_file(tmp_path, [WAN, IMAGE])
    output = tmp_path / "assets"
    cookbook_config.export_catalogs(output_dir=output, models_file=path)
    before = {file.relative_to(output): file.read_bytes() for file in output.rglob("*") if file.is_file()}
    selection_file(tmp_path, model_ids)

    with pytest.raises(ValueError):
        cookbook_config.export_catalogs(output_dir=output, models_file=path)
    assert {file.relative_to(output): file.read_bytes() for file in output.rglob("*") if file.is_file()} == before


def test_failed_catalog_composition_preserves_existing_assets(tmp_path, monkeypatch):
    path = selection_file(tmp_path, [WAN, IMAGE])
    output = tmp_path / "assets"
    cookbook_config.export_catalogs(output_dir=output, models_file=path)
    before = {file.relative_to(output): file.read_bytes() for file in output.rglob("*") if file.is_file()}
    original = cookbook_config._model_overlay

    def fail_second_model(model, shared):
        if model["id"] == IMAGE:
            raise ValueError("Unexportable model metadata")
        return original(model, shared)

    monkeypatch.setattr(cookbook_config, "_model_overlay", fail_second_model)
    with pytest.raises(ValueError, match="Unexportable model metadata"):
        cookbook_config.export_catalogs(output_dir=output, models_file=path)
    assert {file.relative_to(output): file.read_bytes() for file in output.rglob("*") if file.is_file()} == before


def test_default_output_is_gitignored():
    paths = [cookbook_config.OUTPUT_DIR / "index.json", cookbook_config.OUTPUT_DIR / "models" / "example.json"]
    completed = subprocess.run(["git", "check-ignore", "--no-index", *map(str, paths)],
                               cwd=cookbook_config.ROOT, check=True, capture_output=True, text=True)
    assert set(completed.stdout.splitlines()) == {str(path) for path in paths}


def test_new_registration_and_option_need_no_exporter_allowlist(monkeypatch, tmp_path):
    import fastvideo.registry as registry
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.fastvideo_args import WorkloadType

    @dataclasses.dataclass
    class NewPipeline(PipelineConfig):
        custom_gain: float = dataclasses.field(default=0.5, metadata={"gt": 0, "lt": 1})
        custom_mode: str | None = "fast"
        custom_modes: list[str] | None = None

        @staticmethod
        def add_cli_args(parser, prefix=""):
            PipelineConfig.add_cli_args(parser, prefix)
            parser.add_argument("--custom-gain", type=float)
            parser.add_argument("--custom-mode", choices=["fast", "quality"])
            parser.add_argument("--custom-modes", choices=["fast", "quality"], nargs="+")
            return parser

    monkeypatch.setattr(registry, "_CONFIG_REGISTRY", dict(registry._CONFIG_REGISTRY))
    monkeypatch.setattr(registry, "_MODEL_HF_PATH_TO_NAME", dict(registry._MODEL_HF_PATH_TO_NAME))
    aliases = ["Test/New-Registered-Model", "Test/New/Registered-Model", "Test_New/Registered-Model"]
    registry.register_configs(None, NewPipeline, (WorkloadType.T2V,), hf_model_paths=aliases)
    catalog = cookbook_config.build_catalog(aliases[0])
    gain = leaf(catalog["schema"], "generator.pipeline.experimental.custom_gain")
    assert gain["default"] == 0.5
    assert (gain["exclusiveMinimum"], gain["exclusiveMaximum"]) == (0, 1)
    mode = leaf(catalog["schema"], "generator.pipeline.experimental.custom_mode")
    assert mode["default"] == "fast"
    assert mode["anyOf"] == [{"type": "string", "enum": ["fast", "quality"]}, {"type": "null"}]
    modes = leaf(catalog["schema"], "generator.pipeline.experimental.custom_modes")
    assert modes["default"] is None
    assert modes["anyOf"][0]["items"]["enum"] == ["fast", "quality"]
    assert modes["anyOf"][1] == {"type": "null"}

    path = selection_file(tmp_path, aliases)
    output = tmp_path / "assets"
    index = cookbook_config.export_catalogs(output_dir=output, models_file=path)
    urls = {row["id"]: row["catalog_url"] for row in index["models"]}
    assert len(set(urls.values())) == len(aliases)
    for model_id, url in urls.items():
        assert url == f"models/{hashlib.sha256(model_id.encode('utf-8')).hexdigest()}.json"
        exported = json.loads((output / url).read_text())
        assert exported["model"]["id"] == model_id
        assert leaf(exported["schema"], "generator.pipeline.experimental.custom_gain") == gain
    contents = {url: (output / url).read_bytes() for url in urls.values()}

    selection_file(tmp_path, list(reversed(aliases)))
    reordered = cookbook_config.export_catalogs(output_dir=output, models_file=path)
    assert [row["id"] for row in reordered["models"]] == list(reversed(aliases))
    assert {row["id"]: row["catalog_url"] for row in reordered["models"]} == urls
    assert {url: (output / url).read_bytes() for url in urls.values()} == contents


class Mode(str, Enum):
    FAST = "fast"
    QUALITY = "quality"


class Fixture(BaseModel):
    mode: Mode = Mode.FAST
    amount: float | None = Field(default=None, gt=0, lt=1, multiple_of=0.05)


def test_reference_expansion_preserves_annotations_nulls_and_input():
    document = Fixture.model_json_schema()
    document["properties"]["mode"]["title"] = "Mode override"
    before = copy.deepcopy(document)
    expanded = cookbook_config._inline_references(document, document)
    assert document == before
    mode = expanded["properties"]["mode"]
    assert mode["enum"] == ["fast", "quality"]
    assert mode["title"] == "Mode override" and mode["default"] == "fast"
    amount = expanded["properties"]["amount"]
    assert amount["default"] is None
    assert amount["anyOf"][1] == {"type": "null"}
    assert amount["anyOf"][0]["exclusiveMaximum"] == 1


@pytest.mark.parametrize("document", [
    {"$ref": "https://example.com/schema"},
    {"$defs": {"Loop": {"$ref": "#/$defs/Loop"}}, "$ref": "#/$defs/Loop"},
    {
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
        "$ref": "#/$defs/Node",
    },
])
def test_remote_and_cyclic_expansion_rejected(document):
    with pytest.raises(ValueError):
        cookbook_config._inline_references(document, document)


def test_prepare_does_not_rewrite_default_objects_or_open_maps():
    payload = {"type": "integer", "$ref": "this is literal data"}
    source = {"type": "object", "properties": {
        "map": {"type": "object", "additionalProperties": True, "default": payload},
        "count": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
    }}
    before = copy.deepcopy(source)
    expanded = cookbook_config._inline_references(source, source)
    actual = cookbook_config._prepare_schema(expanded)
    assert actual["additionalProperties"] is False
    assert actual["properties"]["map"]["additionalProperties"] is True
    assert actual["properties"]["map"]["default"] == payload
    assert actual["properties"]["count"]["default"] is None
    assert source == before


def test_unknown_model_fails_without_network_lookup(monkeypatch):
    import fastvideo.registry as registry

    def unexpected(*args, **kwargs):
        pytest.fail("Unknown IDs must not trigger checkpoint lookup")

    monkeypatch.setattr(registry, "get_pipeline_config_cls_from_name", unexpected)
    with pytest.raises(ValueError, match="no registered serving workload"):
        cookbook_config.build_catalog("Unknown/Model")


@pytest.mark.parametrize("model_ids", [[WAN, WAN], [WAN, "Unknown/Model"]])
def test_batch_rejects_invalid_ids_before_pipeline_or_network_lookup(monkeypatch, model_ids):
    import fastvideo.registry as registry

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid selections must fail before pipeline or checkpoint lookup")

    monkeypatch.setattr(registry, "get_pipeline_config_cls_from_name", unexpected)
    monkeypatch.setattr(registry, "maybe_download_model_index", unexpected)
    with pytest.raises(ValueError):
        cookbook_config.build_catalogs(model_ids)


def test_export_does_not_use_http_schema_or_load_model_index(monkeypatch):
    import fastvideo.registry as registry
    from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest, ImageGenerationsRequest

    def unexpected(*args, **kwargs):
        pytest.fail("Export must stay on configuration metadata")

    monkeypatch.setattr(VideoGenerationRequest, "model_json_schema", unexpected)
    monkeypatch.setattr(ImageGenerationsRequest, "model_json_schema", unexpected)
    monkeypatch.setattr(registry, "maybe_download_model_index", unexpected)
    assert cookbook_config.build_catalog(WAN)["model"]["id"] == WAN


def test_module_import_needs_no_runtime_dependencies():
    script = (
        "import sys; from docs import cookbook_config; "
        "assert 'fastvideo' not in sys.modules and 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-S", "-c", script],
                   cwd=cookbook_config.ROOT, check=True, capture_output=True, text=True)


def test_cli_exports_selection_to_requested_directory(tmp_path):
    path = selection_file(tmp_path, [IMAGE])
    output = tmp_path / "generated assets"
    subprocess.run([sys.executable, str(cookbook_config.ROOT / "docs/cookbook_config.py"),
                    "--models-file", str(path), "--output-dir", str(output)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    index = json.loads((output / "index.json").read_text())
    assert [row["id"] for row in index["models"]] == [IMAGE]
    catalog = json.loads((output / index["models"][0]["catalog_url"]).read_text())
    assert catalog == cookbook_config.build_catalog(IMAGE)


@pytest.mark.parametrize("failure", [None, "missing", "wrong-id", "missing-schema", "wrong-url", "wrong-order"])
def test_built_site_check_detects_missing_or_mismatched_catalogs(tmp_path, failure):
    path = selection_file(tmp_path, [WAN, IMAGE])
    site = tmp_path / "site"
    output = site / "assets/cookbook-config"
    index = cookbook_config.export_catalogs(output_dir=output, models_file=path)
    catalog_path = output / index["models"][0]["catalog_url"]
    if failure == "missing":
        catalog_path.unlink()
    elif failure in {"wrong-id", "missing-schema"}:
        catalog = json.loads(catalog_path.read_text())
        if failure == "wrong-id":
            catalog["model"]["id"] = IMAGE
        else:
            del catalog["schema"]
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif failure in {"wrong-url", "wrong-order"}:
        if failure == "wrong-url":
            index["models"][0]["catalog_url"] = "../unrelated.json"
        else:
            index["models"].reverse()
        (output / "index.json").write_text(json.dumps(index), encoding="utf-8")

    if failure is None:
        cookbook_config.check_site(site, models_file=path)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            cookbook_config.check_site(site, models_file=path)
