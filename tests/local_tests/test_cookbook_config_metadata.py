"""Check the standard-schema cookbook export without loading model weights."""

import json
import subprocess
import sys
from enum import Enum

import pytest
from pydantic import BaseModel, Field

from docs import cookbook_config


@pytest.fixture(scope="module")
def catalog():
    return cookbook_config.build_catalog()


def leaf(catalog, path):
    return cookbook_config.schema_at_path(catalog["schema"], path)


def test_checkpoint_registration_selects_its_own_preset(catalog):
    assert catalog["model"] == {
        "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "label": "Wan 2.2 TI2V 5B",
        "workload": "t2v",
        "preset": "wan_2_2_ti2v_5b",
    }
    frames = leaf(catalog, "default_request.sampling.num_frames")
    assert frames["default"] == 121
    assert frames["minimum"] == 1
    assert "multipleOf" not in frames  # Alignment is runtime behavior, not rejection.
    assert leaf(catalog, "default_request.sampling.height")["default"] == 704
    assert leaf(catalog, "default_request.sampling.width")["default"] == 1280
    assert leaf(catalog, "default_request.sampling.guidance_scale")["default"] == 5.0
    assert leaf(catalog, "default_request.sampling.seed")["default"] == 1024


def test_payload_is_standard_schema_without_parallel_metadata(catalog):
    assert set(catalog) == {"model", "runtime", "schema"}
    schema = catalog["schema"]
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    paths = set()

    def walk(node, prefix=""):
        assert not {"fields", "base_config", "schema_default", "default_source", "source", "min", "max", "step"} & node.keys()
        if node["type"] == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
            for name, child in node["properties"].items():
                walk(child, f"{prefix}.{name}" if prefix else name)
        else:
            paths.add(prefix)
            assert ("default" in node) != ("const" in node)

    walk(schema)
    assert paths == set(cookbook_config.SELECTED_PATHS) | {
        "generator.model_path", "default_request.output.return_frames"
    }
    assert leaf(catalog, "generator.model_path")["const"] == cookbook_config.MODEL_ID
    assert leaf(catalog, "default_request.output.return_frames")["const"] is False


def test_native_constraints_and_documented_transport_limits(catalog):
    gpu = leaf(catalog, "generator.engine.num_gpus")
    assert (gpu["type"], gpu["default"], gpu["minimum"]) == ("integer", 1, 1)
    assert leaf(catalog, "generator.engine.offload.vae")["type"] == "boolean"
    assert leaf(catalog, "generator.engine.offload.vae")["default"] is True
    host = leaf(catalog, "server.host")
    assert host["type"] == "string"
    assert host["pattern"] == "^[A-Za-z0-9_.:-]+$"
    assert leaf(catalog, "server.port")["maximum"] == 65535
    assert leaf(catalog, "default_request.sampling.num_inference_steps")["maximum"] == 200
    guidance = leaf(catalog, "default_request.sampling.guidance_scale")
    assert (guidance["type"], guidance["minimum"], guidance["maximum"]) == ("number", 0.0, 20.0)
    seed = leaf(catalog, "default_request.sampling.seed")
    assert (seed["minimum"], seed["maximum"]) == (-cookbook_config.JS_SAFE_INTEGER, cookbook_config.JS_SAFE_INTEGER)
    for name in ("height", "width"):
        dimension = leaf(catalog, f"default_request.sampling.{name}")
        assert dimension["minimum"] == dimension["multipleOf"] == 8
        assert dimension["default"] % 8 == 0


class Mode(str, Enum):
    FAST = "fast"
    QUALITY = "quality"


class NestedFixture(BaseModel):
    amount: float = Field(default=0.5, gt=0, lt=1, multiple_of=0.05)
    mode: Mode = Mode.FAST
    count: int | None = Field(default=None, ge=1, le=10)


class SchemaFixture(BaseModel):
    nested: NestedFixture


def test_selected_schema_extraction_preserves_refs_enum_and_exclusive_bounds():
    document = SchemaFixture.model_json_schema()
    amount = cookbook_config.schema_at_path(document, "nested.amount")
    assert amount["exclusiveMinimum"] == 0
    assert amount["exclusiveMaximum"] == 1
    assert amount["multipleOf"] == 0.05
    mode = cookbook_config.schema_at_path(document, "nested.mode")
    assert mode["enum"] == ["fast", "quality"]
    assert mode["default"] == "fast"
    assert "$ref" not in mode
    count = cookbook_config.schema_at_path(document, "nested.count", expected_type="integer")
    assert count["type"] == "integer"
    assert (count["minimum"], count["maximum"]) == (1, 10)
    assert "anyOf" not in count
    assert "default" not in count  # A nullable HTTP default must not erase a config default.


def test_checked_in_asset_matches_actual_python_metadata(catalog):
    assert json.loads(cookbook_config.OUTPUT_PATH.read_text()) == catalog


def test_export_does_not_read_example_or_cookbook_yaml(monkeypatch):
    original = cookbook_config.Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert path.suffix not in {".yaml", ".yml"}, f"Unexpected recipe dependency: {path}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(cookbook_config.Path, "read_text", guarded_read)
    assert cookbook_config.build_catalog()["model"]["id"] == cookbook_config.MODEL_ID


def test_module_import_works_without_site_packages_or_fastvideo_imports():
    script = """
import sys
from docs import cookbook_config
assert not any(name == 'fastvideo' or name.startswith('fastvideo.') or name == 'torch' for name in sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", script], cwd=cookbook_config.ROOT, check=True, capture_output=True, text=True)
