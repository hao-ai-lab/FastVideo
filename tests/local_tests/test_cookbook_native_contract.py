"""Optional native declaration checks in an existing FastVideo environment.

These tests import real configuration classes without loading models. They are
separate from the Node-only documentation build and compare paths/types, not
defaults, deployment bounds, hardware fit, or every runtime consumer.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import pytest

from fastvideo.api.schema import GenerationRequest, ServeConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.entrypoints.openai.mlx_server import MLXServeConfig

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIGS = {
    "fastvideo-cuda-rest": ServeConfig,
    "fastvideo-cuda-streaming": ServeConfig,
    "fastvideo-mlx-rest": MLXServeConfig,
}
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}


def annotation_at_path(config_type: type, path: str) -> Any:
    current = config_type
    for part in path.split("."):
        branches = get_args(current) if get_origin(current) in (UnionType, Union) else (current,)
        objects = [branch for branch in branches if branch is not type(None)]
        if len(objects) != 1 or not isinstance(objects[0], type):
            raise ValueError(f"{path}: cannot traverse native annotation {current}")
        fields = get_type_hints(objects[0])
        if part not in fields:
            raise ValueError(f"{path}: native {objects[0].__name__} has no field {part}")
        current = fields[part]
    return current


def native_annotation(runtime: str, path: str) -> Any:
    if runtime not in RUNTIME_CONFIGS:
        raise ValueError(f"{runtime}: add a native configuration mapping for this runtime")
    config_type = RUNTIME_CONFIGS[runtime]
    if runtime == "fastvideo-mlx-rest" and path.startswith("default_request."):
        # create_mlx_app normalizes this opaque mapping as GenerationRequest.
        assert annotation_at_path(config_type, "default_request") == dict[str, Any]
        return annotation_at_path(GenerationRequest, path.removeprefix("default_request."))
    if config_type is ServeConfig and path == "generator.pipeline.experimental.flow_shift":
        # The compatibility adapter forwards experimental keys to pipeline args.
        assert annotation_at_path(config_type, "generator.pipeline.experimental") == dict[str, Any]
        return annotation_at_path(PipelineConfig, "flow_shift")
    return annotation_at_path(config_type, path)


def native_types(annotation: Any) -> set[str]:
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        return set().union(*(native_types(branch) for branch in get_args(annotation)))
    if origin is Literal:
        return set().union(*(native_types(type(value)) for value in get_args(annotation)))
    if annotation in JSON_TYPES:
        return {JSON_TYPES[annotation]}
    if origin in (list, tuple):
        return {"array"}
    if origin is dict:
        return {"object"}
    raise ValueError(f"No checked JSON type mapping for native annotation {annotation}")


def schema_types(schema: dict) -> set[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return set(declared)
    branches = schema.get("anyOf", schema.get("oneOf"))
    if isinstance(branches, list) and branches:
        return set().union(*(schema_types(branch) for branch in branches))
    raise ValueError("Authored field schema must declare its types")


def check_field(schema: dict, annotation: Any, context: str) -> None:
    allowed = native_types(annotation)
    if "number" in allowed:
        allowed.add("integer")
    declared = schema_types(schema)
    if not declared <= allowed:
        raise ValueError(f"{context}: authored types {sorted(declared)} exceed native types {sorted(allowed)}")


def check_native_field(runtime: str, path: str, schema: dict, catalog_id: str | None = None) -> None:
    context = f"{catalog_id}: {runtime}: {path}" if catalog_id else f"{runtime}: {path}"
    try:
        annotation = native_annotation(runtime, path)
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error
    check_field(schema, annotation, context)


@pytest.fixture(scope="module")
def authored_contract():
    source = """
      import { readFileSync } from 'node:fs';
      import { parse } from 'yaml';
      import { buildCatalogs, mergeOptions } from './js/build-cookbook-config.mjs';
      const definitions = parse(readFileSync('./cookbook/options.yaml', 'utf8'), { merge: true });
      const runtimes = Object.fromEntries(Object.entries(definitions.runtimes).map(([id, runtime]) =>
        [id, mergeOptions(definitions.options, runtime.options)]));
      process.stdout.write(JSON.stringify({ runtimes, catalogs: buildCatalogs() }));
    """
    result = subprocess.run(["node", "--input-type=module"],
                            input=source,
                            text=True,
                            capture_output=True,
                            cwd=ROOT / "docs",
                            check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_all_authored_runtime_fields_match_native_declarations(authored_contract):
    for runtime, options in authored_contract["runtimes"].items():
        for path, schema in options.items():
            check_native_field(runtime, path, schema)


def test_all_discovered_deployment_controls_match_native_declarations(authored_contract):
    for catalog in authored_contract["catalogs"]:
        runtime = catalog["runtime"]["id"]
        for control in catalog["controls"]:
            check_native_field(runtime, control["path"], control["schema"], catalog["id"])


@dataclass
class SampleConfig:
    port: int = 8000
    name: str | None = None


def test_renamed_native_path_reports_field_context():
    with pytest.raises(ValueError, match="missing_port.*SampleConfig.*missing_port"):
        annotation_at_path(SampleConfig, "missing_port")


def test_native_type_drift_reports_runtime_and_field():
    with pytest.raises(ValueError, match="sample-runtime: server.port.*string.*integer"):
        check_field({"type": "string"}, annotation_at_path(SampleConfig, "port"), "sample-runtime: server.port")


def test_renamed_native_path_reports_runtime_and_catalog_context():
    with pytest.raises(ValueError, match="sample/rest: fastvideo-cuda-rest: server.missing_port.*has no field"):
        check_native_field("fastvideo-cuda-rest", "server.missing_port", {"type": "integer"}, "sample/rest")


@pytest.mark.parametrize("schema,annotation", [
    ({"type": "string"}, str | None),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, str | None),
    ({"type": "integer", "minimum": 1}, float),
    ({"type": "string", "enum": ["small"]}, Literal["small", "large"]),
])
def test_recipe_type_narrowing_is_allowed(schema, annotation):
    check_field(schema, annotation, "sample-runtime: field")


def test_recipe_cannot_widen_native_nullability_or_integer_type():
    for schema, annotation in [({"type": ["integer", "null"]}, int), ({"type": "number"}, int)]:
        with pytest.raises(ValueError, match="sample-runtime: field.*exceed native"):
            check_field(schema, annotation, "sample-runtime: field")


def test_opaque_native_fields_require_an_explicit_mapping():
    with pytest.raises(ValueError, match="cannot traverse native annotation"):
        native_annotation("fastvideo-cuda-rest", "generator.pipeline.experimental.new_option")
    with pytest.raises(ValueError, match="No checked JSON type mapping"):
        check_field({"type": "string"}, Any, "sample-runtime: field")
