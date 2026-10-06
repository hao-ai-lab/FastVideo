"""Export a curated Wan serving config as JSON Schema 2020-12.

Types, defaults, and request constraints come from FastVideo's Pydantic-generated
schemas; the registered model preset supplies sampling defaults. Explicit adapter
rules cover transport limits, positive GPU counts, and runtime dimension validity.

Run ``.venv/bin/python docs/cookbook_config.py`` to regenerate the tracked asset.
Export imports metadata without loading weights; importing this module does not
import FastVideo. Documentation builds consume the generated asset directly.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs/assets/cookbook-config.json"
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SELECTED_PATHS = (
    "generator.engine.num_gpus",
    "generator.engine.offload.vae",
    "server.host",
    "server.port",
    "default_request.sampling.num_frames",
    "default_request.sampling.height",
    "default_request.sampling.width",
    "default_request.sampling.fps",
    "default_request.sampling.num_inference_steps",
    "default_request.sampling.guidance_scale",
    "default_request.sampling.seed",
)
JS_SAFE_INTEGER = 2**53 - 1


def _dereference(node: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    """Resolve a selected local JSON pointer; this exporter never fetches URLs."""
    result = copy.deepcopy(node)
    seen = set()
    while "$ref" in result:
        reference = result.pop("$ref")
        if not reference.startswith("#/") or reference in seen:
            raise ValueError(f"Expected a noncyclic local schema reference: {reference}")
        seen.add(reference)
        target = document
        for part in reference[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        # Pydantic uses siblings for field annotations such as title/default.
        # Reject conflicting assertions instead of accidentally weakening one.
        for key in result.keys() & target.keys() - {"title", "description", "default", "examples"}:
            if result[key] != target[key]:
                raise ValueError(f"Conflicting {key} beside schema reference {reference}")
        result = {**copy.deepcopy(target), **result}
    return result


def _inline_references(value: Any, document: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _inline_references(item, document) for key, item in _dereference(value, document).items()}
    if isinstance(value, list):
        return [_inline_references(item, document) for item in value]
    return value


def schema_at_path(document: dict[str, Any], path: str, *, expected_type: str | None = None) -> dict[str, Any]:
    """Resolve a selected field, preserving assertions and optionally excluding null.

    HTTP request fields can be nullable even when their serve-config counterparts
    are not. Select the matching branch without carrying its null default.
    """
    node = document
    for part in path.split("."):
        node = _dereference(node, document)["properties"][part]
    result: dict[str, Any] = _inline_references(node, document)
    if expected_type is not None and "anyOf" in result:
        branches = [branch for branch in result["anyOf"] if branch.get("type") != "null"]
        if len(branches) != 1 or branches[0].get("type") != expected_type:
            raise ValueError(f"Expected one {expected_type} branch for {path}")
        result = {**{key: value for key, value in result.items() if key != "anyOf"}, **branches[0]}
        if result.get("default") is None:
            result.pop("default", None)
    if expected_type is not None and result.get("type") != expected_type:
        raise ValueError(f"Schema type changed for {path}: expected {expected_type}")
    return result


def _object_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _insert_field(schema: dict[str, Any], path: str, field: dict[str, Any]) -> None:
    """Build a closed, required object tree for the complete generated config."""
    owner = schema
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in owner["properties"]:
            owner["properties"][part] = _object_schema()
            owner["required"].append(part)
        owner = owner["properties"][part]
    owner["properties"][parts[-1]] = field
    owner["required"].append(parts[-1])


def build_catalog() -> dict[str, Any]:
    """Select model metadata and curate standard schemas without loading a model."""
    from pydantic import TypeAdapter

    from fastvideo.api.schema import ServeConfig
    from fastvideo.entrypoints.cli.inference_config import _validate_num_gpus
    from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
    from fastvideo.models.wan.definition import WAN_MODEL_DEFINITIONS
    from fastvideo.pipelines.basic.wan.presets import ALL_PRESETS

    definitions = [item for item in WAN_MODEL_DEFINITIONS if MODEL_ID in item.hf_model_paths]
    if len(definitions) != 1 or "t2v" not in definitions[0].workload_types:
        raise ValueError(f"Expected one T2V model registration for {MODEL_ID}")
    presets = [item for item in ALL_PRESETS if item.name == definitions[0].preset]
    if len(presets) != 1 or presets[0].workload_type != "t2v":
        raise ValueError(f"Expected one T2V preset named {definitions[0].preset}")
    preset = presets[0]
    serve_schema = TypeAdapter(ServeConfig).json_schema()
    request_schema = VideoGenerationRequest.model_json_schema()
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **_object_schema()}
    model = schema_at_path(serve_schema, "generator.model_path")
    model.update(const=MODEL_ID)
    model.pop("default", None)
    _insert_field(schema, "generator.model_path", model)

    for path in SELECTED_PATHS:
        field = schema_at_path(serve_schema, path)
        if field.get("type") not in {"integer", "number", "boolean", "string"} or "default" not in field:
            raise ValueError(f"Expected a primitive configuration field with a default: {path}")
        if path.startswith("default_request.sampling."):
            name = path.rsplit(".", 1)[1]
            request = schema_at_path(request_schema, name, expected_type=field["type"])
            for key, value in request.items():
                if key in {"default", "title", "description", "examples"}:
                    continue
                if key in field and field[key] != value:
                    raise ValueError(f"Reconcile differing configuration/request constraints: {path}.{key}")
                field[key] = value
            if name in preset.defaults:
                field["default"] = preset.defaults[name]
        # This bound preserves integer values through JavaScript and JSON/YAML
        # transport. It is intentionally narrower than the backend's int64 range.
        if field["type"] == "integer":
            field["minimum"] = max(field.get("minimum", -JS_SAFE_INTEGER), -JS_SAFE_INTEGER)
            field["maximum"] = min(field.get("maximum", JS_SAFE_INTEGER), JS_SAFE_INTEGER)
        if path == "generator.engine.num_gpus":
            _validate_num_gpus(1)
            try:
                _validate_num_gpus(0)
            except ValueError:
                field["minimum"] = 1
            else:
                raise ValueError("Revisit the cookbook GPU bound: the CLI now permits zero GPUs")
        elif path == "server.port":
            # Cookbook transport policy: no ephemeral port, predictable client URL.
            field.update(minimum=1, maximum=65535)
        elif path == "server.host":
            # Cookbook URL-host policy, not a ServerConfig dataclass constraint.
            field["pattern"] = r"^[A-Za-z0-9_.:-]+$"
        elif path in {"default_request.sampling.height", "default_request.sampling.width"}:
            # InputValidationStage.forward rejects dimensions not divisible by 8.
            field.update(minimum=8, multipleOf=8)
        elif path == "default_request.sampling.num_frames":
            field["description"] = "Requested frame count; the runtime may align it for the model's VAE."
        _insert_field(schema, path, field)

    output = schema_at_path(serve_schema, "default_request.output.return_frames")
    output.update(const=False)
    output.pop("default", None)
    _insert_field(schema, "default_request.output.return_frames", output)
    return {
        "model": {
            "id": MODEL_ID,
            "label": preset.description,
            "workload": "t2v",
            "preset": preset.name
        },
        "runtime": {
            "id": "cuda",
            "label": "CUDA",
            "launch_argv": ["fastvideo", "serve", "--config", "config.yaml"]
        },
        "schema": schema,
    }


def export_catalog(destination: Path = OUTPUT_PATH) -> dict[str, Any]:
    """Write the tracked browser asset after changes to the schema or its adapter."""
    catalog = build_catalog()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return catalog


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    export_catalog()
    print(OUTPUT_PATH)
