"""Export serving schemas and registered model defaults without loading weights.

build_catalog(model_id) describes one registered model. The command-line export
reads the cookbook's selected model IDs and writes an index plus a complete
catalog for each model. Shared schemas and pipeline declarations are discovered
once per export; the browser only fetches the selected model's catalog.

FastVideo imports are delayed until export: importing docs tooling should not
initialize the inference package and its PyTorch/backend dependencies.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

if TYPE_CHECKING:
    from fastvideo.configs.pipelines.base import PipelineConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL_SELECTION_PATH = ROOT / "docs/cookbook/config-builder-models.yaml"
OUTPUT_DIR = ROOT / "docs/assets/cookbook-config"
JS_SAFE_INTEGER = 2**53 - 1
SCHEMA_DATA = {"default", "examples", "enum", "const"}


def _dereference(node: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
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
        for key in result.keys() & target.keys() - {"title", "description", "default", "examples"}:
            if result[key] != target[key]:
                raise ValueError(f"Conflicting {key} beside schema reference {reference}")
        result = {**copy.deepcopy(target), **result}
    return result


def _inline_references(value: Any, document: dict[str, Any], trail: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference in trail:
            raise ValueError(f"Recursive schema cannot be expanded: {reference}")
        if reference:
            trail = (*trail, reference)
        result = {}
        for key, item in _dereference(value, document).items():
            if key == "$defs":
                continue
            if key in SCHEMA_DATA:
                result[key] = copy.deepcopy(item)
            elif key == "properties":
                result[key] = {name: _inline_references(child, document, trail) for name, child in item.items()}
            else:
                result[key] = _inline_references(item, document, trail)
        return result
    if isinstance(value, list):
        return [_inline_references(item, document, trail) for item in value]
    return value


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _put(schema: dict[str, Any], path: tuple[str, ...], value: dict[str, Any]) -> None:
    owner = schema
    for part in path[:-1]:
        owner = owner.setdefault("properties", {}).setdefault(part, {"type": "object"})
    owner.setdefault("properties", {})[path[-1]] = value


def _leaves(schema: dict[str, Any], prefix: tuple[str, ...] = ()):
    for name, field in schema.get("properties", {}).items():
        path = (*prefix, name)
        if field.get("properties"):
            yield from _leaves(field, path)
        else:
            yield path, field


def _prepare_schema(value: Any) -> Any:
    """Match the native parser's closed dataclasses and JS's integer transport."""
    if isinstance(value, list):
        return [_prepare_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in SCHEMA_DATA:
            result[key] = copy.deepcopy(item)
        elif key == "properties":
            result[key] = {name: _prepare_schema(child) for name, child in item.items()}
        else:
            result[key] = _prepare_schema(item)
    if "properties" in result and result.get("type") == "object":
        result.setdefault("additionalProperties", False)
    if result.get("type") == "integer":
        result["minimum"] = max(result.get("minimum", -JS_SAFE_INTEGER), -JS_SAFE_INTEGER)
        result["maximum"] = min(result.get("maximum", JS_SAFE_INTEGER), JS_SAFE_INTEGER)
    return result


def _field_annotation(owner: type, name: str) -> Any:
    # Resolve only this field; unrelated forward references in model internals
    # need not be importable merely to describe a public scalar option.
    for base in owner.__mro__:
        if name in base.__dict__.get("__annotations__", {}):
            carrier = type("_Field", (), {"__annotations__": {name: base.__annotations__[name]}})
            return get_type_hints(carrier, globalns=vars(sys.modules[base.__module__]), localns=dict(vars(base)))[name]
    raise KeyError(name)


def _pipeline_overlay(config_type: type[PipelineConfig], shared: dict[str, Any]) -> dict[str, Any]:
    """Discover legacy options from public argument declarations and model defaults.

    This constructs configuration data, never a model or execution pipeline.
    Python-only objects are not made into JSON controls. Typed public paths win;
    remaining legacy arguments use the runtime's experimental mapping.
    """
    from pydantic import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, TypeAdapter

    config = config_type()
    parser = argparse.ArgumentParser(add_help=False)
    config_type.add_cli_args(parser)
    typed = list(_leaves(shared["properties"]["generator"]["properties"]["pipeline"]))
    overlay: dict[str, Any] = {}
    for action in parser._actions:
        owner = config
        parts = action.dest.split(".")
        try:
            for part in parts[:-1]:
                owner = getattr(owner, part)
            annotation = _field_annotation(type(owner), parts[-1])
            default = getattr(owner, parts[-1])
            default = json.loads(json.dumps(default, allow_nan=False))
            source_field = next(item for item in dataclasses.fields(type(owner)) if item.name == parts[-1])
            option_type = dataclasses.make_dataclass(
                "PipelineOption", [("value", annotation, dataclasses.field(metadata=source_field.metadata))])
            declared = TypeAdapter(option_type).json_schema()
        except (KeyError, AttributeError, TypeError, ValueError, PydanticInvalidForJsonSchema,
                PydanticSchemaGenerationError):
            continue
        field = _inline_references(declared["properties"]["value"], declared)
        matches = [path for path, _ in typed if len(parts) == 1 and path[-1] == action.dest]
        if len(matches) == 1:
            _put(overlay, ("generator", "pipeline", *matches[0]), {"default": default})
            continue
        field.update(default=default, title=action.dest.replace("_", " "), description=action.help or "")
        if action.choices is not None:
            for branch in field.get("anyOf", [field]):
                if branch.get("type") != "null":
                    branch.get("items", branch)["enum"] = list(action.choices)
        _put(overlay, ("generator", "pipeline", "experimental", action.dest), _prepare_schema(field))
    return overlay


def _model_overlay(model: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    from fastvideo.api.presets import get_preset
    from fastvideo.registry import get_preset_selection, get_sampling_param_cls_for_name

    overlay: dict[str, Any] = {}
    _put(overlay, ("generator", "model_path"), {"const": model["id"]})
    workloads = model["workload_types"]
    _put(overlay, ("generator", "pipeline", "workload_type"), {"enum": [None, *workloads], "default": workloads[0]})
    preset_name, family = get_preset_selection(model["id"])
    defaults = {}
    if preset_name and family:
        defaults = get_preset(preset_name, family).defaults
    else:
        sampling_type = get_sampling_param_cls_for_name(model["id"])
        if sampling_type is not None and dataclasses.is_dataclass(sampling_type):
            defaults = {
                field.name: field.default
                for field in dataclasses.fields(sampling_type) if field.default is not dataclasses.MISSING
            }
    request = shared["properties"]["default_request"]
    for path, _ in _leaves(request):
        if path[-1] in defaults:
            try:
                default = json.loads(json.dumps(defaults[path[-1]], allow_nan=False))
            except (TypeError, ValueError):
                continue
            _put(overlay, ("default_request", *path), {"default": default})
    return overlay


def _validate_model_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a nonempty 'models' list")
    seen: set[str] = set()
    for model_id in value:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Every selected model ID must be a nonempty string")
        if model_id in seen:
            raise ValueError(f"Duplicate selected model ID: {model_id}")
        seen.add(model_id)
    return list(value)


def load_model_ids(path: Path = MODEL_SELECTION_PATH) -> list[str]:
    """Read only model selection; all option metadata comes from FastVideo."""
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"models"}:
        raise ValueError("Model selection YAML must contain only a 'models' list")
    return _validate_model_ids(document["models"])


def build_catalogs(model_ids: list[str]) -> list[dict[str, Any]]:
    """Compose complete model catalogs, sharing discovery work within this call."""
    selected = _validate_model_ids(model_ids)
    from pydantic import TypeAdapter

    from fastvideo.api.schema import ServeConfig
    from fastvideo.registry import get_pipeline_config_cls_from_name, get_registered_models_with_workloads

    registered = {model["id"]: model for model in get_registered_models_with_workloads() if model["workload_types"]}
    unknown = set(selected) - registered.keys()
    if unknown:
        raise ValueError(f"Models have no registered serving workload: {', '.join(sorted(unknown))}")
    native = TypeAdapter(ServeConfig).json_schema()
    shared = _prepare_schema(_inline_references(native, native))
    shared["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    pipelines = {}
    catalogs = []
    runtime = {
        "id": "pytorch",
        "label": "FastVideo serve",
        "launch_argv": ["fastvideo", "serve", "--config", "config.yaml"],
    }
    for model_id in selected:
        model = registered[model_id]
        config_type = get_pipeline_config_cls_from_name(model_id)
        key = f"{config_type.__module__}.{config_type.__name__}"
        if key not in pipelines:
            pipelines[key] = _pipeline_overlay(config_type, shared)
        schema = _merge(_merge(shared, pipelines[key]), _model_overlay(model, shared))
        catalogs.append({"model": copy.deepcopy(model), "runtime": copy.deepcopy(runtime), "schema": schema})
    return catalogs


def build_catalog(model_id: str) -> dict[str, Any]:
    """Resolve any registered model ID; the demo's model choice is not defined here."""
    return build_catalogs([model_id])[0]


def _catalog_url(model_id: str) -> str:
    # Hash the complete ID so names with slashes, punctuation or case differences
    # remain distinct, filesystem-safe and stable when YAML order changes.
    return f"models/{hashlib.sha256(model_id.encode('utf-8')).hexdigest()}.json"


def export_catalogs(output_dir: Path = OUTPUT_DIR, models_file: Path = MODEL_SELECTION_PATH) -> dict[str, Any]:
    """Generate selected catalogs; prune only exporter-owned files after success."""
    catalogs = build_catalogs(load_model_ids(models_file))
    index: dict[str, Any] = {"models": []}
    documents = {}
    for catalog in catalogs:
        model = catalog["model"]
        url = _catalog_url(model["id"])
        if url in documents:
            raise ValueError(f"Catalog filename collision for model: {model['id']}")
        documents[url] = json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        index["models"].append({**model, "catalog_url": url})

    # Finish validation/composition before modifying the last successful export.
    if output_dir.is_symlink() or (output_dir / "models").is_symlink():
        raise ValueError("Generated catalog directories must not be symlinks")
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    for url, document in documents.items():
        path = output_dir / url
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)
    manifest = output_dir / "index.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    for path in (output_dir / "models").glob("*.json"):
        if re.fullmatch(r"[0-9a-f]{64}\.json", path.name) and f"models/{path.name}" not in documents:
            path.unlink()
    return index


def check_site(site_dir: Path, models_file: Path = MODEL_SELECTION_PATH) -> None:
    """Check the built site includes every selected catalog before deployment."""
    output_dir = site_dir / "assets/cookbook-config"
    index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    if [model["id"] for model in index["models"]] != load_model_ids(models_file):
        raise ValueError("Built cookbook index does not match the selected model IDs and order")
    for model in index["models"]:
        if model["catalog_url"] != _catalog_url(model["id"]):
            raise ValueError(f"Unexpected catalog URL for model: {model['id']}")
        catalog = json.loads((output_dir / model["catalog_url"]).read_text(encoding="utf-8"))
        if catalog["model"]["id"] != model["id"] or not {"model", "runtime", "schema"} <= catalog.keys():
            raise ValueError(f"Invalid built catalog for model: {model['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-file", type=Path, default=MODEL_SELECTION_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check-site", type=Path, help="Verify a built site instead of generating catalogs.")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.check_site:
        check_site(args.check_site, args.models_file)
        print("Built cookbook catalogs verified.")
    else:
        export_catalogs(args.output_dir, args.models_file)
        print(args.output_dir / "index.json")
