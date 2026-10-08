"""Build browser catalogs from model manifests and native serving deployments.

Each model manifest offers deployments with a native serving YAML and an
explicit list of editable fields. The complete baseline is preserved; public
Python declarations supply metadata only for those controls. Export validates native parsing and serving
translation without loading weights. JSON files are generated docs assets.

FastVideo imports stay inside export helpers because importing its package also
initializes PyTorch/backend dependencies; ordinary docs tooling need not do so.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = ROOT / "docs/cookbook/recipes"
OUTPUT_DIR = ROOT / "docs/assets/cookbook-config"
JS_SAFE_INTEGER = 2**53 - 1
SCHEMA_DATA = {"default", "examples", "enum", "const"}
RUNTIME_IDS = {"fastvideo-cuda-rest", "fastvideo-mlx-rest", "fastvideo-cuda-streaming"}
# These fields change identity/topology or are rejected/overridden by the REST adapter.
UNSUPPORTED_CONTROLS = {
    "generator.pipeline.preset",
    "generator.pipeline.preset_version",
    "generator.pipeline.components.vae_weights",
    "generator.model_path",
    "generator.pipeline.workload_type",
    "generator.engine.num_gpus",
    "default_request.output.save_video",
    "default_request.output.return_frames",
    "default_request.output.output_path",
}


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


def _guide_link(value: Any, root: Path) -> dict[str, str] | None:
    """Reference normal MkDocs output instead of adding a browser Markdown renderer."""
    if value is None:
        return None
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("guide must be a repository-relative Markdown path under docs/")
    path = (root / value).resolve()
    docs = (root / "docs").resolve()
    if not path.is_relative_to(docs) or path.suffix != ".md" or not path.is_file():
        raise ValueError("guide must reference an existing Markdown file under docs/")
    relative = path.relative_to(docs)
    parts = relative.parts[:-1] if relative.name == "index.md" else (*relative.parts[:-1], relative.stem)
    page = "/".join(quote(part, safe="-._~") for part in parts)
    # All catalog links resolve against assets/cookbook-config/index.json.
    return {"url": "../../" + (page + "/" if page else "")}


def load_recipes(recipes_dir: Path = RECIPES_DIR, root: Path = ROOT) -> list[dict[str, Any]]:
    """Flatten model manifests in filename order and deployments in authored order.

    Each deployment is self-contained. Only model-level summary and guide can
    supply presentation fallbacks; configuration, controls and launch settings
    never leak from one deployment to another.
    """
    import yaml

    recipes = []
    required = {"runtime", "config", "workload", "controls"}
    allowed = required | {"summary", "env", "requirements", "guide", "label"}
    paths = sorted([*recipes_dir.glob("*.yaml"), *recipes_dir.glob("*.yml")])
    ids: set[str] = set()
    for path in paths:
        model_key = path.stem
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", model_key) or model_key in ids:
            raise ValueError(f"Invalid or duplicate recipe ID: {model_key}")
        ids.add(model_key)
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or not {"title", "deployments"} <= manifest.keys()
                or manifest.keys() - {"title", "summary", "guide", "deployments"}):
            raise ValueError(f"{path.name}: expected title and deployments; optional summary and guide")
        if not isinstance(manifest["title"], str) or not manifest["title"].strip():
            raise ValueError(f"{path.name}: title must be a nonempty string")
        if "summary" in manifest and not isinstance(manifest["summary"], str):
            raise ValueError(f"{path.name}: summary must be text")
        deployments = manifest["deployments"]
        if not isinstance(deployments, dict) or not deployments:
            raise ValueError(f"{path.name}: deployments must be a nonempty mapping")
        for deployment_key, deployment in deployments.items():
            if not isinstance(deployment_key, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", deployment_key):
                raise ValueError(f"{path.name}: invalid deployment ID: {deployment_key}")
            recipe_id = f"{model_key}/{deployment_key}"
            if not isinstance(deployment, dict) or not required <= deployment.keys() or deployment.keys() - allowed:
                raise ValueError(f"{recipe_id}: expected {sorted(required)}; optional {sorted(allowed - required)}")
            for key in ("config", "runtime", "workload"):
                if not isinstance(deployment[key], str) or not deployment[key].strip():
                    raise ValueError(f"{recipe_id}: {key} must be a nonempty string")
            if deployment["runtime"] not in RUNTIME_IDS:
                raise ValueError(f"{recipe_id}: unsupported runtime {deployment['runtime']}")
            workloads = {"t2v", "i2v"} if deployment["runtime"] == "fastvideo-cuda-rest" else {"t2v"}
            if deployment["workload"] not in workloads:
                raise ValueError(
                    f"{recipe_id}: unsupported workload {deployment['workload']} for {deployment['runtime']}")
            controls = deployment["controls"]
            if not isinstance(controls, list) or any(not isinstance(item, str) for item in controls):
                raise ValueError(f"{recipe_id}: controls must be an explicit list of field paths or namespaces")
            if len(set(controls)) != len(controls):
                raise ValueError(f"{recipe_id}: duplicate control path")
            env = deployment.get("env", {})
            if not isinstance(env, dict) or any(
                    not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                    or not isinstance(value, str) or "\0" in value for key, value in env.items()):
                raise ValueError(f"{recipe_id}: env must map environment names to strings")
            requirements = deployment.get("requirements", [])
            if not isinstance(requirements, list) or any(not isinstance(item, str) or not item.strip()
                                                         for item in requirements):
                raise ValueError(f"{recipe_id}: requirements must be a list of text notes or links")
            if "label" in deployment and (not isinstance(deployment["label"], str) or not deployment["label"].strip()):
                raise ValueError(f"{recipe_id}: label must be nonempty text")
            if "summary" in deployment and not isinstance(deployment["summary"], str):
                raise ValueError(f"{recipe_id}: summary must be text")
            config_path = (root / deployment["config"]).resolve()
            if Path(deployment["config"]).is_absolute() or not config_path.is_relative_to(
                (root / "examples/serving").resolve()):
                raise ValueError(f"{recipe_id}: config must reference a YAML under examples/serving")
            if config_path.suffix not in {".yaml", ".yml"}:
                raise ValueError(f"{recipe_id}: config must reference a native serving YAML")
            baseline = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(baseline, dict):
                raise ValueError(f"{recipe_id}: serving configuration must be a mapping")
            recipes.append({
                "summary": manifest.get("summary", ""),
                "guide": manifest.get("guide"),
                **deployment,
                "id": recipe_id,
                "model_key": model_key,
                "deployment_key": deployment_key,
                "title": manifest["title"],
                "env": env,
                "requirements": requirements,
                "base_config": baseline,
            })
    if not recipes:
        raise ValueError(f"No serving recipe manifests found in {recipes_dir}")
    return recipes


def _control_schema(schema: dict[str, Any], path: str) -> dict[str, Any]:
    """Resolve only typed public leaves; opaque experimental maps stay in the baseline."""
    if path in UNSUPPORTED_CONTROLS or path.startswith("generator.engine.parallelism.") or not re.fullmatch(
            r"(?:generator|server|default_request)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+", path):
        raise ValueError(f"Unsupported serving control: {path}")
    field = schema
    for part in path.split("."):
        field = field.get("properties", {}).get(part, {})
    branches = field.get("anyOf", [field])
    if not field or any(
            branch.get("type") not in {"string", "integer", "number", "boolean", "null", "array"}
            for branch in branches):
        raise ValueError(f"No public editable field metadata for: {path}")
    return copy.deepcopy(field)


def _expand_controls(schema: dict[str, Any], selectors: list[str]) -> list[dict[str, Any]]:
    """Expand explicit namespaces completely, rejecting ineligible or overlapping leaves.

    Manifest order comes first; descendants follow schema declaration order.
    Objects with no declared properties and nullable objects are not namespaces.
    The browser receives ordinary leaf controls and needs no expansion logic.
    """
    controls = []
    owners: dict[str, str] = {}

    def visit(field: dict[str, Any], path: str, selector: str) -> None:
        if field.get("type") == "object" and field.get("properties"):
            for name, child in field["properties"].items():
                visit(child, f"{path}.{name}", selector)
            return
        try:
            editable = _control_schema(schema, path)
        except ValueError as failure:
            raise ValueError(f"Control selector '{selector}' includes unsupported field '{path}'; "
                             "use narrower selectors. " + str(failure)) from failure
        if path in owners:
            raise ValueError(
                f"Overlapping control selector '{selector}': '{path}' is already enabled by '{owners[path]}'")
        owners[path] = selector
        controls.append({"path": path, "schema": editable})

    for selector in selectors:
        if not re.fullmatch(r"(?:generator|server|default_request)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*", selector):
            raise ValueError(f"Unsupported control selector: {selector}")
        node = schema
        for part in selector.split("."):
            node = node.get("properties", {}).get(part, {})
        if not node:
            raise ValueError(f"No public field metadata for control selector: {selector}")
        visit(node, selector, selector)
    return controls


def _check_json_values(value: Any) -> None:
    """Prevent JavaScript from rounding hidden integer settings in the baseline."""
    if isinstance(value, dict):
        for child in value.values():
            _check_json_values(child)
    elif isinstance(value, list):
        for child in value:
            _check_json_values(child)
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > JS_SAFE_INTEGER:
        raise ValueError(f"Configuration integer exceeds browser-safe range: {value}")


def _runtime_metadata(runtime_id: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Use each serving entrypoint's native configuration, without loading its engine."""
    from pydantic import TypeAdapter

    runtime: dict[str, Any]
    if runtime_id in {"fastvideo-cuda-rest", "fastvideo-cuda-streaming"}:
        from fastvideo.api.schema import ServeConfig, ServerConfig

        config_type = ServeConfig
        server = ServerConfig()
        runtime = {
            "id": runtime_id,
            "label": "CUDA streaming" if runtime_id == "fastvideo-cuda-streaming" else "CUDA REST",
            "backend": "cuda",
            "interface": "websocket" if runtime_id == "fastvideo-cuda-streaming" else "rest",
            "launch_argv": ["fastvideo", "serve", "--config", "config.yaml"],
            "install_url": "https://haoailab.com/FastVideo/getting_started/installation/",
        }
    else:
        from fastvideo.entrypoints.openai.mlx_server import MLXServeConfig, MLXServerConfig

        config_type = MLXServeConfig
        server = MLXServerConfig()
        runtime = {
            "id": runtime_id,
            "label": "MLX REST",
            "backend": "mlx",
            "interface": "rest",
            "hardware_label": "Apple Silicon",
            "launch_argv": ["python", "-m", "fastvideo.entrypoints.openai.mlx_server", "--config", "config.yaml"],
            "install_url": "https://haoailab.com/FastVideo/getting_started/installation/mlx/",
        }
    if runtime["interface"] == "websocket":
        runtime["client_guide_url"] = "../../design/server_contracts/streaming/"
    runtime["server_defaults"] = {
        "host": server.host,
        "port": server.port,
        "served_model_name": server.served_model_name,
    }
    adapter = TypeAdapter(config_type)
    declared = adapter.json_schema()
    return adapter, _prepare_schema(_inline_references(declared, declared)), runtime


def build_catalogs(recipes_dir: Path = RECIPES_DIR, root: Path = ROOT) -> list[dict[str, Any]]:
    """Validate deployment baselines and reuse native field metadata per runtime."""
    recipes = load_recipes(recipes_dir, root)
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import ServeConfig
    from fastvideo.registry import get_registered_models_with_workloads

    registered = {model["id"]: model for model in get_registered_models_with_workloads()}
    runtimes = {
        runtime_id: _runtime_metadata(runtime_id)
        for runtime_id in dict.fromkeys(recipe["runtime"] for recipe in recipes)
    }
    model_ids: dict[str, str] = {}
    model_owners: dict[str, str] = {}
    catalogs = []
    for recipe in recipes:
        baseline = recipe["base_config"]
        adapter, shared, runtime = runtimes[recipe["runtime"]]
        guide = _guide_link(recipe.get("guide"), root)
        _check_json_values(baseline)
        native = adapter.validate_python(baseline)
        model_id = native.generator.model_path
        if model_id not in registered:
            raise ValueError(f"{recipe['id']}: unknown registered model {model_id}")
        if recipe["model_key"] in model_ids and model_ids[recipe["model_key"]] != model_id:
            raise ValueError(f"{recipe['id']}: all deployments in a model manifest must use the same model_path")
        if model_id in model_owners and model_owners[model_id] != recipe["model_key"]:
            raise ValueError(f"{recipe['id']}: duplicate model identity {model_id}; "
                             f"add deployments to {model_owners[model_id]}.yaml instead")
        model_ids[recipe["model_key"]] = model_id
        model_owners[model_id] = recipe["model_key"]
        if recipe["workload"] not in registered[model_id]["workload_types"]:
            raise ValueError(f"{recipe['id']}: workload is not registered for {model_id}")
        if runtime["backend"] == "cuda":
            native = parse_config(ServeConfig, baseline)
            if runtime["interface"] == "rest" and native.streaming is not None:
                raise ValueError(f"{recipe['id']}: CUDA REST recipe cannot enable streaming")
            if runtime["interface"] == "websocket" and native.streaming is None:
                raise ValueError(f"{recipe['id']}: CUDA streaming recipe requires a streaming block")
            selected_workload = native.generator.pipeline.workload_type
            if selected_workload is not None and selected_workload != recipe["workload"]:
                raise ValueError(f"{recipe['id']}: workload disagrees with the serving baseline")
            # Exact registered IDs were checked above: no Hub discovery or model weights.
            generator_config_to_fastvideo_args(native.generator)
        else:
            from fastvideo.entrypoints.openai.mlx_server import create_mlx_app

            # The entrypoint validates its opaque default_request here. Creating
            # the app does not enter lifespan or invoke the model factory.
            # Admission creates a sample output directory, so confine that
            # validation side effect without rewriting the published baseline.
            with TemporaryDirectory(prefix="fastvideo-cookbook-mlx-") as output_dir:
                validation_config = native.model_copy(deep=True)
                validation_config.server.output_dir = output_dir
                create_mlx_app(validation_config)
        controls = _expand_controls(shared, recipe["controls"])
        if runtime["interface"] == "websocket":
            for control in controls:
                if control["path"] in {"server.served_model_name", "server.output_dir"}:
                    raise ValueError(f"Unsupported streaming control: {control['path']}")
        catalogs.append({
            "id": recipe["id"],
            "title": recipe["title"],
            "summary": recipe.get("summary", ""),
            "source_config": recipe["config"],
            "model": {
                "id": model_id,
                "key": recipe["model_key"],
                "title": recipe["title"]
            },
            "deployment": {
                "id": recipe["deployment_key"],
                "label": recipe.get("label", runtime["label"])
            },
            "workload": recipe["workload"],
            "runtime": copy.deepcopy(runtime),
            "env": copy.deepcopy(recipe["env"]),
            "base_config": copy.deepcopy(baseline),
            "controls": controls,
            "requirements": copy.deepcopy(recipe["requirements"]),
            "guide": guide,
        })
    return catalogs


def _catalog_url(recipe_id: str) -> str:
    return f"recipes/{recipe_id}.json"


def export_catalogs(output_dir: Path = OUTPUT_DIR,
                    recipes_dir: Path = RECIPES_DIR,
                    root: Path = ROOT) -> dict[str, Any]:
    """Publish catalogs only after composition succeeds; prune generated recipe files."""
    catalogs = build_catalogs(recipes_dir, root)
    index: dict[str, Any] = {"models": []}
    documents = {}
    models: dict[str, dict[str, Any]] = {}
    for catalog in catalogs:
        url = _catalog_url(catalog["id"])
        documents[url] = json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        model_key = catalog["model"]["key"]
        if model_key not in models:
            models[model_key] = {
                "id": model_key,
                "title": catalog["model"]["title"],
                "model_id": catalog["model"]["id"],
                "deployments": [],
            }
            index["models"].append(models[model_key])
        models[model_key]["deployments"].append({
            "id": catalog["id"],
            "label": catalog["deployment"]["label"],
            "workload": catalog["workload"],
            "runtime": catalog["runtime"]["id"],
            "catalog_url": url,
        })
    if output_dir.is_symlink() or (output_dir / "recipes").is_symlink():
        raise ValueError("Generated catalog directories must not be symlinks")
    (output_dir / "recipes").mkdir(parents=True, exist_ok=True)
    for url, document in documents.items():
        path = output_dir / url
        if path.parent.is_symlink():
            raise ValueError("Generated model catalog directories must not be symlinks")
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)
    manifest = output_dir / "index.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    # Prune only this exporter's current model/deployment layout.
    for path in (output_dir / "recipes").glob("*/*.json"):
        relative = path.relative_to(output_dir).as_posix()
        if (not path.parent.is_symlink() and relative not in documents
                and re.fullmatch(r"recipes/[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*\.json", relative)):
            path.unlink()
    return index


def check_site(site_dir: Path, recipes_dir: Path = RECIPES_DIR, root: Path = ROOT) -> None:
    """Check generated catalogs survive the documentation build."""
    output_dir = site_dir / "assets/cookbook-config"
    index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    recipes = load_recipes(recipes_dir, root)
    expected_models = list(dict.fromkeys(recipe["model_key"] for recipe in recipes))
    if [model["id"] for model in index["models"]] != expected_models:
        raise ValueError("Built cookbook models do not match recipe manifests")
    deployments = [deployment for model in index["models"] for deployment in model["deployments"]]
    if [deployment["id"] for deployment in deployments] != [recipe["id"] for recipe in recipes]:
        raise ValueError("Built cookbook index does not match recipe manifests")
    for recipe, source in zip(deployments, recipes, strict=True):
        if recipe["catalog_url"] != _catalog_url(recipe["id"]):
            raise ValueError(f"Unexpected catalog URL for recipe: {recipe['id']}")
        catalog = json.loads((output_dir / recipe["catalog_url"]).read_text(encoding="utf-8"))
        if catalog["id"] != recipe["id"] or not {"base_config", "controls", "runtime"} <= catalog.keys():
            raise ValueError(f"Invalid built catalog for recipe: {recipe['id']}")
        guide = _guide_link(source.get("guide"), root)
        if catalog.get("guide") != guide:
            raise ValueError(f"Built guide link differs from manifest: {recipe['id']}")
        if guide:
            page = unquote(guide["url"][len("../../"):])
            if not (site_dir / page / "index.html").is_file():
                raise ValueError(f"Guide page missing from built site: {page}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes-dir", type=Path, default=RECIPES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check-site", type=Path, help="Verify a built site instead of generating catalogs.")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.check_site:
        check_site(args.check_site, args.recipes_dir)
        print("Built cookbook catalogs verified.")
    else:
        export_catalogs(args.output_dir, args.recipes_dir)
        print(args.output_dir / "index.json")
