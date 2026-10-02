"""Build the serving cookbook's browser data from contributor-authored YAML.

This module is the build-time handoff between recipe authors and the cookbook UI.
Contributors edit three kinds of files under ``docs/cookbook/serving/``:

* ``defaults.yaml`` defines shared controls, allowed values, defaults, tasks,
  request inputs, and runtime launch commands.
* ``hardware.yaml`` lists GPU types and their runtimes; model recipes specify
  GPU counts independently.
* ``models/*.yaml`` supplies each model's identity, baseline ``settings``, and
  option differences, such as a different default or a restricted choice list.

The Pydantic models below validate these files before the site is published.
They check option types and bounds, defaults against choices, references between
files, and conflicting writes to configuration paths. Shared option definitions
and model overrides are combined for validation, but remain separate in the output
so the browser can explain where each value came from.

For example, a shared sampling-step default of 50 and a model override of 9 are
both preserved in the JSON. ``cookbook-serving.js`` later resolves that model to
9, renders its controls, and generates commands from the visitor's selections.

Entry points:

* ``build_serving_example()`` reads and validates YAML, then returns a JSON-ready
  dictionary containing ``defaults``, ``hardware``, and ``recipes``.
* ``generate_serving_example()`` also writes that dictionary to
  ``docs/assets/cookbook-serving-example.json``. This is a generated website
  asset; contributors maintain the YAML rather than editing the JSON.
* ``on_pre_build()`` lets MkDocs regenerate the asset before building the site.
  Running ``python docs/cookbook_serving.py`` from the repository root generates
  the same asset without building the rest of the documentation.

Importing this module does not write files or load FastVideo models. User-selection
resolution and command formatting belong to the browser script; the serving CLI
still validates the final runtime configuration before starting inference.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator
from typing_extensions import Self

SOURCE_DIR = Path(__file__).parent / "cookbook" / "serving"
# TODO(cookbook-ui): Choose the final asset name when the new UI replaces the demo;
# update its consumers and .gitignore together. Retain metadata validation/building.
OUTPUT_PATH = Path(__file__).parent / "assets" / "cookbook-serving-example.json"

NonemptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
ConfigRoot = Annotated[str, Field(pattern=r"^(generator|server|default_request)$")]
Number = int | float
Value = TypeVar("Value")


def _browser_safe_path(path: str) -> str:
    if {"__proto__", "constructor", "prototype"}.intersection(path.split(".")):
        raise ValueError("Configuration path contains a reserved browser property")
    return path


ConfigPath = Annotated[
    str,
    Field(pattern=r"^(generator|server|default_request)(\.[A-Za-z_][A-Za-z0-9_]*)+$"),
    AfterValidator(_browser_safe_path),
]
PATH_ADAPTER = TypeAdapter(ConfigPath)


class MetadataModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class Option(MetadataModel, Generic[Value]):
    """A complete option, either shared or formed by merging a model override."""

    type: Literal["integer", "number", "boolean", "string"]
    label: NonemptyString
    path: ConfigPath
    default: Value
    choices: list[Value] = Field(default_factory=list)
    min: Number | None = None
    max: Number | None = None
    # The browser writes the same resolved value to path and every also_set path.
    also_set: list[ConfigPath] = Field(default_factory=list)
    help: str = ""
    supported: bool = True

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        for bound in {"min", "max"}.intersection(self.model_fields_set):
            if getattr(self, bound) is None:
                raise ValueError("Option bounds cannot be null")
        if {"min", "max"}.intersection(self.model_fields_set) and self.type not in {"integer", "number"}:
            raise ValueError("Only numeric options can have min/max bounds")
        for value in [self.default, *self.choices]:
            if isinstance(value, int | float):
                if self.min is not None and value < self.min:
                    raise ValueError("Option value is below min")
                if self.max is not None and value > self.max:
                    raise ValueError("Option value exceeds max")
        if "choices" in self.model_fields_set and self.default not in self.choices:
            raise ValueError("Option default must be one of choices")
        return self


class IntegerOption(Option[int]):
    type: Literal["integer"]


class NumberOption(Option[Number]):
    type: Literal["number"]


class BooleanOption(Option[bool]):
    type: Literal["boolean"]


class StringOption(Option[str]):
    type: Literal["string"]


OptionDefinition = Annotated[
    IntegerOption | NumberOption | BooleanOption | StringOption,
    Field(discriminator="type"),
]
OPTION_ADAPTER = TypeAdapter(OptionDefinition)


class Runtime(MetadataModel):
    label: NonemptyString
    command: list[NonemptyString] = Field(min_length=1)


class Task(MetadataModel):
    label: NonemptyString
    client: Literal["image", "video"]
    prompt: NonemptyString
    options: list[NonemptyString]
    requires_image: bool = False
    input_reference: NonemptyString | None = None

    @model_validator(mode="after")
    def validate_image_reference(self) -> Self:
        if self.requires_image:
            if self.client != "video":
                raise ValueError("Image references are supported only by the video client")
            if self.input_reference is None:
                raise ValueError("Image task requires a nonempty input_reference")
        return self


class Defaults(MetadataModel):
    runtimes: dict[NonemptyString, Runtime]
    tasks: dict[NonemptyString, Task]
    options: dict[NonemptyString, OptionDefinition]

    @model_validator(mode="after")
    def validate_task_options(self) -> Self:
        for name, task in self.tasks.items():
            if not set(task.options) <= self.options.keys():
                raise ValueError(f"Task references an unknown option: {name}")
        return self


class HardwareProfile(MetadataModel):
    label: NonemptyString
    runtime: NonemptyString


class HardwareCatalog(MetadataModel):
    profiles: dict[NonemptyString, HardwareProfile]


def _settings_paths(settings: Mapping[str, Any], prefix: str = "") -> set[str]:
    """Validate nested keys and collect baseline leaves for ownership checks."""
    paths = set()
    for key, value in settings.items():
        path = f"{prefix}.{key}" if prefix else key
        if prefix:
            PATH_ADAPTER.validate_python(path)
        if isinstance(value, dict):
            paths.update(_settings_paths(value, path))
        else:
            paths.add(path)
    return paths


class Recipe(MetadataModel):
    id: NonemptyString
    label: NonemptyString
    model_id: NonemptyString
    task: NonemptyString
    runtime: NonemptyString
    default_hardware: NonemptyString
    hardware: list[NonemptyString] = Field(min_length=1)
    install: NonemptyString
    evidence: NonemptyString
    notes: str = ""
    # Overrides are partial definitions. Validate them as OptionDefinition only
    # after merging with shared fields, and preserve the authored patches in JSON.
    options: dict[NonemptyString, dict[str, JsonValue]] = Field(default_factory=dict)
    settings: dict[ConfigRoot, dict[str, JsonValue]]

    @model_validator(mode="after")
    def validate_baseline(self) -> Self:
        if self.default_hardware not in self.hardware:
            raise ValueError("Default hardware is not supported")
        _settings_paths(self.settings)
        if "model_path" in self.settings.get("generator", {}):
            raise ValueError("Model identity belongs in model_id")
        return self


class CookbookMetadata(MetadataModel):
    defaults: Defaults
    hardware: dict[NonemptyString, HardwareProfile]
    recipes: list[Recipe] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references_and_ownership(self) -> Self:
        for name, profile in self.hardware.items():
            if profile.runtime not in self.defaults.runtimes:
                raise ValueError(f"Hardware references an unknown runtime: {name}")
        ids = set()
        for recipe in self.recipes:
            if recipe.id in ids:
                raise ValueError(f"Duplicate recipe ID: {recipe.id}")
            ids.add(recipe.id)
            if recipe.task not in self.defaults.tasks:
                raise ValueError(f"Recipe references an unknown task: {recipe.id}")
            if recipe.runtime not in self.defaults.runtimes:
                raise ValueError(f"Recipe references an unknown runtime: {recipe.id}")
            for hardware_id in recipe.hardware:
                if hardware_id not in self.hardware:
                    raise ValueError(f"Unknown hardware profile: {hardware_id}")
                if self.hardware[hardware_id].runtime != recipe.runtime:
                    raise ValueError(f"Hardware/runtime mismatch: {recipe.id}")

            # Reserve the identity path as well as baseline leaves. Parent/child
            # conflicts would overwrite an object or block a later browser write.
            written = _settings_paths(recipe.settings) | {"generator.model_path"}
            names = dict.fromkeys([*self.defaults.tasks[recipe.task].options, *recipe.options])
            for name in names:
                shared = self.defaults.options.get(name)
                definition = shared.model_dump(exclude_unset=True) if shared is not None else {}
                definition.update(recipe.options.get(name, {}))
                try:
                    option = OPTION_ADAPTER.validate_python(definition)
                except ValidationError as error:
                    raise ValueError(f"{recipe.id}.options.{name}: {error}") from error
                if not option.supported:
                    continue
                for path in [option.path, *option.also_set]:
                    if any(path == other or path.startswith(other + ".") or other.startswith(path + ".")
                           for other in written):
                        raise ValueError(f"Configuration path has two owners: {recipe.id}.{name}: {path}")
                    written.add(path)
        return self


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_serving_example(source_dir: str | Path = SOURCE_DIR) -> dict[str, Any]:
    """Validate authored YAML and preserve the browser's shared/override format."""
    source_dir = Path(source_dir)
    catalog = HardwareCatalog.model_validate(_load(source_dir / "hardware.yaml"))
    metadata = CookbookMetadata.model_validate({
        "defaults":
        _load(source_dir / "defaults.yaml"),
        "hardware":
        catalog.profiles,
        "recipes": [_load(path) for path in sorted((source_dir / "models").glob("*.yaml"))],
    })
    # Omitted fields must stay omitted: the browser tracks shared/model origins
    # and treats an explicitly present choices list differently from no choices.
    return metadata.model_dump(mode="json", exclude_unset=True)


def generate_serving_example(destination: str | Path = OUTPUT_PATH) -> dict[str, Any]:
    """Called by the documentation build hook; JSON is generated, never authored."""
    data = build_serving_example()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def on_pre_build(config: object, **kwargs: object) -> None:
    """MkDocs loads this hook directly, without importing the FastVideo package."""
    generate_serving_example()


if __name__ == "__main__":
    generate_serving_example()
