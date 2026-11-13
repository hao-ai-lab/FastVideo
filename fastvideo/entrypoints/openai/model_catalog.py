# SPDX-License-Identifier: Apache-2.0
"""Installed model profiles and serialized residency on a single-device worker."""
from __future__ import annotations

import argparse
import gc
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import fastvideo.envs as envs
from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.api.schema import ServeConfig
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def apply_environment(environment: dict[str, str], baseline: dict[str, tuple[bool, str | None]]) -> None:
    """Set this profile's variables and restore every other baseline variable to the server's start value."""
    for name, (was_set, original) in baseline.items():
        field = getattr(envs, name)
        if name in environment:
            field.set(field.parse(environment[name]))
        elif was_set:
            field.set(original)
        else:
            field.clear()


@dataclass
class ModelProfile:
    """An operator-approved local serve configuration for one model/task."""
    model: str
    task: str
    path: Path
    config: ServeConfig
    args: Any
    environment: dict[str, str]

    @property
    def installed(self) -> bool:
        return Path(self.config.generator.model_path).is_dir()

    @property
    def shares_components(self) -> bool:
        """Whether this profile can hand eagerly loaded components to the next model.

        Lazily loaded or offloaded profiles own their components through loaders and host copies, so a swap tears them
        down completely and builds the next model from scratch (the only way a 60 GB DiT and a different 60 GB DiT
        fit one after the other on a unified-memory machine).
        """
        offload = self.config.generator.engine.offload
        return not (offload.lazy_module_load or offload.dit or offload.vae or offload.text_encoder)


def _component_identity(profile: ModelProfile, name: str) -> str:
    """Compare local component files, precision and loader settings before reuse."""
    args = profile.args
    root = Path(profile.config.generator.model_path)
    # FastVideoArgs names the text encoder override override_text_encoder_safetensors (a file or a directory).
    override = getattr(args, "override_text_encoder_safetensors", None) or getattr(args, "text_encoder_weights", None)
    path = (Path(override) if name == "text_encoder" and override else root / name).resolve()
    if path.is_dir():
        files = [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.rglob("*"))
                 if p.is_file()]
    elif path.is_file():
        files = [(path.name, path.stat().st_size, path.stat().st_mtime_ns)]
    else:
        files = []
    pipeline = args.pipeline_config
    settings = {}
    fields = (["text_encoder_configs", "text_encoder_precisions"]
              if name == "text_encoder" else ["vae_config", "vae_precision", "vae_decode_precision"]
              if name == "vae" else [f"{name}_config", f"{name}_precision"])
    for field in fields:
        value = getattr(pipeline, field, None)
        settings[field] = asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
    if name == "vae":
        decoder = getattr(args, "h3_vae_nvfp4_decoder", None)
        if decoder:
            p = Path(decoder).resolve()
            settings["nvfp4_decoder"] = (str(p), p.stat().st_size, p.stat().st_mtime_ns)
    if name == "text_encoder":
        settings["override_text_encoder_quant"] = getattr(args, "override_text_encoder_quant", None)
    settings["environment"] = tuple(sorted(profile.environment.items()))
    manifest_path = root / "modular_model_index.json"
    if not manifest_path.is_file():
        manifest_path = root / "model_index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return repr((name, str(path), files, settings, manifest.get(name)))


class ModelCatalog:
    """Static model allow-list. Loading never installs or downloads a model."""

    def __init__(self, path: str):
        source = Path(path).resolve()
        document = json.loads(source.read_text())
        self.shared_components = tuple(document.get("shared_components", []))
        if any(name.startswith("transformer") or "scheduler" in name for name in self.shared_components):
            raise ValueError("Transformers and schedulers cannot be retained as shared components")
        models = document["models"]
        for entry in models.values():
            for name, value in entry.get("environment", {}).items():
                field = getattr(envs, name, None)
                if not name.startswith("FASTVIDEO_") or not isinstance(field, envs.EnvField) or not isinstance(
                        value, str):
                    raise ValueError(f"Model environment entry {name!r} must be a registered FASTVIDEO_ variable "
                                     "with a string value")
                field.parse(value)  # refuse a value the variable would reject, at startup
        self.environment_baseline = {
            name: (getattr(envs, name).is_set(), getattr(envs, name).get())
            for entry in models.values()
            for name in entry.get("environment", {})
        }
        self.profiles: dict[tuple[str, str], ModelProfile] = {}
        for model, entry in models.items():
            environment = entry.get("environment", {})
            for task, filename in entry["tasks"].items():
                config_path = (source.parent / filename).resolve()
                config = build_serve_config(argparse.Namespace(config=str(config_path)))
                generator = config.generator
                # -1 (auto) resolves to the GPU count, so on one GPU it means 1.
                parallelism = generator.engine.parallelism
                if (generator.engine.num_gpus != 1 or parallelism.tp_size not in (-1, 1)
                        or parallelism.sp_size not in (-1, 1)):
                    raise ValueError("A model catalog supports one GPU with tp_size=sp_size=1")
                if config.lora or generator.pipeline.components.lora_path:
                    raise ValueError("Model catalogs do not support runtime LoRA selection")
                if config.server.served_model_name != model:
                    raise ValueError(f"Catalog alias {model} differs from {config_path}'s served_model_name")
                if Path(generator.model_path).is_dir():
                    # Env vars such as FASTVIDEO_ATTENTION_BACKEND are folded into FastVideoArgs at
                    # construction, so each profile's arguments must be built under its own environment.
                    apply_environment(environment, self.environment_baseline)
                    try:
                        args = generator_config_to_fastvideo_args(generator)
                    finally:
                        apply_environment({}, self.environment_baseline)
                else:
                    args = None
                self.profiles[model, task] = ModelProfile(model, task, config_path, config, args, dict(environment))
        if not self.profiles:
            raise ValueError("Model catalog is empty")

    def resolve(self, model: str | None, task: str | None) -> ModelProfile:
        candidates = [
            p for p in self.profiles.values()
            if (model is None or p.model == model) and (task is None or p.task == task)
        ]
        if not candidates:
            raise ValueError(f"Model/task is not configured: {model}/{task}")
        if task is None and len({p.path for p in candidates}) > 1:
            raise ValueError(f"Model {model} requires an explicit task")
        if model is None and len({p.model for p in candidates}) > 1:
            raise ValueError("An explicit model is required when several catalog models match a task")
        return candidates[0]


class ResidentModels:
    """Keep shared components while replacing the transformer's pipeline.

    Every mutating method runs inside OpenAIServingEngine's generation lock.
    A failed replacement leaves no loaded model and retains only shared objects.
    """

    def __init__(self, generator: Any, catalog: ModelCatalog, initial_args: Any, alias: str | None):
        from fastvideo.worker.uniproc_executor import UniprocExecutor

        if not isinstance(generator.executor, UniprocExecutor):
            raise ValueError("Model catalogs require the single-device in-process executor")
        self.generator = generator
        self.catalog = catalog
        self.active = next((p for p in catalog.profiles.values()
                            if p.model == alias and p.args is not None and p.args.model_path == initial_args.model_path
                            and p.args.override_pipeline_cls_name == initial_args.override_pipeline_cls_name), None)
        if self.active is None:
            raise ValueError("The startup model/pipeline is missing from the model catalog")
        self.loading: ModelProfile | None = None
        self.error: str | None = None
        self.shared: dict[str, Any] = {}
        # Describe the weights loaded at startup, not the catalog profile's build: a dotted CLI
        # override can select different files while the alias, path and pipeline class still match.
        startup = replace(self.active, args=initial_args)
        self.identities = {name: _component_identity(startup, name) for name in catalog.shared_components}
        self.reused: list[str] = []
        self._baseline = {
            name: (getattr(envs, name).is_set(), getattr(envs, name).get())
            for p in catalog.profiles.values()
            for name in p.environment
        }

    def activate(self, profile: ModelProfile) -> None:
        if self.active is not None and self.active.path == profile.path:
            return
        if not profile.installed:
            raise ValueError(f"Model {profile.model} is not installed locally")
        if profile.args is None:
            # The model was installed after the catalog was read; build its arguments before releasing the old one.
            apply_environment(profile.environment, self._baseline)
            try:
                profile.args = generator_config_to_fastvideo_args(profile.config.generator)
            except Exception:
                apply_environment({}, self._baseline)
                raise
        identities = {name: _component_identity(profile, name) for name in self.catalog.shared_components}
        self.loading, self.error = profile, None
        try:
            sharing = profile.shares_components and (self.active is None or self.active.shares_components)
            if self.active is not None:
                self.shared = self.generator.executor.detach_model(self.catalog.shared_components if sharing else ())
            self.active = None
            self.shared = {
                name: module
                for name, module in self.shared.items() if sharing and identities.get(name) == self.identities.get(name)
            }
            self.reused = sorted(self.shared)
            gc.collect()
            self._apply_environment(profile)
            args = deepcopy(profile.args)
            self.generator.executor.attach_model(args, self.shared)
            self.generator.fastvideo_args = args
            self.generator.config = deepcopy(profile.config.generator)
            self.active, self.identities = profile, identities
            self.shared = {}
            logger.info("Loaded catalog model %s (%s); retained shared components: %s", profile.model, profile.task,
                        self.reused)
        except Exception as error:
            self.error = str(error)
            raise
        finally:
            self.loading = None

    def _apply_environment(self, profile: ModelProfile) -> None:
        apply_environment(profile.environment, self._baseline)

    def unload(self) -> None:
        if self.active is not None:
            self.shared = self.generator.executor.detach_model(self.catalog.shared_components)
            self.active = None
        self.error = None

    def status(self) -> dict[str, Any]:
        return {
            "loaded_model": self.active.model if self.active else None,
            "loaded_task": self.active.task if self.active else None,
            "loading_model": self.loading.model if self.loading else None,
            "reused_components": self.reused,
            "error": self.error
        }
