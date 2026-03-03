# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs

RoleName = str


@dataclass(slots=True)
class RecipeSpec:
    """Selects the model plugin key (``recipe.family``) + training method.

    This is intentionally small: everything else (roles, training args, and
    pipeline config) lives in the run config.
    """

    family: str
    method: str


@dataclass(slots=True)
class RoleSpec:
    """Describes a role's model source and whether it should be trained."""

    family: str
    path: str
    trainable: bool = True
    disable_custom_init_weights: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DistillRunConfig:
    """Parsed distillation run config loaded from schema-v2 YAML."""

    recipe: RecipeSpec
    shared_component_role: RoleName
    roles: dict[RoleName, RoleSpec]
    training_args: TrainingArgs
    validation: dict[str, Any]
    method_config: dict[str, Any]
    raw: dict[str, Any]


def _resolve_existing_file(path: str) -> str:
    """Resolve a user-provided config path and require it exists.

    Distillation intentionally does not perform any "overlay" path rewriting.
    The caller must pass a real file path (typically under
    ``examples/distillation/``).
    """

    if not path:
        return path

    expanded = os.path.expanduser(path)
    resolved = Path(expanded).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Expected a file path, got: {resolved}")
    return str(resolved)


def _require_mapping(raw: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {where}, got {type(raw).__name__}")
    return raw


def _require_str(raw: Any, *, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Expected non-empty string at {where}")
    return raw


def _get_bool(raw: Any, *, where: str, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"Expected bool at {where}, got {type(raw).__name__}")


def get_optional_int(mapping: dict[str, Any], key: str, *, where: str) -> int | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"Expected int at {where}, got bool")
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        return int(raw)
    raise ValueError(f"Expected int at {where}, got {type(raw).__name__}")


def get_optional_float(mapping: dict[str, Any], key: str, *, where: str) -> float | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"Expected float at {where}, got bool")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        return float(raw)
    raise ValueError(f"Expected float at {where}, got {type(raw).__name__}")


def parse_betas(raw: Any, *, where: str) -> tuple[float, float]:
    if raw is None:
        raise ValueError(f"Missing betas for {where}")
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected betas as 'b1,b2' at {where}, got {raw!r}")
        return float(parts[0]), float(parts[1])
    raise ValueError(f"Expected betas as 'b1,b2' at {where}, got {type(raw).__name__}")


def load_distill_run_config(path: str) -> DistillRunConfig:
    """Load a distillation run config from schema-v2 YAML.

    This loader intentionally does **not** merge with legacy CLI args. The YAML
    file is the single source of truth for a run.
    """

    from fastvideo.fastvideo_args import ExecutionMode, TrainingArgs

    path = _resolve_existing_file(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _require_mapping(raw, where=path)

    recipe_raw = _require_mapping(cfg.get("recipe"), where="recipe")
    recipe_family = _require_str(recipe_raw.get("family"), where="recipe.family")
    recipe_method = _require_str(recipe_raw.get("method"), where="recipe.method")
    recipe = RecipeSpec(family=recipe_family, method=recipe_method)

    models_raw = cfg.get("models", None)
    if models_raw is not None:
        raise ValueError(
            "Top-level `models` is not supported in schema-v2. "
            "Use `roles.shared_component_role` and `roles.<role>.*` instead."
        )

    roles_raw = _require_mapping(cfg.get("roles"), where="roles")
    shared_component_role_raw = roles_raw.get("shared_component_role", None)
    if shared_component_role_raw is None:
        shared_component_role = "student"
    else:
        shared_component_role = _require_str(
            shared_component_role_raw,
            where="roles.shared_component_role",
        )
    roles: dict[RoleName, RoleSpec] = {}
    for role, role_cfg_raw in roles_raw.items():
        if role == "shared_component_role":
            continue
        role_str = _require_str(role, where="roles.<role>")
        role_cfg = _require_mapping(role_cfg_raw, where=f"roles.{role_str}")
        if "variant" in role_cfg:
            raise ValueError(
                f"roles.{role_str}.variant is not supported in schema-v2. "
                "Use roles.<role>.family to select the model family instead "
                "(e.g. family: wangame_causal)."
            )
        family = role_cfg.get("family") or recipe_family
        family = _require_str(family, where=f"roles.{role_str}.family")
        model_path = _require_str(role_cfg.get("path"), where=f"roles.{role_str}.path")
        trainable = _get_bool(
            role_cfg.get("trainable"),
            where=f"roles.{role_str}.trainable",
            default=True,
        )
        disable_custom_init_weights = _get_bool(
            role_cfg.get("disable_custom_init_weights"),
            where=f"roles.{role_str}.disable_custom_init_weights",
            default=False,
        )
        extra = {
            key: value
            for key, value in role_cfg.items()
            if key
            not in {
                "family",
                "path",
                "trainable",
                "disable_custom_init_weights",
            }
        }
        roles[role_str] = RoleSpec(
            family=family,
            path=model_path,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            extra=extra,
        )

    shared_component_role = str(shared_component_role).strip()
    if not shared_component_role:
        raise ValueError("roles.shared_component_role cannot be empty")
    if shared_component_role not in roles:
        raise ValueError(
            "roles.shared_component_role must be a role name under roles.*, got "
            f"{shared_component_role!r}"
        )

    training_raw = _require_mapping(cfg.get("training"), where="training")

    legacy_validation_keys = {
        "log_validation",
        "validation_dataset_file",
        "validation_steps",
        "validation_sampling_steps",
        "validation_guidance_scale",
    }
    has_legacy_validation = any(key in training_raw for key in legacy_validation_keys)

    training_validation_raw = training_raw.get("validation", None)
    if training_validation_raw is None:
        if has_legacy_validation:
            raise ValueError(
                "Validation config has moved under training.validation "
                "(enabled/dataset_file/every_steps/sampling_steps/...). "
                "Do not use legacy training.validation_* keys."
            )
        validation: dict[str, Any] = {}
    else:
        if has_legacy_validation:
            raise ValueError(
                "Do not mix training.validation with legacy training.validation_* keys. "
                "Put all validation fields under training.validation."
            )
        validation = _require_mapping(training_validation_raw, where="training.validation")

    method_config_raw = cfg.get("method_config", None)
    if method_config_raw is None:
        method_config: dict[str, Any] = {}
    else:
        method_config = _require_mapping(method_config_raw, where="method_config")

    training_kwargs: dict[str, Any] = dict(training_raw)
    training_kwargs.pop("validation", None)

    # Entrypoint invariants.
    training_kwargs["mode"] = ExecutionMode.DISTILLATION
    training_kwargs["inference_mode"] = False
    # Match the training-mode loader behavior in `ComposedPipelineBase`:
    # training uses fp32 master weights and should not CPU-offload DiT weights.
    training_kwargs.setdefault("dit_precision", "fp32")
    training_kwargs["dit_cpu_offload"] = False

    # Default distributed sizes. These must be set *before* TrainingArgs
    # construction because `check_fastvideo_args()` asserts they are not -1 in
    # training mode.
    num_gpus = int(training_kwargs.get("num_gpus", 1) or 1)
    training_kwargs.setdefault("num_gpus", num_gpus)
    training_kwargs.setdefault("tp_size", 1)
    training_kwargs.setdefault("sp_size", num_gpus)
    training_kwargs.setdefault("hsdp_replicate_dim", 1)
    training_kwargs.setdefault("hsdp_shard_dim", num_gpus)

    # Use the shared-component role path as the default base model_path. This
    # is needed for PipelineConfig registry lookup and shared component loading.
    shared_role_spec = roles.get(shared_component_role)
    if shared_role_spec is None:
        raise ValueError(
            "roles.shared_component_role must reference an existing role under roles.*, got "
            f"{shared_component_role!r}"
        )

    if "model_path" not in training_kwargs:
        training_kwargs["model_path"] = shared_role_spec.path
    else:
        model_path_raw = training_kwargs.get("model_path")
        model_path = _require_str(model_path_raw, where="training.model_path").rstrip("/")
        expected = str(shared_role_spec.path).rstrip("/")
        if model_path != expected:
            raise ValueError(
                "training.model_path must match roles.<shared_component_role>.path. "
                f"Got training.model_path={model_path_raw!r}, "
                f"roles.{shared_component_role}.path={shared_role_spec.path!r}"
            )

    if "pretrained_model_name_or_path" not in training_kwargs:
        training_kwargs["pretrained_model_name_or_path"] = training_kwargs["model_path"]

    default_pipeline_cfg_raw = cfg.get("default_pipeline_config", None)
    default_pipeline_cfg_path = cfg.get("default_pipeline_config_path", None)
    pipeline_cfg_raw = cfg.get("pipeline_config", None)
    pipeline_cfg_path = cfg.get("pipeline_config_path", None)

    if (default_pipeline_cfg_raw is not None or default_pipeline_cfg_path is not None) and (
        pipeline_cfg_raw is not None or pipeline_cfg_path is not None
    ):
        raise ValueError(
            "Provide either default_pipeline_config(_path) or the legacy "
            "pipeline_config(_path), not both"
        )

    cfg_raw = default_pipeline_cfg_raw if default_pipeline_cfg_raw is not None else pipeline_cfg_raw
    cfg_path = (
        default_pipeline_cfg_path if default_pipeline_cfg_path is not None else pipeline_cfg_path
    )

    if cfg_path is not None:
        cfg_path = _require_str(
            cfg_path,
            where=(
                "default_pipeline_config_path"
                if default_pipeline_cfg_path is not None
                else "pipeline_config_path"
            ),
        )
        training_kwargs["pipeline_config"] = _resolve_existing_file(cfg_path)
    elif cfg_raw is not None:
        if isinstance(cfg_raw, str):
            training_kwargs["pipeline_config"] = _resolve_existing_file(cfg_raw)
        elif isinstance(cfg_raw, dict):
            training_kwargs["pipeline_config"] = cfg_raw
        else:
            raise ValueError(
                "default_pipeline_config must be a mapping or a path string"
            )

    training_args = TrainingArgs.from_kwargs(**training_kwargs)

    return DistillRunConfig(
        recipe=recipe,
        shared_component_role=shared_component_role,
        roles=roles,
        training_args=training_args,
        validation=validation,
        method_config=method_config,
        raw=cfg,
    )
