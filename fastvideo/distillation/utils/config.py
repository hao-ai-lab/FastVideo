# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.distillation.methods.base import DistillMethod

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


@dataclass(slots=True)
class DistillRunConfig:
    """Parsed distillation run config loaded from schema-v2 YAML."""

    recipe: RecipeSpec
    roles: dict[RoleName, RoleSpec]
    training_args: TrainingArgs
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

    roles_raw = _require_mapping(cfg.get("roles"), where="roles")
    roles: dict[RoleName, RoleSpec] = {}
    for role, role_cfg_raw in roles_raw.items():
        role_str = _require_str(role, where="roles.<role>")
        role_cfg = _require_mapping(role_cfg_raw, where=f"roles.{role_str}")
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
        roles[role_str] = RoleSpec(
            family=family,
            path=model_path,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
        )

    training_raw = _require_mapping(cfg.get("training"), where="training")

    method_config_raw = cfg.get("method_config", None)
    if method_config_raw is None:
        method_config: dict[str, Any] = {}
    else:
        method_config = _require_mapping(method_config_raw, where="method_config")

    pipeline_cfg_raw = cfg.get("pipeline_config", None)
    pipeline_cfg_path = cfg.get("pipeline_config_path", None)
    if pipeline_cfg_raw is not None and pipeline_cfg_path is not None:
        raise ValueError("Provide either pipeline_config or pipeline_config_path, not both")

    training_kwargs: dict[str, Any] = dict(training_raw)

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

    # Use student path as the default base model_path. This is needed for
    # PipelineConfig registry lookup.
    if "model_path" not in training_kwargs:
        student = roles.get("student")
        if student is None:
            raise ValueError("training.model_path is missing and roles.student is not provided")
        training_kwargs["model_path"] = student.path

    if "pretrained_model_name_or_path" not in training_kwargs:
        training_kwargs["pretrained_model_name_or_path"] = training_kwargs["model_path"]

    if pipeline_cfg_path is not None:
        pipeline_cfg_path = _require_str(pipeline_cfg_path, where="pipeline_config_path")
        training_kwargs["pipeline_config"] = _resolve_existing_file(pipeline_cfg_path)
    elif pipeline_cfg_raw is not None:
        if isinstance(pipeline_cfg_raw, str):
            training_kwargs["pipeline_config"] = _resolve_existing_file(pipeline_cfg_raw)
        elif isinstance(pipeline_cfg_raw, dict):
            training_kwargs["pipeline_config"] = pipeline_cfg_raw
        else:
            raise ValueError("pipeline_config must be a mapping or a path string")

    training_args = TrainingArgs.from_kwargs(**training_kwargs)

    return DistillRunConfig(
        recipe=recipe,
        roles=roles,
        training_args=training_args,
        method_config=method_config,
        raw=cfg,
    )


@dataclass(slots=True)
class DistillRuntime:
    """Fully assembled runtime for `DistillTrainer.run()`."""

    training_args: TrainingArgs
    method: DistillMethod
    dataloader: Any
    tracker: Any
    start_step: int = 0
