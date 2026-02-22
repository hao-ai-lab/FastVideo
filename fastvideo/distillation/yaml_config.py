# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fastvideo.fastvideo_args import ExecutionMode, TrainingArgs
from fastvideo.logger import init_logger

from fastvideo.distillation.specs import DistillSpec, RoleName, RoleSpec

logger = init_logger(__name__)


@dataclass(slots=True)
class DistillRunConfig:
    distill: DistillSpec
    roles: dict[RoleName, RoleSpec]
    training_args: TrainingArgs
    raw: dict[str, Any]


def _distillation_root() -> Path:
    # .../fastvideo/distillation/yaml_config.py -> .../fastvideo/distillation
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    # .../fastvideo/distillation -> .../<repo_root>
    return _distillation_root().parent.parent


def _outside_root() -> Path:
    return _distillation_root() / "outside"


def resolve_outside_overlay(path: str) -> str:
    """Resolve ``path`` via the distillation ``outside/`` overlay if present.

    The overlay root is ``fastvideo/distillation/outside/`` and mirrors the
    repository layout. For example, if the run config references:

        fastvideo/configs/foo.json

    then we first check:

        fastvideo/distillation/outside/fastvideo/configs/foo.json

    and fall back to the original path when the overlay file does not exist.
    """

    if not path:
        return path

    expanded = os.path.expanduser(path)
    candidate: Path | None = None

    p = Path(expanded)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(_repo_root())
        except Exception:
            rel = None
        if rel is not None:
            candidate = _outside_root() / rel
    else:
        candidate = _outside_root() / p

    if candidate is not None and candidate.exists():
        logger.info("Using outside overlay for %s -> %s", path, candidate)
        return str(candidate)

    return expanded


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
    """Load a Phase 2 distillation run config from YAML.

    This loader intentionally does **not** merge with legacy CLI args. The YAML
    file is the single source of truth for a run.
    """

    path = resolve_outside_overlay(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _require_mapping(raw, where=path)

    distill_raw = _require_mapping(cfg.get("distill"), where="distill")
    distill_model = _require_str(distill_raw.get("model"), where="distill.model")
    distill_method = _require_str(distill_raw.get("method"), where="distill.method")
    distill = DistillSpec(model=distill_model, method=distill_method)

    roles_raw = _require_mapping(cfg.get("models"), where="models")
    roles: dict[RoleName, RoleSpec] = {}
    for role, role_cfg_raw in roles_raw.items():
        role_str = _require_str(role, where="models.<role>")
        role_cfg = _require_mapping(role_cfg_raw, where=f"models.{role_str}")
        family = role_cfg.get("family") or distill_model
        family = _require_str(family, where=f"models.{role_str}.family")
        model_path = _require_str(role_cfg.get("path"), where=f"models.{role_str}.path")
        trainable = _get_bool(
            role_cfg.get("trainable"),
            where=f"models.{role_str}.trainable",
            default=True,
        )
        roles[role_str] = RoleSpec(family=family, path=model_path, trainable=trainable)

    training_raw = _require_mapping(cfg.get("training"), where="training")

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

    # Use student path as the default base model_path. This is needed for
    # PipelineConfig registry lookup.
    if "model_path" not in training_kwargs:
        student = roles.get("student")
        if student is None:
            raise ValueError("training.model_path is missing and models.student is not provided")
        training_kwargs["model_path"] = student.path

    if "pretrained_model_name_or_path" not in training_kwargs:
        training_kwargs["pretrained_model_name_or_path"] = training_kwargs["model_path"]

    if pipeline_cfg_path is not None:
        pipeline_cfg_path = _require_str(pipeline_cfg_path, where="pipeline_config_path")
        training_kwargs["pipeline_config"] = resolve_outside_overlay(pipeline_cfg_path)
    elif pipeline_cfg_raw is not None:
        if isinstance(pipeline_cfg_raw, str):
            training_kwargs["pipeline_config"] = resolve_outside_overlay(pipeline_cfg_raw)
        elif isinstance(pipeline_cfg_raw, dict):
            training_kwargs["pipeline_config"] = pipeline_cfg_raw
        else:
            raise ValueError("pipeline_config must be a mapping or a path string")

    training_args = TrainingArgs.from_kwargs(**training_kwargs)

    return DistillRunConfig(
        distill=distill,
        roles=roles,
        training_args=training_args,
        raw=cfg,
    )
