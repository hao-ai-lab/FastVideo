# SPDX-License-Identifier: Apache-2.0

"""Distillation run config (v3 — ``_target_`` based)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs


@dataclass(slots=True)
class RunConfig:
    """Parsed distillation run config loaded from v3 YAML."""

    models: dict[str, dict[str, Any]]
    method: dict[str, Any]
    training_args: TrainingArgs
    validation: dict[str, Any]
    method_config: dict[str, Any]
    raw: dict[str, Any]


# ---- parsing helpers (kept for use by methods) ----


def _resolve_existing_file(path: str) -> str:
    if not path:
        return path
    expanded = os.path.expanduser(path)
    resolved = Path(expanded).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}"
        )
    if not resolved.is_file():
        raise ValueError(
            f"Expected a file path, got: {resolved}"
        )
    return str(resolved)


def _require_mapping(raw: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected mapping at {where}, "
            f"got {type(raw).__name__}"
        )
    return raw


def _require_str(raw: Any, *, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"Expected non-empty string at {where}"
        )
    return raw


def _get_bool(
    raw: Any, *, where: str, default: bool
) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise ValueError(
        f"Expected bool at {where}, "
        f"got {type(raw).__name__}"
    )


def get_optional_int(
    mapping: dict[str, Any], key: str, *, where: str
) -> int | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(
            f"Expected int at {where}, got bool"
        )
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        return int(raw)
    raise ValueError(
        f"Expected int at {where}, "
        f"got {type(raw).__name__}"
    )


def get_optional_float(
    mapping: dict[str, Any], key: str, *, where: str
) -> float | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(
            f"Expected float at {where}, got bool"
        )
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        return float(raw)
    raise ValueError(
        f"Expected float at {where}, "
        f"got {type(raw).__name__}"
    )


def parse_betas(
    raw: Any, *, where: str
) -> tuple[float, float]:
    if raw is None:
        raise ValueError(f"Missing betas for {where}")
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    if isinstance(raw, str):
        parts = [
            p.strip() for p in raw.split(",") if p.strip()
        ]
        if len(parts) != 2:
            raise ValueError(
                f"Expected betas as 'b1,b2' at {where}, "
                f"got {raw!r}"
            )
        return float(parts[0]), float(parts[1])
    raise ValueError(
        f"Expected betas as 'b1,b2' at {where}, "
        f"got {type(raw).__name__}"
    )


def load_run_config(path: str) -> RunConfig:
    """Load a distillation run config from v3 YAML.

    V3 format uses ``models:`` with ``_target_`` per role and
    ``method:`` with ``_target_`` for the algorithm class.
    """
    from fastvideo.fastvideo_args import (
        ExecutionMode,
        TrainingArgs,
    )

    path = _resolve_existing_file(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _require_mapping(raw, where=path)

    # --- models section ---
    models_raw = _require_mapping(
        cfg.get("models"), where="models"
    )
    models: dict[str, dict[str, Any]] = {}
    for role, model_cfg_raw in models_raw.items():
        role_str = _require_str(role, where="models.<role>")
        model_cfg = _require_mapping(
            model_cfg_raw, where=f"models.{role_str}"
        )
        if "_target_" not in model_cfg:
            raise ValueError(
                f"models.{role_str} must have a '_target_' key"
            )
        models[role_str] = dict(model_cfg)

    # --- method section ---
    method_raw = _require_mapping(
        cfg.get("method"), where="method"
    )
    if "_target_" not in method_raw:
        raise ValueError("method must have a '_target_' key")
    method = dict(method_raw)

    # --- method_config section ---
    method_config_raw = cfg.get("method_config", None)
    if method_config_raw is None:
        method_config: dict[str, Any] = {}
    else:
        method_config = _require_mapping(
            method_config_raw, where="method_config"
        )

    # --- training section ---
    training_raw = _require_mapping(
        cfg.get("training"), where="training"
    )

    # Validation sub-section.
    training_validation_raw = training_raw.get(
        "validation", None
    )
    if training_validation_raw is None:
        validation: dict[str, Any] = {}
    else:
        validation = _require_mapping(
            training_validation_raw,
            where="training.validation",
        )

    training_kwargs: dict[str, Any] = dict(training_raw)
    training_kwargs.pop("validation", None)

    # Entrypoint invariants.
    training_kwargs["mode"] = ExecutionMode.DISTILLATION
    training_kwargs["inference_mode"] = False
    training_kwargs.setdefault("dit_precision", "fp32")
    training_kwargs["dit_cpu_offload"] = False

    num_gpus = int(
        training_kwargs.get("num_gpus", 1) or 1
    )
    training_kwargs.setdefault("num_gpus", num_gpus)
    training_kwargs.setdefault("tp_size", 1)
    training_kwargs.setdefault("sp_size", num_gpus)
    training_kwargs.setdefault("hsdp_replicate_dim", 1)
    training_kwargs.setdefault("hsdp_shard_dim", num_gpus)

    # Use the student model path as default model_path.
    student_cfg = models.get("student")
    if student_cfg is not None and "model_path" not in training_kwargs:
        init_from = student_cfg.get("init_from")
        if init_from is not None:
            training_kwargs["model_path"] = str(init_from)

    if "pretrained_model_name_or_path" not in training_kwargs:
        training_kwargs["pretrained_model_name_or_path"] = (
            training_kwargs.get("model_path", "")
        )

    # Pipeline config.
    default_pipeline_cfg_raw = cfg.get(
        "default_pipeline_config", None
    )
    default_pipeline_cfg_path = cfg.get(
        "default_pipeline_config_path", None
    )
    pipeline_cfg_raw = cfg.get("pipeline_config", None)
    pipeline_cfg_path = cfg.get("pipeline_config_path", None)

    if (
        default_pipeline_cfg_raw is not None
        or default_pipeline_cfg_path is not None
    ) and (
        pipeline_cfg_raw is not None
        or pipeline_cfg_path is not None
    ):
        raise ValueError(
            "Provide either default_pipeline_config(_path) or "
            "the legacy pipeline_config(_path), not both"
        )

    cfg_raw = (
        default_pipeline_cfg_raw
        if default_pipeline_cfg_raw is not None
        else pipeline_cfg_raw
    )
    cfg_path = (
        default_pipeline_cfg_path
        if default_pipeline_cfg_path is not None
        else pipeline_cfg_path
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
        training_kwargs["pipeline_config"] = (
            _resolve_existing_file(cfg_path)
        )
    elif cfg_raw is not None:
        if isinstance(cfg_raw, str):
            training_kwargs["pipeline_config"] = (
                _resolve_existing_file(cfg_raw)
            )
        elif isinstance(cfg_raw, dict):
            training_kwargs["pipeline_config"] = cfg_raw
        else:
            raise ValueError(
                "default_pipeline_config must be a mapping "
                "or a path string"
            )

    training_args = TrainingArgs.from_kwargs(**training_kwargs)

    # Stash validation config on training_args for
    # init_preprocessors to pick up.
    training_args._validation_cfg = validation  # type: ignore[attr-defined]

    return RunConfig(
        models=models,
        method=method,
        training_args=training_args,
        validation=validation,
        method_config=method_config,
        raw=cfg,
    )
