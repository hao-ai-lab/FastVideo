# SPDX-License-Identifier: Apache-2.0
"""Distillation run config (v3 — ``_target_`` based)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from fastvideo.distillation.utils.distill_config import (
    CheckpointConfig,
    DataConfig,
    DistillTrainingConfig,
    DistributedConfig,
    ModelTrainingConfig,
    OptimizerConfig,
    TrackerConfig,
    TrainingLoopConfig,
    VSAConfig,
)

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class RunConfig:
    """Parsed distillation run config loaded from v3 YAML."""

    models: dict[str, dict[str, Any]]
    method: dict[str, Any]
    training: DistillTrainingConfig
    validation: dict[str, Any]
    raw: dict[str, Any]


# ---- parsing helpers (kept for use by methods) ----


def _resolve_existing_file(path: str) -> str:
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
        raise ValueError(f"Expected mapping at {where}, "
                         f"got {type(raw).__name__}")
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
    raise ValueError(f"Expected bool at {where}, "
                     f"got {type(raw).__name__}")


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
    raise ValueError(f"Expected int at {where}, "
                     f"got {type(raw).__name__}")


def get_optional_float(mapping: dict[str, Any], key: str, *, where: str) -> float | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"Expected float at {where}, got bool")
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        return float(raw)
    raise ValueError(f"Expected float at {where}, "
                     f"got {type(raw).__name__}")


def parse_betas(raw: Any, *, where: str) -> tuple[float, float]:
    if raw is None:
        raise ValueError(f"Missing betas for {where}")
    if isinstance(raw, tuple | list) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected betas as 'b1,b2' at {where}, "
                             f"got {raw!r}")
        return float(parts[0]), float(parts[1])
    raise ValueError(f"Expected betas as 'b1,b2' at {where}, "
                     f"got {type(raw).__name__}")


# ---- config convenience helpers ----


def require_positive_int(
    mapping: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    where: str | None = None,
) -> int:
    """Read an int that must be > 0."""
    loc = where or key
    raw = mapping.get(key)
    if raw is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required key {loc!r}")
    val = get_optional_int(mapping, key, where=loc)
    if val is None or val <= 0:
        raise ValueError(f"{loc} must be a positive integer, got {raw!r}")
    return val


def require_non_negative_int(
    mapping: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    where: str | None = None,
) -> int:
    """Read an int that must be >= 0."""
    loc = where or key
    raw = mapping.get(key)
    if raw is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required key {loc!r}")
    val = get_optional_int(mapping, key, where=loc)
    if val is None or val < 0:
        raise ValueError(f"{loc} must be a non-negative integer, "
                         f"got {raw!r}")
    return val


def require_non_negative_float(
    mapping: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
    where: str | None = None,
) -> float:
    """Read a float that must be >= 0."""
    loc = where or key
    raw = mapping.get(key)
    if raw is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required key {loc!r}")
    val = get_optional_float(mapping, key, where=loc)
    if val is None or val < 0.0:
        raise ValueError(f"{loc} must be a non-negative float, "
                         f"got {raw!r}")
    return val


def require_choice(
    mapping: dict[str, Any],
    key: str,
    choices: set[str] | frozenset[str],
    *,
    default: str | None = None,
    where: str | None = None,
) -> str:
    """Read a string that must be one of *choices*."""
    loc = where or key
    raw = mapping.get(key)
    if raw is None:
        if default is not None:
            if default not in choices:
                raise ValueError(f"Default {default!r} not in {choices}")
            return default
        raise ValueError(f"Missing required key {loc!r}")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{loc} must be a non-empty string, "
                         f"got {type(raw).__name__}")
    val = raw.strip().lower()
    if val not in choices:
        raise ValueError(f"{loc} must be one of {sorted(choices)}, "
                         f"got {raw!r}")
    return val


def require_bool(
    mapping: dict[str, Any],
    key: str,
    *,
    default: bool | None = None,
    where: str | None = None,
) -> bool:
    """Read a bool value."""
    loc = where or key
    raw = mapping.get(key)
    if raw is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required key {loc!r}")
    if not isinstance(raw, bool):
        raise ValueError(f"{loc} must be a bool, "
                         f"got {type(raw).__name__}")
    return raw


def _parse_pipeline_config(cfg: dict[str, Any], ) -> Any:
    """Resolve PipelineConfig from top-level YAML keys."""
    from fastvideo.configs.pipelines.base import PipelineConfig

    pipeline_raw = cfg.get("pipeline")
    default_pipeline_cfg_raw = cfg.get("default_pipeline_config")
    default_pipeline_cfg_path = cfg.get("default_pipeline_config_path")
    pipeline_cfg_raw = cfg.get("pipeline_config")
    pipeline_cfg_path = cfg.get("pipeline_config_path")

    if pipeline_raw is not None:
        if default_pipeline_cfg_raw is not None:
            raise ValueError("Provide either 'pipeline:' or "
                             "'default_pipeline_config:', not both")
        default_pipeline_cfg_raw = pipeline_raw

    if (default_pipeline_cfg_raw is not None or default_pipeline_cfg_path
            is not None) and (pipeline_cfg_raw is not None or pipeline_cfg_path is not None):
        raise ValueError("Provide either default_pipeline_config(_path) "
                         "or the legacy pipeline_config(_path), not both")

    cfg_raw = (default_pipeline_cfg_raw if default_pipeline_cfg_raw is not None else pipeline_cfg_raw)
    cfg_path = (default_pipeline_cfg_path if default_pipeline_cfg_path is not None else pipeline_cfg_path)

    if cfg_path is not None:
        cfg_path = _require_str(
            cfg_path,
            where=("default_pipeline_config_path" if default_pipeline_cfg_path is not None else "pipeline_config_path"),
        )
        return PipelineConfig.from_kwargs({
            "pipeline_config": _resolve_existing_file(cfg_path),
        })
    if cfg_raw is not None:
        if isinstance(cfg_raw, str):
            return PipelineConfig.from_kwargs({
                "pipeline_config": _resolve_existing_file(cfg_raw),
            })
        if isinstance(cfg_raw, dict):
            return PipelineConfig.from_kwargs({"pipeline_config": cfg_raw})
        raise ValueError("default_pipeline_config must be a mapping "
                         "or a path string")
    return None


def _is_nested_training_format(
    t: dict[str, Any],
) -> bool:
    """Detect whether training: uses nested sub-groups."""
    _nested_keys = {
        "distributed",
        "data",
        "optimizer",
        "loop",
        "checkpoint",
        "tracker",
        "vsa",
        "model",
    }
    return bool(_nested_keys & set(t))


def _build_training_config_nested(
    t: dict[str, Any],
    *,
    models: dict[str, dict[str, Any]],
    pipeline_config: Any,
) -> DistillTrainingConfig:
    """Build DistillTrainingConfig from nested training: YAML."""
    d = dict(t.get("distributed", {}) or {})
    da = dict(t.get("data", {}) or {})
    o = dict(t.get("optimizer", {}) or {})
    lo = dict(t.get("loop", {}) or {})
    ck = dict(t.get("checkpoint", {}) or {})
    tr = dict(t.get("tracker", {}) or {})
    vs = dict(t.get("vsa", {}) or {})
    m = dict(t.get("model", {}) or {})

    num_gpus = int(d.get("num_gpus", 1) or 1)

    betas_raw = o.get("betas", "0.9,0.999")
    betas = parse_betas(betas_raw, where="training.optimizer.betas")

    model_path = str(t.get("model_path", "") or "")
    if not model_path:
        student_cfg = models.get("student")
        if student_cfg is not None:
            init_from = student_cfg.get("init_from")
            if init_from is not None:
                model_path = str(init_from)

    return DistillTrainingConfig(
        distributed=DistributedConfig(
            num_gpus=num_gpus,
            tp_size=int(d.get("tp_size", 1) or 1),
            sp_size=int(d.get("sp_size", num_gpus) or num_gpus),
            hsdp_replicate_dim=int(d.get("hsdp_replicate_dim", 1) or 1),
            hsdp_shard_dim=int(d.get("hsdp_shard_dim", num_gpus) or num_gpus),
            pin_cpu_memory=bool(d.get("pin_cpu_memory", False)),
        ),
        data=DataConfig(
            data_path=str(da.get("data_path", "") or ""),
            train_batch_size=int(da.get("train_batch_size", 1) or 1),
            dataloader_num_workers=int(da.get("dataloader_num_workers", 0) or 0),
            training_cfg_rate=float(da.get("training_cfg_rate", 0.0) or 0.0),
            seed=int(da.get("seed", 0) or 0),
            num_height=int(da.get("num_height", 0) or 0),
            num_width=int(da.get("num_width", 0) or 0),
            num_latent_t=int(da.get("num_latent_t", 0) or 0),
            num_frames=int(da.get("num_frames", 0) or 0),
        ),
        optimizer=OptimizerConfig(
            learning_rate=float(o.get("learning_rate", 0.0) or 0.0),
            betas=betas,
            weight_decay=float(o.get("weight_decay", 0.0) or 0.0),
            lr_scheduler=str(o.get("lr_scheduler", "constant") or "constant"),
            lr_warmup_steps=int(o.get("lr_warmup_steps", 0) or 0),
            lr_num_cycles=int(o.get("lr_num_cycles", 0) or 0),
            lr_power=float(o.get("lr_power", 0.0) or 0.0),
            min_lr_ratio=float(o.get("min_lr_ratio", 0.5) or 0.5),
            max_grad_norm=float(o.get("max_grad_norm", 0.0) or 0.0),
        ),
        loop=TrainingLoopConfig(
            max_train_steps=int(lo.get("max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(lo.get("gradient_accumulation_steps", 1) or 1),
        ),
        checkpoint=CheckpointConfig(
            output_dir=str(ck.get("output_dir", "") or ""),
            resume_from_checkpoint=str(ck.get("resume_from_checkpoint", "") or ""),
            training_state_checkpointing_steps=int(ck.get("training_state_checkpointing_steps", 0) or 0),
            checkpoints_total_limit=int(ck.get("checkpoints_total_limit", 0) or 0),
        ),
        tracker=TrackerConfig(
            trackers=list(tr.get("trackers", []) or []),
            project_name=str(tr.get("project_name", "fastvideo") or "fastvideo"),
            run_name=str(tr.get("run_name", "") or ""),
        ),
        vsa=VSAConfig(
            sparsity=float(vs.get("sparsity", 0.0) or 0.0),
            decay_rate=float(vs.get("decay_rate", 0.0) or 0.0),
            decay_interval_steps=int(vs.get("decay_interval_steps", 0) or 0),
        ),
        model=ModelTrainingConfig(
            weighting_scheme=str(m.get("weighting_scheme", "uniform") or "uniform"),
            logit_mean=float(m.get("logit_mean", 0.0) or 0.0),
            logit_std=float(m.get("logit_std", 1.0) or 1.0),
            mode_scale=float(m.get("mode_scale", 1.0) or 1.0),
            precondition_outputs=bool(m.get("precondition_outputs", False)),
            moba_config=dict(m.get("moba_config", {}) or {}),
            enable_gradient_checkpointing_type=(m.get("enable_gradient_checkpointing_type")),
        ),
        pipeline_config=pipeline_config,
        model_path=model_path,
        dit_precision=str(t.get("dit_precision", "fp32") or "fp32"),
    )


def _build_training_config_flat(
    t: dict[str, Any],
    *,
    models: dict[str, dict[str, Any]],
    pipeline_config: Any,
) -> DistillTrainingConfig:
    """Build DistillTrainingConfig from flat training: YAML."""
    num_gpus = int(t.get("num_gpus", 1) or 1)

    betas_raw = t.get("betas", "0.9,0.999")
    betas = parse_betas(betas_raw, where="training.betas")

    # Use the student model path as default model_path.
    model_path = str(t.get("model_path", "") or "")
    if not model_path:
        student_cfg = models.get("student")
        if student_cfg is not None:
            init_from = student_cfg.get("init_from")
            if init_from is not None:
                model_path = str(init_from)

    return DistillTrainingConfig(
        distributed=DistributedConfig(
            num_gpus=num_gpus,
            tp_size=int(t.get("tp_size", 1) or 1),
            sp_size=int(t.get("sp_size", num_gpus) or num_gpus),
            hsdp_replicate_dim=int(t.get("hsdp_replicate_dim", 1) or 1),
            hsdp_shard_dim=int(t.get("hsdp_shard_dim", num_gpus) or num_gpus),
            pin_cpu_memory=bool(t.get("pin_cpu_memory", False)),
        ),
        data=DataConfig(
            data_path=str(t.get("data_path", "") or ""),
            train_batch_size=int(t.get("train_batch_size", 1) or 1),
            dataloader_num_workers=int(t.get("dataloader_num_workers", 0) or 0),
            training_cfg_rate=float(t.get("training_cfg_rate", 0.0) or 0.0),
            seed=int(t.get("seed", 0) or 0),
            num_height=int(t.get("num_height", 0) or 0),
            num_width=int(t.get("num_width", 0) or 0),
            num_latent_t=int(t.get("num_latent_t", 0) or 0),
            num_frames=int(t.get("num_frames", 0) or 0),
        ),
        optimizer=OptimizerConfig(
            learning_rate=float(t.get("learning_rate", 0.0) or 0.0),
            betas=betas,
            weight_decay=float(t.get("weight_decay", 0.0) or 0.0),
            lr_scheduler=str(t.get("lr_scheduler", "constant") or "constant"),
            lr_warmup_steps=int(t.get("lr_warmup_steps", 0) or 0),
            lr_num_cycles=int(t.get("lr_num_cycles", 0) or 0),
            lr_power=float(t.get("lr_power", 0.0) or 0.0),
            min_lr_ratio=float(t.get("min_lr_ratio", 0.5) or 0.5),
            max_grad_norm=float(t.get("max_grad_norm", 0.0) or 0.0),
        ),
        loop=TrainingLoopConfig(
            max_train_steps=int(t.get("max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1) or 1),
        ),
        checkpoint=CheckpointConfig(
            output_dir=str(t.get("output_dir", "") or ""),
            resume_from_checkpoint=str(t.get("resume_from_checkpoint", "") or ""),
            training_state_checkpointing_steps=int(t.get("training_state_checkpointing_steps", 0) or 0),
            checkpoints_total_limit=int(t.get("checkpoints_total_limit", 0) or 0),
        ),
        tracker=TrackerConfig(
            trackers=list(t.get("trackers", []) or []),
            project_name=str(t.get("tracker_project_name", "fastvideo") or "fastvideo"),
            run_name=str(t.get("wandb_run_name", "") or ""),
        ),
        vsa=VSAConfig(
            sparsity=float(t.get("VSA_sparsity", 0.0) or 0.0),
            decay_rate=float(t.get("VSA_decay_rate", 0.0) or 0.0),
            decay_interval_steps=int(t.get("VSA_decay_interval_steps", 0) or 0),
        ),
        model=ModelTrainingConfig(
            weighting_scheme=str(t.get("weighting_scheme", "uniform") or "uniform"),
            logit_mean=float(t.get("logit_mean", 0.0) or 0.0),
            logit_std=float(t.get("logit_std", 1.0) or 1.0),
            mode_scale=float(t.get("mode_scale", 1.0) or 1.0),
            precondition_outputs=bool(t.get("precondition_outputs", False)),
            moba_config=dict(t.get("moba_config", {}) or {}),
            enable_gradient_checkpointing_type=(t.get("enable_gradient_checkpointing_type")),
        ),
        pipeline_config=pipeline_config,
        model_path=model_path,
        dit_precision=str(t.get("dit_precision", "fp32") or "fp32"),
    )


def _build_training_config(
    training_raw: dict[str, Any],
    *,
    models: dict[str, dict[str, Any]],
    pipeline_config: Any,
) -> DistillTrainingConfig:
    """Build DistillTrainingConfig from training: YAML.

    Supports both nested (new) and flat (legacy) formats.
    """
    t = dict(training_raw)
    t.pop("validation", None)

    if _is_nested_training_format(t):
        return _build_training_config_nested(
            t, models=models, pipeline_config=pipeline_config)
    return _build_training_config_flat(
        t, models=models, pipeline_config=pipeline_config)


def load_run_config(path: str) -> RunConfig:
    """Load a distillation run config from v3 YAML.

    V3 format uses ``models:`` with ``_target_`` per role and
    ``method:`` with ``_target_`` for the algorithm class.
    """
    path = _resolve_existing_file(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _require_mapping(raw, where=path)

    # --- models section ---
    models_raw = _require_mapping(cfg.get("models"), where="models")
    models: dict[str, dict[str, Any]] = {}
    for role, model_cfg_raw in models_raw.items():
        role_str = _require_str(role, where="models.<role>")
        model_cfg = _require_mapping(model_cfg_raw, where=f"models.{role_str}")
        if "_target_" not in model_cfg:
            raise ValueError(f"models.{role_str} must have a "
                             "'_target_' key")
        models[role_str] = dict(model_cfg)

    # --- method section ---
    method_raw = _require_mapping(cfg.get("method"), where="method")
    if "_target_" not in method_raw:
        raise ValueError("method must have a '_target_' key")
    method = dict(method_raw)

    # --- backward compat: merge method_config ---
    method_config_raw = cfg.get("method_config", None)
    if method_config_raw is not None:
        warnings.warn(
            "The top-level 'method_config:' section is "
            "deprecated.  Move its keys into 'method:' "
            "directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        mc = _require_mapping(method_config_raw, where="method_config")
        for k, v in mc.items():
            if k in method and k != "_target_":
                warnings.warn(
                    f"method_config.{k} overrides "
                    f"method.{k} — prefer using "
                    "method: only",
                    DeprecationWarning,
                    stacklevel=2,
                )
            method.setdefault(k, v)

    # --- validation section ---
    validation_raw = cfg.get("validation", None)
    training_raw = _require_mapping(cfg.get("training"), where="training")

    training_validation_raw = training_raw.get("validation", None)
    if training_validation_raw is not None:
        if validation_raw is not None:
            raise ValueError("Provide 'validation:' at top-level or "
                             "under 'training:', not both")
        warnings.warn(
            "Nesting 'validation:' under 'training:' is "
            "deprecated.  Move it to the top level.",
            DeprecationWarning,
            stacklevel=2,
        )
        validation_raw = training_validation_raw

    if validation_raw is None:
        validation: dict[str, Any] = {}
    else:
        validation = _require_mapping(validation_raw, where="validation")

    # --- pipeline config ---
    pipeline_config = _parse_pipeline_config(cfg)

    # --- build typed training config ---
    training = _build_training_config(
        training_raw,
        models=models,
        pipeline_config=pipeline_config,
    )

    return RunConfig(
        models=models,
        method=method,
        training=training,
        validation=validation,
        raw=cfg,
    )
