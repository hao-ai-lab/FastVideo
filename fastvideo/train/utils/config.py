# SPDX-License-Identifier: Apache-2.0
"""Training run config (``_target_`` based YAML)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fastvideo.train.utils.training_config import (
    CheckpointConfig,
    DataConfig,
    TrainingConfig,
    DistributedConfig,
    ModelTrainingConfig,
    OptimizerConfig,
    TrackerConfig,
    TrainingLoopConfig,
    VSAConfig,
)


@dataclass(slots=True)
class RunConfig:
    """Parsed run config loaded from YAML."""

    models: dict[str, dict[str, Any]]
    method: dict[str, Any]
    training: TrainingConfig
    callbacks: dict[str, dict[str, Any]]
    raw: dict[str, Any]

    def resolved_config(self) -> dict[str, Any]:
        """Return a fully-resolved config dict with defaults.

        Suitable for logging to W&B so that every parameter
        (including defaults) is visible.
        """
        import dataclasses

        def _safe_asdict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {
                    f.name: _safe_asdict(getattr(obj, f.name))
                    for f in dataclasses.fields(obj)
                    if not callable(getattr(obj, f.name))
                }
            if isinstance(obj, dict):
                return {k: _safe_asdict(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_safe_asdict(v) for v in obj)
            return obj

        resolved: dict[str, Any] = {}
        resolved["models"] = dict(self.models)
        resolved["method"] = dict(self.method)
        resolved["training"] = _safe_asdict(self.training)
        resolved["callbacks"] = dict(self.callbacks)
        return resolved


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


def _parse_pipeline_config(
    cfg: dict[str, Any],
    *,
    models: dict[str, dict[str, Any]],
) -> Any:
    """Resolve PipelineConfig from the ``pipeline:`` YAML key."""
    from fastvideo.configs.pipelines.base import PipelineConfig

    pipeline_raw = cfg.get("pipeline")
    if pipeline_raw is None:
        return None

    # Derive model_path from models.student.init_from —
    # needed by PipelineConfig.from_kwargs.
    model_path: str | None = None
    student_cfg = models.get("student")
    if student_cfg is not None:
        init_from = student_cfg.get("init_from")
        if init_from is not None:
            model_path = str(init_from)

    kwargs: dict[str, Any] = {"pipeline_config": pipeline_raw}
    if model_path is not None:
        kwargs["model_path"] = model_path

    if isinstance(pipeline_raw, str):
        kwargs["pipeline_config"] = _resolve_existing_file(
            pipeline_raw)

    return PipelineConfig.from_kwargs(kwargs)


def _build_training_config(
    t: dict[str, Any],
    *,
    models: dict[str, dict[str, Any]],
    pipeline_config: Any,
) -> TrainingConfig:
    """Build TrainingConfig from nested training: YAML."""
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
    betas = parse_betas(betas_raw,
                        where="training.optimizer.betas")

    model_path = str(t.get("model_path", "") or "")
    if not model_path:
        student_cfg = models.get("student")
        if student_cfg is not None:
            init_from = student_cfg.get("init_from")
            if init_from is not None:
                model_path = str(init_from)

    return TrainingConfig(
        distributed=DistributedConfig(
            num_gpus=num_gpus,
            tp_size=int(d.get("tp_size", 1) or 1),
            sp_size=int(
                d.get("sp_size", num_gpus) or num_gpus),
            hsdp_replicate_dim=int(
                d.get("hsdp_replicate_dim", 1) or 1),
            hsdp_shard_dim=int(
                d.get("hsdp_shard_dim", num_gpus)
                or num_gpus),
            pin_cpu_memory=bool(
                d.get("pin_cpu_memory", False)),
        ),
        data=DataConfig(
            data_path=str(da.get("data_path", "") or ""),
            train_batch_size=int(
                da.get("train_batch_size", 1) or 1),
            dataloader_num_workers=int(
                da.get("dataloader_num_workers", 0) or 0),
            training_cfg_rate=float(
                da.get("training_cfg_rate", 0.0) or 0.0),
            seed=int(da.get("seed", 0) or 0),
            num_height=int(
                da.get("num_height", 0) or 0),
            num_width=int(da.get("num_width", 0) or 0),
            num_latent_t=int(
                da.get("num_latent_t", 0) or 0),
            num_frames=int(
                da.get("num_frames", 0) or 0),
        ),
        optimizer=OptimizerConfig(
            learning_rate=float(
                o.get("learning_rate", 0.0) or 0.0),
            betas=betas,
            weight_decay=float(
                o.get("weight_decay", 0.0) or 0.0),
            lr_scheduler=str(
                o.get("lr_scheduler", "constant")
                or "constant"),
            lr_warmup_steps=int(
                o.get("lr_warmup_steps", 0) or 0),
            lr_num_cycles=int(
                o.get("lr_num_cycles", 0) or 0),
            lr_power=float(
                o.get("lr_power", 0.0) or 0.0),
            min_lr_ratio=float(
                o.get("min_lr_ratio", 0.5) or 0.5),
        ),
        loop=TrainingLoopConfig(
            max_train_steps=int(
                lo.get("max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(
                lo.get("gradient_accumulation_steps", 1)
                or 1),
        ),
        checkpoint=CheckpointConfig(
            output_dir=str(
                ck.get("output_dir", "") or ""),
            resume_from_checkpoint=str(
                ck.get("resume_from_checkpoint", "")
                or ""),
            training_state_checkpointing_steps=int(
                ck.get(
                    "training_state_checkpointing_steps",
                    0) or 0),
            checkpoints_total_limit=int(
                ck.get("checkpoints_total_limit", 0)
                or 0),
        ),
        tracker=TrackerConfig(
            trackers=list(
                tr.get("trackers", []) or []),
            project_name=str(
                tr.get("project_name", "fastvideo")
                or "fastvideo"),
            run_name=str(tr.get("run_name", "") or ""),
        ),
        vsa=VSAConfig(
            sparsity=float(
                vs.get("sparsity", 0.0) or 0.0),
            decay_rate=float(
                vs.get("decay_rate", 0.0) or 0.0),
            decay_interval_steps=int(
                vs.get("decay_interval_steps", 0) or 0),
        ),
        model=ModelTrainingConfig(
            weighting_scheme=str(
                m.get("weighting_scheme", "uniform")
                or "uniform"),
            logit_mean=float(
                m.get("logit_mean", 0.0) or 0.0),
            logit_std=float(
                m.get("logit_std", 1.0) or 1.0),
            mode_scale=float(
                m.get("mode_scale", 1.0) or 1.0),
            precondition_outputs=bool(
                m.get("precondition_outputs", False)),
            moba_config=dict(
                m.get("moba_config", {}) or {}),
            enable_gradient_checkpointing_type=(
                m.get(
                    "enable_gradient_checkpointing_type"
                )),
        ),
        pipeline_config=pipeline_config,
        model_path=model_path,
        dit_precision=str(
            t.get("dit_precision", "fp32") or "fp32"),
    )


def load_run_config(path: str) -> RunConfig:
    """Load a run config from YAML.

    Expected top-level keys: ``models``, ``method``,
    ``training`` (nested), and optionally ``callbacks``
    and ``pipeline``.
    """
    path = _resolve_existing_file(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _require_mapping(raw, where=path)

    # --- models ---
    models_raw = _require_mapping(
        cfg.get("models"), where="models")
    models: dict[str, dict[str, Any]] = {}
    for role, model_cfg_raw in models_raw.items():
        role_str = _require_str(
            role, where="models.<role>")
        model_cfg = _require_mapping(
            model_cfg_raw, where=f"models.{role_str}")
        if "_target_" not in model_cfg:
            raise ValueError(
                f"models.{role_str} must have a "
                "'_target_' key")
        models[role_str] = dict(model_cfg)

    # --- method ---
    method_raw = _require_mapping(
        cfg.get("method"), where="method")
    if "_target_" not in method_raw:
        raise ValueError(
            "method must have a '_target_' key")
    method = dict(method_raw)

    # --- callbacks ---
    callbacks_raw = cfg.get("callbacks", None)
    if callbacks_raw is None:
        callbacks: dict[str, dict[str, Any]] = {}
    else:
        callbacks = _require_mapping(
            callbacks_raw, where="callbacks")

    # --- pipeline config ---
    pipeline_config = _parse_pipeline_config(
        cfg, models=models)

    # --- training config ---
    training_raw = _require_mapping(
        cfg.get("training"), where="training")
    t = dict(training_raw)
    training = _build_training_config(
        t, models=models,
        pipeline_config=pipeline_config)

    return RunConfig(
        models=models,
        method=method,
        training=training,
        callbacks=callbacks,
        raw=cfg,
    )
