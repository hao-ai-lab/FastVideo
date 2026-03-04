# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, cast

from fastvideo.train.utils.config import get_optional_int


def is_validation_enabled(cfg: dict[str, Any]) -> bool:
    if not cfg:
        return False
    enabled = cfg.get("enabled")
    if enabled is None:
        return True
    if isinstance(enabled, bool):
        return bool(enabled)
    raise ValueError("training.validation.enabled must be a bool when set, got "
                     f"{type(enabled).__name__}")


def parse_validation_every_steps(cfg: dict[str, Any]) -> int:
    raw = cfg.get("every_steps")
    if raw is None:
        raise ValueError("training.validation.every_steps must be set when validation is enabled")
    if isinstance(raw, bool):
        raise ValueError("training.validation.every_steps must be an int, got bool")
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        return int(raw)
    raise ValueError("training.validation.every_steps must be an int, got "
                     f"{type(raw).__name__}")


def parse_validation_dataset_file(cfg: dict[str, Any]) -> str:
    raw = cfg.get("dataset_file")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("training.validation.dataset_file must be set when validation is enabled")
    return raw.strip()


def parse_validation_sampling_steps(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("sampling_steps")
    steps: list[int] = []
    if raw is None or raw == "":
        raise ValueError("training.validation.sampling_steps must be set for validation")
    if isinstance(raw, bool):
        raise ValueError("validation sampling_steps must be an int/list/str, got bool")
    if isinstance(raw, int) or (isinstance(raw, float) and raw.is_integer()):
        steps = [int(raw)]
    elif isinstance(raw, str):
        steps = [int(s) for s in raw.split(",") if str(s).strip()]
    elif isinstance(raw, list):
        steps = [int(s) for s in raw]
    else:
        raise ValueError("validation sampling_steps must be an int/list/str, got "
                         f"{type(raw).__name__}")
    return [s for s in steps if int(s) > 0]


def parse_validation_guidance_scale(cfg: dict[str, Any]) -> float | None:
    raw = cfg.get("guidance_scale")
    if raw in (None, ""):
        return None
    if isinstance(raw, bool):
        raise ValueError("validation guidance_scale must be a number/string, got bool")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        return float(raw)
    raise ValueError("validation guidance_scale must be a number/string, got "
                     f"{type(raw).__name__}")


def parse_validation_sampler_kind(
    cfg: dict[str, Any],
    *,
    default: Literal["ode", "sde"],
) -> Literal["ode", "sde"]:
    raw = cfg.get("sampler_kind", default)
    if raw is None:
        raw = default
    if not isinstance(raw, str):
        raise ValueError("training.validation.sampler_kind must be a string when set, got "
                         f"{type(raw).__name__}")
    kind = raw.strip().lower()
    if kind not in {"ode", "sde"}:
        raise ValueError("training.validation.sampler_kind must be one of {ode, sde}, got "
                         f"{raw!r}")
    return cast(Literal["ode", "sde"], kind)


def parse_validation_rollout_mode(
    cfg: dict[str, Any],
    *,
    default: Literal["parallel", "streaming"] = "parallel",
) -> Literal["parallel", "streaming"]:
    raw = cfg.get("rollout_mode", default)
    if raw is None:
        raw = default
    if not isinstance(raw, str):
        raise ValueError("training.validation.rollout_mode must be a string when set, got "
                         f"{type(raw).__name__}")
    mode = raw.strip().lower()
    if mode not in {"parallel", "streaming"}:
        raise ValueError("training.validation.rollout_mode must be one of {parallel, streaming}, "
                         f"got {raw!r}")
    return cast(Literal["parallel", "streaming"], mode)


def parse_validation_ode_solver(
    cfg: dict[str, Any],
    *,
    sampler_kind: Literal["ode", "sde"],
) -> str | None:
    raw = cfg.get("ode_solver")
    if raw in (None, ""):
        return None
    if sampler_kind != "ode":
        raise ValueError("training.validation.ode_solver is only valid when "
                         "training.validation.sampler_kind='ode'")
    if not isinstance(raw, str):
        raise ValueError("training.validation.ode_solver must be a string when set, got "
                         f"{type(raw).__name__}")
    solver = raw.strip().lower()
    if solver in {"unipc", "unipc_multistep", "multistep"}:
        return "unipc"
    if solver in {"euler", "flowmatch", "flowmatch_euler"}:
        return "euler"
    raise ValueError("training.validation.ode_solver must be one of {unipc, euler}, got "
                     f"{raw!r}")


def parse_validation_output_dir(cfg: dict[str, Any]) -> str | None:
    raw = cfg.get("output_dir")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("training.validation.output_dir must be a string when set, got "
                         f"{type(raw).__name__}")
    return raw


def parse_validation_num_frames(cfg: dict[str, Any]) -> int | None:
    num_frames = get_optional_int(cfg, "num_frames", where="training.validation.num_frames")
    if num_frames is not None and num_frames <= 0:
        raise ValueError("training.validation.num_frames must be > 0 when set")
    return num_frames
