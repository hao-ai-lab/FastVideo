# SPDX-License-Identifier: Apache-2.0
"""Fixed, versioned calibration for heterogeneous video reward components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from fastvideo.train.methods.rl.rewards.media import RewardScorer


_CALIBRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RewardCalibrationEntry:
    """Affine calibration parameters for one raw reward component."""

    center: float
    scale: float
    count: int | None = None
    method: str = "median_mad"

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Mapping[str, Any],
        *,
        eps: float,
    ) -> "RewardCalibrationEntry":
        try:
            center = float(raw["center"])
            scale = float(raw["scale"])
        except KeyError as exc:
            raise ValueError(
                f"Calibration entry {name!r} must define center and scale"
            ) from exc
        if not math.isfinite(center):
            raise ValueError(
                f"Calibration center for {name!r} must be finite"
            )
        if not math.isfinite(scale) or scale <= float(eps):
            raise ValueError(
                f"Calibration scale for {name!r} must be finite and > {eps}, "
                f"got {scale}"
            )
        count_value = raw.get("count")
        count = None if count_value is None else int(count_value)
        if count is not None and count <= 0:
            raise ValueError(
                f"Calibration count for {name!r} must be positive"
            )
        return cls(
            center=center,
            scale=scale,
            count=count,
            method=str(raw.get("method", "median_mad")),
        )


@dataclass(frozen=True, slots=True)
class RewardCalibration:
    """Parsed calibration artifact and provenance."""

    entries: dict[str, RewardCalibrationEntry]
    metadata: dict[str, Any]
    source_path: str | None = None


class CalibratedRewardScorer:
    """Apply fixed affine calibration without changing the wrapped scorer."""

    def __init__(
        self,
        scorer: RewardScorer,
        entry: RewardCalibrationEntry,
        *,
        clip: float | None = None,
    ) -> None:
        if clip is not None and (
            not math.isfinite(float(clip)) or float(clip) <= 0.0
        ):
            raise ValueError("calibration clip must be finite and positive")
        self.scorer = scorer
        self.entry = entry
        self.clip = None if clip is None else float(clip)
        nested = tuple(
            str(value).strip().lower()
            for value in getattr(scorer, "diagnostic_names", ())
        )
        if "unnormalized" in nested:
            raise ValueError(
                "Wrapped reward scorer already declares an unnormalized diagnostic"
            )
        self.diagnostic_names = ("unnormalized", *nested)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        raw = self.scorer(media, prompts).detach().float()
        calibrated = (raw - self.entry.center) / self.entry.scale
        if self.clip is not None:
            calibrated = calibrated.clamp(
                min=-self.clip,
                max=self.clip,
            )
        diagnostics: dict[str, torch.Tensor] = {
            "unnormalized": raw,
        }
        nested = getattr(self.scorer, "last_diagnostics", None)
        if isinstance(nested, Mapping):
            for name, values in nested.items():
                key = str(name).strip().lower()
                if key == "unnormalized":
                    raise ValueError(
                        "Wrapped reward diagnostic collides with unnormalized"
                    )
                diagnostics[key] = torch.as_tensor(
                    values,
                    device=raw.device,
                    dtype=torch.float32,
                ).detach()
        self.last_diagnostics = diagnostics
        return calibrated


def load_reward_calibration(
    raw: Mapping[str, Any],
    *,
    reward_names: Sequence[str],
) -> tuple[RewardCalibration | None, bool, float | None]:
    """Load optional fixed reward calibration from a JSON artifact or inline entries."""

    calibration_raw = raw.get("calibration")
    if calibration_raw in (None, False):
        return None, False, None
    if not isinstance(calibration_raw, Mapping):
        raise ValueError("reward_fn.calibration must be a mapping")

    required = bool(calibration_raw.get("required", True))
    eps = float(calibration_raw.get("eps", 1e-6))
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("reward calibration eps must be finite and positive")
    clip_value = calibration_raw.get("clip")
    clip = None if clip_value is None else float(clip_value)
    if clip is not None and (
        not math.isfinite(clip) or clip <= 0.0
    ):
        raise ValueError("reward calibration clip must be finite and positive")

    path_value = calibration_raw.get("path")
    inline = calibration_raw.get("entries")
    if path_value not in (None, "") and inline is not None:
        raise ValueError(
            "reward_fn.calibration must define only one of path or entries"
        )

    source_path: str | None = None
    if path_value not in (None, ""):
        path = Path(str(path_value)).expanduser()
        source_path = str(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    f"Required reward calibration artifact is missing: {path}"
                )
            return None, False, clip
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif inline is not None:
        payload = {
            "schema_version": _CALIBRATION_SCHEMA_VERSION,
            "components": inline,
        }
    elif required:
        raise ValueError(
            "reward_fn.calibration requires path or entries when required=true"
        )
    else:
        return None, False, clip

    if not isinstance(payload, Mapping):
        raise ValueError("reward calibration artifact must be a JSON object")
    version = int(payload.get("schema_version", 0))
    if version != _CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported reward calibration schema_version "
            f"{version}; expected {_CALIBRATION_SCHEMA_VERSION}"
        )
    component_raw = payload.get("components")
    if not isinstance(component_raw, Mapping):
        raise ValueError(
            "reward calibration artifact must define a components mapping"
        )

    entries: dict[str, RewardCalibrationEntry] = {}
    for raw_name, value in component_raw.items():
        name = str(raw_name).strip().lower()
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Calibration entry {name!r} must be a mapping"
            )
        entries[name] = RewardCalibrationEntry.from_mapping(
            name,
            value,
            eps=eps,
        )

    requested = [str(name).strip().lower() for name in reward_names]
    missing = sorted(set(requested) - set(entries))
    if required and missing:
        raise ValueError(
            "Required calibration entries are missing for rewards: "
            f"{missing}"
        )
    selected = {
        name: entries[name]
        for name in requested
        if name in entries
    }
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key != "components"
    }
    return (
        RewardCalibration(
            entries=selected,
            metadata=metadata,
            source_path=source_path,
        ),
        required,
        clip,
    )


__all__ = [
    "CalibratedRewardScorer",
    "RewardCalibration",
    "RewardCalibrationEntry",
    "load_reward_calibration",
]
