# SPDX-License-Identifier: Apache-2.0
"""RVM validation metrics persisted for unattended pilot runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from fastvideo.distributed import get_world_group
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.rvm_faithful import RVMFaithfulMethod


def _scalar(value: LogScalar) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Expected a scalar validation metric, got {tuple(value.shape)}"
            )
        return float(value.detach().float().cpu())
    return float(value)


def validation_metrics_path(
    output_dir: str | Path,
    iteration: int,
) -> Path:
    return (
        Path(output_dir)
        / "validation"
        / f"step-{int(iteration):06d}"
        / "metrics.json"
    )


def collect_initial_reward_results(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Collect baseline/latest validation metrics and their deltas."""
    root = Path(output_dir) / "validation"
    records: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(
            root.glob("step-*/metrics.json")
        ):
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            if not isinstance(payload, dict):
                raise ValueError(
                    "Validation metrics must be a JSON object: "
                    f"{path}"
                )
            payload = dict(payload)
            payload["path"] = str(path)
            records.append(payload)

    result: dict[str, Any] = {
        "output_dir": str(Path(output_dir)),
        "num_evaluations": len(records),
        "evaluations": records,
    }
    if not records:
        return result

    baseline = records[0]
    latest = records[-1]
    baseline_metrics = dict(
        baseline.get("metrics", {})
    )
    latest_metrics = dict(
        latest.get("metrics", {})
    )
    delta = {
        key: float(latest_metrics[key])
        - float(baseline_metrics[key])
        for key in sorted(
            set(baseline_metrics)
            & set(latest_metrics)
        )
        if key.startswith("validation/reward/")
    }
    result.update(
        {
            "baseline_iteration": int(
                baseline["iteration"]
            ),
            "latest_iteration": int(
                latest["iteration"]
            ),
            "baseline": baseline_metrics,
            "latest": latest_metrics,
            "reward_delta": delta,
        }
    )
    return result


class RVMWithLocalMetricsMethod(RVMFaithfulMethod):
    """Paper-faithful RVM with validation metrics saved beside MP4s."""

    @torch.no_grad()
    def _run_validation(
        self,
        iteration: int,
    ) -> dict[str, LogScalar]:
        metrics = super()._run_validation(iteration)
        if int(get_world_group().rank) != 0:
            return metrics

        path = validation_metrics_path(
            self.training_config.checkpoint.output_dir,
            iteration,
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        payload = {
            "iteration": int(iteration),
            "metrics": {
                key: _scalar(value)
                for key, value in sorted(
                    metrics.items()
                )
            },
        }
        temporary = path.with_suffix(
            ".json.tmp"
        )
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

        latest = (
            path.parent.parent
            / "latest_metrics.json"
        )
        latest_temporary = latest.with_suffix(
            ".json.tmp"
        )
        latest_temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        latest_temporary.replace(latest)
        return metrics


__all__ = [
    "RVMWithLocalMetricsMethod",
    "collect_initial_reward_results",
    "validation_metrics_path",
]
