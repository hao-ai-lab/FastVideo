# SPDX-License-Identifier: Apache-2.0
"""Reuse a QAD run's step-0 evaluation (PTQ baseline, seed+1 noise floor, reconstruction normalizers).

A Stage-2 run from a fixed checkpoint spends most of its first ~1.75 h on step 0: the full held-out evaluation,
the same evaluation at seed+1 for the noise floor, and the reconstruction calibration. All of it is a deterministic
function of the teacher and student checkpoints (``init_from``, plus ``init_student_dcp``), the calibration table
(or the provisional-calibration rows), the quantization plan, the evaluation set and the score weights, so a rerun
with the same inputs can load it. The cache is keyed by a fingerprint of exactly those inputs; any change (another
init checkpoint, eval rows, tolerances, modality weights, ...) is a mismatch and the run refuses to start.

No torch import, so the fingerprint can be computed offline from a run's ``cfg.yaml``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

STEP0_FORMAT = 2
METHOD_KEYS = ("init_student_dcp", "max_eval_rows", "max_rollout_rows", "noise_floor_eval", "min_eligibility_tolerance",
               "eligibility_keys", "fidelity_tolerance", "objective", "recon_normalize", "recon_norm_rows",
               "recon_group_weights", "rungs_per_row")
DATA_KEYS = ("manifests", "cases", "max_frames", "eval_manifest", "eval_resolutions", "eval_per_group",
             "eval_max_frames", "eval_seed", "width", "height", "num_frames")
# The training-row plan; it only reaches step 0 through provisional FFN calibration (no amax_json).
PROVISIONAL_DATA_KEYS = ("seed", "per_group", "max_train_rows", "train_manifest")
QAD_KEYS = ("linears", "fp4_attention")
# Logged per evaluation but not part of the step-0 result.
VOLATILE = ("eval/seconds", "eval/score_vs_ptq")


def _plain(value: Any) -> Any:
    """OmegaConf / mapping / sequence -> JSON-native containers."""
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple) or (hasattr(value, "__iter__") and not isinstance(value, str | bytes)):
        return [_plain(v) for v in value]
    return value


def step0_fingerprint_inputs(method_config: Any, data_config: Any, qad_config: Any, amax_sha256: str,
                             num_eval_rows: int, *, student_init_from: str, teacher_init_from: str) -> dict[str, Any]:
    """Everything step 0 depends on. The init paths stand for their weights and ``fastvideo_inference.json``."""
    method, data, qad = _plain(method_config), _plain(data_config), _plain(qad_config)
    linears = qad.get("linears") or {}
    provisional = int(linears.get("provisional_calibration_rows") or 0) > 0 and not linears.get("amax_json")
    return {
        "format": STEP0_FORMAT,
        "models": {
            "student_init_from": str(student_init_from),
            "teacher_init_from": str(teacher_init_from)
        },
        "method": {
            key: method.get(key)
            for key in METHOD_KEYS
        },
        # ``summarize`` weights the score with these; best_score and the guards compare against it.
        "score_weights": {
            modality: float(method.get(f"{modality}_weight", 1.0))
            for modality in ("video", "audio")
        },
        "data": {
            key: data.get(key)
            for key in DATA_KEYS
        },
        "provisional_plan": {
            key: data.get(key)
            for key in PROVISIONAL_DATA_KEYS
        } if provisional else None,
        "qad": {
            key: qad.get(key)
            for key in QAD_KEYS
        },
        "amax_sha256": amax_sha256,
        "num_eval_rows": int(num_eval_rows),
    }


def fingerprint(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def save_step0(path: str | Path, inputs: dict[str, Any], metrics: dict[str, float]) -> None:
    payload = {
        "format": STEP0_FORMAT,
        "fingerprint": fingerprint(inputs),
        "inputs": inputs,
        "metrics": {
            key: float(value)
            for key, value in metrics.items() if key not in VOLATILE
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def load_step0(path: str | Path, inputs: dict[str, Any]) -> dict[str, float] | None:
    """The cached step-0 metrics, None if there is no cache; raises on a fingerprint mismatch."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    expected = fingerprint(inputs)
    if payload.get("format") != STEP0_FORMAT or payload.get("fingerprint") != expected:
        cached = payload.get("inputs", {})
        diff = sorted(k for k in set(cached) | set(inputs) if cached.get(k) != inputs.get(k))
        raise ValueError(f"step-0 cache {path} was computed for other inputs (differs in {diff}); "
                         f"delete it or point step0_cache elsewhere")
    return {key: float(value) for key, value in payload["metrics"].items()}


def restore_step0(metrics: dict[str, float], num_rungs: int, recon: bool, noise_floor: bool,
                  min_tolerance: float) -> tuple[dict[str, float], list[float] | None, dict[str, float]]:
    """(eval summary, reconstruction normalizers, eligibility tolerances) from cached step-0 metrics."""
    summary = {key[len("eval/"):]: value for key, value in metrics.items() if key.startswith("eval/")}
    if "score" not in summary:
        raise ValueError("step-0 cache has no eval/score")
    recon_norm = None
    if recon:
        recon_norm = [max(metrics[f"init/recon_step0/rung{rung}"], 1e-12) for rung in range(num_rungs)]
    tolerances: dict[str, float] = {}
    if noise_floor:
        prefix = "init/noise_sigma/"
        tolerances = {
            key[len(prefix):]: max(2.0 * value, min_tolerance)
            for key, value in metrics.items() if key.startswith(prefix) and math.isfinite(value)
        }
        if not tolerances:
            raise ValueError("noise_floor_eval is on but the step-0 cache has no noise-floor sigmas")
    return summary, recon_norm, tolerances
