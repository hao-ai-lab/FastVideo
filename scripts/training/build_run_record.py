#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a JSON record for the training run log workbook."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    OmegaConf = None


def _strip_yaml_comment(line: str) -> str:
    result: list[str] = []
    in_single = False
    in_double = False
    depth = 0
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "[{" and not in_single and not in_double:
            depth += 1
        elif ch in "]}" and not in_single and not in_double and depth > 0:
            depth -= 1
        elif ch == "#" and not in_single and not in_double and depth == 0:
            break
        result.append(ch)
    return "".join(result).rstrip()


def _coerce_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if text.startswith(("[", "{", "(", "'", '"')):
        normalized = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.IGNORECASE)
        try:
            return ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            pass

    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def _parse_mapping_item(text: str) -> tuple[str, Any]:
    key, value = text.split(":", 1)
    value = value.strip()
    if value == "":
        return key.strip(), {}
    return key.strip(), _coerce_scalar(value)


def _minimal_yaml_load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines: list[tuple[int, str]] = []
    for raw in raw_lines:
        line = _strip_yaml_comment(raw.rstrip("\n"))
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        is_list = lines[index][1].startswith("- ")
        if is_list:
            result: list[Any] = []
            while index < len(lines):
                cur_indent, cur_text = lines[index]
                if cur_indent < indent or not cur_text.startswith("- "):
                    break
                if cur_indent > indent:
                    raise ValueError(f"Unexpected indentation in {path}: {cur_text}")
                item_text = cur_text[2:].strip()
                index += 1
                if item_text == "":
                    if index < len(lines) and lines[index][0] > cur_indent:
                        value, index = parse_block(index, lines[index][0])
                    else:
                        value = None
                elif ": " in item_text or item_text.endswith(":"):
                    key, value = _parse_mapping_item(item_text)
                    item: dict[str, Any] = {key: value}
                    if value == {} and index < len(lines) and lines[index][0] > cur_indent:
                        nested, index = parse_block(index, lines[index][0])
                        item[key] = nested
                    result.append(item)
                    continue
                else:
                    value = _coerce_scalar(item_text)
                result.append(value)
            return result, index

        result: dict[str, Any] = {}
        while index < len(lines):
            cur_indent, cur_text = lines[index]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ValueError(f"Unexpected indentation in {path}: {cur_text}")
            if ":" not in cur_text:
                raise ValueError(f"Expected mapping entry in {path}: {cur_text}")
            key, value = _parse_mapping_item(cur_text)
            index += 1
            if value == {} and index < len(lines) and lines[index][0] > cur_indent:
                value, index = parse_block(index, lines[index][0])
            result[key] = value
        return result, index

    parsed, _ = parse_block(0, lines[0][0] if lines else 0)
    return parsed if isinstance(parsed, dict) else {}


def _read_yaml(path: str) -> dict[str, Any]:
    if yaml is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    if OmegaConf is not None:
        data = OmegaConf.to_container(
            OmegaConf.load(path),
            resolve=True,
        )
        return data if isinstance(data, dict) else {}

    data = _minimal_yaml_load(path)
    return data if isinstance(data, dict) else {}


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_log_tail(path: str, max_lines: int = 200) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return [line.rstrip() for line in lines[-max_lines:]]


def _extract_training_run_summary(log_tail: list[str]) -> dict[str, Any]:
    marker = "TRAINING_RUN_SUMMARY "
    for line in reversed(log_tail):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else {}
    return {}


def _infer_status(exit_code: int, log_tail: list[str]) -> str:
    if exit_code == 0:
        return "success"
    if exit_code == 130:
        return "stopped"
    text = "\n".join(log_tail).lower()
    if (
        "cuda out of memory" in text
        or "out of memory" in text
        or re.search(r"\boom\b", text) is not None
    ):
        return "oom"
    return "error"


def _infer_failure_reason(status: str, exit_code: int, log_tail: list[str]) -> str:
    if status == "success":
        return ""
    if status == "oom":
        for line in reversed(log_tail):
            if "out of memory" in line.lower():
                return line.strip()
        return "out of memory"
    for line in reversed(log_tail):
        line = line.strip()
        if line:
            return line
    return f"training exited with code {exit_code}"


def _dataset_name_from_path(data_path: str) -> str:
    clean = str(data_path or "").strip().rstrip("/")
    if not clean:
        return ""
    parts = Path(clean).parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return Path(clean).name


def _infer_task_type(raw_cfg: dict[str, Any], config_path: str) -> str:
    sources = [
        str(config_path),
        str(_get(raw_cfg, "models", "student", "init_from", default="")),
        str(_get(raw_cfg, "models", "student", "_target_", default="")),
        str(
            _get(
                raw_cfg,
                "callbacks",
                "validation",
                "pipeline_target",
                default="",
            )
        ),
    ]
    text = " ".join(sources).lower()
    if "ti2v" in text:
        return "TI2V"
    if "i2v" in text or "image_to_video" in text:
        return "I2V"
    return "T2V"


def _infer_family(raw_cfg: dict[str, Any], task_type: str) -> str:
    student_target = str(_get(raw_cfg, "models", "student", "_target_", default=""))
    init_from = str(_get(raw_cfg, "models", "student", "init_from", default=""))
    text = f"{student_target} {init_from}".lower()
    if "hunyuan" in text:
        prefix = "hunyuan"
    elif "ltx2" in text:
        prefix = "ltx2"
    elif "matrix" in text:
        prefix = "matrixgame"
    elif "longcat" in text:
        prefix = "longcat"
    elif "gen3c" in text or "cosmos" in text:
        prefix = "gen3c"
    else:
        prefix = "wan"
    return f"{prefix}_{task_type.lower()}"


def _infer_training_type(raw_cfg: dict[str, Any]) -> str:
    student = _get(raw_cfg, "models", "student", default={}) or {}
    method_target = str(_get(raw_cfg, "method", "_target_", default="")).lower()
    if isinstance(student, dict) and student.get("lora_rank") is not None:
        return "lora"
    if "distillation" in method_target or "self_forcing" in method_target or "dmd" in method_target:
        return "distill"
    return "full_finetune"


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return ""


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _get_gpu_model() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return names[0] if names else ""


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training run log record JSON.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--wall-time-sec", type=float, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--owner", default=os.environ.get("TRAINING_RUN_LOG_OWNER", os.environ.get("USER", "")))
    args = parser.parse_args()

    raw_cfg = _read_yaml(args.config)
    log_tail = _read_log_tail(args.log_file, max_lines=400)
    log_summary = _extract_training_run_summary(log_tail)
    status = _infer_status(args.exit_code, log_tail)
    failure_reason = _infer_failure_reason(status, args.exit_code, log_tail)

    student_cfg = _get(raw_cfg, "models", "student", default={}) or {}
    train_cfg = _get(raw_cfg, "training", default={}) or {}
    data_cfg = _get(train_cfg, "data", default={}) or {}
    dist_cfg = _get(train_cfg, "distributed", default={}) or {}
    opt_cfg = _get(train_cfg, "optimizer", default={}) or {}
    loop_cfg = _get(train_cfg, "loop", default={}) or {}
    ckpt_cfg = _get(train_cfg, "checkpoint", default={}) or {}
    val_cfg = _get(raw_cfg, "callbacks", "validation", default={}) or {}

    output_dir = str(_get(ckpt_cfg, "output_dir", default=""))
    summary_path = (
        Path(output_dir)
        / "tracker"
        / "wandb"
        / "latest-run"
        / "files"
        / "wandb-summary.json"
    )
    summary = _load_json(summary_path)

    task_type = _infer_task_type(raw_cfg, args.config)
    model_family = _infer_family(raw_cfg, task_type)
    training_type = _infer_training_type(raw_cfg)

    now = datetime.now()
    run_id = (
        f"{now:%Y%m%d_%H%M%S}_{model_family}_{training_type}"
    )

    record = {
        "run_id": run_id,
        "train_date": f"{now:%Y-%m-%d}",
        "owner": args.owner,
        "model_family": model_family,
        "base_model": _get(student_cfg, "init_from", default=""),
        "task_type": task_type,
        "training_type": training_type,
        "dataset_name": _dataset_name_from_path(_get(data_cfg, "data_path", default="")),
        "data_path": _get(data_cfg, "data_path", default=""),
        "validation_source": _get(val_cfg, "dataset_file", default=""),
        "output_dir": output_dir,
        "status": status,
        "gpu_model": _get_gpu_model(),
        "gpu_count": _get(dist_cfg, "num_gpus", default=""),
        "peak_vram_gb": _first_nonempty(
            _first_present(
                summary,
                "peak_vram_gb",
                "train/peak_vram_gb",
            ),
            _first_present(log_summary, "peak_vram_gb"),
        ),
        "peak_ram_gb": _first_present(summary, "peak_ram_gb"),
        "wall_time_hours": round(float(args.wall_time_sec) / 3600.0, 4),
        "avg_step_time_sec": _first_nonempty(
            _first_present(
                summary,
                "avg_step_time_sec",
                "avg_step_time",
                "step_time_sec",
                "step_time",
            ),
            _first_present(
                log_summary,
                "avg_step_time_sec",
                "avg_step_time",
                "wall_clock_step_time_sec",
            ),
        ),
        "num_height": _get(data_cfg, "num_height", default=""),
        "num_width": _get(data_cfg, "num_width", default=""),
        "num_frames": _get(data_cfg, "num_frames", default=""),
        "fps": _get(data_cfg, "fps", default=""),
        "num_latent_t": _get(data_cfg, "num_latent_t", default=""),
        "train_batch_size": _get(data_cfg, "train_batch_size", default=""),
        "gradient_accumulation_steps": _get(loop_cfg, "gradient_accumulation_steps", default=""),
        "learning_rate": _get(opt_cfg, "learning_rate", default=""),
        "lr_scheduler": _get(opt_cfg, "lr_scheduler", default=""),
        "max_train_steps": _get(loop_cfg, "max_train_steps", default=""),
        "mixed_precision": _get(train_cfg, "dit_precision", default="fp32"),
        "sp_size": _get(dist_cfg, "sp_size", default=""),
        "tp_size": _get(dist_cfg, "tp_size", default=""),
        "hsdp_replicate_dim": _get(dist_cfg, "hsdp_replicate_dim", default=""),
        "hsdp_shard_dim": _get(dist_cfg, "hsdp_shard_dim", default=""),
        "lora_rank": _get(student_cfg, "lora_rank", default=""),
        "lora_alpha": _get(student_cfg, "lora_alpha", default=""),
        "lora_target_modules": _get(student_cfg, "lora_target_modules", default=""),
        "seed": _get(data_cfg, "seed", default=""),
        "final_train_loss": _first_nonempty(
            _first_present(
                summary,
                "final_train_loss",
                "finetune_loss",
                "total_loss",
                "train_loss",
            ),
            _first_present(
                log_summary,
                "final_train_loss",
                "train_loss",
            ),
        ),
        "failure_reason": failure_reason,
        "log_file": args.log_file,
        "config_path": args.config,
    }

    normalized = {
        key: _normalize_value(value)
        for key, value in record.items()
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
