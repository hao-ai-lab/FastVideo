# SPDX-License-Identifier: Apache-2.0
"""Prompt-set runner and summary metrics for native Interleave workflows."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.interleave_thinker.generator import ImageGeneratorBackend
from apps.interleave_thinker.runner import (
    build_critic,
    build_image_backend,
    build_planner,
)
from apps.interleave_thinker.schema import InterleaveTrace
from apps.interleave_thinker.trace import save_trace


@dataclass(frozen=True)
class InterleavePromptItem:
    """One prompt-set row for end-to-end Interleave evaluation."""

    sample_id: str
    instruction: str
    initial_image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterleavePromptResult:
    sample_id: str
    instruction: str
    trace_path: str
    success: bool
    attempts: int
    final_image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    resumed: bool = False


@dataclass(frozen=True)
class InterleavePromptSetSummary:
    output_dir: str
    summary_path: str
    num_samples: int
    num_success: int
    success_rate: float
    total_attempts: int
    average_attempts: float
    num_resumed: int
    results: list[InterleavePromptResult]


def load_interleave_prompt_set(path: str | Path) -> list[InterleavePromptItem]:
    """Load prompt rows from JSONL, JSON, or plain text files."""

    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt set not found: {prompt_path}")

    suffix = prompt_path.suffix.lower()
    if suffix == ".jsonl":
        raw_items = _load_jsonl(prompt_path)
    elif suffix == ".json":
        raw_items = _load_json(prompt_path)
    elif suffix in {".txt", ".prompts"}:
        raw_items = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported prompt-set file format: {prompt_path}")

    items = [_coerce_prompt_item(raw, index) for index, raw in enumerate(raw_items)]
    if not items:
        raise ValueError(f"Prompt set is empty: {prompt_path}")
    return items


def run_interleave_prompt_set_config(
    config: Any,
    prompt_set_path: str | Path,
    *,
    output_dir: str | None = None,
    summary_path: str | None = None,
    limit: int | None = None,
    resume: bool = False,
    image_backend: ImageGeneratorBackend | None = None,
) -> InterleavePromptSetSummary:
    """Run a typed Interleave config over a prompt-set file."""

    prompt_items = load_interleave_prompt_set(prompt_set_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be > 0; got {limit}")
        prompt_items = prompt_items[:limit]
    return run_interleave_prompt_set(
        config,
        prompt_items,
        output_dir=output_dir,
        summary_path=summary_path,
        resume=resume,
        image_backend=image_backend,
    )


def run_interleave_prompt_set(
    config: Any,
    prompt_items: Sequence[InterleavePromptItem],
    *,
    output_dir: str | None = None,
    summary_path: str | None = None,
    resume: bool = False,
    image_backend: ImageGeneratorBackend | None = None,
) -> InterleavePromptSetSummary:
    """Run multiple Interleave traces while reusing planner/generator/critic backends."""

    if not prompt_items:
        raise ValueError("prompt_items must not be empty")

    run_config = deepcopy(config)
    root = Path(output_dir or run_config.interleave.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_config.interleave.output_dir = str(root)

    planned_rows = _planned_trace_rows(prompt_items, root)
    if resume and all(trace_path.exists() for _, item, trace_path in planned_rows):
        resumed_results = [_result_from_saved_trace(item, trace_path) for _, item, trace_path in planned_rows]
        resolved_summary_path = Path(summary_path or root / "summary.json")
        summary = _build_summary(
            resumed_results,
            output_dir=root,
            summary_path=resolved_summary_path,
        )
        save_prompt_set_summary(summary, resolved_summary_path)
        return summary

    cleanup: Callable[[], None] = _noop_cleanup
    if image_backend is None:
        image_backend, cleanup = build_image_backend(run_config)

    try:
        orchestrator = _build_prompt_set_orchestrator(run_config, image_backend)
        results: list[InterleavePromptResult] = []
        for index, item, trace_path in planned_rows:
            if resume and trace_path.exists():
                results.append(_result_from_saved_trace(item, trace_path))
                continue

            trace = orchestrator.run(
                item.instruction,
                initial_image_path=item.initial_image_path or run_config.interleave.initial_image_path,
                metadata=_trace_metadata(item, index),
            )
            trace.metadata.update(_trace_metadata(item, index))
            save_trace(
                trace,
                trace_path,
                include_images=run_config.interleave.include_images_in_trace,
            )
            results.append(_result_from_trace(item, trace, trace_path))

        resolved_summary_path = Path(summary_path or root / "summary.json")
        summary = _build_summary(
            results,
            output_dir=root,
            summary_path=resolved_summary_path,
        )
        save_prompt_set_summary(summary, resolved_summary_path)
        return summary
    finally:
        cleanup()


def save_prompt_set_summary(
    summary: InterleavePromptSetSummary,
    path: str | Path | None = None,
) -> None:
    output_path = Path(path or summary.summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            prompt_set_summary_to_dict(summary),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def prompt_set_summary_to_dict(summary: InterleavePromptSetSummary) -> dict[str, Any]:
    return {
        "output_dir": summary.output_dir,
        "summary_path": summary.summary_path,
        "num_samples": summary.num_samples,
        "num_success": summary.num_success,
        "success_rate": summary.success_rate,
        "total_attempts": summary.total_attempts,
        "average_attempts": summary.average_attempts,
        "num_resumed": summary.num_resumed,
        "results": [_prompt_result_to_dict(result) for result in summary.results],
    }


def _build_prompt_set_orchestrator(
    config: Any,
    image_backend: ImageGeneratorBackend,
) -> Any:
    from apps.interleave_thinker.orchestrator import InterleaveOrchestrator

    return InterleaveOrchestrator(
        planner=build_planner(config.planner),
        generator=image_backend,
        critic=build_critic(config.critic),
    )


def _noop_cleanup() -> None:
    pass


def _load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL row in {path}:{line_number}: {exc}") from exc
    return rows


def _load_json(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, Mapping):
        for key in ("items", "prompts", "samples"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return [raw]
    raise ValueError(f"{path} must contain a prompt list or mapping")


def _coerce_prompt_item(raw: Any, index: int) -> InterleavePromptItem:
    if isinstance(raw, str):
        return InterleavePromptItem(
            sample_id=f"sample_{index:05d}",
            instruction=raw,
        )
    if not isinstance(raw, Mapping):
        raise ValueError(f"Prompt row {index} must be a mapping or string")

    instruction = _first_text(raw, "instruction", "prompt", "text")
    if not instruction:
        raise ValueError(f"Prompt row {index} requires instruction, prompt, or text")
    sample_id = _first_text(raw, "id", "sample_id", "name") or f"sample_{index:05d}"
    initial_image_path = _first_text(raw, "initial_image_path", "input_image", "image_path", "image")

    metadata: dict[str, Any] = {}
    raw_metadata = raw.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update(dict(raw_metadata))
    reserved = {
        "id",
        "sample_id",
        "name",
        "instruction",
        "prompt",
        "text",
        "initial_image_path",
        "input_image",
        "image_path",
        "image",
        "metadata",
    }
    for key, value in raw.items():
        if key not in reserved:
            metadata[str(key)] = value

    return InterleavePromptItem(
        sample_id=str(sample_id),
        instruction=str(instruction),
        initial_image_path=str(initial_image_path) if initial_image_path else None,
        metadata=metadata,
    )


def _first_text(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _trace_metadata(item: InterleavePromptItem, index: int) -> dict[str, Any]:
    return {
        "prompt_set_id": item.sample_id,
        "prompt_set_index": index,
        "prompt_set_metadata": dict(item.metadata),
    }


def _result_from_trace(
    item: InterleavePromptItem,
    trace: InterleaveTrace,
    trace_path: Path,
) -> InterleavePromptResult:
    return InterleavePromptResult(
        sample_id=item.sample_id,
        instruction=item.instruction,
        trace_path=str(trace_path),
        success=trace.success,
        attempts=len(trace.attempts),
        final_image_path=(trace.final_image.file_path if trace.final_image is not None else None),
        metadata=dict(item.metadata),
    )


def _result_from_saved_trace(
    item: InterleavePromptItem,
    trace_path: Path,
) -> InterleavePromptResult:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    final_image = payload.get("final_image")
    return InterleavePromptResult(
        sample_id=item.sample_id,
        instruction=item.instruction,
        trace_path=str(trace_path),
        success=bool(payload.get("success")),
        attempts=len(payload.get("attempts") or []),
        final_image_path=(final_image.get("file_path") if isinstance(final_image, Mapping) else None),
        metadata=dict(item.metadata),
        resumed=True,
    )


def _build_summary(
    results: Sequence[InterleavePromptResult],
    *,
    output_dir: Path,
    summary_path: Path,
) -> InterleavePromptSetSummary:
    num_samples = len(results)
    num_success = sum(1 for result in results if result.success)
    total_attempts = sum(result.attempts for result in results)
    return InterleavePromptSetSummary(
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        num_samples=num_samples,
        num_success=num_success,
        success_rate=(num_success / num_samples if num_samples else 0.0),
        total_attempts=total_attempts,
        average_attempts=(total_attempts / num_samples if num_samples else 0.0),
        num_resumed=sum(1 for result in results if result.resumed),
        results=list(results),
    )


def _prompt_result_to_dict(result: InterleavePromptResult) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "instruction": result.instruction,
        "trace_path": result.trace_path,
        "success": result.success,
        "attempts": result.attempts,
        "final_image_path": result.final_image_path,
        "metadata": dict(result.metadata),
        "resumed": result.resumed,
    }


def _planned_trace_rows(
    prompt_items: Sequence[InterleavePromptItem],
    root: Path,
) -> list[tuple[int, InterleavePromptItem, Path]]:
    seen_ids: dict[str, int] = {}
    rows: list[tuple[int, InterleavePromptItem, Path]] = []
    for index, item in enumerate(prompt_items):
        sample_dir = root / _unique_sample_dir_name(item.sample_id, index, seen_ids)
        rows.append((index, item, sample_dir / "trace.json"))
    return rows


def _unique_sample_dir_name(
    sample_id: str,
    index: int,
    seen_ids: dict[str, int],
) -> str:
    base = _safe_sample_id(sample_id) or f"sample_{index:05d}"
    count = seen_ids.get(base, 0)
    seen_ids[base] = count + 1
    if count:
        return f"{base}_{count + 1}"
    return base


def _safe_sample_id(sample_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._-")
    return sanitized[:96]


__all__ = [
    "InterleavePromptItem",
    "InterleavePromptResult",
    "InterleavePromptSetSummary",
    "load_interleave_prompt_set",
    "prompt_set_summary_to_dict",
    "run_interleave_prompt_set",
    "run_interleave_prompt_set_config",
    "save_prompt_set_summary",
]
