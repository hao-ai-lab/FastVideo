# SPDX-License-Identifier: Apache-2.0
"""Trace-level evaluation helpers for Interleave prompt-set outputs."""

from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class InterleaveTraceMetrics:
    trace_path: str
    instruction: str
    success: bool
    attempts: int
    steps: int
    retry_attempts: int
    failed_step_index: int | None = None
    failure_reason: str | None = None
    final_image_path: str | None = None
    final_prompt: str | None = None
    total_inference_time_s: float | None = None
    prompt_set_id: str | None = None
    prompt_set_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterleaveTraceEvaluationSummary:
    input_paths: list[str]
    num_traces: int
    num_success: int
    success_rate: float
    total_attempts: int
    average_attempts: float
    total_retry_attempts: int
    average_retry_attempts: float
    traces_with_final_image: int
    total_inference_time_s: float | None
    average_inference_time_s: float | None
    failure_reasons: dict[str, int]
    success_by_category: dict[str, dict[str, float]]
    traces: list[InterleaveTraceMetrics]


def discover_interleave_trace_paths(paths: Sequence[str | Path]) -> list[Path]:
    """Discover trace JSON files from trace files, summaries, or output dirs."""

    if not paths:
        raise ValueError("At least one trace, summary, or output directory is required")

    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace input not found: {path}")
        if path.is_dir():
            summary_path = path / "summary.json"
            if summary_path.is_file():
                discovered.extend(_trace_paths_from_summary(summary_path))
            else:
                discovered.extend(sorted(path.rglob("trace.json")))
            continue
        if path.name == "summary.json":
            discovered.extend(_trace_paths_from_summary(path))
            continue
        discovered.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for trace_path in discovered:
        resolved = trace_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(trace_path)
    if not unique:
        raise ValueError(f"No trace files found in inputs: {[str(path) for path in paths]}")
    return unique


def evaluate_interleave_traces(paths: Sequence[str | Path]) -> InterleaveTraceEvaluationSummary:
    """Evaluate saved Interleave traces and return aggregate metrics."""

    trace_paths = discover_interleave_trace_paths(paths)
    traces = [load_interleave_trace_metrics(path) for path in trace_paths]
    num_traces = len(traces)
    num_success = sum(1 for trace in traces if trace.success)
    total_attempts = sum(trace.attempts for trace in traces)
    total_retry_attempts = sum(trace.retry_attempts for trace in traces)
    inference_times = [trace.total_inference_time_s for trace in traces if trace.total_inference_time_s is not None]

    return InterleaveTraceEvaluationSummary(
        input_paths=[str(path) for path in paths],
        num_traces=num_traces,
        num_success=num_success,
        success_rate=(num_success / num_traces if num_traces else 0.0),
        total_attempts=total_attempts,
        average_attempts=(total_attempts / num_traces if num_traces else 0.0),
        total_retry_attempts=total_retry_attempts,
        average_retry_attempts=(total_retry_attempts / num_traces if num_traces else 0.0),
        traces_with_final_image=sum(1 for trace in traces if trace.final_image_path),
        total_inference_time_s=(sum(inference_times) if inference_times else None),
        average_inference_time_s=((sum(inference_times) / len(inference_times)) if inference_times else None),
        failure_reasons=_failure_reason_counts(traces),
        success_by_category=_success_by_category(traces),
        traces=traces,
    )


def load_interleave_trace_metrics(path: str | Path) -> InterleaveTraceMetrics:
    trace_path = Path(path)
    payload = _load_json_mapping(trace_path)
    attempts = _mapping_list(payload.get("attempts"))
    metadata = _string_mapping(payload.get("metadata"))
    final_image = _optional_mapping(payload.get("final_image"))
    final_image_path = _string_value(final_image.get("file_path")) if final_image is not None else None
    final_prompt = _string_value(final_image.get("prompt")) if final_image is not None else None
    total_time = _sum_attempt_inference_time(attempts)

    return InterleaveTraceMetrics(
        trace_path=str(trace_path),
        instruction=_string_value(payload.get("instruction")) or "",
        success=bool(payload.get("success")),
        attempts=len(attempts),
        steps=_count_steps(attempts),
        retry_attempts=sum(1 for attempt in attempts if _int_value(attempt.get("attempt_index")) not in (None, 0)),
        failed_step_index=_int_value(metadata.get("failed_step_index")),
        failure_reason=_failure_reason(payload, attempts, metadata),
        final_image_path=final_image_path,
        final_prompt=final_prompt,
        total_inference_time_s=total_time,
        prompt_set_id=_string_value(metadata.get("prompt_set_id")),
        prompt_set_index=_int_value(metadata.get("prompt_set_index")),
        metadata=dict(metadata),
    )


def interleave_trace_evaluation_to_dict(summary: InterleaveTraceEvaluationSummary) -> dict[str, Any]:
    return {
        "input_paths": list(summary.input_paths),
        "num_traces": summary.num_traces,
        "num_success": summary.num_success,
        "success_rate": summary.success_rate,
        "total_attempts": summary.total_attempts,
        "average_attempts": summary.average_attempts,
        "total_retry_attempts": summary.total_retry_attempts,
        "average_retry_attempts": summary.average_retry_attempts,
        "traces_with_final_image": summary.traces_with_final_image,
        "total_inference_time_s": summary.total_inference_time_s,
        "average_inference_time_s": summary.average_inference_time_s,
        "failure_reasons": dict(summary.failure_reasons),
        "success_by_category": dict(summary.success_by_category),
        "traces": [_trace_metrics_to_dict(trace) for trace in summary.traces],
    }


def write_interleave_trace_evaluation(
    summary: InterleaveTraceEvaluationSummary,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            interleave_trace_evaluation_to_dict(summary),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def write_interleave_trace_html_report(
    summary: InterleaveTraceEvaluationSummary,
    output_path: str | Path,
    *,
    title: str = "Interleave Trace Evaluation",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_html_report(summary, path.parent, title=title), encoding="utf-8")


def _trace_paths_from_summary(summary_path: Path) -> list[Path]:
    payload = _load_json_mapping(summary_path)
    results = _mapping_list(payload.get("results"))
    trace_paths: list[Path] = []
    for result in results:
        raw_trace_path = _string_value(result.get("trace_path"))
        if not raw_trace_path:
            continue
        candidate = Path(raw_trace_path)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = summary_path.parent / candidate
        if candidate.is_file():
            trace_paths.append(candidate)
    return trace_paths


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], payload)


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(cast(Mapping[str, Any], item))
    return rows


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def _string_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _count_steps(attempts: Sequence[Mapping[str, Any]]) -> int:
    step_indices = {_int_value(attempt.get("step_index")) for attempt in attempts}
    step_indices.discard(None)
    return len(step_indices)


def _sum_attempt_inference_time(attempts: Sequence[Mapping[str, Any]]) -> float | None:
    total = 0.0
    found = False
    for attempt in attempts:
        generated = _optional_mapping(attempt.get("generated"))
        if generated is None:
            continue
        value = _float_value(generated.get("inference_time_s"))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _failure_reason(
    payload: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> str | None:
    if bool(payload.get("success")):
        return None
    explicit_error = _string_value(metadata.get("error"))
    if explicit_error:
        return explicit_error
    for attempt in reversed(attempts):
        decision = _optional_mapping(attempt.get("decision"))
        if decision is None:
            continue
        reason = _string_value(decision.get("reason"))
        if reason:
            return reason
    failed_step = _int_value(metadata.get("failed_step_index"))
    if failed_step is not None:
        return f"failed_step_{failed_step}"
    return "unknown"


def _failure_reason_counts(traces: Sequence[InterleaveTraceMetrics]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trace in traces:
        if trace.success:
            continue
        counts[trace.failure_reason or "unknown"] += 1
    return dict(sorted(counts.items()))


def _success_by_category(traces: Sequence[InterleaveTraceMetrics]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[InterleaveTraceMetrics]] = {}
    for trace in traces:
        category = _metadata_category(trace.metadata)
        if category is None:
            continue
        grouped.setdefault(category, []).append(trace)

    result: dict[str, dict[str, float]] = {}
    for category, category_traces in sorted(grouped.items()):
        total = len(category_traces)
        success = sum(1 for trace in category_traces if trace.success)
        result[category] = {
            "num_traces": float(total),
            "num_success": float(success),
            "success_rate": success / total if total else 0.0,
        }
    return result


def _metadata_category(metadata: Mapping[str, Any]) -> str | None:
    prompt_metadata = _optional_mapping(metadata.get("prompt_set_metadata"))
    if prompt_metadata is None:
        return None
    return _string_value(prompt_metadata.get("category"))


def _trace_metrics_to_dict(trace: InterleaveTraceMetrics) -> dict[str, Any]:
    return {
        "trace_path": trace.trace_path,
        "instruction": trace.instruction,
        "success": trace.success,
        "attempts": trace.attempts,
        "steps": trace.steps,
        "retry_attempts": trace.retry_attempts,
        "failed_step_index": trace.failed_step_index,
        "failure_reason": trace.failure_reason,
        "final_image_path": trace.final_image_path,
        "final_prompt": trace.final_prompt,
        "total_inference_time_s": trace.total_inference_time_s,
        "prompt_set_id": trace.prompt_set_id,
        "prompt_set_index": trace.prompt_set_index,
        "metadata": dict(trace.metadata),
    }


def _render_html_report(
    summary: InterleaveTraceEvaluationSummary,
    html_dir: Path,
    *,
    title: str,
) -> str:
    rows = "\n".join(_render_trace_row(trace, html_dir) for trace in summary.traces)
    failure_rows = "\n".join(f"<li>{html.escape(reason)}: {count}</li>"
                             for reason, count in sorted(summary.failure_reasons.items()))
    if not failure_rows:
        failure_rows = "<li>None</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 180px; max-height: 120px; object-fit: contain; border: 1px solid #bcccdc; }}
    .ok {{ color: #1f7a4d; font-weight: 600; }}
    .fail {{ color: #b42318; font-weight: 600; }}
    .summary {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px; }}
    .metric {{ background: #f8fafc; border: 1px solid #d9e2ec; padding: 10px 12px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <section class="summary">
    <div class="metric">Traces: {summary.num_traces}</div>
    <div class="metric">Success: {summary.num_success}</div>
    <div class="metric">Success rate: {summary.success_rate:.4f}</div>
    <div class="metric">Avg attempts: {summary.average_attempts:.2f}</div>
    <div class="metric">Avg retries: {summary.average_retry_attempts:.2f}</div>
  </section>
  <h2>Failure Reasons</h2>
  <ul>{failure_rows}</ul>
  <h2>Traces</h2>
  <table>
    <thead>
      <tr>
        <th>Sample</th>
        <th>Status</th>
        <th>Attempts</th>
        <th>Instruction</th>
        <th>Final image</th>
        <th>Trace</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""


def _render_trace_row(trace: InterleaveTraceMetrics, html_dir: Path) -> str:
    sample = trace.prompt_set_id or Path(trace.trace_path).parent.name
    status_class = "ok" if trace.success else "fail"
    status_text = "success" if trace.success else f"failed: {trace.failure_reason or 'unknown'}"
    image_html = _image_html(trace.final_image_path, html_dir)
    trace_link = _path_link(trace.trace_path, html_dir)
    return ("      <tr>"
            f"<td>{html.escape(sample)}</td>"
            f"<td class=\"{status_class}\">{html.escape(status_text)}</td>"
            f"<td>{trace.attempts} ({trace.retry_attempts} retries)</td>"
            f"<td>{html.escape(trace.instruction)}</td>"
            f"<td>{image_html}</td>"
            f"<td>{trace_link}</td>"
            "</tr>")


def _image_html(image_path: str | None, html_dir: Path) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    href = _relative_or_raw_path(path, html_dir)
    return f"<a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"final image\"></a>"


def _path_link(raw_path: str, html_dir: Path) -> str:
    href = _relative_or_raw_path(Path(raw_path), html_dir)
    return f"<a href=\"{html.escape(href)}\">trace</a>"


def _relative_or_raw_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)


__all__ = [
    "InterleaveTraceEvaluationSummary",
    "InterleaveTraceMetrics",
    "discover_interleave_trace_paths",
    "evaluate_interleave_traces",
    "interleave_trace_evaluation_to_dict",
    "load_interleave_trace_metrics",
    "write_interleave_trace_evaluation",
    "write_interleave_trace_html_report",
]
