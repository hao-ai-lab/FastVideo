# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastvideo.entrypoints.interleave import (
    discover_interleave_trace_paths,
    evaluate_interleave_traces,
    interleave_trace_evaluation_to_dict,
    write_interleave_trace_html_report,
)


def test_evaluate_interleave_traces_from_summary(tmp_path: Path) -> None:
    output_dir = _write_trace_fixture(tmp_path)

    summary = evaluate_interleave_traces([output_dir / "summary.json"])

    assert summary.num_traces == 2
    assert summary.num_success == 1
    assert summary.success_rate == 0.5
    assert summary.total_attempts == 3
    assert summary.average_attempts == 1.5
    assert summary.total_retry_attempts == 1
    assert summary.traces_with_final_image == 1
    assert summary.total_inference_time_s == 1.5
    assert summary.failure_reasons == {"critic rejected final attempt": 1}
    assert summary.success_by_category["product"] == {
        "num_traces": 1.0,
        "num_success": 1.0,
        "success_rate": 1.0,
    }
    payload = interleave_trace_evaluation_to_dict(summary)
    assert payload["traces"][0]["prompt_set_id"] == "mug"
    assert payload["traces"][1]["failure_reason"] == "critic rejected final attempt"


def test_discover_interleave_trace_paths_from_directory(tmp_path: Path) -> None:
    output_dir = _write_trace_fixture(tmp_path)

    paths = discover_interleave_trace_paths([output_dir])

    assert [path.name for path in paths] == ["trace.json", "trace.json"]
    assert {path.parent.name for path in paths} == {"mug", "poster"}


def test_write_interleave_trace_html_report(tmp_path: Path) -> None:
    output_dir = _write_trace_fixture(tmp_path)
    summary = evaluate_interleave_traces([output_dir])
    html_path = tmp_path / "report.html"

    write_interleave_trace_html_report(summary, html_path, title="Smoke Report")

    html_text = html_path.read_text(encoding="utf-8")
    assert "Smoke Report" in html_text
    assert "mug" in html_text
    assert "critic rejected final attempt" in html_text
    assert "<img" in html_text


def test_evaluate_traces_script_writes_json_and_html(tmp_path: Path) -> None:
    output_dir = _write_trace_fixture(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    html_path = tmp_path / "report.html"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/interleave_thinker/evaluate_traces.py",
            str(output_dir),
            "--output",
            str(metrics_path),
            "--html-output",
            str(html_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote metrics" in result.stdout
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["num_traces"] == 2
    assert html_path.exists()


def _write_trace_fixture(tmp_path: Path) -> Path:
    output_dir = tmp_path / "eval"
    image_path = output_dir / "mug" / "final.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image")
    _write_json(
        output_dir / "mug" / "trace.json",
        {
            "instruction": "draw a mug",
            "success": True,
            "metadata": {
                "prompt_set_id": "mug",
                "prompt_set_index": 0,
                "prompt_set_metadata": {
                    "category": "product",
                },
            },
            "final_image": {
                "prompt": "refined mug",
                "file_path": str(image_path),
                "inference_time_s": 0.4,
                "metadata": {},
            },
            "attempts": [
                {
                    "step_index": 0,
                    "attempt_index": 0,
                    "prompt": "draw a mug",
                    "generated": {
                        "prompt": "draw a mug",
                        "file_path": str(output_dir / "mug" / "attempt0.png"),
                        "inference_time_s": 0.5,
                        "metadata": {},
                    },
                    "decision": {
                        "success": False,
                        "refine_prompt": "refined mug",
                        "reason": "needs refinement",
                        "metadata": {},
                    },
                },
                {
                    "step_index": 0,
                    "attempt_index": 1,
                    "prompt": "refined mug",
                    "generated": {
                        "prompt": "refined mug",
                        "file_path": str(image_path),
                        "inference_time_s": 0.7,
                        "metadata": {},
                    },
                    "decision": {
                        "success": True,
                        "refine_prompt": None,
                        "reason": None,
                        "metadata": {},
                    },
                },
            ],
        },
    )
    _write_json(
        output_dir / "poster" / "trace.json",
        {
            "instruction": "draw a poster",
            "success": False,
            "metadata": {
                "prompt_set_id": "poster",
                "prompt_set_index": 1,
                "failed_step_index": 0,
                "prompt_set_metadata": {
                    "category": "poster",
                },
            },
            "final_image": None,
            "attempts": [
                {
                    "step_index": 0,
                    "attempt_index": 0,
                    "prompt": "draw a poster",
                    "generated": {
                        "prompt": "draw a poster",
                        "file_path": str(output_dir / "poster" / "attempt0.png"),
                        "inference_time_s": 0.3,
                        "metadata": {},
                    },
                    "decision": {
                        "success": False,
                        "refine_prompt": None,
                        "reason": "critic rejected final attempt",
                        "metadata": {},
                    },
                },
            ],
        },
    )
    _write_json(
        output_dir / "summary.json",
        {
            "num_samples": 2,
            "num_success": 1,
            "results": [
                {
                    "sample_id": "mug",
                    "trace_path": str(output_dir / "mug" / "trace.json"),
                },
                {
                    "sample_id": "poster",
                    "trace_path": str(output_dir / "poster" / "trace.json"),
                },
            ],
        },
    )
    return output_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
