# SPDX-License-Identifier: Apache-2.0
"""Serialization helpers for interleaved generation traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastvideo.workflows.interleave_thinker.schema import (
    CriticDecision,
    GeneratedImage,
    InterleaveAttempt,
    InterleaveTrace,
)


def trace_to_dict(
    trace: InterleaveTrace,
    *,
    include_images: bool = False,
) -> dict[str, Any]:
    return {
        "instruction": trace.instruction,
        "success": trace.success,
        "final_image": _generated_image_to_dict(
            trace.final_image,
            include_images=include_images,
        ),
        "attempts": [
            _attempt_to_dict(
                attempt,
                include_images=include_images,
            ) for attempt in trace.attempts
        ],
        "metadata": dict(trace.metadata),
    }


def save_trace(
    trace: InterleaveTrace,
    path: str | Path,
    *,
    include_images: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            trace_to_dict(
                trace,
                include_images=include_images,
            ),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _attempt_to_dict(
    attempt: InterleaveAttempt,
    *,
    include_images: bool,
) -> dict[str, Any]:
    return {
        "step_index": attempt.step_index,
        "attempt_index": attempt.attempt_index,
        "prompt": attempt.prompt,
        "generated": _generated_image_to_dict(
            attempt.generated,
            include_images=include_images,
        ),
        "decision": _critic_decision_to_dict(attempt.decision),
    }


def _generated_image_to_dict(
    image: GeneratedImage | None,
    *,
    include_images: bool,
) -> dict[str, Any] | None:
    if image is None:
        return None
    result = {
        "prompt": image.prompt,
        "file_path": image.file_path,
        "inference_time_s": image.inference_time_s,
        "metadata": dict(image.metadata),
    }
    if include_images:
        result["image_base64"] = image.image_base64
    return result


def _critic_decision_to_dict(decision: CriticDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "success": decision.success,
        "refine_prompt": decision.refine_prompt,
        "reason": decision.reason,
        "metadata": dict(decision.metadata),
    }
