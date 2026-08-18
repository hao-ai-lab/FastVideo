# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for local prompt enrichment."""

from __future__ import annotations

import json

from fastvideo.mlx_runtime.prompt_enhance import (
    DEFAULT_ENHANCE_SYSTEM_PROMPT,
    enhance_prompt,
    enhance_prompt_template,
    enhance_result_as_metrics,
    load_or_enhance_prompt,
)


def test_template_adds_cinematic_cues_to_thin_prompt() -> None:
    raw = "A red fox in the snow"
    out = enhance_prompt_template(raw)
    assert raw in out
    lower = out.lower()
    assert "lens" in lower or "camera" in lower or "cinematic" in lower
    assert "light" in lower
    assert out.endswith(".")


def test_template_is_idempotent_on_rich_prompts() -> None:
    rich = (
        "A red fox trotting through a snowy pine forest at golden hour, "
        "cinematic wide shot on a 35mm lens, soft volumetric lighting, "
        "gentle camera dolly, highly detailed film grain and color graded."
    )
    assert enhance_prompt_template(rich) == " ".join(rich.split())


def test_enhance_prompt_template_backend() -> None:
    result = enhance_prompt("a paper boat on a stream", backend="template")
    assert result.backend == "template"
    assert result.changed
    assert result.original == "a paper boat on a stream"
    assert "paper boat" in result.enhanced.lower()


def test_enhance_prompt_rejects_empty() -> None:
    try:
        enhance_prompt("   ", backend="template")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_load_or_enhance_prompt_caches(tmp_path) -> None:
    result1 = load_or_enhance_prompt(
        "a lantern in the fog",
        backend="template",
        cache=True,
        cache_dir=tmp_path,
    )
    assert result1.backend == "template"
    cached_files = list(tmp_path.glob("*.json"))
    assert len(cached_files) == 1
    payload = json.loads(cached_files[0].read_text())
    assert payload["enhanced"] == result1.enhanced

    result2 = load_or_enhance_prompt(
        "a lantern in the fog",
        backend="template",
        cache=True,
        cache_dir=tmp_path,
    )
    assert result2.enhanced == result1.enhanced
    assert result2.backend == "cache"
    assert result2.elapsed_s == 0.0


def test_enhance_result_as_metrics_none() -> None:
    metrics = enhance_result_as_metrics(None)
    assert metrics["enhance_prompt"] is False
    assert metrics["prompt_enhanced"] is None


def test_default_system_prompt_matches_streaming_contract() -> None:
    # Lockstep with fastvideo/entrypoints/streaming/prompt/enhancer.py
    assert "cinematic video generation" in DEFAULT_ENHANCE_SYSTEM_PROMPT.lower()
    assert "just the enhanced prompt" in DEFAULT_ENHANCE_SYSTEM_PROMPT.lower()
