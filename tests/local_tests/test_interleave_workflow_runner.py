# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from apps.interleave_thinker import (
    GeneratedImage,
    InterleaveEditRequest,
    InterleavePromptItem,
    load_interleave_prompt_set,
    load_interleave_run_config,
    resolve_interleave_instruction,
    run_interleave_prompt_set,
    run_interleave_config,
)
from apps.interleave_thinker.orchestrator import SinglePromptPlanner
from apps.interleave_thinker.schema import PlannerInput


def test_interleave_run_config_loads_prompt_and_request_defaults(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path)

    config = load_interleave_run_config(str(config_path))

    assert resolve_interleave_instruction(config) == "draw a red mug"
    assert config.generator is not None
    assert config.generator.model_path == "black-forest-labs/FLUX.2-klein-4B"
    assert config.request.sampling.width == 512
    assert config.request.sampling.num_inference_steps == 4


def test_interleave_run_config_accepts_runtime_fields_and_dotted_overrides(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path)

    config = load_interleave_run_config(
        str(config_path),
        prompt="draw a blue mug",
        output_dir=str(tmp_path / "override_outputs"),
        trace_path=str(tmp_path / "trace_override.json"),
        overrides=[
            "--request.sampling.seed",
            "99",
            "--planner.max-attempts-per-step",
            "3",
        ],
    )

    assert resolve_interleave_instruction(config) == "draw a blue mug"
    assert config.interleave.output_dir == str(tmp_path / "override_outputs")
    assert config.interleave.trace_path == str(tmp_path / "trace_override.json")
    assert config.request.sampling.seed == 99
    assert config.planner.max_attempts_per_step == 3


def test_interleave_run_config_rejects_unknown_override_prefix(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path)

    with pytest.raises(ValueError, match="Unsupported override path"):
        load_interleave_run_config(
            str(config_path),
            overrides=["--server.port", "9000"],
        )


def test_interleave_eval_config_can_omit_single_instruction(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path, include_prompt=False)

    config = load_interleave_run_config(
        str(config_path),
        require_instruction=False,
    )

    with pytest.raises(ValueError, match="requires interleave.instruction"):
        resolve_interleave_instruction(config)


def test_run_interleave_config_with_injected_backend_writes_trace(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path)
    config = load_interleave_run_config(str(config_path))
    backend = _FakeImageBackend(tmp_path / "generated.png")

    result = run_interleave_config(config, image_backend=backend)

    assert result.trace.success is True
    assert result.trace.final_image is not None
    assert result.trace.final_image.file_path == str(tmp_path / "generated.png")
    assert Path(result.trace_path).exists()
    trace_text = Path(result.trace_path).read_text(encoding="utf-8")
    assert "draw a red mug" in trace_text
    assert "image_base64" not in trace_text
    assert backend.requests[0].prompt == "draw a red mug"


def test_load_interleave_prompt_set_accepts_jsonl_rows(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"id": "mug", "prompt": "draw a mug", "metadata": {"split": "smoke"}, "difficulty": "easy"}\n'
        '"draw a kettle"\n',
        encoding="utf-8",
    )

    items = load_interleave_prompt_set(prompt_path)

    assert [item.sample_id for item in items] == ["mug", "sample_00001"]
    assert items[0].instruction == "draw a mug"
    assert items[0].metadata == {
        "split": "smoke",
        "difficulty": "easy",
    }
    assert items[1].instruction == "draw a kettle"


def test_run_interleave_prompt_set_writes_traces_and_summary(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path, include_prompt=False)
    config = load_interleave_run_config(
        str(config_path),
        require_instruction=False,
    )
    backend = _FakeImageBackend(tmp_path / "generated.png")
    items = [
        InterleavePromptItem(sample_id="red/mug", instruction="draw a red mug"),
        InterleavePromptItem(sample_id="blue mug", instruction="draw a blue mug"),
    ]

    summary = run_interleave_prompt_set(
        config,
        items,
        output_dir=str(tmp_path / "eval"),
        image_backend=backend,
    )

    assert summary.num_samples == 2
    assert summary.num_success == 2
    assert summary.success_rate == 1.0
    assert summary.total_attempts == 2
    assert [request.prompt for request in backend.requests] == ["draw a red mug", "draw a blue mug"]
    assert Path(summary.summary_path).exists()
    assert Path(summary.results[0].trace_path).exists()
    assert "red_mug" in summary.results[0].trace_path
    summary_text = Path(summary.summary_path).read_text(encoding="utf-8")
    assert "draw a blue mug" in summary_text


def test_run_interleave_prompt_set_resume_uses_existing_trace(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path, include_prompt=False)
    config = load_interleave_run_config(
        str(config_path),
        require_instruction=False,
    )
    item = InterleavePromptItem(sample_id="mug", instruction="draw a mug")
    backend = _FakeImageBackend(tmp_path / "generated.png")
    first = run_interleave_prompt_set(
        config,
        [item],
        output_dir=str(tmp_path / "eval"),
        image_backend=backend,
    )

    resumed = run_interleave_prompt_set(
        config,
        [item],
        output_dir=str(tmp_path / "eval"),
        image_backend=_FakeImageBackend(tmp_path / "unused.png"),
        resume=True,
    )

    assert first.num_resumed == 0
    assert resumed.num_resumed == 1
    assert resumed.results[0].resumed is True
    assert resumed.results[0].trace_path == first.results[0].trace_path


def test_single_prompt_planner_uses_configured_attempt_count() -> None:
    planner = SinglePromptPlanner(max_attempts=4)

    steps = list(planner.plan(PlannerInput(instruction="draw a red mug")))

    assert len(steps) == 1
    assert steps[0].max_attempts == 4


class _FakeImageBackend:

    def __init__(self, output_path: Path) -> None:
        self.requests: list[InterleaveEditRequest] = []
        self.output_path = output_path

    def generate(
        self,
        request: InterleaveEditRequest,
        *,
        request_id: str | None = None,
    ) -> GeneratedImage:
        del request_id
        self.requests.append(request)
        self.output_path.write_bytes(b"fake-image")
        return GeneratedImage(
            prompt=request.prompt,
            image_base64=base64.b64encode(b"fake-image").decode("utf-8"),
            file_path=str(self.output_path),
            metadata={"backend": "fake"},
        )


def _write_run_config(tmp_path: Path, *, include_prompt: bool = True) -> Path:
    output_path = tmp_path / "generated.png"
    config_path = tmp_path / "interleave_run.yaml"
    request_prompt = "  prompt: draw a red mug\n" if include_prompt else ""
    config_path.write_text(
        f"""
generator:
  model_path: black-forest-labs/FLUX.2-klein-4B
  engine:
    num_gpus: 1
  pipeline:
    workload_type: t2i

image_backend:
  kind: fastvideo

planner:
  max_attempts_per_step: 1

critic:
  kind: accept_all

interleave:
  output_dir: {tmp_path}
  trace_path: {tmp_path / "trace.json"}

request:
{request_prompt}  extensions:
    test_output_path: {output_path}
  sampling:
    width: 512
    height: 512
    seed: 7
    num_inference_steps: 4
""",
        encoding="utf-8",
    )
    return config_path
