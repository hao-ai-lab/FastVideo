# SPDX-License-Identifier: Apache-2.0
"""VideoScore2 reward service for PromptRL.

Runs as a pluggable service (canonically on a ninth GPU).  Training
ranks POST their generated videos to ``/v1/rewards:score`` with group
metadata; the service collects complete groups, evaluates each video
with the official VideoScore2 inference procedure, and returns a
composite score plus visual-quality, text-alignment, and
physical-consistency details. Completed request IDs are cached so
client retries are idempotent. Videos transfer over HTTP only — no
shared filesystem.

Endpoints:

* ``GET /healthz``
* ``POST /v1/rewards:score`` (multipart form; fields ``group_id``,
  ``request_id``, ``sample_id``, ``expected_group_size``,
  ``original_prompt``, ``reward_tag``, ``fps`` + ``video`` file)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from fastvideo.logger import init_logger

logger = init_logger(__name__)

#: Judge payload keys produced per scored sample.
COMPONENT_KEYS = ("visual_quality", "text_alignment", "physical_consistency")


def _read_video_with_pyav(
    path: str,
    *,
    start_pts: float = 0.0,
    end_pts: float | None = None,
    pts_unit: str = "sec",
    output_format: str = "THWC",
) -> tuple[Any, Any, dict[str, float]]:
    """PyAV implementation of the removed ``torchvision.io.read_video`` API."""
    import av
    import numpy as np
    import torch

    if pts_unit != "sec":
        raise ValueError("PromptRL's PyAV video fallback only supports pts_unit='sec'")
    if output_format not in {"THWC", "TCHW"}:
        raise ValueError(f"Unsupported video output format: {output_format!r}")

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate or stream.base_rate or 0.0)
        frames = []
        for frame in container.decode(video=0):
            timestamp = frame.time
            if timestamp is not None and timestamp < float(start_pts):
                continue
            if end_pts is not None and timestamp is not None and timestamp > float(end_pts):
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"No video frames decoded from {path}")

    video = torch.from_numpy(np.stack(frames))
    if output_format == "TCHW":
        video = video.permute(0, 3, 1, 2)
    audio = torch.empty((1, 0), dtype=torch.float32)
    return video, audio, {"video_fps": video_fps}


def _install_torchvision_read_video_fallback() -> None:
    """Polyfill torchvision >=0.26 for qwen-vl-utils' video reader."""
    import torchvision.io

    if callable(getattr(torchvision.io, "read_video", None)):
        return
    torchvision.io.read_video = _read_video_with_pyav  # type: ignore[attr-defined]
    logger.warning(
        "torchvision.io.read_video is unavailable; using the PyAV compatibility decoder",
    )


class RewardRequestError(RuntimeError):
    """Client-facing request failure carrying an HTTP status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class RewardTimeoutError(RewardRequestError):
    def __init__(self, message: str) -> None:
        super().__init__(504, message)


@runtime_checkable
class VideoJudge(Protocol):
    """Group video scoring backend (e.g. VideoScore2)."""

    def score_batch(
        self,
        videos: Sequence[bytes],
        prompts: Sequence[str],
        *,
        fps: int,
    ) -> list[dict[str, float]]:
        """Score *videos* against *prompts*.

        Returns one dict per video with ``composite`` plus the
        :data:`COMPONENT_KEYS` component scores.
        """
        ...


@dataclass(slots=True)
class _SubmittedSample:
    sample_id: str
    original_prompt: str
    reward_tag: str
    fps: int
    video_bytes: bytes


@dataclass(slots=True)
class _GroupState:
    expected: int
    samples: dict[str, _SubmittedSample] = field(default_factory=dict)
    scoring_started: bool = False
    payloads: dict[str, dict[str, Any]] | None = None
    error: BaseException | None = None



class RewardGroupCoordinator:
    """Thread-safe group batching + idempotency core (framework-free)."""

    def __init__(
        self,
        judge: VideoJudge,
        *,
        max_wait_sec: float = 300.0,
        max_completed_requests: int = 1024,
    ) -> None:
        self._judge = judge
        self._max_wait_sec = float(max_wait_sec)
        self._max_completed = int(max_completed_requests)
        self._cond = threading.Condition()
        self._groups: dict[tuple[str, str], _GroupState] = {}
        self._completed: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()

    def submit(
        self,
        *,
        group_id: str,
        request_id: str,
        sample_id: str,
        expected_group_size: int,
        original_prompt: str,
        reward_tag: str,
        fps: int,
        video_bytes: bytes,
    ) -> dict[str, Any]:
        """Submit one sample; block until the whole group is scored."""
        if expected_group_size <= 0:
            raise RewardRequestError(400, "expected_group_size must be positive")
        key = (str(group_id), str(request_id))
        with self._cond:
            cached = self._completed.get(str(request_id))
            if cached is not None:
                payload = cached.get(sample_id)
                if payload is None:
                    raise RewardRequestError(
                        409, f"request_id {request_id!r} completed without "
                        f"sample_id {sample_id!r}")
                return payload

            state = self._groups.get(key)
            if state is None:
                state = _GroupState(expected=int(expected_group_size))
                self._groups[key] = state
            elif state.expected != int(expected_group_size):
                raise RewardRequestError(400, f"expected_group_size mismatch for group "
                                         f"{group_id!r}: {state.expected} vs {expected_group_size}")
            elif state.scoring_started and sample_id not in state.samples:
                raise RewardRequestError(409, f"group {group_id!r} request {request_id!r} "
                                         f"is already scoring and sample_id {sample_id!r} "
                                         "was not part of the submitted group")

            if sample_id not in state.samples:
                state.samples[sample_id] = _SubmittedSample(
                    sample_id=sample_id,
                    original_prompt=original_prompt,
                    reward_tag=reward_tag,
                    fps=int(fps),
                    video_bytes=video_bytes,
                )

            complete_now = (not state.scoring_started and len(state.samples) == state.expected)
            if complete_now:
                state.scoring_started = True

        # Score outside the lock so other groups proceed concurrently.
        if complete_now:
            self._score_group(key, state)

        with self._cond:
            ready = self._cond.wait_for(
                lambda: state.payloads is not None or state.error is not None,
                timeout=self._max_wait_sec,
            )
            if not ready:
                raise RewardTimeoutError(f"Timed out after {self._max_wait_sec:.0f}s waiting "
                                         f"for group {group_id!r} to complete scoring")
            if state.error is not None:
                raise RewardRequestError(500, f"Video judge failed for group "
                                         f"{group_id!r}: {state.error}")
            assert state.payloads is not None
            return state.payloads[sample_id]

    # ------------------------------------------------------------------

    def _score_group(self, key: tuple[str, str], state: _GroupState) -> None:
        group_id, request_id = key
        try:
            # Deterministic sample order: sorted by sample_id so every
            # group scores identically regardless of arrival order.
            ordered = [state.samples[k] for k in sorted(state.samples)]
            scores = self._judge.score_batch(
                [s.video_bytes for s in ordered],
                [s.original_prompt for s in ordered],
                fps=ordered[0].fps,
            )
            if len(scores) != len(ordered):
                raise RewardRequestError(500, f"Judge returned {len(scores)} score(s) "
                                         f"for {len(ordered)} sample(s)")
            payloads: dict[str, dict[str, Any]] = {}
            for sample, score_row in zip(ordered, scores, strict=True):
                composite = float(score_row["composite"])
                detail_keys = (*COMPONENT_KEYS, "judge_fallback")
                details = {k: float(score_row[k]) for k in detail_keys if k in score_row}
                payloads[sample.sample_id] = {
                    "request_id": request_id,
                    "sample_id": sample.sample_id,
                    "score": composite,
                    "details": details,
                }
            with self._cond:
                state.payloads = payloads
                self._cache_completed(request_id, payloads)
                self._groups.pop(key, None)
                self._cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - propagated to waiters
            logger.warning("Scoring failed for group %s request %s: %s", group_id, request_id, exc)
            with self._cond:
                state.error = exc
                self._groups.pop(key, None)
                self._cond.notify_all()

    def _cache_completed(self, request_id: str, payloads: dict[str, dict[str, Any]]) -> None:
        self._completed[request_id] = payloads
        self._completed.move_to_end(request_id)
        while len(self._completed) > self._max_completed:
            self._completed.popitem(last=False)


def create_reward_app(
    judge: VideoJudge,
    *,
    max_wait_sec: float = 300.0,
) -> Any:
    """Build the FastAPI reward service app around *judge*."""
    coordinator = RewardGroupCoordinator(judge, max_wait_sec=max_wait_sec)
    app = FastAPI(title="fastvideo-promptrl-reward", version="1")
    app.state.coordinator = coordinator

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/rewards:score")
    async def rewards_score(
        group_id: str = Form(...),
        request_id: str = Form(...),
        sample_id: str = Form(...),
        expected_group_size: int = Form(...),
        original_prompt: str = Form(...),
        reward_tag: str = Form(""),
        fps: int = Form(16),
        video: UploadFile = File(...),  # noqa: B008 - FastAPI dependency idiom
    ) -> dict[str, Any]:
        video_bytes = await video.read()
        try:
            # ``submit`` intentionally waits for all group members and may
            # execute the synchronous GPU judge. Keep both operations off the
            # ASGI event loop so the remaining uploads can be accepted.
            return await run_in_threadpool(
                coordinator.submit,
                group_id=group_id,
                request_id=request_id,
                sample_id=sample_id,
                expected_group_size=int(expected_group_size),
                original_prompt=original_prompt,
                reward_tag=reward_tag,
                fps=int(fps),
                video_bytes=video_bytes,
            )
        except RewardRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return app


_VIDEOSCORE2_QUERY_TEMPLATE = """
You are an expert for evaluating AI-generated videos from three dimensions:
(1) visual quality – clarity, smoothness, artifacts;
(2) text-to-video alignment – fidelity to the prompt;
(3) physical/common-sense consistency – naturalness and physics plausibility.

Video prompt: {prompt}

Please output in this format:
visual quality: <v_score>;
text-to-video alignment: <t_score>,
physical/common-sense consistency: <p_score>
""".strip()

_VIDEOSCORE2_LABELS = {
    "visual_quality": "visual quality:",
    "text_alignment": "text-to-video alignment:",
    "physical_consistency": "physical/common-sense consistency:",
}

_VIDEOSCORE2_LABEL_ALIASES = {
    "visual_quality": ("visual quality",),
    "text_alignment": (
        "text-to-video alignment",
        "text to video alignment",
        "text alignment",
    ),
    "physical_consistency": (
        "physical/common-sense consistency",
        "physical consistency / common-sense",
        "physical consistency",
        "common-sense consistency",
    ),
}


def _videoscore2_score_patterns(label_base: str) -> tuple[str, ...]:
    """Accepted VideoScore2 score renderings for one dimension label."""
    import re

    label = re.escape(label_base)
    value = r"([1-5])(?:\s*/\s*5)?"
    return (
        # Requested/final forms: ``Visual Quality: 3`` or
        # ``Ground-truth scores: Visual Quality 3``. Formatting characters
        # cover Markdown output such as ``**Visual Quality:** 3``.
        label + r"[\s*_`-]*[:=]?[\s*_`-]*" + value + r"\b",
        # Numbered rubric form:
        # ``(1) visual quality – clarity, smoothness, artifacts: 3``.
        r"(?:\(\d+\)\s*)?"
        + label
        + r"(?:\s*[-–—]\s*[^:\n]+)?\s*:[\s*_`-]*"
        + value
        + r"\b",
        # Narrative forms: ``visual quality is moderate (3/5)`` and
        # ``visual quality is moderate (score 3)``.
        label
        + r"[^\n]{0,500}?\(\s*(?:score(?:\s+of)?\s*[:=]?\s*)?"
        + value
        + r"\s*\)",
        # Narrative forms without parentheses: ``alignment ... score of 4``.
        label
        + r"[^\n]{0,500}?\bscore(?:\s+of)?\s*[:=]?\s*"
        + value
        + r"\b",
    )


def _videoscore2_aliases(prompt_text: str) -> tuple[str, ...]:
    normalized = prompt_text.rstrip(":").lower()
    for component, label in _VIDEOSCORE2_LABELS.items():
        if normalized == label.rstrip(":").lower():
            return _VIDEOSCORE2_LABEL_ALIASES[component]
    return (prompt_text.rstrip(":"),)


def _find_score_token_index(
    prompt_text: str,
    tokenizer: Any,
    generated_token_ids: Sequence[int],
) -> int:
    """Locate the generated numeric score token after *prompt_text*.

    This follows the official VideoScore2 inference implementation.
    """
    import re

    def decode(token_ids: Sequence[int]) -> str:
        try:
            return str(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ))
        except TypeError:
            # Lightweight injected tokenizers and older implementations may
            # not expose the cleanup keyword.
            return str(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                ))

    generated_text = decode(generated_token_ids)
    matches = [
        match
        for alias in _videoscore2_aliases(prompt_text)
        for pattern in _videoscore2_score_patterns(alias)
        for match in re.finditer(pattern, generated_text, flags=re.IGNORECASE)
    ]
    if not matches:
        return -1
    match = max(matches, key=lambda candidate: candidate.start())
    target_end = match.end(1)
    target = generated_text[:target_end]
    for index in range(len(generated_token_ids)):
        partial = decode(generated_token_ids[:index + 1])
        if partial == target:
            return index
        # Transformers 5 tokenizers can decode the newest token together with
        # punctuation or whitespace, so an exact string boundary is not
        # guaranteed even though this is the score-producing token. Select the
        # first token whose decoded prefix contains the complete target.
        if len(partial) >= target_end and partial[:target_end] == target:
            return index
    return -1


def _confidence_weighted_score(
    hard_value: int | None,
    token_index: int,
    generation_scores: Sequence[Any],
    tokenizer: Any,
) -> float | None:
    """Compute the official confidence-weighted score over tokens 1–5."""
    if hard_value is None or token_index < 0 or token_index >= len(generation_scores):
        return None

    import torch

    logits = generation_scores[token_index][0]
    values: list[int] = []
    token_ids: list[int] = []
    for value in range(1, 6):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) == 1:
            values.append(value)
            token_ids.append(int(encoded[0]))
    if not token_ids:
        return None

    # The official implementation applies log_softmax over the vocabulary,
    # then renormalizes the five score-token probabilities. The vocabulary
    # denominator cancels, so a softmax over these five logits is equivalent.
    probabilities = torch.softmax(
        logits[token_ids].float(),
        dim=0,
    )
    best_index = int(probabilities.argmax().item())
    return round(
        float(values[best_index]) * float(probabilities[best_index].item()),
        4,
    )


def parse_videoscore2_output(
    output_text: str,
    *,
    generated_token_ids: Sequence[int],
    generation_scores: Sequence[Any],
    tokenizer: Any,
) -> dict[str, float]:
    """Parse one official VideoScore2 generation into reward components."""
    import re

    hard_values: dict[str, int | None] = {}
    for component, label in _VIDEOSCORE2_LABELS.items():
        # The model sometimes wraps its requested final labels in Markdown,
        # yielding text such as ``Visual Quality:** 1``. The official regex
        # accepts only whitespace after the colon, even though this response
        # is semantically identical. Select the last formatted label so
        # explanatory headings in the chain of thought cannot shadow the
        # requested final summary.
        matches = [
            match
            for alias in _VIDEOSCORE2_LABEL_ALIASES[component]
            for pattern in _videoscore2_score_patterns(alias)
            for match in re.finditer(
                pattern,
                output_text,
                flags=re.IGNORECASE,
            )
        ]
        hard_values[component] = (
            int(max(matches, key=lambda candidate: candidate.start()).group(1))
            if matches
            else None
        )
    parsed: dict[str, float] = {}
    for component, label in _VIDEOSCORE2_LABELS.items():
        token_index = _find_score_token_index(
            label,
            tokenizer,
            generated_token_ids,
        )
        score = _confidence_weighted_score(
            hard_values[component],
            token_index,
            generation_scores,
            tokenizer,
        )
        if score is None:
            try:
                decoded_generation = tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                decoded_generation = tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=False,
                )
            raise RewardRequestError(
                500,
                f"VideoScore2 output missing a usable {component} score "
                f"(hard_value={hard_values[component]!r}, "
                f"token_index={token_index}): output={output_text[:2000]!r}; "
                f"decoded_generation={str(decoded_generation)[:2000]!r}",
            )
        parsed[component] = score
    parsed["composite"] = (
        sum(parsed[key] for key in COMPONENT_KEYS) / len(COMPONENT_KEYS)
    )
    return parsed


class VideoScore2Judge:
    """Official VideoScore2 inference wrapped as a group reward judge.

    The upstream procedure evaluates one video at a time at 2 FPS and derives
    confidence-weighted dimension scores from generation logits. Model loading
    is lazy so service startup and unit tests do not touch Hugging Face.
    """

    def __init__(
        self,
        model_id: str = "TIGER-Lab/VideoScore2",
        *,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        infer_fps: float = 2.0,
        temperature: float = 0.7,
        seed: int | None = 0,
        parse_failure_score: float | None = 3.0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.infer_fps = float(infer_fps)
        self.temperature = float(temperature)
        self.seed = seed
        self.parse_failure_score = (
            None if parse_failure_score is None else float(parse_failure_score)
        )
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor, AutoTokenizer

        # AutoModelForVision2Seq is the class in the official transformers
        # 4.53.2 example. Transformers 5 renamed it to
        # AutoModelForImageTextToText, which FastVideo currently uses.
        model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            model_cls = transformers.AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._tokenizer = (
            getattr(self._processor, "tokenizer", None)
            or AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=False,
            )
        )
        self._model = model_cls.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self._model.eval()

    def _score_one(
        self,
        video_path: str,
        prompt: str,
    ) -> dict[str, float]:
        import contextlib

        import torch

        _install_torchvision_read_video_fallback()
        from qwen_vl_utils import process_vision_info

        assert self._model is not None
        assert self._processor is not None
        assert self._tokenizer is not None

        messages = [{
            "role": "user",
            "content": [{
                "type": "video",
                "video": video_path,
                "fps": self.infer_fps,
            }, {
                "type": "text",
                "text": _VIDEOSCORE2_QUERY_TEMPLATE.format(prompt=prompt),
            }],
        }]
        rendered = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            fps=self.infer_fps,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        if self.seed is None:
            rng_context = contextlib.nullcontext()
        elif torch.device(self.device).type == "cuda":
            rng_context = torch.random.fork_rng(devices=[torch.device(self.device)])
        else:
            rng_context = torch.random.fork_rng(devices=[])
        with rng_context:
            if self.seed is not None:
                torch.manual_seed(int(self.seed))
                if torch.device(self.device).type == "cuda":
                    torch.cuda.manual_seed_all(int(self.seed))
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=self.temperature,
                )

        input_length = int(inputs["input_ids"].shape[1])
        generated_ids = generated.sequences[0, input_length:].tolist()
        output_text = self._processor.batch_decode(
            generated.sequences[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_videoscore2_output(
            output_text,
            generated_token_ids=generated_ids,
            generation_scores=generated.scores,
            tokenizer=self._tokenizer,
        )

    def score_batch(
        self,
        videos: Sequence[bytes],
        prompts: Sequence[str],
        *,
        fps: int,
    ) -> list[dict[str, float]]:
        import tempfile
        from pathlib import Path

        del fps  # Official VideoScore2 inference samples each video at 2 FPS.
        if len(videos) != len(prompts):
            raise RewardRequestError(
                400,
                f"Video/prompt cardinality mismatch: {len(videos)} vs {len(prompts)}",
            )
        self._load()
        with tempfile.TemporaryDirectory(prefix="promptrl_reward_") as tmpdir:
            results: list[dict[str, float]] = []
            try:
                for idx, (video_bytes, prompt) in enumerate(zip(videos, prompts, strict=True)):
                    video_path = Path(tmpdir) / f"video_{idx}.mp4"
                    video_path.write_bytes(video_bytes)
                    result = self._score_one(str(video_path), prompt)
                    result["judge_fallback"] = 0.0
                    results.append(result)
            except RewardRequestError as exc:
                is_parse_failure = (
                    exc.status_code == 500
                    and "VideoScore2 output missing a usable" in str(exc)
                )
                if not is_parse_failure or self.parse_failure_score is None:
                    raise
                # A malformed generation is a judge-side failure, not evidence
                # that one video is worse. Give the entire comparison group the
                # same neutral VideoScore2 value so this component contributes
                # zero group-relative advantage, while format reward can still
                # train and the long-running job remains healthy.
                logger.warning(
                    "VideoScore2 output was unscorable; using neutral score %.2f "
                    "for all %d samples in the group: %s",
                    self.parse_failure_score,
                    len(videos),
                    exc,
                )
                neutral = {
                    key: self.parse_failure_score
                    for key in COMPONENT_KEYS
                }
                neutral["composite"] = self.parse_failure_score
                neutral["judge_fallback"] = 1.0
                results = [dict(neutral) for _ in videos]
        return results


# Backward-compatible name for early PromptRL configs and imports.
RubricVideoScore2Judge = VideoScore2Judge


def parse_judge_scores(output_text: str, *, expected: int) -> list[dict[str, float]]:
    """Parse the judge JSON array into per-sample score dicts."""
    import json
    import re

    match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if match is None:
        raise RewardRequestError(500, f"Judge output missing JSON array: {output_text[:200]!r}")
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RewardRequestError(500, f"Judge output is not valid JSON: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != expected:
        raise RewardRequestError(500, f"Judge returned {len(rows) if isinstance(rows, list) else 'non-list'} "
                                 f"score row(s), expected {expected}")
    results: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RewardRequestError(500, f"Judge score row is not an object: {row!r}")
        try:
            parsed = {key: float(row[key]) for key in COMPONENT_KEYS}
        except (KeyError, TypeError, ValueError) as exc:
            raise RewardRequestError(500, f"Judge score row missing components: {row!r}") from exc
        parsed["composite"] = sum(parsed[key] for key in COMPONENT_KEYS) / len(COMPONENT_KEYS)
        results.append(parsed)
    return results
