# SPDX-License-Identifier: Apache-2.0
"""Reward service batching/idempotency + provider retry/validation tests."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from fastvideo.train.methods.rl.promptrl.rewards import (
    COMPONENT_KEYS,
    HttpRewardProvider,
    RewardRequestError,
    RewardResult,
    RewardSample,
    RewardServiceError,
    create_reward_app,
    parse_judge_scores,
    parse_videoscore2_output,
    validate_reward_results,
)
from fastvideo.train.methods.rl.promptrl.rewards.provider import _parse_reward_payload
from fastvideo.train.methods.rl.promptrl.rewards.service import (
    _read_video_with_pyav,
)
from fastvideo.train.methods.rl.promptrl.video_io import encode_video_bytes


class FakeJudge:
    """Deterministic fake VideoScore2 judge recording its batches."""

    def __init__(self):
        self.calls: list[tuple[int, tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def score_batch(self, videos, prompts, *, fps):
        with self._lock:
            self.calls.append((len(videos), tuple(prompts)))
        results = []
        for idx, prompt in enumerate(prompts):
            base = 3.0 + 0.1 * idx + (0.5 if "good" in prompt else 0.0)
            results.append({
                "visual_quality": base,
                "text_alignment": base + 0.1,
                "physical_consistency": base - 0.1,
                "composite": base,
            })
        return results


class SlowJudge(FakeJudge):
    """Fake judge that blocks while a duplicate in-flight submit attaches."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def score_batch(self, videos, prompts, *, fps):
        self.started.set()
        assert self.release.wait(timeout=5.0)
        return super().score_batch(videos, prompts, fps=fps)


def _sample(sample_id: str, *, group_id="g", request_id="r", expected=2,
            prompt="a good cat") -> RewardSample:
    return RewardSample(
        group_id=group_id,
        request_id=request_id,
        sample_id=sample_id,
        expected_group_size=expected,
        original_prompt=prompt,
        reward_tag="tag",
        fps=16,
        video_bytes=b"fake-mp4-" + sample_id.encode(),
    )


def _post(client: TestClient, sample: RewardSample):
    return client.post(
        "/v1/rewards:score",
        data={
            "group_id": sample.group_id,
            "request_id": sample.request_id,
            "sample_id": sample.sample_id,
            "expected_group_size": str(sample.expected_group_size),
            "original_prompt": sample.original_prompt,
            "reward_tag": sample.reward_tag,
            "fps": str(sample.fps),
        },
        files={"video": ("sample.mp4", sample.video_bytes, "video/mp4")},
    )


class TestRewardService:
    def test_healthz(self):
        app = create_reward_app(FakeJudge())
        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_group_batched_into_single_judge_call(self):
        judge = FakeJudge()
        client = TestClient(create_reward_app(judge))
        responses: dict[str, dict] = {}
        errors: list[BaseException] = []

        def worker(sample_id: str):
            try:
                responses[sample_id] = _post(client, _sample(sample_id)).json()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"slot-{i}", )) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors
        # The coordinator invokes its group judge exactly once.
        assert judge.calls == [(2, ("a good cat", "a good cat"))]
        assert set(responses) == {"slot-0", "slot-1"}
        for sample_id, payload in responses.items():
            assert payload["sample_id"] == sample_id
            assert payload["request_id"] == "r"
            assert payload["score"] > 0
            for key in COMPONENT_KEYS:
                assert key in payload["details"]

    def test_async_server_accepts_group_members_concurrently(self):
        """A waiting upload must not block the ASGI event loop."""
        judge = FakeJudge()
        app = create_reward_app(judge, max_wait_sec=1.0)

        async def exercise():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://reward.test",
            ) as client:

                async def post(sample_id: str):
                    sample = _sample(sample_id)
                    return await client.post(
                        "/v1/rewards:score",
                        data={
                            "group_id": sample.group_id,
                            "request_id": sample.request_id,
                            "sample_id": sample.sample_id,
                            "expected_group_size": str(sample.expected_group_size),
                            "original_prompt": sample.original_prompt,
                            "reward_tag": sample.reward_tag,
                            "fps": str(sample.fps),
                        },
                        files={
                            "video": (
                                "sample.mp4",
                                sample.video_bytes,
                                "video/mp4",
                            ),
                        },
                    )

                return await asyncio.gather(post("slot-0"), post("slot-1"))

        responses = asyncio.run(exercise())
        assert [response.status_code for response in responses] == [200, 200]
        assert judge.calls == [(2, ("a good cat", "a good cat"))]

    def test_completed_request_id_is_idempotent(self):
        judge = FakeJudge()
        client = TestClient(create_reward_app(judge))
        first = _post(client, _sample("slot-0", expected=1)).json()
        # Retry of the completed request returns the cached payload and
        # never re-invokes the judge.
        second = _post(client, _sample("slot-0", expected=1)).json()
        assert first == second
        assert len(judge.calls) == 1

    def test_duplicate_sample_retry_waits_while_group_is_scoring(self):
        judge = SlowJudge()
        client = TestClient(create_reward_app(judge))
        responses: dict[str, dict] = {}
        errors: list[BaseException] = []

        def worker(sample_id: str, key: str):
            try:
                responses[key] = _post(client, _sample(sample_id)).json()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        originals = [
            threading.Thread(target=worker, args=("slot-0", "slot-0")),
            threading.Thread(target=worker, args=("slot-1", "slot-1")),
        ]
        for thread in originals:
            thread.start()
        assert judge.started.wait(timeout=5.0)

        duplicate = threading.Thread(target=worker, args=("slot-0", "slot-0-retry"))
        duplicate.start()
        judge.release.set()

        for thread in [*originals, duplicate]:
            thread.join(timeout=10.0)
        assert not errors
        assert judge.calls == [(2, ("a good cat", "a good cat"))]
        assert responses["slot-0-retry"] == responses["slot-0"]

    def test_expected_group_size_mismatch_rejected(self):
        client = TestClient(create_reward_app(FakeJudge()))
        response = _post(client, _sample("slot-0", expected=0))
        assert response.status_code == 400

    def test_judge_failure_surfaces_500(self):
        class BrokenJudge:
            def score_batch(self, videos, prompts, *, fps):
                raise RuntimeError("judge exploded")

        client = TestClient(create_reward_app(BrokenJudge()))
        response = _post(client, _sample("slot-0", expected=1))
        assert response.status_code == 500


def test_pyav_torchvision_read_video_compatibility(tmp_path):
    frames = torch.linspace(0.0, 1.0, 4).view(1, 1, 4, 1, 1)
    video = frames.expand(1, 3, 4, 16, 16).contiguous()
    path = tmp_path / "sample.mp4"
    path.write_bytes(encode_video_bytes(video, fps=4))

    decoded, audio, info = _read_video_with_pyav(
        str(path),
        start_pts=0.0,
        pts_unit="sec",
        output_format="TCHW",
    )

    assert decoded.shape == (4, 3, 16, 16)
    assert decoded.dtype == torch.uint8
    assert audio.numel() == 0
    assert info["video_fps"] == pytest.approx(4.0)


class TestJudgeScoreParsing:
    def test_parse_valid_json_array(self):
        text = ('Here are the scores: [{"visual_quality": 3, "text_alignment": 4, '
                '"physical_consistency": 5}] done')
        rows = parse_judge_scores(text, expected=1)
        assert rows[0]["visual_quality"] == 3.0
        assert rows[0]["composite"] == pytest.approx(4.0)

    def test_parse_rejects_missing_array(self):
        with pytest.raises(RewardRequestError):
            parse_judge_scores("no json here", expected=1)

    def test_parse_rejects_wrong_cardinality(self):
        with pytest.raises(RewardRequestError):
            parse_judge_scores('[{"visual_quality": 1, "text_alignment": 1, '
                               '"physical_consistency": 1}]', expected=2)

    def test_parse_rejects_missing_components(self):
        with pytest.raises(RewardRequestError):
            parse_judge_scores('[{"visual_quality": 3}]', expected=1)


class _CharacterTokenizer:

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


class _ContextualDecodeTokenizer(_CharacterTokenizer):
    """Mimic tokenizers that expose punctuation with the newest token."""

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    ):
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
        if not clean_up_tokenization_spaces and decoded.endswith(tuple("12345")):
            return decoded + ";"
        return decoded


class TestOfficialVideoScore2Parsing:

    def test_confidence_weighted_dimension_scores(self):
        output = (
            "visual quality: 4; "
            "text-to-video alignment: 3, "
            "physical/common-sense consistency: 2"
        )
        token_ids = [ord(character) for character in output]
        generation_scores = []
        for character in output:
            logits = torch.full((1, 128), -10.0)
            for value in range(1, 6):
                logits[0, ord(str(value))] = 0.0
            if character in "12345":
                logits[0, ord(character)] = 10.0
            generation_scores.append(logits)

        parsed = parse_videoscore2_output(
            output,
            generated_token_ids=token_ids,
            generation_scores=generation_scores,
            tokenizer=_CharacterTokenizer(),
        )

        assert parsed["visual_quality"] == pytest.approx(4.0, abs=1e-3)
        assert parsed["text_alignment"] == pytest.approx(3.0, abs=1e-3)
        assert parsed["physical_consistency"] == pytest.approx(2.0, abs=1e-3)
        assert parsed["composite"] == pytest.approx(3.0, abs=1e-3)

    def test_contextual_decode_boundary_maps_to_score_token(self):
        output = (
            "visual quality: 4; "
            "text-to-video alignment: 3, "
            "physical/common-sense consistency: 2"
        )
        token_ids = [ord(character) for character in output]
        generation_scores = [torch.zeros((1, 128)) for _ in output]
        for index, character in enumerate(output):
            if character in "12345":
                generation_scores[index][0, ord(character)] = 10.0

        parsed = parse_videoscore2_output(
            output,
            generated_token_ids=token_ids,
            generation_scores=generation_scores,
            tokenizer=_ContextualDecodeTokenizer(),
        )

        assert parsed["visual_quality"] == pytest.approx(4.0, abs=1e-3)
        assert parsed["text_alignment"] == pytest.approx(3.0, abs=1e-3)
        assert parsed["physical_consistency"] == pytest.approx(2.0, abs=1e-3)

    def test_markdown_wrapped_final_labels_are_accepted(self):
        output = (
            "Analysis with no numeric summary.\n"
            "**Visual Quality:** 4;\n"
            "**Text-to-Video Alignment:** 3,\n"
            "**Physical/Common-Sense Consistency:** 2"
        )
        token_ids = [ord(character) for character in output]
        generation_scores = [torch.zeros((1, 128)) for _ in output]
        for index, character in enumerate(output):
            if character in "12345":
                generation_scores[index][0, ord(character)] = 10.0

        parsed = parse_videoscore2_output(
            output,
            generated_token_ids=token_ids,
            generation_scores=generation_scores,
            tokenizer=_CharacterTokenizer(),
        )

        assert parsed["visual_quality"] == pytest.approx(4.0, abs=1e-3)
        assert parsed["text_alignment"] == pytest.approx(3.0, abs=1e-3)
        assert parsed["physical_consistency"] == pytest.approx(2.0, abs=1e-3)

    def test_numbered_rubric_response_is_accepted(self):
        output = (
            "(1) visual quality – clarity, smoothness, artifacts: 4\n"
            "(2) text-to-video alignment – fidelity to the prompt: 3\n"
            "(3) physical/common-sense consistency – naturalness and "
            "physics plausibility: 2"
        )
        token_ids = [ord(character) for character in output]
        generation_scores = [torch.zeros((1, 128)) for _ in output]
        for index, character in enumerate(output):
            if character in "12345":
                generation_scores[index][0, ord(character)] = 10.0

        parsed = parse_videoscore2_output(
            output,
            generated_token_ids=token_ids,
            generation_scores=generation_scores,
            tokenizer=_CharacterTokenizer(),
        )

        assert parsed["visual_quality"] == pytest.approx(4.0, abs=1e-3)
        assert parsed["text_alignment"] == pytest.approx(3.0, abs=1e-3)
        assert parsed["physical_consistency"] == pytest.approx(2.0, abs=1e-3)

    def test_prose_dimension_scores_are_accepted(self):
        output = (
            "Visual Quality Analysis:\n"
            "The overall visual quality is moderate (score 3).\n\n"
            "Text-to-Video Alignment Analysis:\n"
            "The text-to-video alignment is strong (score 4).\n\n"
            "Physical/Common-Sense Consistency Analysis:\n"
            "The physical/common-sense consistency is weak (score 2)."
        )
        token_ids = [ord(character) for character in output]
        generation_scores = [torch.zeros((1, 128)) for _ in output]
        for index, character in enumerate(output):
            if character in "12345":
                generation_scores[index][0, ord(character)] = 10.0

        parsed = parse_videoscore2_output(
            output,
            generated_token_ids=token_ids,
            generation_scores=generation_scores,
            tokenizer=_CharacterTokenizer(),
        )

        assert parsed["visual_quality"] == pytest.approx(3.0, abs=1e-3)
        assert parsed["text_alignment"] == pytest.approx(4.0, abs=1e-3)
        assert parsed["physical_consistency"] == pytest.approx(2.0, abs=1e-3)

    def test_fraction_scores_and_short_physical_label_are_accepted(self):
        output = (
            "Visual Quality Analysis:\n"
            "The overall visual quality is moderate (3/5).\n"
            "Text-to-Video Alignment Analysis:\n"
            "Overall alignment corresponds to a text-to-video alignment "
            "score of 4/5.\n"
            "Physical Consistency Analysis:\n"
            "Overall Physical Consistency score: 3/5.\n"
            "Ground-truth scores: Visual Quality 3, "
            "Text-to-Video Alignment 4, Physical Consistency 3."
        )
        token_ids = [ord(character) for character in output]
        generation_scores = [torch.zeros((1, 128)) for _ in output]
        for index, character in enumerate(output):
            if character in "12345":
                generation_scores[index][0, ord(character)] = 10.0

        parsed = parse_videoscore2_output(
            output,
            generated_token_ids=token_ids,
            generation_scores=generation_scores,
            tokenizer=_CharacterTokenizer(),
        )

        assert parsed["visual_quality"] == pytest.approx(3.0, abs=1e-3)
        assert parsed["text_alignment"] == pytest.approx(4.0, abs=1e-3)
        assert parsed["physical_consistency"] == pytest.approx(3.0, abs=1e-3)

    def test_missing_official_fields_are_rejected(self):
        output = "visual quality: 4"
        token_ids = [ord(character) for character in output]
        generation_scores = [torch.zeros((1, 128)) for _ in output]
        with pytest.raises(RewardRequestError, match="missing a usable"):
            parse_videoscore2_output(
                output,
                generated_token_ids=token_ids,
                generation_scores=generation_scores,
                tokenizer=_CharacterTokenizer(),
            )


def test_videoscore2_unscorable_generation_neutralizes_entire_group(
    monkeypatch,
):
    from fastvideo.train.methods.rl.promptrl.rewards import VideoScore2Judge

    judge = VideoScore2Judge(device="cpu", parse_failure_score=3.0)
    monkeypatch.setattr(judge, "_load", lambda: None)
    calls = 0

    def score_one(path, prompt):
        nonlocal calls
        del path, prompt
        calls += 1
        if calls == 1:
            return {
                "visual_quality": 4.0,
                "text_alignment": 4.0,
                "physical_consistency": 4.0,
                "composite": 4.0,
            }
        raise RewardRequestError(
            500,
            "VideoScore2 output missing a usable visual_quality score",
        )

    monkeypatch.setattr(judge, "_score_one", score_one)
    results = judge.score_batch(
        [b"video-a", b"video-b"],
        ["prompt a", "prompt b"],
        fps=4,
    )

    assert [result["composite"] for result in results] == [3.0, 3.0]
    assert [result["judge_fallback"] for result in results] == [1.0, 1.0]


class _StubResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FlakySession:
    """requests.Session stand-in: fails N times, then serves ``payload``."""

    def __init__(self, failures: int, payload: dict):
        self.failures_left = failures
        self.payload = payload
        self.attempts = 0

    def post(self, url, data=None, files=None, timeout=None):
        self.attempts += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise ConnectionError("simulated connection drop")
        return _StubResponse(200, dict(self.payload))

    def get(self, url, timeout=None):
        return _StubResponse(200, {"status": "ok"})


class _ConcurrentGroupSession:
    """Requires both local group uploads to be in flight together."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=2.0)

    def post(self, url, data=None, files=None, timeout=None):
        del url, files, timeout
        self.barrier.wait()
        return _StubResponse(
            200,
            {
                "request_id": data["request_id"],
                "sample_id": data["sample_id"],
                "score": 4.0,
                "details": {},
            },
        )


class TestHttpRewardProvider:
    def _provider(self, session) -> HttpRewardProvider:
        provider = HttpRewardProvider(
            endpoint_url="http://reward.test",
            timeout_sec=1.0,
            retries=2,
            retry_backoff_sec=0.0,
        )
        provider._session = session
        return provider

    def test_retries_then_succeeds(self):
        payload = {
            "request_id": "r",
            "sample_id": "slot-0",
            "score": 4.2,
            "details": {"visual_quality": 4.0},
        }
        session = _FlakySession(failures=2, payload=payload)
        provider = self._provider(session)
        results = provider.score([_sample("slot-0", expected=1)])
        assert session.attempts == 3  # 2 failures + final success
        assert results[0].score == 4.2

    def test_exhausted_retries_raise(self):
        session = _FlakySession(failures=99, payload={})
        provider = self._provider(session)
        with pytest.raises(RewardServiceError, match="failed after"):
            provider.score([_sample("slot-0", expected=1)])
        assert session.attempts == 3  # retries=2 -> 3 attempts

    def test_invalid_payload_raises(self):
        session = _FlakySession(failures=0, payload={"request_id": "r"})
        provider = self._provider(session)
        with pytest.raises(RewardServiceError):
            provider.score([_sample("slot-0", expected=1)])

    def test_health(self):
        provider = self._provider(_FlakySession(0, {}))
        assert provider.health()

    def test_multiple_local_samples_are_submitted_concurrently(self):
        provider = self._provider(_ConcurrentGroupSession())
        results = provider.score([
            _sample("slot-0"),
            _sample("slot-1"),
        ])
        assert [result.sample_id for result in results] == ["slot-0", "slot-1"]


class TestProviderValidation:
    def _result(self, sample_id, score=1.0, details=None) -> RewardResult:
        return RewardResult(
            score=score,
            details=details or {},
            sample_id=sample_id,
            request_id="r",
        )

    def test_cardinality_mismatch(self):
        with pytest.raises(RewardServiceError, match="cardinality"):
            validate_reward_results(
                [_sample("a"), _sample("b")],
                [self._result("a")],
            )

    def test_missing_and_duplicate_samples(self):
        samples = [_sample("a"), _sample("b")]
        with pytest.raises(RewardServiceError, match="missing"):
            validate_reward_results(samples, [self._result("a"), self._result("a")])

    def test_non_finite_score(self):
        samples = [_sample("a")]
        with pytest.raises(RewardServiceError, match="Non-finite"):
            validate_reward_results(samples, [self._result("a", score=float("nan"))])

    def test_non_finite_detail(self):
        samples = [_sample("a")]
        with pytest.raises(RewardServiceError, match="Non-finite reward detail"):
            validate_reward_results(
                samples, [self._result("a", details={"vq": float("inf")})])

    def test_valid_passes(self):
        samples = [_sample("a"), _sample("b")]
        validate_reward_results(samples, [self._result("b"), self._result("a")])

    def test_payload_request_id_mismatch(self):
        with pytest.raises(RewardServiceError, match="request_id"):
            _parse_reward_payload(
                {"request_id": "other", "sample_id": "slot-0", "score": 1.0},
                sample=_sample("slot-0"),
            )


class TestGroupFailureSynchronization:
    """Every rank runs identical validation on the gathered results."""

    def _results(self, count: int) -> list[RewardResult]:
        return [
            RewardResult(score=float(i), details={}, sample_id=f"slot-{i}", request_id="r")
            for i in range(count)
        ]

    def test_cardinality_mismatch_fails(self):
        from fastvideo.train.methods.rl.promptrl.distributed import (
            RewardConsistencyError,
            validate_group_reward_results,
        )

        with pytest.raises(RewardConsistencyError, match="cardinality"):
            validate_group_reward_results(
                self._results(7), group_id="g", expected_group_size=8)

    def test_duplicate_sample_fails(self):
        from fastvideo.train.methods.rl.promptrl.distributed import (
            RewardConsistencyError,
            validate_group_reward_results,
        )

        results = self._results(8)
        results[3] = RewardResult(score=1.0, details={}, sample_id="slot-0", request_id="r")
        with pytest.raises(RewardConsistencyError, match="duplicate"):
            validate_group_reward_results(
                results, group_id="g", expected_group_size=8)

    def test_non_finite_score_fails(self):
        from fastvideo.train.methods.rl.promptrl.distributed import (
            RewardConsistencyError,
            validate_group_reward_results,
        )

        results = self._results(8)
        results[5] = RewardResult(
            score=float("nan"), details={}, sample_id="slot-5", request_id="r")
        with pytest.raises(RewardConsistencyError, match="non-finite"):
            validate_group_reward_results(
                results, group_id="g", expected_group_size=8)

    def test_rank_local_reward_failure_fails_all_ranks(self):
        from fastvideo.train.methods.rl.promptrl.distributed import (
            RewardConsistencyError,
            RewardFailure,
            validate_group_reward_results,
        )

        results = self._results(7)
        results.append(
            RewardFailure(rank=3, error_type="RewardServiceError", message="timeout"))
        with pytest.raises(RewardConsistencyError, match="rank 3 reward failure"):
            validate_group_reward_results(
                results, group_id="g", expected_group_size=8)

    def test_valid_group_passes(self):
        from fastvideo.train.methods.rl.promptrl.distributed import (
            validate_group_reward_results,
        )

        validate_group_reward_results(
            self._results(8), group_id="g", expected_group_size=8)

    def test_single_process_gather_matches_expectation(self):
        from fastvideo.train.methods.rl.promptrl.distributed import (
            gather_group_reward_results,
        )

        result = self._results(1)[0]
        gathered = gather_group_reward_results(
            result, group_id="g", expected_group_size=1)
        assert gathered == [result]
