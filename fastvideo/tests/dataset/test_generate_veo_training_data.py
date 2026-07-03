import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "dataset_preparation"
sys.path.insert(0, str(SCRIPT_DIR))
import generate_veo_training_data as veo


FAKE_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp41" + b"\x00" * 1_200
FAKE_WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 1_200
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-data"
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"png-data"
FAKE_VIDEO_METADATA = veo.VideoMetadata(
    width=1280,
    height=720,
    fps=24.0,
    duration=8.0,
    num_frames=192,
)


class FakeRequestException(Exception):
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


class FakeResponse:
    def __init__(
        self,
        payload=None,
        *,
        content=b"",
        status_code=200,
        headers=None,
        json_error=None,
    ):
        self.payload = payload
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.json_error = json_error
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeRequestException(f"HTTP {self.status_code}", response=self)

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, *, post_responses=(), get_responses=()):
        self.post_responses = list(post_responses)
        self.get_responses = list(get_responses)
        self.post_calls = []
        self.get_calls = []
        self.closed = False

    def post(self, url, **kwargs):
        captured = {"url": url, **kwargs}
        files = kwargs.get("files")
        if files:
            filename, payload, mime_type = files["input_reference"]
            if hasattr(payload, "read"):
                position = payload.tell()
                file_data = payload.read()
                payload.seek(position)
            else:
                file_data = bytes(payload)
            captured["uploaded_file"] = (filename, file_data, mime_type)
        self.post_calls.append(captured)
        response = self.post_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        response = self.get_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.closed = True


class RequestsModuleMixin:
    def setUp(self):
        fake_requests = SimpleNamespace(
            RequestException=FakeRequestException,
            Session=mock.Mock(),
        )
        self.requests_patch = mock.patch.dict(sys.modules, {"requests": fake_requests})
        self.requests_patch.start()

    def tearDown(self):
        self.requests_patch.stop()


class InputTests(unittest.TestCase):
    def test_jsonl_record_supports_i2v_fields(self):
        records = list(
            veo.iter_prompt_records(
                [
                    '{"id":"one","prompt":" Move ","input_image":"subject.jpg",'
                    '"seconds":6,"model":"custom/model","metadata":{"split":"train"}}'
                ],
                "jsonl",
            )
        )

        self.assertEqual(records[0].record_id, "one")
        self.assertEqual(records[0].prompt, "Move")
        self.assertEqual(records[0].input_image, "subject.jpg")
        self.assertEqual(records[0].seconds, 6)
        self.assertEqual(records[0].model, "custom/model")
        self.assertEqual(records[0].metadata, {"split": "train"})

    def test_text_records_keep_duplicate_prompts_but_have_distinct_ids(self):
        records = list(veo.iter_prompt_records(["same\n", "same\n"], "text"))
        self.assertEqual([record.prompt for record in records], ["same", "same"])
        self.assertNotEqual(records[0].record_id, records[1].record_id)

    def test_duplicate_explicit_id_is_rejected(self):
        with self.assertRaisesRegex(veo.InputError, "duplicate record id"):
            list(
                veo.iter_prompt_records(
                    [
                        '{"id":"same","prompt":"one"}',
                        '{"id":"same","prompt":"two"}',
                    ],
                    "jsonl",
                )
            )

    def test_invalid_seconds_is_rejected(self):
        with self.assertRaisesRegex(veo.InputError, "must be 4, 6, or 8"):
            list(
                veo.iter_prompt_records(
                    ['{"prompt":"move","input_image":"x.jpg","seconds":5}'],
                    "jsonl",
                )
            )

    def test_seconds_without_image_is_rejected_when_specs_are_built(self):
        record = veo.PromptRecord("one", "Move", {}, 1, seconds=6)
        args = veo.build_parser().parse_args(["prompts.jsonl"])

        with self.assertRaisesRegex(veo.InputError, "requires 'input_image'"):
            veo.build_generation_specs([record], args)

    def test_i2v_image_is_resolved_relative_to_jsonl_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompts.jsonl"
            prompt_path.write_text("", encoding="utf-8")
            image_path = root / "subject.jpg"
            image_path.write_bytes(FAKE_JPEG)
            record = veo.PromptRecord("one", "Move", {}, 1, input_image="subject.jpg")
            args = veo.build_parser().parse_args([str(prompt_path)])

            spec = veo.build_generation_specs([record], args)[0]

            self.assertEqual(spec.mode, "image-to-video")
            self.assertEqual(spec.model, veo.DEFAULT_I2V_MODEL)
            self.assertEqual(spec.base_url, veo.DEFAULT_I2V_BASE_URL)
            self.assertEqual(spec.seconds, 8)
            self.assertEqual(spec.input_image.path, image_path.resolve())
            self.assertEqual(
                spec.input_image.sha256,
                hashlib.sha256(FAKE_JPEG).hexdigest(),
            )


class ApiContractTests(RequestsModuleMixin, unittest.TestCase):
    def test_t2v_uses_json_v1_submission_contract(self):
        session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-1", "status": "queued"})]
        )
        record = veo.PromptRecord("one", "Hello!", {}, 1)
        args = veo.build_parser().parse_args(["prompts.jsonl"])
        spec = veo.build_generation_specs([record], args)[0]

        job = veo.submit_video(session, "secret", spec, 120)

        self.assertEqual(job["id"], "vid-1")
        self.assertEqual(len(session.post_calls), 1)
        call = session.post_calls[0]
        self.assertEqual(call["url"], "https://inference-api.nvidia.com/v1/videos")
        self.assertEqual(
            call["json"],
            {
                "model": "gcp/google/veo-3.0-generate-001",
                "prompt": "Hello!",
            },
        )
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertNotIn("files", call)

    def test_i2v_uses_multipart_root_submission_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "subject.png"
            image_path.write_bytes(FAKE_PNG)
            image = veo.inspect_input_image(image_path, "subject.png")
            spec = veo.GenerationSpec(
                record=veo.PromptRecord("one", "Turn slowly", {}, 1),
                mode="image-to-video",
                model=veo.DEFAULT_I2V_MODEL,
                base_url=veo.DEFAULT_I2V_BASE_URL,
                seconds=8,
                input_image=image,
            )
            session = FakeSession(
                post_responses=[FakeResponse({"id": "vid-i2v", "status": "queued"})]
            )

            job = veo.submit_video(session, "secret", spec, 120)

        self.assertEqual(job["id"], "vid-i2v")
        call = session.post_calls[0]
        self.assertEqual(call["url"], "https://inference-api.nvidia.com/videos")
        self.assertEqual(
            call["data"],
            {
                "model": "gcp/google/veo-3.1-generate-001",
                "prompt": "Turn slowly",
                "seconds": "8",
            },
        )
        self.assertEqual(call["uploaded_file"], ("subject.png", FAKE_PNG, "image/png"))
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(call["headers"]["Accept"], "application/json")
        self.assertNotIn("Content-Type", call["headers"])
        self.assertNotIn("json", call)

    def test_submission_http_errors_are_classified_by_duplicate_risk(self):
        record = veo.PromptRecord("one", "Hello!", {}, 1)
        args = veo.build_parser().parse_args(["prompts.jsonl"])
        spec = veo.build_generation_specs([record], args)[0]

        with self.assertRaises(veo.SubmissionRejected):
            veo.submit_video(
                FakeSession(post_responses=[FakeResponse(status_code=400)]),
                "secret",
                spec,
                120,
            )
        for status_code in (429, 500):
            with (
                self.subTest(status_code=status_code),
                self.assertRaises(veo.SubmissionUnknown),
            ):
                veo.submit_video(
                    FakeSession(post_responses=[FakeResponse(status_code=status_code)]),
                    "secret",
                    spec,
                    120,
                )

    def test_submission_failure_without_id_is_terminal(self):
        record = veo.PromptRecord("one", "Hello!", {}, 1)
        args = veo.build_parser().parse_args(["prompts.jsonl"])
        spec = veo.build_generation_specs([record], args)[0]

        with self.assertRaises(veo.ProviderVideoFailed):
            veo.submit_video(
                FakeSession(
                    post_responses=[
                        FakeResponse({"status": "failed", "error": "blocked"})
                    ]
                ),
                "secret",
                spec,
                120,
            )

    def test_poll_sequence_uses_status_endpoint_until_completed(self):
        session = FakeSession(
            get_responses=[
                FakeResponse({"id": "vid/1", "status": "processing"}),
                FakeResponse({"id": "vid/1", "status": "completed"}),
            ]
        )
        events = []

        with mock.patch.object(veo.time, "sleep") as sleep:
            result = veo.poll_video(
                session,
                "secret",
                veo.DEFAULT_T2V_BASE_URL,
                "vid/1",
                {"id": "vid/1", "status": "queued"},
                10,
                300,
                30,
                events.append,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(events), 2)
        self.assertEqual(
            [call["url"] for call in session.get_calls],
            [
                "https://inference-api.nvidia.com/v1/videos/vid%2F1",
                "https://inference-api.nvidia.com/v1/videos/vid%2F1",
            ],
        )

    def test_failed_provider_status_raises(self):
        with self.assertRaises(veo.ProviderVideoFailed):
            veo.poll_video(
                FakeSession(),
                "secret",
                veo.DEFAULT_T2V_BASE_URL,
                "vid-1",
                {"id": "vid-1", "status": "failed", "error": "blocked"},
                10,
                300,
                30,
                lambda event: None,
            )

    def test_poll_timeout_occurs_without_status_request_after_deadline(self):
        session = FakeSession()
        with (
            mock.patch.object(veo.time, "monotonic", side_effect=[0.0, 301.0]),
            self.assertRaises(TimeoutError),
        ):
            veo.poll_video(
                session,
                "secret",
                veo.DEFAULT_T2V_BASE_URL,
                "vid-1",
                {"id": "vid-1", "status": "queued"},
                10,
                300,
                30,
                lambda event: None,
            )

        self.assertEqual(session.get_calls, [])

    def test_poll_error_trace_redacts_api_key(self):
        session = FakeSession(
            get_responses=[
                FakeRequestException("request exposed secret-key"),
                FakeResponse({"id": "vid-1", "status": "completed"}),
            ]
        )
        events = []

        with mock.patch.object(veo.time, "sleep"):
            veo.poll_video(
                session,
                "secret-key",
                veo.DEFAULT_T2V_BASE_URL,
                "vid-1",
                {"id": "vid-1", "status": "queued"},
                10,
                300,
                30,
                events.append,
            )

        self.assertNotIn("secret-key", json.dumps(events))
        self.assertIn("[REDACTED]", json.dumps(events))

    def test_download_uses_authenticated_content_endpoint_and_saves_mp4(self):
        response = FakeResponse(
            content=FAKE_MP4,
            headers={"Content-Type": "video/mp4", "Content-Length": str(len(FAKE_MP4))},
        )
        session = FakeSession(get_responses=[response])

        with tempfile.TemporaryDirectory() as directory:
            saved = veo.download_video(
                session,
                "secret",
                veo.DEFAULT_T2V_BASE_URL,
                "vid-1",
                Path(directory),
                "sample",
                120,
                10_000,
            )

            self.assertEqual(saved.path.read_bytes(), FAKE_MP4)
            self.assertEqual(saved.path.suffix, ".mp4")

        call = session.get_calls[0]
        self.assertEqual(
            call["url"],
            "https://inference-api.nvidia.com/v1/videos/vid-1/content",
        )
        self.assertEqual(call["headers"], {"Authorization": "Bearer secret"})
        self.assertTrue(call["stream"])
        self.assertTrue(response.closed)

    def test_download_rejects_too_small_content(self):
        session = FakeSession(
            get_responses=[
                FakeResponse(
                    content=FAKE_MP4[:100], headers={"Content-Type": "video/mp4"}
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(veo.VideoAPIError, "too small"):
                veo.download_video(
                    session,
                    "secret",
                    veo.DEFAULT_T2V_BASE_URL,
                    "vid-1",
                    Path(directory),
                    "sample",
                    120,
                    10_000,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_download_rejects_non_mp4_content(self):
        response = FakeResponse(
            content=FAKE_WEBM,
            headers={"Content-Type": "video/webm"},
        )
        session = FakeSession(get_responses=[response])

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                veo.VideoAPIError, "FastVideo preprocessing requires MP4"
            ):
                veo.download_video(
                    session,
                    "secret",
                    veo.DEFAULT_T2V_BASE_URL,
                    "vid-1",
                    Path(directory),
                    "sample",
                    120,
                    10_000,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertTrue(response.closed)

    def test_streamed_download_size_limit_removes_partial_file(self):
        response = FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
        session = FakeSession(get_responses=[response])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(veo.VideoAPIError, "size limit"):
                veo.download_video(
                    session,
                    "secret",
                    veo.DEFAULT_T2V_BASE_URL,
                    "vid-1",
                    Path(directory),
                    "sample",
                    120,
                    1_000,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertTrue(response.closed)

    def test_changed_image_is_not_uploaded_with_stale_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "subject.jpg"
            image_path.write_bytes(FAKE_JPEG)
            image = veo.inspect_input_image(image_path, "subject.jpg")
            image_path.write_bytes(b"\xff\xd8\xff\xe0changed")
            spec = veo.GenerationSpec(
                record=veo.PromptRecord("one", "Move", {}, 1),
                mode="image-to-video",
                model=veo.DEFAULT_I2V_MODEL,
                base_url=veo.DEFAULT_I2V_BASE_URL,
                seconds=8,
                input_image=image,
            )
            session = FakeSession()

            with self.assertRaisesRegex(veo.InputError, "changed after validation"):
                veo.submit_video(session, "secret", spec, 120)

        self.assertEqual(session.post_calls, [])

    def test_non_image_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-an-image.jpg"
            path.write_bytes(b"plain text")
            with self.assertRaisesRegex(veo.InputError, "not a recognized image"):
                veo.inspect_input_image(path, "not-an-image.jpg")

    def test_parser_defaults_follow_both_supplied_contracts(self):
        args = veo.build_parser().parse_args(["prompts.jsonl"])
        self.assertEqual(
            veo.DEFAULT_T2V_BASE_URL, "https://inference-api.nvidia.com/v1"
        )
        self.assertEqual(veo.DEFAULT_I2V_BASE_URL, "https://inference-api.nvidia.com")
        self.assertIsNone(args.model)
        self.assertEqual(args.seconds, 8)
        self.assertIsNone(args.poll_interval)
        self.assertEqual(args.timeout, 300)


class DatasetTests(RequestsModuleMixin, unittest.TestCase):
    def make_args(self, input_path, output_path, *extra):
        return veo.build_parser().parse_args(
            [str(input_path), "--output-dir", str(output_path), *extra]
        )

    def run_dataset(self, args, session):
        with (
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
            mock.patch.object(veo, "create_http_session", return_value=session),
            mock.patch.object(
                veo, "inspect_video", return_value=FAKE_VIDEO_METADATA
            ),
            mock.patch.object(veo.time, "sleep"),
        ):
            return veo.generate_dataset(args)

    def test_fastvideo_index_contains_only_successful_mp4s(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            videos_path = output_path / "videos"
            videos_path.mkdir()
            (videos_path / "good.mp4").write_bytes(FAKE_MP4)
            metadata = veo.video_metadata_dict(FAKE_VIDEO_METADATA)
            latest = {
                "good": {
                    "id": "good",
                    "status": "succeeded",
                    "prompt": "A usable clip",
                    "video_path": "videos/good.mp4",
                    "video_metadata": metadata,
                },
                "failed": {
                    "id": "failed",
                    "status": "failed",
                    "prompt": "Do not index",
                },
                "pending": {
                    "id": "pending",
                    "status": "pending",
                    "prompt": "Not ready",
                },
            }

            rows = veo.sync_fastvideo_dataset(output_path, latest)
            saved_rows = json.loads(
                (output_path / "videos2caption.json").read_text(encoding="utf-8")
            )

        self.assertEqual(rows, saved_rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["path"], "good.mp4")
        self.assertEqual(rows[0]["cap"], ["A usable clip"])
        self.assertEqual(rows[0]["resolution"], {"width": 1280, "height": 720})

    def test_end_to_end_t2v_generation_and_resume(self):
        session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-1", "status": "queued"})],
            get_responses=[
                FakeResponse({"id": "vid-1", "status": "completed"}),
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"}),
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"clip-1","prompt":"A wave","metadata":{"split":"train"}}\n',
                encoding="utf-8",
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)

            self.assertEqual(self.run_dataset(args, session), 0)
            self.assertEqual(self.run_dataset(args, session), 0)

            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            video_exists = (output_path / rows[-1]["video_path"]).is_file()
            response_exists = (output_path / rows[-1]["response_path"]).is_file()
            annotations = json.loads(
                (output_path / "videos2caption.json").read_text(encoding="utf-8")
            )
            merge_line = (output_path / "merge.txt").read_text(encoding="utf-8")

        self.assertEqual(len(session.post_calls), 1)
        self.assertEqual([row["status"] for row in rows], ["submitted", "succeeded"])
        self.assertEqual([row["attempt"] for row in rows], [1, 1])
        self.assertEqual(rows[-1]["provider_video_id"], "vid-1")
        self.assertEqual(rows[-1]["mode"], "text-to-video")
        self.assertEqual(rows[-1]["video_metadata"]["num_frames"], 192)
        self.assertTrue(video_exists)
        self.assertTrue(response_exists)
        self.assertEqual(
            annotations,
            [
                {
                    "path": Path(rows[-1]["video_path"]).name,
                    "resolution": {"width": 1280, "height": 720},
                    "fps": 24.0,
                    "duration": 8.0,
                    "num_frames": 192,
                    "size": len(FAKE_MP4),
                    "cap": ["A wave"],
                }
            ],
        )
        self.assertIn("videos,", merge_line)
        self.assertTrue(merge_line.rstrip().endswith("videos2caption.json"))

    def test_i2v_record_runs_multipart_lifecycle(self):
        session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-i2v", "status": "queued"})],
            get_responses=[
                FakeResponse({"id": "vid-i2v", "status": "completed"}),
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"}),
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "subject.jpg").write_bytes(FAKE_JPEG)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"clip-1","prompt":"Blink","input_image":"subject.jpg",'
                '"seconds":6}\n',
                encoding="utf-8",
            )
            output_path = root / "dataset"

            self.assertEqual(
                self.run_dataset(self.make_args(input_path, output_path), session), 0
            )
            row = json.loads(
                (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )

        self.assertEqual(
            session.post_calls[0]["url"], f"{veo.DEFAULT_I2V_BASE_URL}/videos"
        )
        self.assertEqual(session.post_calls[0]["data"]["seconds"], "6")
        self.assertEqual(row["mode"], "image-to-video")
        self.assertEqual(row["model"], veo.DEFAULT_I2V_MODEL)
        self.assertIn("input_reference", row["request"])

    def test_pending_i2v_job_resumes_after_source_image_is_deleted(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-i2v", "status": "queued"})]
        )
        second_session = FakeSession(
            get_responses=[
                FakeResponse({"id": "vid-i2v", "status": "completed"}),
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"}),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "subject.jpg"
            image_path.write_bytes(FAKE_JPEG)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"clip-1","prompt":"Blink","input_image":"subject.jpg"}\n',
                encoding="utf-8",
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)

            with mock.patch.object(
                veo, "poll_video", side_effect=TimeoutError("still running")
            ):
                self.assertEqual(self.run_dataset(args, first_session), 1)
            image_path.unlink()

            self.assertEqual(self.run_dataset(args, second_session), 0)

        self.assertEqual(second_session.post_calls, [])
        self.assertEqual(len(second_session.get_calls), 2)

    def test_pending_job_is_resumed_without_duplicate_post(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-pending", "status": "queued"})]
        )
        second_session = FakeSession(
            get_responses=[
                FakeResponse({"id": "vid-pending", "status": "completed"}),
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"}),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"clip-1","prompt":"A wave"}\n', encoding="utf-8"
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)
            overwrite_args = self.make_args(input_path, output_path, "--overwrite")

            with mock.patch.object(
                veo, "poll_video", side_effect=TimeoutError("still running")
            ):
                self.assertEqual(self.run_dataset(args, first_session), 1)
            self.assertEqual(self.run_dataset(overwrite_args, second_session), 0)

            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(first_session.post_calls), 1)
        self.assertEqual(len(second_session.post_calls), 0)
        self.assertEqual(
            [row["status"] for row in rows],
            ["submitted", "pending", "succeeded"],
        )
        self.assertEqual({row["attempt"] for row in rows}, {1})

    def test_ambiguous_post_counts_toward_limit_and_blocks_automatic_retry(self):
        first_session = FakeSession(
            post_responses=[
                FakeRequestException("connection lost after request"),
                FakeResponse({"id": "should-not-submit", "status": "completed"}),
            ]
        )
        second_session = FakeSession()
        third_session = FakeSession(
            post_responses=[FakeResponse({"id": "replacement", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"one","prompt":"First"}\n{"id":"two","prompt":"Second"}\n',
                encoding="utf-8",
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path, "--limit", "1")

            self.assertEqual(self.run_dataset(args, first_session), 1)
            input_path.write_text('{"id":"one","prompt":"First"}\n', encoding="utf-8")
            self.assertEqual(self.run_dataset(args, second_session), 1)
            abandon_args = self.make_args(
                input_path, output_path, "--limit", "1", "--abandon-inflight"
            )
            self.assertEqual(self.run_dataset(abandon_args, third_session), 0)
            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(first_session.post_calls), 1)
        self.assertEqual(second_session.post_calls, [])
        self.assertEqual(len(third_session.post_calls), 1)
        self.assertEqual(rows[0]["status"], "submission_unknown")
        self.assertEqual(
            [row["attempt"] for row in rows],
            [1, 2, 2],
        )

    def test_interrupt_during_post_records_ambiguous_submission(self):
        session = FakeSession(post_responses=[KeyboardInterrupt()])
        retry_session = FakeSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"one","prompt":"First"}\n', encoding="utf-8"
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)

            with self.assertRaises(KeyboardInterrupt):
                self.run_dataset(args, session)
            self.assertEqual(self.run_dataset(args, retry_session), 1)

            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(session.post_calls), 1)
        self.assertEqual(retry_session.post_calls, [])
        self.assertEqual(rows[-1]["status"], "submission_unknown")
        self.assertEqual(rows[-1]["attempt"], 1)

    def test_interrupt_after_success_does_not_shadow_completed_attempt(self):
        session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-1", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )
        retry_session = FakeSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"one","prompt":"First"}\n', encoding="utf-8"
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)

            with mock.patch.object(
                veo,
                "sync_fastvideo_dataset",
                side_effect=[[], [], KeyboardInterrupt()],
            ):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_dataset(args, session)

            self.assertEqual(self.run_dataset(args, retry_session), 0)
            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual([row["status"] for row in rows], ["submitted", "succeeded"])
        self.assertEqual(retry_session.post_calls, [])

    def test_changed_pending_input_does_not_submit_new_job(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-pending", "status": "queued"})]
        )
        second_session = FakeSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"clip-1","prompt":"First"}\n', encoding="utf-8"
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)
            with mock.patch.object(
                veo, "poll_video", side_effect=TimeoutError("still running")
            ):
                self.assertEqual(self.run_dataset(args, first_session), 1)

            input_path.write_text(
                '{"id":"clip-1","prompt":"Changed"}\n', encoding="utf-8"
            )
            self.assertEqual(self.run_dataset(args, second_session), 1)

        self.assertEqual(len(first_session.post_calls), 1)
        self.assertEqual(second_session.post_calls, [])

    def test_missing_success_video_is_redownloaded_without_post(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-1", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )
        second_session = FakeSession(
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text('{"id":"one","prompt":"First"}\n', encoding="utf-8")
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)
            self.assertEqual(self.run_dataset(args, first_session), 0)
            succeeded = json.loads(
                (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )
            (output_path / succeeded["video_path"]).unlink()

            self.assertEqual(self.run_dataset(args, second_session), 0)

        self.assertEqual(second_session.post_calls, [])
        self.assertEqual(len(second_session.get_calls), 1)

    def test_poll_404_keeps_job_pending_without_new_post(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-pending", "status": "queued"})]
        )
        second_session = FakeSession(get_responses=[FakeResponse(status_code=404)])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text('{"id":"one","prompt":"First"}\n', encoding="utf-8")
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path)
            with mock.patch.object(
                veo, "poll_video", side_effect=TimeoutError("still running")
            ):
                self.assertEqual(self.run_dataset(args, first_session), 1)

            self.assertEqual(self.run_dataset(args, second_session), 1)
            latest = json.loads(
                (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )

        self.assertEqual(second_session.post_calls, [])
        self.assertEqual(latest["status"], "pending")
        self.assertEqual(latest["provider_video_id"], "vid-pending")
        self.assertIn("ProviderJobGone", latest["error"])

    def test_overwrite_success_creates_new_attempt_and_artifacts(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "first-job", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )
        second_session = FakeSession(
            post_responses=[FakeResponse({"id": "second-job", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text('{"id":"one","prompt":"First"}\n', encoding="utf-8")
            output_path = root / "dataset"
            self.assertEqual(
                self.run_dataset(
                    self.make_args(input_path, output_path), first_session
                ),
                0,
            )
            self.assertEqual(
                self.run_dataset(
                    self.make_args(input_path, output_path, "--overwrite"),
                    second_session,
                ),
                0,
            )
            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            successes = [row for row in rows if row["status"] == "succeeded"]
            paths_exist = [
                (output_path / row["video_path"]).is_file() for row in successes
            ]

        self.assertEqual([row["attempt"] for row in successes], [1, 2])
        self.assertNotEqual(successes[0]["video_path"], successes[1]["video_path"])
        self.assertEqual(paths_exist, [True, True])

    def test_provider_failure_is_terminal_for_attempt(self):
        session = FakeSession(
            post_responses=[
                FakeResponse(
                    {"id": "vid-failed", "status": "failed", "error": "blocked"}
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text('{"id":"one","prompt":"x"}\n', encoding="utf-8")
            output_path = root / "dataset"

            self.assertEqual(
                self.run_dataset(self.make_args(input_path, output_path), session), 1
            )
            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual([row["status"] for row in rows], ["submitted", "failed"])
        self.assertIn("blocked", rows[-1]["error"])

    def test_limit_counts_new_submissions_not_completed_rows(self):
        first_session = FakeSession(
            post_responses=[FakeResponse({"id": "one-job", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )
        second_session = FakeSession(
            post_responses=[FakeResponse({"id": "two-job", "status": "completed"})],
            get_responses=[
                FakeResponse(content=FAKE_MP4, headers={"Content-Type": "video/mp4"})
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id":"one","prompt":"First"}\n{"id":"two","prompt":"Second"}\n',
                encoding="utf-8",
            )
            output_path = root / "dataset"
            args = self.make_args(input_path, output_path, "--limit", "1")

            self.assertEqual(self.run_dataset(args, first_session), 0)
            self.assertEqual(self.run_dataset(args, second_session), 0)
            rows = [
                json.loads(line)
                for line in (output_path / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(first_session.post_calls), 1)
        self.assertEqual(len(second_session.post_calls), 1)
        self.assertEqual(
            [row["id"] for row in rows if row["status"] == "succeeded"],
            ["one", "two"],
        )

    def test_missing_api_key_fails_before_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.txt"
            input_path.write_text("A wave\n", encoding="utf-8")
            args = self.make_args(input_path, root / "dataset")

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(veo, "create_http_session") as create_session,
            ):
                exit_code = veo.generate_dataset(args)

            self.assertEqual(exit_code, 2)
            create_session.assert_not_called()

    def test_nvapikey_fallback_is_supported(self):
        session = FakeSession(
            post_responses=[FakeResponse({"id": "vid-1", "status": "failed"})]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.txt"
            input_path.write_text("A wave\n", encoding="utf-8")
            args = self.make_args(input_path, root / "dataset")

            with (
                mock.patch.dict(os.environ, {"NVAPIKEY": "fallback-key"}, clear=True),
                mock.patch.object(veo, "create_http_session", return_value=session),
            ):
                exit_code = veo.generate_dataset(args)

            self.assertEqual(exit_code, 1)
            self.assertEqual(
                session.post_calls[0]["headers"]["Authorization"],
                "Bearer fallback-key",
            )

    def test_dry_run_validates_image_without_requests_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "subject.jpg"
            image_path.write_bytes(FAKE_JPEG)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"prompt":"Move","input_image":"subject.jpg"}\n', encoding="utf-8"
            )
            args = self.make_args(input_path, root / "dataset", "--dry-run")

            with mock.patch.object(veo, "create_http_session") as create_session:
                exit_code = veo.generate_dataset(args)

            self.assertEqual(exit_code, 0)
            create_session.assert_not_called()
            self.assertFalse((root / "dataset").exists())

    def test_invalid_output_path_fails_before_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.txt"
            input_path.write_text("A wave\n", encoding="utf-8")
            output_path = root / "not-a-directory"
            output_path.write_text("occupied", encoding="utf-8")
            args = self.make_args(input_path, output_path)

            with (
                mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "key"}),
                mock.patch.object(veo, "create_http_session") as create_session,
            ):
                exit_code = veo.generate_dataset(args)

            self.assertEqual(exit_code, 2)
            create_session.assert_not_called()

    def test_output_lock_rejects_concurrent_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            first = veo.acquire_output_lock(output_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "already using"):
                    veo.acquire_output_lock(output_dir)
            finally:
                first.release()

            second = veo.acquire_output_lock(output_dir)
            second.release()

    def test_resume_ignores_and_removes_truncated_manifest_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                '{"id":"good","status":"failed","attempt":2}\n{"id":"partial"',
                encoding="utf-8",
            )
            counts = {}

            latest = veo.load_latest_manifest_records(manifest, counts)

            self.assertEqual(set(latest), {"good"})
            self.assertEqual(counts, {"good": 2})
            self.assertEqual(
                manifest.read_text(encoding="utf-8"),
                '{"id":"good","status":"failed","attempt":2}\n',
            )

    def test_structurally_invalid_manifest_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                '{"id":"one","status":"submitted","attempt":1}\n'
                '{"id":"one"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(veo.InputError, "lifecycle record"):
                veo.load_latest_manifest_records(manifest)


if __name__ == "__main__":
    unittest.main()
