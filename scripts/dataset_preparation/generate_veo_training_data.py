#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate resumable NVIDIA Veo datasets for FastVideo preprocessing."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote


DEFAULT_T2V_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_I2V_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_T2V_MODEL = "gcp/google/veo-3.0-generate-001"
DEFAULT_I2V_MODEL = "gcp/google/veo-3.1-generate-001"
DEFAULT_OUTPUT_DIR = Path("veo_training_data")
DEFAULT_API_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_MAX_VIDEO_BYTES = 512 * 1024 * 1024
MIN_VIDEO_BYTES = 1_000
COMPLETED_STATUSES = {"completed", "succeeded", "success", "done"}
FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}
RESUMABLE_MANIFEST_STATUSES = {"submitted", "pending"}
MANIFEST_STATUSES = RESUMABLE_MANIFEST_STATUSES | {
    "submission_unknown",
    "failed",
    "succeeded",
}

LOGGER = logging.getLogger("veo-data")


class InputError(ValueError):
    """Raised when an input record or CLI value is invalid."""


class VideoAPIError(RuntimeError):
    """Raised when NVIDIA's Videos API returns an unusable response."""


class ProviderVideoFailed(VideoAPIError):
    """Raised when NVIDIA reports that a submitted video job failed."""

    def __init__(self, job: Mapping[str, Any]):
        self.job = dict(job)
        detail = job.get("error") or job
        super().__init__(
            f"video generation failed: {json.dumps(detail, ensure_ascii=False)}"
        )


class ProviderJobGone(VideoAPIError):
    """Raised when a previously submitted provider job no longer exists."""


class SubmissionUnknown(VideoAPIError):
    """Raised when a POST may have succeeded but no provider id was received."""


class SubmissionRejected(VideoAPIError):
    """Raised when NVIDIA definitively rejects a POST before creating a job."""


@dataclass(frozen=True)
class PromptRecord:
    record_id: str
    prompt: str
    metadata: dict[str, Any]
    line_number: int
    input_image: str | None = None
    seconds: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class InputImage:
    path: Path
    display_path: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GenerationSpec:
    record: PromptRecord
    mode: str
    model: str
    base_url: str
    seconds: int | None
    input_image: InputImage | None

    @property
    def request_identity(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.input_image is not None:
            result["seconds"] = self.seconds
            result["input_reference"] = {
                "path": self.input_image.display_path,
                "sha256": self.input_image.sha256,
                "size_bytes": self.input_image.size_bytes,
                "mime_type": self.input_image.mime_type,
            }
        return result


@dataclass(frozen=True)
class SavedVideo:
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration: float
    num_frames: int


@dataclass
class OutputLock:
    path: Path
    handle: Any

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_auto_id(line_number: int, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return f"row-{line_number:06d}-{digest}"


def safe_file_stem(record_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", record_id).strip("-._")
    cleaned = cleaned[:72] or "record"
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}-{digest}"


def _optional_nonempty_string(
    value: Mapping[str, Any], key: str, line_number: int
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise InputError(f"line {line_number}: '{key}' must be a non-empty string")
    return item.strip()


def _parse_json_record(value: Any, line_number: int) -> PromptRecord:
    if isinstance(value, str):
        prompt = value.strip()
        if not prompt:
            raise InputError(f"line {line_number}: prompt must not be empty")
        return PromptRecord(
            record_id=stable_auto_id(line_number, prompt),
            prompt=prompt,
            metadata={},
            line_number=line_number,
        )

    if not isinstance(value, Mapping):
        raise InputError(f"line {line_number}: expected a JSON object or string")

    prompt_value = value.get("prompt")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise InputError(f"line {line_number}: 'prompt' must be a non-empty string")
    prompt = prompt_value.strip()

    record_id_value = value.get("id")
    if record_id_value is None:
        record_id = stable_auto_id(line_number, prompt)
    elif (
        not isinstance(record_id_value, bool)
        and isinstance(record_id_value, (str, int))
        and str(record_id_value).strip()
    ):
        record_id = str(record_id_value).strip()
    else:
        raise InputError(
            f"line {line_number}: 'id' must be a non-empty string or integer"
        )

    metadata_value = value.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise InputError(f"line {line_number}: 'metadata' must be a JSON object")

    seconds_value = value.get("seconds")
    if seconds_value is not None:
        if isinstance(seconds_value, bool) or not isinstance(seconds_value, int):
            raise InputError(f"line {line_number}: 'seconds' must be 4, 6, or 8")
        if seconds_value not in {4, 6, 8}:
            raise InputError(f"line {line_number}: 'seconds' must be 4, 6, or 8")

    return PromptRecord(
        record_id=record_id,
        prompt=prompt,
        metadata=dict(metadata_value),
        line_number=line_number,
        input_image=_optional_nonempty_string(value, "input_image", line_number),
        seconds=seconds_value,
        model=_optional_nonempty_string(value, "model", line_number),
    )


def iter_prompt_records(
    lines: Iterable[str], input_format: str
) -> Iterator[PromptRecord]:
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if input_format == "text":
            record = PromptRecord(
                record_id=stable_auto_id(line_number, line),
                prompt=line,
                metadata={},
                line_number=line_number,
            )
        else:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(
                    f"line {line_number}: invalid JSON: {exc.msg} at column {exc.colno}"
                ) from exc
            record = _parse_json_record(value, line_number)

        if record.record_id in seen_ids:
            raise InputError(
                f"line {line_number}: duplicate record id {record.record_id!r}"
            )
        seen_ids.add(record.record_id)
        yield record


def read_prompt_records(path: str, input_format: str) -> list[PromptRecord]:
    resolved_format = input_format
    if resolved_format == "auto":
        resolved_format = (
            "text" if path != "-" and Path(path).suffix.lower() == ".txt" else "jsonl"
        )

    if path == "-":
        return list(iter_prompt_records(sys.stdin, resolved_format))

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return list(iter_prompt_records(handle, resolved_format))
    except OSError as exc:
        raise InputError(f"could not read {path!r}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def detect_image_mime(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(16)
    except OSError as exc:
        raise InputError(f"could not read input image {str(path)!r}: {exc}") from exc
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    if prefix.startswith(b"BM"):
        return "image/bmp"
    raise InputError(f"input reference is not a recognized image: {path}")


def inspect_input_image(path: Path, display_path: str) -> InputImage:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise InputError(f"input image not found: {display_path}")
    try:
        size_bytes = resolved.stat().st_size
        if size_bytes <= 0:
            raise InputError(f"input image is empty: {display_path}")
        mime_type = detect_image_mime(resolved)
        digest = sha256_file(resolved)
    except OSError as exc:
        raise InputError(f"could not read input image {display_path!r}: {exc}") from exc
    extension_mime = mimetypes.guess_type(resolved.name)[0]
    if extension_mime and not extension_mime.startswith("image/"):
        raise InputError(f"input reference is not an image: {display_path}")
    return InputImage(
        path=resolved,
        display_path=display_path,
        filename=resolved.name,
        mime_type=mime_type,
        size_bytes=size_bytes,
        sha256=digest,
    )


def prompt_file_directory(input_path: str) -> Path:
    return (
        Path.cwd()
        if input_path == "-"
        else Path(input_path).expanduser().resolve().parent
    )


def recover_input_image(
    previous: Mapping[str, Any] | None,
    display_path: str,
    resolved_path: Path,
) -> InputImage | None:
    if previous is None or previous.get("status") not in (
        RESUMABLE_MANIFEST_STATUSES | {"succeeded", "submission_unknown"}
    ):
        return None
    request = previous.get("request")
    if not isinstance(request, Mapping):
        return None
    reference = request.get("input_reference")
    if not isinstance(reference, Mapping) or reference.get("path") != display_path:
        return None
    sha256 = reference.get("sha256")
    size_bytes = reference.get("size_bytes")
    mime_type = reference.get("mime_type")
    if (
        not isinstance(sha256, str)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or not isinstance(mime_type, str)
        or not mime_type.startswith("image/")
    ):
        return None
    return InputImage(
        path=resolved_path,
        display_path=display_path,
        filename=resolved_path.name,
        mime_type=mime_type,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def build_generation_specs(
    records: Sequence[PromptRecord],
    args: argparse.Namespace,
    previous_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[GenerationSpec]:
    record_base = prompt_file_directory(args.input)
    specs: list[GenerationSpec] = []
    for record in records:
        image_value: str | None
        image_base: Path
        if record.input_image is not None:
            image_value = record.input_image
            image_base = record_base
        elif args.input_image is not None:
            image_value = str(args.input_image)
            image_base = Path.cwd()
        else:
            image_value = None
            image_base = Path.cwd()

        image: InputImage | None = None
        if image_value is not None:
            image_path = Path(image_value).expanduser()
            if not image_path.is_absolute():
                image_path = image_base / image_path
            try:
                image = inspect_input_image(image_path, image_value)
            except InputError:
                previous = (
                    previous_records.get(record.record_id)
                    if previous_records is not None
                    else None
                )
                image = recover_input_image(
                    previous, image_value, image_path.expanduser().resolve()
                )
                if image is None:
                    raise

        if image is None and record.seconds is not None:
            raise InputError(
                f"line {record.line_number}: 'seconds' requires 'input_image'"
            )

        mode = "image-to-video" if image is not None else "text-to-video"
        model = record.model or args.model
        if model is None:
            model = DEFAULT_I2V_MODEL if image is not None else DEFAULT_T2V_MODEL
        base_url = DEFAULT_I2V_BASE_URL if image is not None else DEFAULT_T2V_BASE_URL
        specs.append(
            GenerationSpec(
                record=record,
                mode=mode,
                model=model,
                base_url=base_url.rstrip("/"),
                seconds=(record.seconds or args.seconds) if image is not None else None,
                input_image=image,
            )
        )
    return specs


def create_http_session() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is required. Install it with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    return requests.Session()


def auth_headers(api_key: str, *, json_content: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def json_object_response(response: Any, operation: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
        try:
            value = response.json()
        except (TypeError, ValueError) as exc:
            raise VideoAPIError(f"{operation} returned invalid JSON") from exc
    finally:
        response.close()
    if not isinstance(value, Mapping):
        raise VideoAPIError(f"{operation} returned a non-object JSON response")
    return dict(value)


def normalize_status(job: Mapping[str, Any]) -> str:
    value = job.get("status", "unknown")
    return str(value).strip().lower() or "unknown"


def validate_provider_job(job: Mapping[str, Any], require_id: bool = True) -> str:
    video_id = job.get("id")
    if require_id and (not isinstance(video_id, str) or not video_id.strip()):
        raise VideoAPIError(
            f"video submission returned no id: {json.dumps(job, ensure_ascii=False)}"
        )
    return video_id.strip() if isinstance(video_id, str) else ""


def raise_for_terminal_failure(job: Mapping[str, Any]) -> None:
    if job.get("error") or normalize_status(job) in FAILED_STATUSES:
        raise ProviderVideoFailed(job)


def submit_video(
    session: Any,
    api_key: str,
    spec: GenerationSpec,
    request_timeout: float,
    on_post_attempt: Callable[[], None] | None = None,
) -> dict[str, Any]:
    try:
        from requests import RequestException
    except ImportError as exc:
        raise RuntimeError("The 'requests' package is required") from exc

    url = f"{spec.base_url}/videos"
    try:
        if spec.input_image is None:
            if on_post_attempt is not None:
                on_post_attempt()
            response = session.post(
                url,
                headers=auth_headers(api_key, json_content=True),
                json={"model": spec.model, "prompt": spec.record.prompt},
                timeout=request_timeout,
            )
        else:
            try:
                image_data = spec.input_image.path.read_bytes()
            except OSError as exc:
                raise InputError(
                    f"could not read input image {spec.input_image.display_path!r}: {exc}"
                ) from exc
            actual_sha256 = hashlib.sha256(image_data).hexdigest()
            if (
                len(image_data) != spec.input_image.size_bytes
                or actual_sha256 != spec.input_image.sha256
            ):
                raise InputError(
                    f"input image changed after validation: {spec.input_image.display_path}"
                )
            fields = {
                "model": spec.model,
                "prompt": spec.record.prompt,
                "seconds": str(spec.seconds),
            }
            files = {
                "input_reference": (
                    spec.input_image.filename,
                    image_data,
                    spec.input_image.mime_type,
                )
            }
            if on_post_attempt is not None:
                on_post_attempt()
            response = session.post(
                url,
                headers={**auth_headers(api_key), "Accept": "application/json"},
                data=fields,
                files=files,
                timeout=request_timeout,
            )
    except RequestException as exc:
        raise SubmissionUnknown(f"video submission outcome is unknown: {exc}") from exc

    try:
        job = json_object_response(response, "video submission")
    except RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if (
            status_code is not None
            and 400 <= status_code < 500
            and status_code
            not in {
                408,
                429,
            }
        ):
            raise SubmissionRejected(f"video submission was rejected: {exc}") from exc
        raise SubmissionUnknown(f"video submission outcome is unknown: {exc}") from exc
    except VideoAPIError as exc:
        raise SubmissionUnknown(f"video submission outcome is unknown: {exc}") from exc

    try:
        raise_for_terminal_failure(job)
    except ProviderVideoFailed:
        if isinstance(job.get("id"), str) and str(job["id"]).strip():
            return job
        raise
    try:
        validate_provider_job(job)
    except VideoAPIError as exc:
        raise SubmissionUnknown(f"video submission outcome is unknown: {exc}") from exc
    return job


def poll_video(
    session: Any,
    api_key: str,
    base_url: str,
    video_id: str,
    initial_job: Mapping[str, Any],
    poll_interval: float,
    timeout: float,
    request_timeout: float,
    on_event: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    job = dict(initial_job)
    status = normalize_status(job)
    raise_for_terminal_failure(job)
    if status in COMPLETED_STATUSES:
        return job

    try:
        from requests import RequestException
    except ImportError as exc:
        raise RuntimeError("The 'requests' package is required") from exc

    deadline = time.monotonic() + timeout
    encoded_id = quote(video_id, safe="")
    poll_url = f"{base_url.rstrip('/')}/videos/{encoded_id}"
    headers = {**auth_headers(api_key), "Accept": "application/json"}

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"video generation timed out after {timeout:g}s")
        time.sleep(min(poll_interval, remaining))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"video generation timed out after {timeout:g}s")

        try:
            response = session.get(
                poll_url,
                headers=headers,
                timeout=min(request_timeout, max(0.001, deadline - time.monotonic())),
            )
            job = json_object_response(response, "video status poll")
        except RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {404, 410}:
                raise ProviderJobGone(
                    f"provider video job {video_id!r} no longer exists"
                ) from exc
            if (
                status_code is not None
                and 400 <= status_code < 500
                and status_code != 429
            ):
                raise VideoAPIError(f"video status poll failed: {exc}") from exc
            on_event({"at": utc_now(), "request_error": sanitize_error(exc, api_key)})
            continue
        except VideoAPIError as exc:
            on_event({"at": utc_now(), "response_error": sanitize_error(exc, api_key)})
            continue

        event = {"at": utc_now(), "response": job}
        on_event(event)
        status = normalize_status(job)
        raise_for_terminal_failure(job)
        if status in COMPLETED_STATUSES:
            return job
        LOGGER.info("provider job %s status: %s", video_id, status)


def extension_for_video(data_prefix: bytes, mime_type: str | None) -> str | None:
    normalized_mime = mime_type.split(";", 1)[0].lower() if mime_type else None
    if len(data_prefix) >= 12 and data_prefix[4:8] == b"ftyp":
        return ".mov" if normalized_mime == "video/quicktime" else ".mp4"
    if data_prefix.startswith(b"\x1a\x45\xdf\xa3"):
        return ".mkv" if normalized_mime == "video/x-matroska" else ".webm"
    if data_prefix.startswith(b"RIFF") and data_prefix[8:12] == b"AVI ":
        return ".avi"
    if data_prefix.startswith(b"OggS"):
        return ".ogv"
    return None


def download_video(
    session: Any,
    api_key: str,
    base_url: str,
    video_id: str,
    videos_dir: Path,
    stem: str,
    request_timeout: float,
    max_bytes: int,
) -> SavedVideo:
    encoded_id = quote(video_id, safe="")
    url = f"{base_url.rstrip('/')}/videos/{encoded_id}/content"
    response = session.get(
        url,
        headers=auth_headers(api_key),
        timeout=request_timeout,
        stream=True,
    )
    temp_path: Path | None = None
    try:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type")
        content_length_value = response.headers.get("Content-Length")
        if content_length_value:
            try:
                content_length = int(content_length_value)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > max_bytes:
                raise VideoAPIError(
                    f"video download exceeds the {max_bytes}-byte size limit"
                )

        videos_dir.mkdir(parents=True, exist_ok=True)
        prefix = bytearray()
        total_bytes = 0
        with tempfile.NamedTemporaryFile(
            "wb", dir=videos_dir, prefix=f".{stem}.", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise VideoAPIError(
                        f"video download exceeds the {max_bytes}-byte size limit"
                    )
                if len(prefix) < 64:
                    prefix.extend(chunk[: 64 - len(prefix)])
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        if total_bytes < MIN_VIDEO_BYTES:
            raise VideoAPIError(f"downloaded video is too small ({total_bytes} bytes)")
        extension = extension_for_video(bytes(prefix), content_type)
        if extension is None:
            raise VideoAPIError(
                f"downloaded content is not a recognized video ({content_type!r})"
            )
        if extension != ".mp4":
            raise VideoAPIError(
                f"FastVideo preprocessing requires MP4 content, got {extension}"
            )
        destination = videos_dir / f"{stem}{extension}"
        temp_path.replace(destination)
        return SavedVideo(path=destination, size_bytes=total_bytes)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        response.close()


def inspect_video(path: Path) -> VideoMetadata:
    """Decode an MP4 once to collect the metadata FastVideo validates."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("The 'av' package is required to index generated videos") from exc

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoAPIError(f"downloaded MP4 has no video stream: {path}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate is not None else 0.0
            width = int(stream.width)
            height = int(stream.height)
            num_frames = sum(1 for _ in container.decode(video=0))
    except VideoAPIError:
        raise
    except Exception as exc:
        raise VideoAPIError(f"could not decode generated MP4 {path}: {exc}") from exc

    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or num_frames <= 0:
        raise VideoAPIError(
            f"generated MP4 has invalid metadata: {width}x{height}, "
            f"{fps:g} fps, {num_frames} frames"
        )
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration=num_frames / fps,
        num_frames=num_frames,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_latest_manifest_records(
    path: Path, attempt_counts: dict[str, int] | None = None
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    truncate_at: int | None = None

    def consume_line(
        line_bytes: bytes, line_number: int, allow_truncated: bool
    ) -> bool:
        try:
            line = line_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            if allow_truncated and not line_bytes.endswith(b"\n"):
                LOGGER.warning("ignoring an incomplete final line in manifest %s", path)
                return False
            raise InputError(
                f"manifest {path} line {line_number} is not valid UTF-8"
            ) from exc
        if not line.strip():
            return True
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if allow_truncated and not line.endswith("\n"):
                LOGGER.warning("ignoring an incomplete final line in manifest %s", path)
                return False
            raise InputError(
                f"manifest {path} line {line_number} is invalid JSON"
            ) from exc
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("id"), str)
            or not value["id"]
            or value.get("status") not in MANIFEST_STATUSES
        ):
            raise InputError(
                f"manifest {path} line {line_number} is not a valid lifecycle record"
            )
        attempt_value = value.get("attempt")
        if attempt_value is not None and (
            not isinstance(attempt_value, int)
            or isinstance(attempt_value, bool)
            or attempt_value <= 0
        ):
            raise InputError(
                f"manifest {path} line {line_number} has an invalid attempt number"
            )
        latest[value["id"]] = value
        if attempt_counts is not None:
            if attempt_value is not None:
                attempt_counts[value["id"]] = max(
                    attempt_counts.get(value["id"], 0), attempt_value
                )
            else:
                attempt_counts[value["id"]] = attempt_counts.get(value["id"], 0) + 1
        return True

    try:
        with path.open("rb") as handle:
            pending_line: bytes | None = None
            pending_number = 0
            pending_offset = 0
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if pending_line is not None:
                    consume_line(pending_line, pending_number, allow_truncated=False)
                pending_line = line
                pending_number = line_number
                pending_offset = offset
            if pending_line is not None:
                if not consume_line(pending_line, pending_number, allow_truncated=True):
                    truncate_at = pending_offset
    except OSError as exc:
        raise InputError(f"could not read manifest {path}: {exc}") from exc
    if truncate_at is not None:
        try:
            with path.open("r+b") as handle:
                handle.truncate(truncate_at)
        except OSError as exc:
            raise InputError(f"could not repair manifest {path}: {exc}") from exc
    return latest


def completed_record_exists(record: Mapping[str, Any], output_dir: Path) -> bool:
    if record.get("status") != "succeeded":
        return False
    video_path = record.get("video_path")
    if not isinstance(video_path, str):
        return False
    path = output_dir / video_path
    return path.is_file() and path.stat().st_size >= MIN_VIDEO_BYTES


def generation_matches(previous: Mapping[str, Any], spec: GenerationSpec) -> bool:
    return (
        previous.get("prompt") == spec.record.prompt
        and previous.get("metadata") == spec.record.metadata
        and previous.get("model") == spec.model
        and previous.get("base_url") == spec.base_url
        and previous.get("request") == spec.request_identity
    )


def sanitize_error(exc: BaseException, api_key: str) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


def relative_to_output(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def video_metadata_dict(metadata: VideoMetadata) -> dict[str, Any]:
    return {
        "resolution": {"width": metadata.width, "height": metadata.height},
        "fps": metadata.fps,
        "duration": metadata.duration,
        "num_frames": metadata.num_frames,
    }


def saved_video_metadata(record: Mapping[str, Any]) -> VideoMetadata | None:
    value = record.get("video_metadata")
    if not isinstance(value, Mapping):
        return None
    resolution = value.get("resolution")
    if not isinstance(resolution, Mapping):
        return None
    try:
        metadata = VideoMetadata(
            width=int(resolution["width"]),
            height=int(resolution["height"]),
            fps=float(value["fps"]),
            duration=float(value["duration"]),
            num_frames=int(value["num_frames"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        metadata.width <= 0
        or metadata.height <= 0
        or not math.isfinite(metadata.fps)
        or metadata.fps <= 0
        or not math.isfinite(metadata.duration)
        or metadata.duration <= 0
        or metadata.num_frames <= 0
    ):
        return None
    return metadata


def sync_fastvideo_dataset(
    output_dir: Path, latest: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Write the merged-dataset index consumed by FastVideo preprocessors."""
    annotations: list[dict[str, Any]] = []
    for record in latest.values():
        if not completed_record_exists(record, output_dir):
            continue
        video_path_value = record.get("video_path")
        prompt = record.get("prompt")
        if not isinstance(video_path_value, str) or not isinstance(prompt, str):
            raise InputError("successful manifest record is missing video_path or prompt")

        relative_path = Path(video_path_value)
        if (
            len(relative_path.parts) != 2
            or relative_path.parts[0] != "videos"
            or relative_path.suffix.lower() != ".mp4"
        ):
            raise InputError(
                f"successful video path is not a FastVideo MP4: {video_path_value!r}"
            )

        absolute_path = output_dir / relative_path
        metadata = saved_video_metadata(record) or inspect_video(absolute_path)
        annotations.append({
            "path": relative_path.name,
            **video_metadata_dict(metadata),
            "size": absolute_path.stat().st_size,
            "cap": [prompt],
        })

    annotation_path = output_dir / "videos2caption.json"
    atomic_write_json(annotation_path, annotations)
    atomic_write_text(
        output_dir / "merge.txt",
        f"{output_dir / 'videos'},{annotation_path}\n",
    )
    return annotations


def prepare_output(
    output_dir: Path, videos_dir: Path, responses_dir: Path, manifest_path: Path
) -> None:
    for directory in (output_dir, videos_dir, responses_dir):
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise OSError(f"output path is not a directory: {directory}")

    for directory in (videos_dir, responses_dir):
        probe_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory, prefix=".write-test.", delete=False
            ) as handle:
                probe_path = Path(handle.name)
            probe_path.unlink()
        finally:
            if probe_path is not None:
                probe_path.unlink(missing_ok=True)

    with manifest_path.open("a", encoding="utf-8"):
        pass


def acquire_output_lock(output_dir: Path) -> OutputLock:
    lock_path = output_dir / ".veo-generator.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise RuntimeError(
            f"another generator process is already using {output_dir}"
        ) from exc
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise
    return OutputLock(path=lock_path, handle=handle)


def load_trace(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return fallback
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return value if isinstance(value, dict) else fallback


def common_manifest_record(
    spec: GenerationSpec, attempt: int, started_at: str
) -> dict[str, Any]:
    return {
        "id": spec.record.record_id,
        "prompt": spec.record.prompt,
        "metadata": spec.record.metadata,
        "mode": spec.mode,
        "model": spec.model,
        "base_url": spec.base_url,
        "request": spec.request_identity,
        "attempt": attempt,
        "started_at": started_at,
    }


def append_manifest_or_raise(path: Path, value: Mapping[str, Any]) -> None:
    try:
        append_jsonl(path, value)
    except OSError as exc:
        raise RuntimeError(f"could not write manifest {path}: {exc}") from exc


def generate_dataset(args: argparse.Namespace) -> int:
    try:
        records = read_prompt_records(args.input, args.input_format)
    except InputError as exc:
        LOGGER.error("%s", exc)
        return 2

    if not records:
        LOGGER.error("no non-empty prompt records found")
        return 2

    if args.dry_run:
        try:
            specs = build_generation_specs(records, args)
        except InputError as exc:
            LOGGER.error("%s", exc)
            return 2
        t2v_count = sum(spec.mode == "text-to-video" for spec in specs)
        LOGGER.info(
            "validated %d record(s): %d text-to-video, %d image-to-video; no API calls made",
            len(specs),
            t2v_count,
            len(specs) - t2v_count,
        )
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key and args.api_key_env == DEFAULT_API_KEY_ENV:
        api_key = os.environ.get("NVAPIKEY")
    if not api_key:
        LOGGER.error("set %s to your NVIDIA API key", args.api_key_env)
        return 2

    output_dir = args.output_dir.resolve()
    videos_dir = output_dir / "videos"
    responses_dir = output_dir / "responses"
    manifest_path = output_dir / "manifest.jsonl"
    attempt_counts: dict[str, int] = {}
    output_lock: OutputLock | None = None
    try:
        prepare_output(output_dir, videos_dir, responses_dir, manifest_path)
        output_lock = acquire_output_lock(output_dir)
        latest = load_latest_manifest_records(manifest_path, attempt_counts)
        sync_fastvideo_dataset(output_dir, latest)
        specs = build_generation_specs(records, args, latest)
        session = create_http_session()
    except (InputError, OSError, RuntimeError) as exc:
        if output_lock is not None:
            output_lock.release()
        LOGGER.error("%s", sanitize_error(exc, api_key))
        return 2

    succeeded = 0
    failed = 0
    skipped = 0
    submitted = 0

    try:
        for index, spec in enumerate(specs, start=1):
            record = spec.record
            previous = latest.get(record.record_id)
            previous_status = previous.get("status") if previous else None
            matches_previous = (
                generation_matches(previous, spec) if previous is not None else False
            )
            can_resume = False

            if previous_status == "succeeded" and not args.overwrite:
                if not matches_previous:
                    LOGGER.error(
                        "[%d/%d] %s already exists with different input or request "
                        "settings; use a new id or --overwrite",
                        index,
                        len(specs),
                        record.record_id,
                    )
                    failed += 1
                    if args.fail_fast:
                        break
                    continue
                if completed_record_exists(previous, output_dir):
                    LOGGER.info(
                        "[%d/%d] skip %s (already completed)",
                        index,
                        len(specs),
                        record.record_id,
                    )
                    skipped += 1
                    continue
                if isinstance(previous.get("provider_video_id"), str):
                    can_resume = True
                else:
                    LOGGER.error(
                        "[%d/%d] %s has a missing local video and no provider id; "
                        "use --overwrite to submit a new job",
                        index,
                        len(specs),
                        record.record_id,
                    )
                    failed += 1
                    if args.fail_fast:
                        break
                    continue

            if (
                previous_status in RESUMABLE_MANIFEST_STATUSES
                and not args.abandon_inflight
            ):
                if not matches_previous:
                    LOGGER.error(
                        "[%d/%d] %s has an in-flight provider job for different "
                        "input or request settings; use a new id or "
                        "--abandon-inflight",
                        index,
                        len(specs),
                        record.record_id,
                    )
                    failed += 1
                    if args.fail_fast:
                        break
                    continue
                if not isinstance(previous.get("provider_video_id"), str):
                    LOGGER.error(
                        "[%d/%d] %s has an in-flight state without a provider id; "
                        "use --abandon-inflight only if a duplicate job is acceptable",
                        index,
                        len(specs),
                        record.record_id,
                    )
                    failed += 1
                    if args.fail_fast:
                        break
                    continue
                can_resume = True

            if previous_status == "submission_unknown" and not args.abandon_inflight:
                LOGGER.error(
                    "[%d/%d] %s has an ambiguous earlier POST outcome; use a new "
                    "id or --abandon-inflight only if a duplicate job is acceptable",
                    index,
                    len(specs),
                    record.record_id,
                )
                failed += 1
                if args.fail_fast:
                    break
                continue

            if can_resume:
                attempt = int(previous.get("attempt", 1))
                video_id = str(previous["provider_video_id"])
                started_at = str(previous.get("started_at") or utc_now())
                LOGGER.info(
                    "[%d/%d] resume %s (provider job %s)",
                    index,
                    len(specs),
                    record.record_id,
                    video_id,
                )
            else:
                if args.limit is not None and submitted >= args.limit:
                    continue
                attempt = attempt_counts.get(record.record_id, 0) + 1
                video_id = ""
                started_at = utc_now()

            stem = f"{safe_file_stem(record.record_id)}-attempt-{attempt:04d}"
            response_path = responses_dir / f"{stem}.json"
            common = common_manifest_record(spec, attempt, started_at)
            trace_fallback = {
                "id": record.record_id,
                "mode": spec.mode,
                "model": spec.model,
                "events": [],
            }
            trace = (
                load_trace(response_path, trace_fallback)
                if can_resume
                else trace_fallback
            )
            post_attempted = False
            try:
                if can_resume:
                    saved_final = trace.get("final")
                    saved_submission = previous.get("submission_response")
                    initial_job = (
                        dict(saved_final)
                        if isinstance(saved_final, Mapping)
                        else dict(saved_submission)
                        if isinstance(saved_submission, Mapping)
                        else {
                            "id": video_id,
                            "status": previous.get("provider_status", "submitted"),
                        }
                    )
                else:
                    LOGGER.info(
                        "[%d/%d] submit %s (%s)",
                        index,
                        len(specs),
                        record.record_id,
                        spec.mode,
                    )

                    def mark_post_attempt() -> None:
                        nonlocal post_attempted, submitted
                        post_attempted = True
                        submitted += 1

                    initial_job = submit_video(
                        session,
                        api_key,
                        spec,
                        args.request_timeout,
                        on_post_attempt=mark_post_attempt,
                    )
                    video_id = validate_provider_job(initial_job)
                    trace["submission"] = initial_job
                    trace["provider_video_id"] = video_id
                    submitted_record = {
                        **common,
                        "status": "submitted",
                        "submitted_at": utc_now(),
                        "provider_video_id": video_id,
                        "provider_status": normalize_status(initial_job),
                        "submission_response": initial_job,
                        "response_path": relative_to_output(response_path, output_dir),
                    }
                    append_manifest_or_raise(manifest_path, submitted_record)
                    latest[record.record_id] = submitted_record
                    attempt_counts[record.record_id] = attempt
                    sync_fastvideo_dataset(output_dir, latest)
                    atomic_write_json(response_path, trace)

                def on_poll_event(event: dict[str, Any]) -> None:
                    trace.setdefault("events", []).append(event)
                    atomic_write_json(response_path, trace)

                final_job = poll_video(
                    session,
                    api_key,
                    spec.base_url,
                    video_id,
                    initial_job,
                    (
                        args.poll_interval
                        if args.poll_interval is not None
                        else 10.0
                        if spec.mode == "image-to-video"
                        else 15.0
                    ),
                    args.timeout,
                    args.poll_request_timeout,
                    on_poll_event,
                )
                trace["final"] = final_job
                trace.pop("pending_error", None)
                atomic_write_json(response_path, trace)
                saved = download_video(
                    session,
                    api_key,
                    spec.base_url,
                    video_id,
                    videos_dir,
                    stem,
                    args.download_timeout,
                    int(args.max_video_mb * 1024 * 1024),
                )
                video_metadata = inspect_video(saved.path)
                manifest_record = {
                    **common,
                    "status": "succeeded",
                    "completed_at": utc_now(),
                    "provider_video_id": video_id,
                    "provider_status": normalize_status(final_job),
                    "video_path": relative_to_output(saved.path, output_dir),
                    "video_size_bytes": saved.size_bytes,
                    "video_metadata": video_metadata_dict(video_metadata),
                    "response_path": relative_to_output(response_path, output_dir),
                    "source_kind": "nvidia-content-api",
                }
                append_manifest_or_raise(manifest_path, manifest_record)
                latest[record.record_id] = manifest_record
                attempt_counts[record.record_id] = attempt
                sync_fastvideo_dataset(output_dir, latest)
                succeeded += 1
                LOGGER.info("[%d/%d] saved %s", index, len(specs), saved.path)
            except (KeyboardInterrupt, SystemExit) as exc:
                if not can_resume and post_attempted:
                    current = latest.get(record.record_id)
                    current_attempt = current.get("attempt") if current else None
                    if current_attempt != attempt:
                        if video_id:
                            interrupt_record = {
                                **common,
                                "status": "submitted",
                                "submitted_at": utc_now(),
                                "provider_video_id": video_id,
                                "provider_status": normalize_status(initial_job),
                                "submission_response": initial_job,
                                "response_path": relative_to_output(
                                    response_path, output_dir
                                ),
                            }
                        else:
                            interrupt_record = {
                                **common,
                                "status": "submission_unknown",
                                "updated_at": utc_now(),
                                "error": sanitize_error(exc, api_key),
                            }
                        append_manifest_or_raise(manifest_path, interrupt_record)
                        latest[record.record_id] = interrupt_record
                        attempt_counts[record.record_id] = attempt
                raise
            except ProviderVideoFailed as exc:
                error = sanitize_error(exc, api_key)
                trace["provider_failure"] = exc.job
                atomic_write_json(response_path, trace)
                manifest_record = {
                    **common,
                    "status": "failed",
                    "completed_at": utc_now(),
                    "provider_video_id": video_id or None,
                    "provider_status": normalize_status(exc.job),
                    "error": error,
                    "response_path": relative_to_output(response_path, output_dir),
                }
                append_manifest_or_raise(manifest_path, manifest_record)
                latest[record.record_id] = manifest_record
                attempt_counts[record.record_id] = attempt
                sync_fastvideo_dataset(output_dir, latest)
                failed += 1
                LOGGER.error(
                    "[%d/%d] %s failed: %s", index, len(specs), record.record_id, error
                )
                if args.fail_fast:
                    break
            except Exception as exc:
                error = sanitize_error(exc, api_key)
                response_written = response_path.exists()
                if video_id:
                    trace["pending_error"] = error
                    atomic_write_json(response_path, trace)
                    response_written = True
                    manifest_record = {
                        **common,
                        "status": "pending",
                        "updated_at": utc_now(),
                        "provider_video_id": video_id,
                        "provider_status": normalize_status(
                            trace.get("final", initial_job)
                        ),
                        "error": error,
                        "response_path": relative_to_output(response_path, output_dir),
                    }
                else:
                    terminal_status = (
                        "submission_unknown"
                        if isinstance(exc, SubmissionUnknown)
                        else "failed"
                    )
                    manifest_record = {
                        **common,
                        "status": terminal_status,
                        (
                            "updated_at"
                            if terminal_status == "submission_unknown"
                            else "completed_at"
                        ): utc_now(),
                        "error": error,
                    }
                    if response_written:
                        manifest_record["response_path"] = relative_to_output(
                            response_path, output_dir
                        )
                append_manifest_or_raise(manifest_path, manifest_record)
                latest[record.record_id] = manifest_record
                attempt_counts[record.record_id] = attempt
                sync_fastvideo_dataset(output_dir, latest)
                failed += 1
                LOGGER.error(
                    "[%d/%d] %s failed: %s", index, len(specs), record.record_id, error
                )
                if args.fail_fast:
                    break
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()
        if output_lock is not None:
            output_lock.release()

    LOGGER.info(
        "done: %d succeeded, %d skipped, %d failed, %d new submission(s); manifest: %s",
        succeeded,
        skipped,
        failed,
        submitted,
        manifest_path,
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a resumable NVIDIA Veo prompt-to-video dataset using the "
            "asynchronous Videos API."
        )
    )
    parser.add_argument(
        "input",
        help="JSONL/TXT prompt file, or '-' to read JSONL from standard input",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "jsonl", "text"),
        default="auto",
        help="input format (default: infer .txt as text, otherwise JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"dataset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--input-image",
        "-i",
        type=Path,
        help="input image applied to records without an input_image field",
    )
    parser.add_argument(
        "--seconds",
        "-s",
        type=int,
        choices=(4, 6, 8),
        default=8,
        help="image-to-video duration (default: 8)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help=(
            f"override model (defaults: T2V {DEFAULT_T2V_MODEL}; "
            f"I2V {DEFAULT_I2V_MODEL})"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"environment variable containing the API key (default: {DEFAULT_API_KEY_ENV})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        help="seconds between status polls (defaults: T2V 15, I2V 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="overall generation wait per record in seconds (default: 300)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="submission request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--poll-request-timeout",
        type=float,
        default=30.0,
        help="individual status request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=120.0,
        help="content download request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-video-mb",
        type=float,
        default=512.0,
        help="maximum downloaded size per video in MiB (default: 512)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="submit at most this many new provider jobs (poll/download not counted)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="submit a new job for records whose latest attempt succeeded",
    )
    parser.add_argument(
        "--abandon-inflight",
        action="store_true",
        help=(
            "submit a new job instead of resuming pending/unknown state; may "
            "duplicate a paid job"
        ),
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failed record"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate prompts and input images without making API calls",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.poll_interval is not None and (
        not math.isfinite(args.poll_interval) or args.poll_interval <= 0
    ):
        parser.error("--poll-interval must be greater than zero")
    for name in (
        "timeout",
        "request_timeout",
        "poll_request_timeout",
        "download_timeout",
        "max_video_mb",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return generate_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
