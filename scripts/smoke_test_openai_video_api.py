# SPDX-License-Identifier: Apache-2.0
"""Smoke-test a running OpenAI-compatible FastVideo video server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _json_request(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"content-type": "application/json"} if body is not None else {},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {error.code}: {detail}") from error


def _wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = _json_request(f"{base_url}/health")
            if response.get("status") == "ok":
                return
        except (RuntimeError, URLError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout:g}s: {last_error}")


def _download_video(url: str, output: Path) -> int:
    try:
        with urlopen(url, timeout=120) as response:
            content_type = response.headers.get_content_type()
            data = response.read()
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"GET {url} returned HTTP {error.code}: {detail}") from error
    if content_type != "video/mp4":
        raise RuntimeError(f"Expected video/mp4 content, got {content_type!r}")
    if len(data) <= 8 or data[4:8] != b"ftyp":
        raise RuntimeError("Downloaded content is not an MP4 file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return len(data)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="A fox runs through fresh snow under soft morning light.")
    parser.add_argument("--size", default="1344x768")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--lora-name")
    parser.add_argument("--lora-path")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--generation-timeout", type=float, default=1200)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    _wait_for_health(base_url, args.startup_timeout)

    models = _json_request(f"{base_url}/v1/models")
    model_ids = [model.get("id") for model in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"Requested model {args.model!r} is absent from /v1/models: {model_ids}")

    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    if args.lora_name or args.lora_path:
        payload["lora"] = {
            "name": args.lora_name,
            "path": args.lora_path,
            "scale": args.lora_scale,
        }

    job = _json_request(f"{base_url}/v1/videos", method="POST", payload=payload)
    video_id = job.get("id")
    if not video_id:
        raise RuntimeError(f"Video submission returned no id: {job}")

    deadline = time.monotonic() + args.generation_timeout
    while job.get("status") not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(args.poll_interval)
        job = _json_request(f"{base_url}/v1/videos/{video_id}")
    if job.get("status") != "completed":
        raise RuntimeError(f"Video job did not complete successfully: {job}")

    listing = _json_request(f"{base_url}/v1/videos?limit=100")
    if video_id not in {item.get("id") for item in listing.get("data", [])}:
        raise RuntimeError(f"Completed job {video_id!r} is absent from /v1/videos")
    num_bytes = _download_video(f"{base_url}/v1/videos/{video_id}/content", args.output)
    print(
        json.dumps(
            {
                "id": video_id,
                "status": job["status"],
                "model": job.get("model"),
                "inference_time_s": job.get("inference_time_s"),
                "output": str(args.output),
                "bytes": num_bytes,
            },
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
