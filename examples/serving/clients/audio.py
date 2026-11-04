# SPDX-License-Identifier: Apache-2.0
"""Generate audio through a local FastVideo media-job endpoint.

Audio uses the same asynchronous job routes as video because the queue,
polling and deletion contracts are shared. Read ``media_type`` from the job:
successful audio jobs return ``audio/wav`` and a WAV download.

Install HTTP dependencies with: python -m pip install httpx
"""

import os
import time
from pathlib import Path

import httpx

BASE_URL = os.environ.get("FASTVIDEO_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("FASTVIDEO_MODEL", "stable-audio")
PROMPT = os.environ.get("FASTVIDEO_PROMPT", "Rain on a metal roof, distant thunder.")


def main() -> None:
    body = {
        "model": MODEL,
        "prompt": PROMPT,
        "seconds": int(os.environ.get("FASTVIDEO_SECONDS", "6")),
    }
    video_reference = os.environ.get("FASTVIDEO_VIDEO_REFERENCE")
    if video_reference:
        body["video_reference"] = {"video_url": video_reference}

    headers = {"Authorization": f"Bearer {os.environ.get('FASTVIDEO_API_KEY', 'local')}"}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60.0) as client:
        response = client.post("/videos", json=body)
        response.raise_for_status()
        job = response.json()
        job_id = job["id"]
        print(f"Submitted {job_id}")

        deadline = time.monotonic() + 1800
        while job["status"] in {"queued", "in_progress"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Polling timed out; job {job_id} may still be running")
            time.sleep(min(2, remaining))
            job = client.get(f"/videos/{job_id}", timeout=min(60, remaining)).json()

        if job["status"] != "completed":
            message = (job.get("error") or {}).get("message", job["status"])
            raise RuntimeError(f"Audio job {job_id} failed: {message}")

        media_type = job.get("media_type")
        if media_type != "audio/wav":
            raise RuntimeError(f"Expected audio/wav, got {media_type!r}")

        output = Path(f"{job_id}.wav")
        with client.stream("GET", f"/videos/{job_id}/content", timeout=300.0) as download:
            download.raise_for_status()
            with output.open("wb") as stream:
                for chunk in download.iter_bytes():
                    stream.write(chunk)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
