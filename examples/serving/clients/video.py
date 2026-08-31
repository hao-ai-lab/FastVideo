# SPDX-License-Identifier: Apache-2.0
"""Generate with a local FastVideo server through the OpenAI Python client.

Install the tested client with: python -m pip install openai==3.6.0
Start the server first. No OpenAI account or cloud API key is needed.
"""

import os
import time
from pathlib import Path

from openai import OpenAI


def main() -> None:
    with OpenAI(
            base_url=os.environ.get("FASTVIDEO_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("FASTVIDEO_API_KEY", "local"),
            timeout=60.0,
            max_retries=0,
    ) as client:
        video = client.videos.create(
            model=os.environ.get("FASTVIDEO_MODEL", "fasth3"),
            prompt="A fox runs through fresh snow.",
        )
        print(f"Submitted {video.id}")

        # This deadline limits polling, not GPU execution. Increase it for
        # longer jobs. On timeout, use the printed ID to retrieve the job.
        deadline = time.monotonic() + 1800
        while video.status in {"queued", "in_progress"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Polling timed out; job {video.id} may still be running")
            time.sleep(min(2, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Polling timed out; job {video.id} may still be running")
            video = client.videos.retrieve(video.id, timeout=min(60, remaining))

        if video.status != "completed":
            message = video.error.message if video.error else video.status
            raise RuntimeError(f"Video {video.id} failed: {message}")

        output = Path(f"{video.id}.mp4")
        content = client.videos.download_content(video.id)
        content.write_to_file(output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
