# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the OpenAI-compatible API server.

These tests spin up a real ``fastvideo serve`` process with a model,
send HTTP requests, and validate responses.  They require a GPU and
network access to download the model weights.
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest
import requests

REQUIRED_GPUS = 1

MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Keep compute low — CI budget is limited.
NUM_INFERENCE_STEPS = 4
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 17
SEED = 42
GUIDANCE_SCALE = 5.0

# How long to wait for the server to become healthy (model download +
# load can take a while on a cold cache).
SERVER_STARTUP_TIMEOUT_S = 600  # 10 minutes

# How long to wait for a single generation request to finish.
GENERATION_TIMEOUT_S = 300  # 5 minutes

# Polling interval when waiting for async video jobs.
POLL_INTERVAL_S = 5

TEST_PROMPT = ("A small orange cat sitting on a windowsill watching raindrops.")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    """Start a ``fastvideo serve`` subprocess and wait for health."""
    port = _find_free_port()
    output_dir = tempfile.mkdtemp(prefix="fv_api_test_")
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(_find_free_port()))

    cmd = [
        sys.executable,
        "-m",
        "fastvideo.entrypoints.openai.api_server",
        "--model-path",
        MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--output-dir",
        output_dir,
        "--num-gpus",
        "1",
        "--dit-cpu-offload",
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Poll /health until the server is ready.
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_S
        healthy = False
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    healthy = True
                    break
            except requests.ConnectionError:
                pass
            # Check if the process died.
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode(
                    errors="replace") if proc.stdout else ""
                pytest.fail(
                    f"Server exited with code {proc.returncode} "
                    f"before becoming healthy.\n--- stdout ---\n{stdout}")
            time.sleep(2)

        if not healthy:
            stdout = proc.stdout.read().decode(
                errors="replace") if proc.stdout else ""
            proc.kill()
            proc.wait()
            pytest.fail(
                "Server did not become healthy within "
                f"{SERVER_STARTUP_TIMEOUT_S}s.\n--- stdout ---\n{stdout}")

        yield {"base_url": base_url, "output_dir": output_dir}

    finally:
        # Graceful shutdown.
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestServerHealth:

    def test_health(self, server):
        resp = requests.get(f"{server['base_url']}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_models(self, server):
        resp = requests.get(f"{server['base_url']}/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) >= 1
        assert body["data"][0]["id"] == MODEL_PATH

    def test_model_info(self, server):
        resp = requests.get(f"{server['base_url']}/v1/model_info")
        assert resp.status_code == 200
        assert resp.json()["model_path"] == MODEL_PATH


class TestVideoGeneration:

    def test_video_generation_e2e(self, server):
        """Submit a video job, poll until done, download content."""
        base_url = server["base_url"]

        # 1. Submit generation request.
        payload = {
            "prompt": TEST_PROMPT,
            "size": f"{WIDTH}x{HEIGHT}",
            "num_frames": NUM_FRAMES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
            "guidance_scale": GUIDANCE_SCALE,
        }
        resp = requests.post(
            f"{base_url}/v1/videos",
            json=payload,
            timeout=30,
        )
        assert resp.status_code == 200, resp.text
        job = resp.json()
        video_id = job["id"]
        assert job["status"] in ("queued", "completed")

        # 2. Poll until completed.
        deadline = time.monotonic() + GENERATION_TIMEOUT_S
        status = job["status"]
        while status != "completed" and time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_S)
            poll_resp = requests.get(f"{base_url}/v1/videos/{video_id}",
                                     timeout=10)
            assert poll_resp.status_code == 200
            status = poll_resp.json()["status"]
            if status == "failed":
                pytest.fail(f"Video generation failed: {poll_resp.json()}")

        assert status == "completed", (
            f"Video not completed within {GENERATION_TIMEOUT_S}s")

        # 3. Verify the video file exists on the server filesystem.
        detail = requests.get(f"{base_url}/v1/videos/{video_id}",
                              timeout=10).json()
        assert detail.get("file_path") is not None
        assert detail["file_path"].endswith(".mp4")

        # 4. Download content and verify it looks like an MP4.
        content_resp = requests.get(
            f"{base_url}/v1/videos/{video_id}/content",
            timeout=30,
        )
        assert content_resp.status_code == 200
        assert content_resp.headers["content-type"] == "video/mp4"
        # MP4 files start with a "ftyp" box; the signature is at byte 4.
        assert len(content_resp.content) > 8
        assert content_resp.content[4:8] == b"ftyp"

    def test_list_videos_after_generation(self, server):
        """After the e2e test above, listing should return >= 1 video."""
        resp = requests.get(f"{server['base_url']}/v1/videos", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) >= 1

    def test_video_not_found(self, server):
        resp = requests.get(
            f"{server['base_url']}/v1/videos/nonexistent_id",
            timeout=10,
        )
        assert resp.status_code == 404
