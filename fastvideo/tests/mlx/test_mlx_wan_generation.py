# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-weight serving smoke tests on Apple Silicon with ffmpeg installed.

Run with FASTVIDEO_TEST_MLX_WAN_SMOKE=1 after downloading FastMetal weights.
Select one family with pytest -k wan21 or -k wan22. Both fixtures have 17
frames; Wan2.1 uses 256x256 and Wan2.2 uses its recipe resolution, 1280x704.
These do not establish memory needs for the 81-frame example defaults.
Set FASTVIDEO_TEST_MLX_WAN_COMPARE_CLI=1 for 5B CLI pixel parity.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pytest

from fastvideo import envs

ROOT = Path(__file__).resolve().parents[3]


def _probe_and_decode(path: Path, fps: int, width: int, height: int) -> np.ndarray:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_read_frames"])) == (width, height, 17)
    assert stream["r_frame_rate"] == f"{fps}/1"
    decoded = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], check=True, capture_output=True)
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(17, height, width, 3)
    assert frames.std() > 1.0, "Generated video is blank or constant"
    print(f"Verified MP4 {path}: {stream}", flush=True)
    return frames


@pytest.mark.parametrize("family,config_name,model_dir,fps,width,height", [
    ("wan21", "mlx_wan21_1_3b.yaml", "FastMetal-1.3B-QAD", 16, 256, 256),
    ("wan22", "mlx_wan22_5b.yaml", "FastMetal-5B-QAD", 24, 1280, 704),
])
def test_real_wan_serving_generation(tmp_path, family, config_name, model_dir, fps, width, height):
    if not envs.FASTVIDEO_TEST_MLX_WAN_SMOKE.get():
        pytest.skip("Set FASTVIDEO_TEST_MLX_WAN_SMOKE=1 to run real-weight Apple Silicon generation")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Real Wan MLX generation requires Apple Silicon")
    assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "Install ffmpeg for the hardware smoke test"

    from fastapi.testclient import TestClient
    from fastvideo.entrypoints.openai.mlx_wan_server import create_mlx_wan_app, load_config

    if family == "wan21":
        root_override = envs.FASTVIDEO_TEST_MLX_WAN21_MODEL_ROOT.get()
        checkpoint_override = envs.FASTVIDEO_TEST_MLX_WAN21_CHECKPOINT.get()
    else:
        root_override = envs.FASTVIDEO_TEST_MLX_WAN22_MODEL_ROOT.get()
        checkpoint_override = envs.FASTVIDEO_TEST_MLX_WAN22_CHECKPOINT.get()
    model_root = Path(root_override) if root_override else ROOT / model_dir
    checkpoint = Path(checkpoint_override) if checkpoint_override else model_root
    assert (checkpoint / "mlx_dit.safetensors").is_file(), f"Missing packed DiT under {checkpoint}"
    config = load_config(str(ROOT / "examples" / "serving" / config_name))
    config.generator.model_root = str(model_root)
    config.generator.mlx_checkpoint = str(checkpoint)
    config.generator.prompt_cache_dir = str(envs.FASTVIDEO_TEST_MLX_WAN_PROMPT_CACHE_DIR.get() or tmp_path / "prompt_cache")
    config.server.output_dir = str(tmp_path / "server")
    artifacts = Path(envs.FASTVIDEO_TEST_MLX_WAN_OUTPUT_DIR.get() or tmp_path) / family
    artifacts.mkdir(parents=True, exist_ok=True)
    prompt = "A fox runs through fresh snow."
    with TestClient(create_mlx_wan_app(config)) as client:
        response = client.post("/v1/videos/sync", json={
            "model": config.server.served_model_name, "prompt": prompt,
            "size": f"{width}x{height}", "num_frames": 17, "fps": fps, "seed": 1234,
        })
        assert response.status_code == 200, response.text[:1000] if response.status_code != 200 else ""
    video = artifacts / f"{family}_server.mp4"
    video.write_bytes(response.content)
    server_frames = _probe_and_decode(video, fps, width, height)

    if family == "wan22" and envs.FASTVIDEO_TEST_MLX_WAN_COMPARE_CLI.get():
        reference = artifacts / "wan22_cli.mp4"
        subprocess.run([
            sys.executable, str(ROOT / "examples/inference/basic/mlx_wan22_generate.py"),
            "--mlx-checkpoint", str(checkpoint), "--text-encoder-root", str(model_root),
            "--text-encoder-device", "cpu", "--prompt", prompt, "--seed", "1234",
            "--height", str(height), "--width", str(width), "--num-frames", "17", "--fps", str(fps),
            "--output-path", str(reference), "--prompt-embeds-cache", str(tmp_path / "cli_prompt.npy"),
            "--compile",
        ], check=True, cwd=ROOT)
        reference_frames = _probe_and_decode(reference, fps, width, height)
        np.testing.assert_array_equal(server_frames, reference_frames)
        print("5B server/CLI decoded pixels are identical", flush=True)
