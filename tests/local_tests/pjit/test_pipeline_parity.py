# SPDX-License-Identifier: Apache-2.0
"""Compare the public FastVideo example with Linum's seed-2, 50-step portrait.

Set PJIT_RUN_CUDA_PARITY=1 for the unchanged Linum CLI, or PJIT_RUN_MPS_PARITY=1
for the Metal reference helper. Set PJIT_REFERENCE_ROOT to the pinned Linum checkout.
Set PJIT_MODEL_DIR / PJIT_QWEN_MODEL_DIR to local official weights or cache the
pinned Hub snapshots below. Run pytest -s to see output paths and pixel metrics.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from fastvideo.tests.utils import compute_video_ssim_torchvision

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "95403933f3727abe947977c75915a53292e1d841"
PARITY_SCOPE = "pipeline"


def _weights(env: str, repo: str, revision: str) -> str:
    if local := os.environ.get(env):
        path = Path(local).resolve()
        assert path.is_dir(), f"{env}={path} does not exist"
        return str(path)
    return snapshot_download(repo, revision=revision, local_files_only=True)


@pytest.mark.parametrize("device", [
    pytest.param("cuda", marks=pytest.mark.skipif(os.environ.get("PJIT_RUN_CUDA_PARITY") != "1",
                                                   reason="set PJIT_RUN_CUDA_PARITY=1 on CUDA")),
    pytest.param("mps", marks=pytest.mark.skipif(os.environ.get("PJIT_RUN_MPS_PARITY") != "1",
                                                  reason="set PJIT_RUN_MPS_PARITY=1 on Metal")),
])
def test_pjit_official_seed2_portrait(tmp_path, device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    assert available, f"Requested full-quality PJiT parity requires {device}"
    assert "PJIT_REFERENCE_ROOT" in os.environ, "Set PJIT_REFERENCE_ROOT to Linum's reference checkout"
    upstream = Path(os.environ["PJIT_REFERENCE_ROOT"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION, f"Unexpected Linum reference revision {revision}"
    weights = _weights("PJIT_MODEL_DIR", "Linum-AI/pjit-21k", "251725ad9a0618f0a4ba3cdf4a488d35312be352")
    qwen = _weights("PJIT_QWEN_MODEL_DIR", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ref_dir = tmp_path / "linum"
    native_png = tmp_path / "fastvideo.png"
    if device == "cuda":
        subprocess.run([
            sys.executable, "generate.py", "--weights", weights, "--qwen_model_path", qwen, "--no_upsample",
            "--aspect_ratio", "tall", "--seeds", "2", "--sampling_steps", "50", "--guidance_scale", "15",
            "--attention_backend", "sdpa", "--out_dir", str(ref_dir),
        ], cwd=upstream, check=True)
    else:
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("run_linum_mps.py")), "--reference", str(upstream),
            "--weights", weights, "--qwen", qwen, "--output-dir", str(ref_dir),
        ], check=True)
    # Use the literal caption saved by Linum, without invoking a prompt upsampler.
    recipe = json.loads((ref_dir / "prompt.json").read_text())
    assert recipe["seeds"] == [2] and recipe["upsampled"] is False
    subprocess.run([
        sys.executable, str(ROOT / "examples/inference/basic/basic_pjit_21k.py"), "--model-path", weights,
        "--qwen-model-path", qwen, "--prompt", recipe["caption"], "--width", "864", "--height", "1216",
        "--steps", "50", "--seed", "2", "--output", str(native_png),
    ], cwd=ROOT, env={**os.environ, "FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA"}, check=True)

    ref_png = ref_dir / "seed_2.png"
    with Image.open(ref_png) as ref, Image.open(native_png) as native:
        assert ref.size == native.size == (864, 1216)
        assert ref.mode == native.mode == "RGB"
        delta = np.abs(np.asarray(ref, dtype=np.int16) - np.asarray(native, dtype=np.int16))
    ms_ssim = float(compute_video_ssim_torchvision(str(ref_png), str(native_png))[0])
    report = {
        "reference_png": str(ref_png), "fastvideo_png": str(native_png),
        "device": torch.cuda.get_device_name() if device == "cuda" else "mps", "torch": torch.__version__, "ms_ssim": ms_ssim,
        "max_abs_8bit": int(delta.max()), "mean_abs_8bit": float(delta.mean()),
        "identical_rgb_pixel_fraction": float((delta == 0).all(axis=-1).mean()),
    }
    print(json.dumps(report, indent=2))
    assert ms_ssim >= 0.98, report
