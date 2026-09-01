# pyright: reportAttributeAccessIssue=false
"""Minimal Modal wrapper for the portable FastH3 RVM smoke runner.

All environment setup, asset preparation, configs, hyperparameters, training,
validation, and result collection live outside this file. Production 8/16-H100
runs use the same repository scripts directly on custom nodes; Modal is only a
1/4-GPU integration-test backend.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import modal


REPOSITORY = "https://github.com/Abecid/FastVideo.git"
DEFAULT_BRANCH = "adam/h3-rvm-posttraining"
IMAGE_REF = os.environ.get(
    "FASTVIDEO_MODAL_IMAGE",
    "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest",
)
GPU_1 = os.environ.get("H3_RVM_MODAL_GPU_1", "H100!")
GPU_4 = os.environ.get("H3_RVM_MODAL_GPU_4", "H100!:4")
SECRET_NAMES = os.environ.get(
    "H3_RVM_MODAL_SECRETS",
    "fastvideo-training",
)

image = modal.Image.from_registry(
    IMAGE_REF,
    add_python="3.12",
).apt_install(
    "ffmpeg",
    "git",
    "git-lfs",
    "libgl1",
    "libglib2.0-0",
    "ninja-build",
)
app = modal.App("fastvideo-h3-rvm-smoke")
assets = modal.Volume.from_name(
    "fastvideo-h3-rvm-assets",
    create_if_missing=True,
)
runs = modal.Volume.from_name(
    "fastvideo-h3-rvm-runs",
    create_if_missing=True,
)
secrets = [
    modal.Secret.from_name(name.strip())
    for name in SECRET_NAMES.split(",")
    if name.strip()
]


def _run(
    *,
    gpus: int,
    mode: str,
    branch: str,
    commit: str,
    run_name: str,
    config: str,
    max_steps: int,
    eval_prompts: int,
    max_train_prompts: int,
    smoke_prompts: int,
    learning_rate: float,
    force_prepare: bool,
    resume: bool,
    skip_preflight: bool,
) -> dict[str, Any]:
    name = run_name.strip()
    if not name or name == "auto":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"h3-rvm-{gpus}gpu-{stamp}"

    repository = Path("/workspace/FastVideo")
    shutil.rmtree(repository, ignore_errors=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-tags",
            "--branch",
            branch,
            REPOSITORY,
            str(repository),
        ],
        check=True,
    )
    if commit:
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", commit],
            cwd=repository,
            check=True,
        )

    env = os.environ.copy()
    env.update(
        {
            "RVM_ARTIFACT_ROOT": "/cache/rvm_h3",
            "RVM_SKIP_CONDA": "1",
            "RVM_SMOKE_MODE": mode,
            "RVM_SMOKE_GPUS": str(gpus),
            "RVM_SMOKE_RUN_NAME": name,
            "RVM_SMOKE_RUN_ROOT": "/runs/h3-rvm",
            "RVM_SMOKE_CONFIG": config,
            "RVM_SMOKE_MAX_STEPS": str(max_steps),
            "RVM_SMOKE_EVAL_PROMPTS": str(eval_prompts),
            "RVM_SMOKE_MAX_TRAIN_PROMPTS": str(max_train_prompts),
            "RVM_SMOKE_PROMPTS": str(smoke_prompts),
            "RVM_SMOKE_LEARNING_RATE": str(learning_rate),
            "RVM_SMOKE_FORCE_PREPARE": str(int(force_prepare)),
            "RVM_SMOKE_RESUME": str(int(resume)),
            "RVM_SMOKE_SKIP_PREFLIGHT": str(int(skip_preflight)),
        }
    )

    try:
        subprocess.run(
            ["bash", "examples/train/rvm_h3/12_run_portable_smoke.sh"],
            cwd=repository,
            env=env,
            check=True,
        )
    finally:
        with suppress(Exception):
            assets.commit()
        with suppress(Exception):
            runs.commit()

    manifest = Path("/runs/h3-rvm") / name / "run_manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))


_COMMON = {
    "image": image,
    "cpu": 32,
    "memory": 262_144,
    "ephemeral_disk": 524_288,
    "timeout": 43_000,
    "startup_timeout": 4_800,
    "secrets": secrets,
    "volumes": {"/cache": assets, "/runs": runs},
}


@app.function(gpu=GPU_1, **_COMMON)
def run_1gpu(options: dict[str, Any]) -> dict[str, Any]:
    return _run(gpus=1, **options)


@app.function(gpu=GPU_4, **_COMMON)
def run_4gpu(options: dict[str, Any]) -> dict[str, Any]:
    return _run(gpus=4, **options)


@app.local_entrypoint()
def main(
    gpus: int = 1,
    mode: str = "smoke",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    run_name: str = "auto",
    config: str = "",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> None:
    if gpus not in {1, 4}:
        raise ValueError(
            "--gpus must be 1 or 4; use custom-node scripts for 8/16 GPUs"
        )
    kwargs = {
        "mode": mode,
        "branch": branch,
        "commit": commit,
        "run_name": run_name,
        "config": config,
        "max_steps": max_steps,
        "eval_prompts": eval_prompts,
        "max_train_prompts": max_train_prompts,
        "smoke_prompts": smoke_prompts,
        "learning_rate": learning_rate,
        "force_prepare": force_prepare,
        "resume": resume,
        "skip_preflight": skip_preflight,
    }
    result = (
        run_1gpu if gpus == 1 else run_4gpu
    ).remote(kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
