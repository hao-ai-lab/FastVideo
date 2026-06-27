"""Run the GenRL HPSv3 + VideoAlign training recipe on Modal.

This launcher intentionally mirrors the public recipe in examples/train/README.md:

1. Install FastVideo and the GenRL reward-stack requirements.
2. Run examples/train/prepare_genrl_assets.py.
3. Run examples/train/run.sh with the GenRL config.
"""

from __future__ import annotations

import modal

app = modal.App("fastvideo-genrl-hpsv3-videoalign")

PROJECT_ROOT = "/root/FastVideo"
OLD_PROJECT_ROOT = "/root/FastVideoOld"
OLD_REPRO_COMMIT = "17aecbe2dd07245333a1c0ea85f89b2b7a4a1f88"
CONFIG_PATH = "examples/train/configs/rl/wan/genrl_hpsv3_videoalign.yaml"
OUTPUT_DIR = "outputs/genrl_longcat"
LEGACY_PROMPT_DIR = "/data/filtered_prompts"
CURRENT_PROMPT_DIR = f"{PROJECT_ROOT}/.cache/genrl_filtered_prompts"
WANDB_ENTITY = "adamlee00"
WANDB_SECRET_NAME = "wandb-adamlee00"
HF_SECRET_NAME = "hf-adamlee00"
MODAL_CONTEXT_IGNORE = [
    ".git/fsmonitor--daemon.ipc",
    ".cache",
    ".cache/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "logs",
    "logs/**",
    "outputs",
    "outputs/**",
    "wandb",
    "wandb/**",
]

data_vol = modal.Volume.from_name("fastvideo-data")
runs_vol = modal.Volume.from_name("fastvideo-runs")
cache_vol = modal.Volume.from_name("fastvideo-cache")

image = (modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",
).entrypoint([]).apt_install(
    "git",
    "git-lfs",
    "ffmpeg",
    "libgl1",
    "libglib2.0-0",
    "build-essential",
    "ninja-build",
    "cmake",
).pip_install("uv").add_local_dir(
    ".",
    PROJECT_ROOT,
    copy=True,
    ignore=MODAL_CONTEXT_IGNORE,
).run_commands(
    f"cd {PROJECT_ROOT} && uv pip install --system --prerelease=allow -e .",
    "uv pip install --system --prerelease=allow "
    "--index-url https://download.pytorch.org/whl/cu128 "
    "--upgrade torch torchvision torchaudio",
    "uv pip install --system --no-cache-dir "
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
    "releases/download/v0.7.16/"
    "flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl",
    f"cd {PROJECT_ROOT} && uv pip install --system "
    "-r examples/train/requirements-genrl.txt",
    "python -c 'import flash_attn; print(\"flash_attn ok\")'",
    "python -c 'import cv2, imageio; print(\"video io ok\")'",
    "python -c 'import datasets, matplotlib, peft, qwen_vl_utils, "
    "safetensors, timm, trl; "
    "from transformers import Qwen2VLForConditionalGeneration; "
    "print(\"genrl deps ok\")'",
    "python -c 'import diffusers, peft; "
    "print(f\"diffusers {diffusers.__version__} peft {peft.__version__}\")'",
).env({
    "WANDB_MODE": "online",
    "WANDB_ENTITY": WANDB_ENTITY,
    "TOKENIZERS_PARALLELISM": "false",
    "NUM_GPUS": "4",
    "HF_HOME": f"{PROJECT_ROOT}/.cache/huggingface",
    "HF_HUB_CACHE": f"{PROJECT_ROOT}/.cache/huggingface",
    "TRANSFORMERS_CACHE": f"{PROJECT_ROOT}/.cache/huggingface",
    "VIDEOALIGN_CHECKPOINT_PATH": f"{PROJECT_ROOT}/.cache/VideoReward",
    "FORCE_QWENVL_VIDEO_READER": "opencv",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}))

old_repro_image = (modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",
).entrypoint([]).apt_install(
    "git",
    "git-lfs",
    "ffmpeg",
    "libgl1",
    "libglib2.0-0",
    "build-essential",
    "ninja-build",
    "cmake",
).pip_install("uv").pip_install(
    "fire>=0.7.0",
    "datasets==3.6.0",
    "matplotlib==3.10.3",
    "peft==0.10.0",
    "prettytable==3.8.0",
    "qwen-vl-utils==0.0.11",
    "safetensors==0.5.3",
    "timm==1.0.15",
    "trl==0.8.6",
).add_local_dir(
    ".",
    OLD_PROJECT_ROOT,
    copy=True,
    ignore=MODAL_CONTEXT_IGNORE,
).run_commands(
    f"cd {OLD_PROJECT_ROOT} && git checkout -f {OLD_REPRO_COMMIT}",
    f"cd {OLD_PROJECT_ROOT} && git submodule update --init --recursive "
    "fastvideo/train/methods/rl/reward/HPSv3 "
    "fastvideo/train/methods/rl/reward/VideoAlign",
    f"cd {OLD_PROJECT_ROOT} && uv pip install --system --prerelease=allow -e .",
    "uv pip install --system --prerelease=allow "
    "--index-url https://download.pytorch.org/whl/cu128 "
    "--upgrade torch torchvision torchaudio",
    "uv pip install --system --no-cache-dir "
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
    "releases/download/v0.7.16/"
    "flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl",
    "uv pip install --system "
    "datasets==3.6.0 "
    "diffusers==0.33.1 "
    "numpy==1.26.4 "
    "peft==0.10.0 "
    "qwen-vl-utils==0.0.11 "
    "safetensors==0.5.3 "
    "scipy==1.15.2 "
    "timm==1.0.15 "
    "transformers==4.57.3 "
    "trl==0.8.6",
    "python -c 'import flash_attn; print(\"old flash_attn ok\")'",
    "python -c 'import cv2, imageio; print(\"old video io ok\")'",
    "python -c 'import datasets, matplotlib, peft, qwen_vl_utils, "
    "safetensors, timm, trl; "
    "from transformers import Qwen2VLForConditionalGeneration; "
    "print(\"old genrl deps ok\")'",
).env({
    "TOKENIZERS_PARALLELISM": "false",
    "NUM_GPUS": "4",
    "HF_HOME": "/cache/huggingface",
    "HF_HUB_CACHE": "/cache/huggingface",
    "TRANSFORMERS_CACHE": "/cache/huggingface",
    "VIDEOALIGN_CHECKPOINT_PATH": "/cache/VideoReward",
    "FORCE_QWENVL_VIDEO_READER": "opencv",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}))


def _fixed_reward_inputs():
    import numpy as np

    num_videos = 4
    num_frames = 8
    height = 224
    width = 224
    videos = np.zeros((num_videos, num_frames, height, width, 3), dtype=np.uint8)
    videos[1, ...] = 255
    x = np.linspace(0, 255, width, dtype=np.uint8)
    videos[2, ...] = x[None, None, :, None]
    for t in range(num_frames):
        start = 16 + t * 16
        videos[3, t, 48:176, start:start + 32, 0] = 255
        videos[3, t, 48:176, start:start + 32, 1] = 64
    prompts = [
        "A black blank video.",
        "A white blank video.",
        "A smooth horizontal grayscale gradient.",
        "A red block moves steadily from left to right.",
    ]
    return videos, prompts


def _score_reward_suite(device):
    import gc
    import importlib.metadata
    import torch

    videos, prompts = _fixed_reward_inputs()
    metadata = [{} for _ in prompts]
    from fastvideo.train.methods.rl.reward.hpsv3 import (
        _HPSV3_INFERENCERS,
        hpsv3_general_score,
        hpsv3_percentile_score,
        set_hpsv3_device,
    )
    from fastvideo.train.methods.rl.reward.videoalign import (
        _VIDEOALIGN_INFERENCERS,
        set_videoalign_device,
        videoalign_mq_score,
        videoalign_ta_score,
    )

    versions = {}
    for package_name in (
        "transformers",
        "peft",
        "torch",
        "qwen-vl-utils",
        "timm",
        "safetensors",
    ):
        versions[package_name] = importlib.metadata.version(package_name)

    results = {}
    for name, factory in (
        ("hpsv3_general", hpsv3_general_score),
        ("hpsv3_percentile", hpsv3_percentile_score),
        ("videoalign_mq", videoalign_mq_score),
        ("videoalign_ta", videoalign_ta_score),
    ):
        reward_fn = factory(device)
        scores, _ = reward_fn(videos, prompts, metadata)
        values = scores["avg"].detach().float().cpu().tolist()
        results[name] = values

    set_hpsv3_device("cpu")
    set_videoalign_device("cpu")
    _HPSV3_INFERENCERS.clear()
    _VIDEOALIGN_INFERENCERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "prompts": prompts,
        "versions": versions,
        "scores": results,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={
        f"{PROJECT_ROOT}/.cache": cache_vol,
    },
    secrets=[
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def score_current_rewards():
    import torch

    device = torch.device("cuda:0")
    return _score_reward_suite(device)


@app.function(
    image=old_repro_image,
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={
        "/cache": cache_vol,
    },
    secrets=[
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def score_old_rewards():
    import os
    import sys
    import torch

    os.chdir(OLD_PROJECT_ROOT)
    sys.path.insert(0, OLD_PROJECT_ROOT)
    device = torch.device("cuda:0")
    return _score_reward_suite(device)


@app.function(
    image=image,
    gpu="H100:4",
    timeout=24 * 60 * 60,
    volumes={
        "/data": data_vol,
        f"{PROJECT_ROOT}/.cache": cache_vol,
        f"{PROJECT_ROOT}/outputs": runs_vol,
    },
    secrets=[
        modal.Secret.from_name(WANDB_SECRET_NAME),
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def train(
    output_dir: str = OUTPUT_DIR,
    max_train_steps: int | None = None,
    checkpoint_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
    check_rewards: bool = True,
    reward_diagnostics_only: bool = False,
    asset_diagnostics_only: bool = False,
):
    import hashlib
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    def summarize_prompt_file(path: Path) -> dict[str, object]:
        digest = hashlib.sha256()
        prompts = 0
        lines = 0
        first_prompt = None
        last_prompt = None
        with path.open("rb") as f:
            for raw_line in f:
                digest.update(raw_line)
                line = raw_line.strip()
                if not line:
                    continue
                lines += 1
                item = json.loads(line)
                prompt = item.get("prompt")
                if prompt:
                    prompts += 1
                    if first_prompt is None:
                        first_prompt = prompt
                    last_prompt = prompt
        return {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "jsonl_lines": lines,
            "prompt_count": prompts,
            "first_prompt": first_prompt,
            "last_prompt": last_prompt,
        }

    def compare_prompt_dirs() -> None:
        for split in ("train", "test"):
            legacy = Path(LEGACY_PROMPT_DIR) / f"{split}.json"
            current = Path(CURRENT_PROMPT_DIR) / f"{split}.json"
            if not legacy.exists():
                raise FileNotFoundError(f"Missing legacy prompt file: {legacy}")
            if not current.exists():
                raise FileNotFoundError(f"Missing current prompt file: {current}")
            legacy_summary = summarize_prompt_file(legacy)
            current_summary = summarize_prompt_file(current)
            byte_equal = legacy.read_bytes() == current.read_bytes()
            print(f"=== Prompt parity: {split}.json ===", flush=True)
            print(f"legacy={legacy_summary}", flush=True)
            print(f"current={current_summary}", flush=True)
            print(f"byte_equal={byte_equal}", flush=True)
            if not byte_equal:
                raise RuntimeError(
                    f"Prompt mismatch for {split}.json between "
                    f"{LEGACY_PROMPT_DIR} and {CURRENT_PROMPT_DIR}."
                )

    prepare_cmd = ["python", "examples/train/prepare_genrl_assets.py"]
    if check_rewards:
        prepare_cmd.append("--check-rewards")
    subprocess.run(
        prepare_cmd,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )
    if asset_diagnostics_only:
        compare_prompt_dirs()
        cache_vol.commit()
        data_vol.commit()
        return
    if reward_diagnostics_only:
        cache_vol.commit()
        return

    cmd = [
        "bash",
        "examples/train/run.sh",
        CONFIG_PATH,
        "--training.checkpoint.output_dir",
        output_dir,
    ]
    if max_train_steps is not None:
        cmd.extend(["--training.loop.max_train_steps", str(max_train_steps)])
    if checkpoint_steps is not None:
        cmd.extend([
            "--training.checkpoint.training_state_checkpointing_steps",
            str(checkpoint_steps),
        ])
    if resume_from_checkpoint:
        cmd.extend([
            "--training.checkpoint.resume_from_checkpoint",
            resume_from_checkpoint,
        ])

    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    finally:
        runs_vol.commit()
        cache_vol.commit()


@app.local_entrypoint()
def main(
    output_dir: str = OUTPUT_DIR,
    max_train_steps: int | None = None,
    checkpoint_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
    check_rewards: bool = True,
    reward_diagnostics_only: bool = False,
    asset_diagnostics_only: bool = False,
    reward_parity_only: bool = False,
):
    if reward_parity_only:
        import json

        old = score_old_rewards.remote()
        current = score_current_rewards.remote()
        print("=== GenRL fixed-video reward parity ===")
        print(json.dumps({"old": old, "current": current}, indent=2))
        for reward_name, old_values in old["scores"].items():
            current_values = current["scores"][reward_name]
            deltas = [
                current_value - old_value
                for old_value, current_value in zip(
                    old_values,
                    current_values,
                    strict=False,
                )
            ]
            print(
                f"{reward_name}: "
                f"old={old_values} current={current_values} "
                f"delta={deltas}"
            )
        return

    train.remote(
        output_dir=output_dir,
        max_train_steps=max_train_steps,
        checkpoint_steps=checkpoint_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        check_rewards=check_rewards,
        reward_diagnostics_only=reward_diagnostics_only,
        asset_diagnostics_only=asset_diagnostics_only,
    )
