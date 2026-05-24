"""Run a tiny DiffusionNFT Wan RL training job on Modal.

This is the DiffusionNFT smoke/debug companion to ``modal_train_genrl.py``.
Wan is used as an image generator by setting ``num_frames=1``; an image is a
one-frame video.

Expected Modal resources in the hao-ai-lab workspace:
- Volume `fastvideo-data` for generated text-only parquet prompts.
- Volume `fastvideo-runs` for checkpoints/logs.
- Volume `fastvideo-cache` for Hugging Face caches.
- Secrets `wandb-adamlee00` and `hf-adamlee00`.
"""

from __future__ import annotations

import modal

app = modal.App("fastvideo-diffusion-nft-wan-debug")

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CONFIG_PATH = (
    "examples/train/configs/diffusion_nft_wan2.1_t2i_text_only.yaml"
)
DATASET_ROOT = "/data/diffusion_nft_text_only_debug"
PROMPT_FILE = "/tmp/diffusion_nft_prompts_debug.txt"
OUTPUT_DIR_BASE = "/outputs/diffusion_nft_wan_debug"
WANDB_ENTITY = "adamlee00"
WANDB_SECRET_NAME = "wandb-adamlee00"
HF_SECRET_NAME = "hf-adamlee00"
DEBUG_PROMPT_COUNT = 4
DEBUG_PROMPTS = [
    "a cinematic photo of a red sports car parked on a wet city street",
    "a watercolor painting of a cozy cabin under northern lights",
    "a close-up studio photo of a glass teapot filled with flowers",
    "a small robot reading a book in a sunny library",
]

data_vol = modal.Volume.from_name("fastvideo-data")
runs_vol = modal.Volume.from_name("fastvideo-runs")
cache_vol = modal.Volume.from_name("fastvideo-cache")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
        "ninja-build",
        "cmake",
    )
    .pip_install("uv")
    .add_local_dir(".", "/root/FastVideo", copy=True)
    .run_commands(
        "cd /root/FastVideo && uv pip install --system --prerelease=allow -e .",
        "uv pip install --system --prerelease=allow "
        "--index-url https://download.pytorch.org/whl/cu128 "
        "--upgrade torch torchvision torchaudio",
        "uv pip install --system --no-cache-dir "
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
        "releases/download/v0.7.16/"
        "flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl",
        "uv pip install --system numpy==1.26.4 scipy==1.15.2",
        "python -c 'import flash_attn; print(\"flash_attn ok\")'",
        "python -c 'import cv2, imageio; print(\"video io ok\")'",
        "python -c 'import cloudpickle, pyarrow, torchdata; "
        "print(\"training deps ok\")'",
    )
    .env(
        {
            "WANDB_MODE": "online",
            "WANDB_ENTITY": WANDB_ENTITY,
            "TOKENIZERS_PARALLELISM": "false",
            "NUM_GPUS": "1",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


@app.function(
    image=image,
    gpu="H100:1",  # use "A100-80GB:1" if H100 queue is annoying
    timeout=12 * 60 * 60,
    volumes={
        "/data": data_vol,
        "/outputs": runs_vol,
        "/cache": cache_vol,
    },
    secrets=[
        modal.Secret.from_name(WANDB_SECRET_NAME),
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def train():
    from datetime import datetime, timezone
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path("/root/FastVideo")
    dataset_root = Path(DATASET_ROOT)
    parquet_dir = dataset_root / "combined_parquet_dataset"

    def preflight_runtime() -> None:
        import torch
        import transformers

        print("=== DiffusionNFT Modal Preflight ===", flush=True)
        print(
            "Versions: "
            f"torch={torch.__version__} "
            f"transformers={transformers.__version__}",
            flush=True,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in Modal preflight.")
        print("=== DiffusionNFT Modal Preflight OK ===", flush=True)

    def write_debug_prompt_file() -> None:
        if len(DEBUG_PROMPTS) < DEBUG_PROMPT_COUNT:
            raise RuntimeError(
                "DEBUG_PROMPTS must contain at least "
                f"{DEBUG_PROMPT_COUNT} prompts."
            )
        Path(PROMPT_FILE).write_text(
            "\n".join(DEBUG_PROMPTS[:DEBUG_PROMPT_COUNT]) + "\n",
            encoding="utf-8",
        )

    def ensure_text_only_dataset() -> None:
        if parquet_dir.exists() and list(parquet_dir.glob("*.parquet")):
            print(
                f"Using cached text-only parquet at {parquet_dir}",
                flush=True,
            )
            return

        write_debug_prompt_file()
        dataset_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            "torchrun",
            "--nnodes=1",
            "--nproc_per_node=1",
            "--master_port=29541",
            "fastvideo/pipelines/preprocess/v1_preprocess.py",
            "--model_path",
            MODEL_ID,
            "--data_merge_path",
            PROMPT_FILE,
            "--preprocess_video_batch_size",
            "2",
            "--seed",
            "42",
            "--max_height",
            "448",
            "--max_width",
            "832",
            "--num_frames",
            "1",
            "--num_latent_t",
            "1",
            "--dataloader_num_workers",
            "0",
            "--output_dir",
            DATASET_ROOT,
            "--samples_per_file",
            str(DEBUG_PROMPT_COUNT),
            "--flush_frequency",
            str(DEBUG_PROMPT_COUNT),
            "--preprocess_task",
            "text_only",
        ]
        subprocess.run(
            cmd,
            cwd=repo,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
        if not parquet_dir.exists() or not list(parquet_dir.glob("*.parquet")):
            raise RuntimeError(
                f"Text-only preprocessing produced no parquet under {parquet_dir}"
            )
        data_vol.commit()

    preflight_runtime()
    ensure_text_only_dataset()

    output_dir = (
        f"{OUTPUT_DIR_BASE}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Output dir: {output_dir}", flush=True)

    cmd = [
        "bash",
        "examples/train/run.sh",
        CONFIG_PATH,
        "--training.data.data_path",
        str(parquet_dir),
        "--training.checkpoint.output_dir",
        output_dir,
        "--training.tracker.run_name",
        Path(output_dir).name,
        "--training.loop.max_train_steps",
        "2",
        "--method.num_images_per_prompt",
        "2",
        "--method.inner_epochs",
        "1",
        "--method.train_batch_size",
        "1",
    ]

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        cmd,
        cwd=repo,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )

    runs_vol.commit()
    cache_vol.commit()


@app.local_entrypoint()
def main():
    train.remote()
