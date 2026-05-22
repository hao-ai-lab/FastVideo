"""Run the GenRL LongCat RL training job on Modal.

Expected Modal resources in the hao-ai-lab workspace:
- Volume `fastvideo-data` containing `/filtered_prompts/train.json` and
  `/filtered_prompts/test.json` with the real LongCat prompt JSON files.
- Volume `fastvideo-runs` for checkpoints/logs.
- Volume `fastvideo-cache` for Hugging Face and reward-model caches.
- Secrets `wandb-secret` and `hf-secret`.
"""

import modal

app = modal.App("fastvideo-genrl-longcat")

DATASET_DIR = "/data/filtered_prompts"
OUTPUT_DIR = "/outputs/genrl_longcat_probe_001"
RUN_NAME = "genrl_longcat_probe_001"
VIDEOALIGN_DIR = "/cache/VideoReward"

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
    .pip_install(
        "fire>=0.7.0",
        "trl>=0.7.0",
    )
    .add_local_dir(".", "/root/FastVideo", copy=True)
    .run_commands(
        "cd /root/FastVideo && uv pip install --system --prerelease=allow -e .",
        "uv pip install --system --prerelease=allow "
        "--index-url https://download.pytorch.org/whl/cu128 "
        "--upgrade torch torchvision torchaudio",
        "uv pip install --system numpy==1.26.4 scipy==1.15.2",
        "python -c 'from torchvision.transforms import InterpolationMode; "
        "print(InterpolationMode.BICUBIC)'",
    )
    .env(
        {
            "WANDB_MODE": "online",
            "TOKENIZERS_PARALLELISM": "false",
            "NUM_GPUS": "4",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "VIDEOALIGN_CHECKPOINT_PATH": VIDEOALIGN_DIR,
        }
    )
)

@app.function(
    image=image,
    gpu="H100:4",  # use "A100-80GB:4" if H100 queue is annoying
    timeout=24 * 60 * 60,
    volumes={
        "/data": data_vol,
        "/outputs": runs_vol,
        "/cache": cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
)
def train():
    import json
    import subprocess
    import sys
    from pathlib import Path

    def require_prompt_dataset(dataset_dir: str) -> None:
        train_path = Path(dataset_dir) / "train.json"
        test_path = Path(dataset_dir) / "test.json"
        if not train_path.exists() or not test_path.exists():
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        "set -euo pipefail; "
                        "if [ ! -d /cache/GenRL/.git ]; then "
                        "git clone https://github.com/ModelTC/GenRL.git "
                        "/cache/GenRL; "
                        "fi; "
                        "cd /cache/GenRL; "
                        "git lfs install; "
                        "git lfs pull -I 'datasets/filtered_prompts/*'"
                    ),
                ],
                check=True,
            )
            Path(dataset_dir).mkdir(parents=True, exist_ok=True)
            for name in ("train.json", "test.json"):
                src = (
                    Path("/cache/GenRL")
                    / "datasets"
                    / "filtered_prompts"
                    / name
                )
                dst = Path(dataset_dir) / name
                if src.exists() and not dst.exists():
                    dst.write_bytes(src.read_bytes())
            data_vol.commit()

        for path in (train_path, test_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Upload GenRL filtered_prompts "
                    "to the `fastvideo-data` Modal volume."
                )
            with path.open(encoding="utf-8") as f:
                first = f.readline().strip()
            if first.startswith("version https://git-lfs.github.com"):
                raise RuntimeError(
                    f"{path} is a Git LFS pointer, not the real prompt "
                    "JSON. Pull LFS locally or upload the real file to "
                    "the Modal volume."
                )
            json.loads(first)

    def ensure_videoalign_checkpoint() -> None:
        model_config = Path(VIDEOALIGN_DIR) / "model_config.json"
        if model_config.exists():
            return

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="KwaiVGI/VideoReward",
            repo_type="model",
            local_dir=VIDEOALIGN_DIR,
            local_dir_use_symlinks=False,
        )

    require_prompt_dataset(DATASET_DIR)
    ensure_videoalign_checkpoint()
    cmd = [
        "bash",
        "examples/train/run.sh",
        "examples/train/configs/genrl_wan2.1_t2v_1.3B_longcat.yaml",
        "--method.prompt_dataset_path",
        DATASET_DIR,
        "--training.checkpoint.output_dir",
        OUTPUT_DIR,
        "--training.tracker.run_name",
        RUN_NAME,
    ]

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        cmd,
        cwd="/root/FastVideo",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )

    runs_vol.commit()
    cache_vol.commit()


@app.local_entrypoint()
def main():
    train.remote()
