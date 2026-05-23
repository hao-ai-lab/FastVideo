"""Run the GenRL LongCat RL training job on Modal.

Expected Modal resources in the hao-ai-lab workspace:
- Volume `fastvideo-data` containing `/filtered_prompts/train.json` and
  `/filtered_prompts/test.json` with the real LongCat prompt JSON files.
- Volume `fastvideo-runs` for checkpoints/logs.
- Volume `fastvideo-cache` for Hugging Face and reward-model caches.
- Secrets `wandb-adamlee00` and `hf-adamlee00`.
"""

import modal

app = modal.App("fastvideo-genrl-longcat")

DATASET_DIR = "/data/filtered_prompts"
OUTPUT_DIR_BASE = "/outputs/genrl_longcat"
VIDEOALIGN_DIR = "/cache/VideoReward"
WANDB_ENTITY = "adamlee00"
WANDB_SECRET_NAME = "wandb-adamlee00"
HF_SECRET_NAME = "hf-adamlee00"
MIN_TRAIN_PROMPTS = 4

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
        "datasets==3.6.0",
        "matplotlib==3.10.3",
        "peft==0.10.0",
        "prettytable==3.8.0",
        "qwen-vl-utils==0.0.11",
        "safetensors==0.5.3",
        "timm==1.0.15",
        "trl==0.8.6",
    )
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
        "python -c 'from torchvision.transforms import InterpolationMode; "
        "print(InterpolationMode.BICUBIC)'",
        "python -c 'import datasets, matplotlib, peft, "
        "qwen_vl_utils, safetensors, timm, trl; "
        "from transformers import Qwen2VLForConditionalGeneration'",
    )
    .env(
        {
            "WANDB_MODE": "online",
            "WANDB_ENTITY": WANDB_ENTITY,
            "TOKENIZERS_PARALLELISM": "false",
            "NUM_GPUS": "4",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "VIDEOALIGN_CHECKPOINT_PATH": VIDEOALIGN_DIR,
            "FORCE_QWENVL_VIDEO_READER": "opencv",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
        modal.Secret.from_name(WANDB_SECRET_NAME),
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def train():
    from datetime import datetime, timezone
    import gc
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    def require_prompt_dataset(dataset_dir: str) -> None:
        def count_json_prompts(path: Path) -> int:
            count = 0
            with path.open(encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("prompt"):
                        count += 1
            return count

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

        prompt_count = count_json_prompts(train_path)
        if prompt_count < MIN_TRAIN_PROMPTS:
            raise RuntimeError(
                f"{train_path} has only {prompt_count} usable prompts; "
                f"this launch needs at least {MIN_TRAIN_PROMPTS} so all "
                "distributed ranks receive a non-empty prompt batch."
            )

    def ensure_videoalign_checkpoint() -> None:
        def has_complete_checkpoint(root: Path) -> bool:
            for ckpt in root.glob("checkpoint-*"):
                if (ckpt / "model.pth").exists():
                    return True
                if (
                    (ckpt / "adapter_model.safetensors").exists()
                    and (ckpt / "non_lora_state_dict.pth").exists()
                ):
                    return True
            return False

        model_config = Path(VIDEOALIGN_DIR) / "model_config.json"
        if (
            model_config.exists()
            and has_complete_checkpoint(Path(VIDEOALIGN_DIR))
        ):
            return

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="KwaiVGI/VideoReward",
            repo_type="model",
            local_dir=VIDEOALIGN_DIR,
            local_dir_use_symlinks=False,
        )

    def preflight_runtime() -> None:
        import numpy as np
        import torch

        print("=== GenRL Modal Preflight ===", flush=True)
        import peft
        import transformers
        import torchvision

        print(
            "Versions: "
            f"torch={torch.__version__} "
            f"torchvision={torchvision.__version__} "
            f"transformers={transformers.__version__} "
            f"peft={peft.__version__}",
            flush=True,
        )

        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError

        hf_token_present = bool(
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )
        for attempt in range(3):
            try:
                whoami = HfApi().whoami()
                print(
                    f"HF auth user: {whoami.get('name', '<unknown>')}",
                    flush=True,
                )
                break
            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (401, 403):
                    raise RuntimeError(
                        "Hugging Face token was rejected. Verify the "
                        f"`{HF_SECRET_NAME}` Modal secret."
                    ) from exc
                if attempt == 2 and hf_token_present:
                    print(
                        "HF whoami check failed after retries; continuing "
                        "because HF token env is present.",
                        flush=True,
                    )
                elif attempt == 2:
                    raise
                else:
                    time.sleep(2**attempt)
            except Exception:
                if attempt == 2 and hf_token_present:
                    print(
                        "HF whoami check hit a transient network error; "
                        "continuing because HF token env is present.",
                        flush=True,
                    )
                elif attempt == 2:
                    raise
                else:
                    time.sleep(2**attempt)

        print(
            "W&B target: "
            f"entity={os.environ.get('WANDB_ENTITY')} "
            f"project=VideoRL",
            flush=True,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in Modal preflight.")
        device = torch.device("cuda:0")

        # HPSv3 catches checkpoint/key-layout drift before Wan sampling.
        from fastvideo.train.methods.rl.reward.hpsv3 import (
            _HPSV3_INFERENCERS,
            hpsv3_general_score,
            hpsv3_percentile_score,
            set_hpsv3_device,
        )

        dummy_video = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)
        for name, factory in (
            ("HPSv3-general", hpsv3_general_score),
            ("HPSv3-percentile", hpsv3_percentile_score),
        ):
            reward = factory(device)
            scores, _ = reward(
                dummy_video,
                ["preflight prompt"],
                {},
            )
            value = float(scores["avg"].detach().cpu()[0])
            print(f"{name} preflight score: {value:.4f}", flush=True)
        set_hpsv3_device("cpu")
        _HPSV3_INFERENCERS.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # VideoAlign catches checkpoint/dependency issues before training.
        from fastvideo.train.methods.rl.reward.videoalign import (
            _VIDEOALIGN_INFERENCERS,
            set_videoalign_device,
            videoalign_mq_score,
            videoalign_ta_score,
        )

        dummy_video = np.zeros((1, 8, 224, 224, 3), dtype=np.uint8)
        for name, factory in (
            ("VideoAlign-MQ", videoalign_mq_score),
            ("VideoAlign-TA", videoalign_ta_score),
        ):
            reward = factory(device)
            scores, _ = reward(
                dummy_video,
                ["preflight prompt"],
                {},
            )
            value = float(scores["avg"].detach().cpu()[0])
            print(f"{name} preflight score: {value:.4f}", flush=True)

        set_videoalign_device("cpu")
        _VIDEOALIGN_INFERENCERS.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print("=== GenRL Modal Preflight OK ===", flush=True)

    require_prompt_dataset(DATASET_DIR)
    ensure_videoalign_checkpoint()
    preflight_runtime()
    output_dir = (
        f"{OUTPUT_DIR_BASE}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Output dir: {output_dir}", flush=True)
    cmd = [
        "bash",
        "examples/train/run.sh",
        "examples/train/configs/genrl_wan2.1_t2v_1.3B_longcat.yaml",
        "--method.prompt_dataset_path",
        DATASET_DIR,
        "--training.checkpoint.output_dir",
        output_dir,
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
