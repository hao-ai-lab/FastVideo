"""Run DiffusionNFT Wan RL training on Modal.

By default this uses the real NVlabs/DiffusionNFT prompt datasets and
Flow-GRPO reward implementations. ``num_frames`` is exposed as a Modal
argument so the same video-policy path can run cheap one-frame debug jobs or
multi-frame video generation.

Expected Modal resources in the hao-ai-lab workspace:
- Volume `fastvideo-data` mounted under the repo for text-only parquet prompts.
- Volume `fastvideo-runs` mounted under the repo for checkpoints/logs.
- Volume `fastvideo-cache` mounted under the repo for Hugging Face/DiffusionNFT caches.
- Secrets `wandb-adamlee00` and `hf-adamlee00`.
"""

from __future__ import annotations

import modal

app = modal.App("fastvideo-diffusion-nft-wan")

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CONFIG_PATH = (
    "examples/train/configs/diffusion_nft_wan2.1_t2v_text_only.yaml"
)
PROJECT_ROOT = "/root/FastVideo"
MODAL_DATA_ROOT = f"{PROJECT_ROOT}/.modal_data"
MODAL_CACHE_ROOT = f"{PROJECT_ROOT}/.modal_cache"
DIFFUSION_NFT_REPO = "https://github.com/NVlabs/DiffusionNFT.git"
DIFFUSION_NFT_ROOT = f"{MODAL_CACHE_ROOT}/DiffusionNFT"
OUTPUT_DIR_BASE = f"{PROJECT_ROOT}/outputs/diffusion_nft_wan"
WANDB_ENTITY = "adamlee00"
WANDB_SECRET_NAME = "wandb-adamlee00"
HF_SECRET_NAME = "hf-adamlee00"
DEFAULT_GPU_TYPE = "A100-80GB"
DEFAULT_NUM_GPUS = 4
DEFAULT_MAX_TRAIN_STEPS = 100
DEFAULT_NUM_SAMPLES_PER_PROMPT = 24
DEFAULT_COLLECTION_BATCH_SIZE = 6
DEFAULT_INNER_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 6
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 48
DEFAULT_NUM_FRAMES = 1
DEFAULT_NUM_LATENT_T = 0
DEFAULT_PREPROCESS_BATCH_SIZE = 128
DEFAULT_PREPROCESS_NUM_GPUS = 1
DEFAULT_DATASET = "pickscore"
DEFAULT_REWARD = "pickscore"
DEFAULT_MAX_PROMPTS = "quarter"
MULTI_REWARD_NAMES = ("pickscore", "hpsv2", "clipscore")

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
    .add_local_dir(".", PROJECT_ROOT, copy=True)
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
            "NUM_GPUS": str(DEFAULT_NUM_GPUS),
            "HF_HOME": f"{MODAL_CACHE_ROOT}/huggingface",
            "HF_HUB_CACHE": f"{MODAL_CACHE_ROOT}/huggingface",
            "TRANSFORMERS_CACHE": f"{MODAL_CACHE_ROOT}/huggingface",
            "DIFFUSION_NFT_ROOT": DIFFUSION_NFT_ROOT,
            "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


@app.function(
    image=image,
    gpu=f"{DEFAULT_GPU_TYPE}:{DEFAULT_NUM_GPUS}",
    timeout=12 * 60 * 60,
    volumes={
        MODAL_DATA_ROOT: data_vol,
        f"{PROJECT_ROOT}/outputs": runs_vol,
        MODAL_CACHE_ROOT: cache_vol,
    },
    secrets=[
        modal.Secret.from_name(WANDB_SECRET_NAME),
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def train(
    max_train_steps: int = DEFAULT_MAX_TRAIN_STEPS,
    num_samples_per_prompt: int = DEFAULT_NUM_SAMPLES_PER_PROMPT,
    collection_batch_size: int = DEFAULT_COLLECTION_BATCH_SIZE,
    inner_epochs: int = DEFAULT_INNER_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_latent_t: int = DEFAULT_NUM_LATENT_T,
    preprocess_batch_size: int = DEFAULT_PREPROCESS_BATCH_SIZE,
    preprocess_num_gpus: int = DEFAULT_PREPROCESS_NUM_GPUS,
    dataset: str = DEFAULT_DATASET,
    reward: str = DEFAULT_REWARD,
    max_prompts: str = DEFAULT_MAX_PROMPTS,
):
    from datetime import datetime, timezone
    import os
    import subprocess
    import sys
    from pathlib import Path

    def derive_wan_num_latent_t(frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("--num-frames must be positive")
        if (frame_count - 1) % 4 != 0:
            raise ValueError(
                "Wan frame counts must satisfy "
                "num_frames = (num_latent_t - 1) * 4 + 1. "
                f"Got num_frames={frame_count}; try 1, 5, 9, 13, 17, ..."
            )
        return ((frame_count - 1) // 4) + 1

    num_frames = int(num_frames)
    requested_num_latent_t = int(num_latent_t)
    derived_num_latent_t = derive_wan_num_latent_t(num_frames)
    if requested_num_latent_t <= 0:
        num_latent_t = derived_num_latent_t
    elif requested_num_latent_t != derived_num_latent_t:
        raise ValueError(
            f"For Wan num_frames={num_frames} implies "
            f"num_latent_t={derived_num_latent_t}, but got "
            f"num_latent_t={requested_num_latent_t}. Leave "
            "--num-latent-t unset/0 to derive it automatically."
        )
    else:
        num_latent_t = requested_num_latent_t

    repo = Path(PROJECT_ROOT)
    diffusion_nft_root = Path(DIFFUSION_NFT_ROOT)
    dataset = dataset.strip().lower()
    reward = reward.strip().lower()
    max_prompts_mode = str(max_prompts).strip().lower()
    preprocess_batch_size = int(preprocess_batch_size)
    preprocess_num_gpus = int(preprocess_num_gpus)
    if preprocess_batch_size <= 0:
        raise ValueError("--preprocess-batch-size must be positive")
    if preprocess_num_gpus != 1:
        raise ValueError(
            "FastVideo text preprocessing currently supports "
            "--preprocess-num-gpus 1 only."
        )

    def resolve_max_prompts(total_prompts: int) -> int:
        if max_prompts_mode in {"", "0", "all", "full", "none"}:
            return 0
        estimated_prompt_batches = (
            int(max_train_steps) * int(gradient_accumulation_steps)
        )
        if max_prompts_mode in {"tenth", "1/10"}:
            return max(1, min(total_prompts, estimated_prompt_batches // 10))
        if max_prompts_mode in {"quarter", "1/4"}:
            return max(1, min(total_prompts, estimated_prompt_batches // 4))
        if max_prompts_mode in {"half", "1/2"}:
            return max(1, min(total_prompts, estimated_prompt_batches // 2))
        if max_prompts_mode in {"steps", "used"}:
            return max(1, min(total_prompts, estimated_prompt_batches))
        value = int(max_prompts_mode)
        if value < 0:
            raise ValueError(
                "--max-prompts must be >= 0, tenth, quarter, half, "
                "steps, or full"
            )
        return min(total_prompts, value)

    prompt_suffix = "pending"
    dataset_root = Path(MODAL_DATA_ROOT)
    parquet_dir = dataset_root / "combined_parquet_dataset"
    prompt_file = Path(MODAL_DATA_ROOT) / "prompts" / "pending.txt"

    def has_parquet(path: Path) -> bool:
        return path.exists() and any(path.rglob("*.parquet"))

    def verify_text_only_dataset(expected_rows: int) -> None:
        import pyarrow.parquet as pq

        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(
                f"Text-only preprocessing produced no parquet under {parquet_dir}"
            )
        row_count = sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in parquet_files
        )
        if row_count < expected_rows:
            raise RuntimeError(
                f"Expected at least {expected_rows} prompt rows in "
                f"{parquet_dir}, found {row_count}."
            )
        print(
            "Text-only dataset ready: "
            f"{len(parquet_files)} parquet file(s), {row_count} row(s), "
            f"root={parquet_dir}",
            flush=True,
        )

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

    def ensure_diffusion_nft_repo() -> None:
        if (diffusion_nft_root / "flow_grpo").is_dir():
            print(
                f"Using cached DiffusionNFT repo at {diffusion_nft_root}",
                flush=True,
            )
        else:
            diffusion_nft_root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    DIFFUSION_NFT_REPO,
                    str(diffusion_nft_root),
                ],
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
            cache_vol.commit()

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        paths = [str(diffusion_nft_root)]
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(paths)
        os.environ["DIFFUSION_NFT_ROOT"] = str(diffusion_nft_root)

    def prepare_prompt_file() -> int:
        nonlocal dataset_root, parquet_dir, prompt_file, prompt_suffix

        source = diffusion_nft_root / "dataset" / dataset / "train.txt"
        if not source.is_file():
            raise RuntimeError(
                f"DiffusionNFT dataset {dataset!r} does not provide "
                f"{source}. The launcher expects a train.txt prompt file."
            )
        prompts = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompt_limit = resolve_max_prompts(len(prompts))
        prompt_suffix = (
            "full"
            if prompt_limit <= 0
            else f"first{prompt_limit}"
        )
        dataset_root = (
            Path(MODAL_DATA_ROOT)
            / f"diffusion_nft_{dataset}_text_only_f{num_frames}_{prompt_suffix}"
        )
        parquet_dir = dataset_root / "combined_parquet_dataset"
        prompt_file = (
            Path(MODAL_DATA_ROOT)
            / "prompts"
            / f"diffusion_nft_{dataset}_{prompt_suffix}_train.txt"
        )
        if prompt_limit > 0:
            prompts = prompts[:prompt_limit]
        if not prompts:
            raise RuntimeError(f"No prompts found in {source}")
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(
            "\n".join(prompts) + "\n", encoding="utf-8"
        )
        print(
            f"Prepared {len(prompts)} prompt(s) from {source} "
            f"(max_prompts={max_prompts_mode})",
            flush=True,
        )
        return len(prompts)

    def prepare_run_config() -> Path:
        import yaml

        if reward == "multi_reward":
            reward_map = {name: 1.0 for name in MULTI_REWARD_NAMES}
        else:
            reward_map = {reward: 1.0}

        config_path = repo / CONFIG_PATH
        run_config_dir = repo / "outputs" / "diffusion_nft_run_configs"
        run_config_dir.mkdir(parents=True, exist_ok=True)
        run_config_path = run_config_dir / "diffusion_nft_wan_run.yaml"
        raw_config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        method_config = raw_config.setdefault("method", {})
        method_config["reward_backend"] = "diffusion_nft"
        method_config["reward_fn"] = {"rewards": reward_map}
        if reward == "multi_reward":
            method_config["nft_beta"] = 0.1
            method_config["sample_timesteps"] = list(range(1000, 0, -40))
        run_config_path.write_text(
            yaml.safe_dump(raw_config, sort_keys=False),
            encoding="utf-8",
        )
        return run_config_path

    def ensure_text_only_dataset(expected_rows: int) -> None:
        if has_parquet(parquet_dir):
            print(
                f"Using cached text-only parquet at {parquet_dir}",
                flush=True,
            )
            verify_text_only_dataset(expected_rows)
            return

        dataset_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            "torchrun",
            "--nnodes=1",
            "--nproc_per_node",
            str(preprocess_num_gpus),
            "--master_port=29541",
            "fastvideo/pipelines/preprocess/v1_preprocess.py",
            "--model_path",
            MODEL_ID,
            "--data_merge_path",
            str(prompt_file),
            "--preprocess_video_batch_size",
            str(preprocess_batch_size),
            "--seed",
            "42",
            "--max_height",
            "448",
            "--max_width",
            "832",
            "--num_frames",
            str(int(num_frames)),
            "--num_latent_t",
            str(int(num_latent_t)),
            "--dataloader_num_workers",
            "0",
            "--output_dir",
            str(dataset_root),
            "--samples_per_file",
            "1024",
            "--flush_frequency",
            "1024",
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
        verify_text_only_dataset(expected_rows)
        data_vol.commit()

    preflight_runtime()
    ensure_diffusion_nft_repo()
    expected_prompt_rows = prepare_prompt_file()
    ensure_text_only_dataset(expected_prompt_rows)
    run_config_path = prepare_run_config()

    output_dir = (
        f"{OUTPUT_DIR_BASE}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Output dir: {output_dir}", flush=True)
    print(
        "DiffusionNFT probe settings: "
        f"max_train_steps={max_train_steps} "
        f"num_samples_per_prompt={num_samples_per_prompt} "
        f"collection_batch_size={collection_batch_size} "
        f"inner_epochs={inner_epochs} "
        f"train_batch_size={train_batch_size} "
        f"gradient_accumulation_steps={gradient_accumulation_steps} "
        f"num_frames={num_frames} "
        f"num_latent_t={num_latent_t} "
        f"preprocess_batch_size={preprocess_batch_size} "
        f"preprocess_num_gpus={preprocess_num_gpus} "
        f"dataset={dataset} "
        f"reward={reward} "
        f"max_prompts={max_prompts_mode} "
        f"resolved_prompts={expected_prompt_rows}",
        flush=True,
    )

    cmd = [
        "bash",
        "examples/train/run.sh",
        str(run_config_path),
        "--training.data.data_path",
        str(parquet_dir),
        "--training.checkpoint.output_dir",
        output_dir,
        "--training.tracker.run_name",
        Path(output_dir).name,
        "--training.loop.max_train_steps",
        str(int(max_train_steps)),
        "--training.distributed.num_gpus",
        str(DEFAULT_NUM_GPUS),
        "--training.loop.gradient_accumulation_steps",
        str(int(gradient_accumulation_steps)),
        "--training.data.num_frames",
        str(int(num_frames)),
        "--training.data.num_latent_t",
        str(int(num_latent_t)),
        "--method.num_samples_per_prompt",
        str(int(num_samples_per_prompt)),
        "--method.collection_batch_size",
        str(int(collection_batch_size)),
        "--method.inner_epochs",
        str(int(inner_epochs)),
        "--method.train_batch_size",
        str(int(train_batch_size)),
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
def main(
    max_train_steps: int = DEFAULT_MAX_TRAIN_STEPS,
    num_samples_per_prompt: int = DEFAULT_NUM_SAMPLES_PER_PROMPT,
    collection_batch_size: int = DEFAULT_COLLECTION_BATCH_SIZE,
    inner_epochs: int = DEFAULT_INNER_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_latent_t: int = DEFAULT_NUM_LATENT_T,
    preprocess_batch_size: int = DEFAULT_PREPROCESS_BATCH_SIZE,
    preprocess_num_gpus: int = DEFAULT_PREPROCESS_NUM_GPUS,
    dataset: str = DEFAULT_DATASET,
    reward: str = DEFAULT_REWARD,
    max_prompts: str = DEFAULT_MAX_PROMPTS,
):
    train.remote(
        max_train_steps=max_train_steps,
        num_samples_per_prompt=num_samples_per_prompt,
        collection_batch_size=collection_batch_size,
        inner_epochs=inner_epochs,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_frames=num_frames,
        num_latent_t=num_latent_t,
        preprocess_batch_size=preprocess_batch_size,
        preprocess_num_gpus=preprocess_num_gpus,
        dataset=dataset,
        reward=reward,
        max_prompts=max_prompts,
    )
