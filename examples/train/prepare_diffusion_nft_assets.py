# SPDX-License-Identifier: Apache-2.0
"""Prepare reproducible assets for DiffusionNFT Wan RL training.

This script is intentionally environment-agnostic. It contains the prompt,
preprocessing, reward-checkpoint, and run-config preparation needed to reproduce
the DiffusionNFT video run without relying on private Modal launchers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DIFFUSION_NFT_REPO = "https://github.com/NVlabs/DiffusionNFT.git"
VIDEO_REWARD_REPO = "KwaiVGI/VideoReward"
DEFAULT_CONFIG = "examples/train/configs/rl/wan/diffusion_nft_videoalign.yaml"
DEFAULT_OUTPUT_DIR = "outputs/wan2.1_diffusion_nft_videoalign"

IMAGE_MULTI_REWARD_NAMES = ("pickscore", "hpsv2", "clipscore")
VIDEO_MULTI_REWARD_NAMES = (
    "videoalign_vq",
    "videoalign_mq",
    "videoalign_ta",
)
VIDEO_BALANCED_REWARD_WEIGHTS = {
    "videoalign_ta": 1.0,
    "videoalign_mq": 1.0,
    "videoalign_vq": 0.75,
    "hpsv3_general": 0.25,
}
GENRL_REWARD_NAMES = frozenset({
    "video_ocr",
    "hpsv3_general",
    "hpsv3_percentile",
    "videoalign_vq",
    "videoalign_mq",
    "videoalign_ta",
})


def derive_wan_num_latent_t(frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("--num-frames must be positive")
    if (frame_count - 1) % 4 != 0:
        raise ValueError("Wan frame counts must satisfy num_frames = (num_latent_t - 1) * 4 + 1. "
                         f"Got num_frames={frame_count}; try 1, 5, 9, 13, 17, ...")
    return ((frame_count - 1) // 4) + 1


def resolve_reward_map(reward: str) -> tuple[dict[str, float], str]:
    reward = reward.strip().lower()
    if reward in {"videoalign", "video_reward", "video_multi_reward"}:
        reward_map = {name: 1.0 for name in VIDEO_MULTI_REWARD_NAMES}
    elif reward in {"videoalign_hpsv3", "video_reward_hpsv3", "balanced_video", "quality_video"}:
        reward_map = dict(VIDEO_BALANCED_REWARD_WEIGHTS)
    elif reward in {"multi_reward", "image_multi_reward"}:
        reward_map = {name: 1.0 for name in IMAGE_MULTI_REWARD_NAMES}
    else:
        reward_map = {reward: 1.0}
    backend = "genrl" if any(name in GENRL_REWARD_NAMES for name in reward_map) else "diffusion_nft"
    return reward_map, backend


def resolve_max_prompts(
    max_prompts: str,
    *,
    total_prompts: int,
    max_train_steps: int,
    gradient_accumulation_steps: int,
) -> int:
    mode = str(max_prompts).strip().lower()
    if mode in {"", "0", "all", "full", "none"}:
        return 0
    estimated_prompt_batches = int(max_train_steps) * int(gradient_accumulation_steps)
    if mode in {"tenth", "1/10"}:
        return max(1, min(total_prompts, estimated_prompt_batches // 10))
    if mode in {"quarter", "1/4"}:
        return max(1, min(total_prompts, estimated_prompt_batches // 4))
    if mode in {"half", "1/2"}:
        return max(1, min(total_prompts, estimated_prompt_batches // 2))
    if mode in {"steps", "used"}:
        return max(1, min(total_prompts, estimated_prompt_batches))
    value = int(mode)
    if value < 0:
        raise ValueError("--max-prompts must be >= 0, tenth, quarter, half, steps, or full")
    return min(total_prompts, value)


def has_parquet(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def verify_text_only_dataset(path: Path, expected_rows: int) -> int:
    import pyarrow.parquet as pq

    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"Text-only preprocessing produced no parquet under {path}")
    row_count = sum(pq.ParquetFile(file_path).metadata.num_rows for file_path in parquet_files)
    if row_count < expected_rows:
        raise RuntimeError(f"Expected at least {expected_rows} prompt rows in {path}, found {row_count}.")
    return int(row_count)


def ensure_diffusion_nft_repo(root: Path) -> None:
    if (root / "flow_grpo").is_dir():
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", DIFFUSION_NFT_REPO, str(root)],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def ensure_videoalign_checkpoint(path: Path) -> None:
    if has_video_reward_checkpoint(path):
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(f"huggingface_hub is required to download {VIDEO_REWARD_REPO}. "
                          "Install examples/train/requirements-diffusion-nft.txt and rerun.") from exc
    snapshot_download(
        repo_id=VIDEO_REWARD_REPO,
        repo_type="model",
        local_dir=str(path),
        local_dir_use_symlinks=False,
    )
    if not has_video_reward_checkpoint(path):
        raise RuntimeError(f"Downloaded {VIDEO_REWARD_REPO}, but no VideoReward checkpoint was found under {path}.")


def has_video_reward_checkpoint(root: Path) -> bool:
    model_config = root / "model_config.json"
    if not model_config.exists():
        return False
    if (root / "model.pth").exists():
        return True
    if ((root / "adapter_model.safetensors").exists() and (root / "non_lora_state_dict.pth").exists()):
        return True
    for checkpoint in root.glob("checkpoint-*"):
        if (checkpoint / "model.pth").exists():
            return True
        if ((checkpoint / "adapter_model.safetensors").exists()
                and (checkpoint / "non_lora_state_dict.pth").exists()):
            return True
    return False


def load_prompts(
    dataset: str,
    *,
    diffusion_nft_root: Path,
) -> tuple[list[str], str]:
    dataset = dataset.strip().lower()
    if dataset in {"world-r1", "world-r1-final", "world_r1", "world_r1_final"}:
        from datasets import load_dataset

        rows = load_dataset("microsoft/World-R1", "final", split="train")
        source_label = "microsoft/World-R1 final/train"
    elif dataset in {"world-r1-dynamic", "world_r1_dynamic", "world-r1-final-dynamic", "world_r1_final_dynamic"}:
        from datasets import load_dataset

        rows = load_dataset("microsoft/World-R1", "final", split="dynamic")
        source_label = "microsoft/World-R1 final/dynamic"
    elif dataset in {"world-r1-enhanced", "world_r1_enhanced"}:
        from datasets import load_dataset

        rows = load_dataset("microsoft/World-R1", "enhanced", split="train")
        source_label = "microsoft/World-R1 enhanced/train"
    elif dataset in {"world-r1-enhanced-dynamic", "world_r1_enhanced_dynamic"}:
        from datasets import load_dataset

        rows = load_dataset("microsoft/World-R1", "enhanced", split="dynamic")
        source_label = "microsoft/World-R1 enhanced/dynamic"
    else:
        source = diffusion_nft_root / "dataset" / dataset / "train.txt"
        if not source.is_file():
            raise RuntimeError(f"Dataset {dataset!r} is not a built-in World-R1 dataset and does not provide "
                               f"{source}. Use a World-R1 dataset alias or prepare a DiffusionNFT checkout.")
        prompts = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        return prompts, str(source)

    prompts = [str(row["prompt"]).strip() for row in rows if str(row.get("prompt", "")).strip()]
    return prompts, source_label


def prepare_prompt_file(args: argparse.Namespace, *, num_latent_t: int) -> tuple[Path, Path, int, str]:
    del num_latent_t
    prompts, source_label = load_prompts(args.dataset, diffusion_nft_root=args.diffusion_nft_root)
    prompt_limit = resolve_max_prompts(
        args.max_prompts,
        total_prompts=len(prompts),
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if prompt_limit > 0:
        prompts = prompts[:prompt_limit]
    if not prompts:
        raise RuntimeError(f"No prompts found in {source_label}")
    if len(prompts) < args.num_gpus:
        raise RuntimeError(f"Need at least {args.num_gpus} prompts for {args.num_gpus} ranks with drop_last=True; "
                           f"resolved only {len(prompts)} prompt(s). Increase --max-prompts.")

    prompt_suffix = "full" if prompt_limit <= 0 else f"first{prompt_limit}"
    dataset_root = args.data_root / f"diffusion_nft_{args.dataset}_text_only_f{args.num_frames}_{prompt_suffix}"
    parquet_dir = dataset_root / "combined_parquet_dataset"
    prompt_file = args.data_root / "prompts" / f"diffusion_nft_{args.dataset}_{prompt_suffix}_train.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return prompt_file, parquet_dir, len(prompts), source_label


def run_text_only_preprocess(
    args: argparse.Namespace,
    *,
    prompt_file: Path,
    dataset_root: Path,
    num_latent_t: int,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node",
        str(args.preprocess_num_gpus),
        "--master_port",
        str(args.preprocess_master_port),
        "fastvideo/pipelines/preprocess/v1_preprocess.py",
        "--model_path",
        args.model_id,
        "--data_merge_path",
        str(prompt_file),
        "--preprocess_video_batch_size",
        str(args.preprocess_batch_size),
        "--seed",
        str(args.seed),
        "--max_height",
        str(args.num_height),
        "--max_width",
        str(args.num_width),
        "--num_frames",
        str(args.num_frames),
        "--num_latent_t",
        str(num_latent_t),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--output_dir",
        str(dataset_root),
        "--samples_per_file",
        str(args.samples_per_file),
        "--flush_frequency",
        str(args.flush_frequency),
        "--preprocess_task",
        "text_only",
    ]
    subprocess.run(cmd, cwd=args.repo_root, stdout=sys.stdout, stderr=sys.stderr, check=True)


def write_run_config(
    args: argparse.Namespace,
    *,
    parquet_dir: Path,
    num_latent_t: int,
    reward_map: dict[str, float],
    reward_backend: str,
) -> Path:
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method_config = raw_config.setdefault("method", {})
    training_config = raw_config.setdefault("training", {})
    distributed_config = training_config.setdefault("distributed", {})
    data_config = training_config.setdefault("data", {})
    loop_config = training_config.setdefault("loop", {})
    optimizer_config = training_config.setdefault("optimizer", {})
    checkpoint_config = training_config.setdefault("checkpoint", {})
    tracker_config = training_config.setdefault("tracker", {})
    sampling_config = method_config.setdefault("sampling", {})

    method_config["reward_backend"] = reward_backend
    method_config["reward_fn"] = {"rewards": reward_map}
    method_config["num_video_per_prompt"] = int(args.num_samples_per_prompt)
    method_config["sample_train_batch_size"] = int(args.collection_batch_size)
    method_config["num_inner_epochs"] = int(args.inner_epochs)
    method_config["train_batch_size"] = int(args.train_batch_size)
    validation_config = method_config.setdefault("validation", {})
    validation_config["log_samples"] = bool(args.log_sample_max_videos > 0)
    if args.sample_num_steps is not None:
        sampling_config["num_steps"] = int(args.sample_num_steps)
    if args.sample_flow_shift is not None:
        sampling_config["flow_shift"] = float(args.sample_flow_shift)
    if args.sample_guidance_scale is not None:
        sampling_config["guidance_scale"] = float(args.sample_guidance_scale)
    if args.reward in {"multi_reward", "image_multi_reward"}:
        method_config["beta"] = 0.1

    distributed_config["num_gpus"] = int(args.num_gpus)
    distributed_config["tp_size"] = int(args.tp_size)
    distributed_config["sp_size"] = int(args.sp_size)
    distributed_config["hsdp_replicate_dim"] = int(args.hsdp_replicate_dim)
    distributed_config["hsdp_shard_dim"] = int(args.hsdp_shard_dim)
    data_config["data_path"] = str(parquet_dir)
    data_config["preprocessed_data_type"] = "text_only"
    data_config["num_frames"] = int(args.num_frames)
    data_config["num_latent_t"] = int(num_latent_t)
    data_config["num_height"] = int(args.num_height)
    data_config["num_width"] = int(args.num_width)
    data_config["dataloader_num_workers"] = int(args.dataloader_num_workers)
    loop_config["max_train_steps"] = int(args.max_train_steps)
    loop_config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.learning_rate is not None:
        optimizer_config["learning_rate"] = float(args.learning_rate)
    checkpoint_config["output_dir"] = str(args.output_dir)
    tracker_config["project_name"] = args.project_name
    tracker_config["run_name"] = args.run_name or args.output_dir.name

    args.run_config_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.run_config_dir / "diffusion_nft_wan_run.yaml"
    run_config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    return run_config_path


def check_reward_runtime(
    reward_map: dict[str, float],
    *,
    reward_backend: str,
    device: str = "auto",
) -> None:
    import torch

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(selected_device)
    print("=== DiffusionNFT reward preflight ===", flush=True)
    print(f"reward_backend={reward_backend} reward_map={reward_map}", flush=True)

    from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer

    media = torch.zeros(1, 3, 8, 224, 224, device=torch_device)
    prompts = ["A small red block moves steadily from left to right."]
    scorer = build_multi_reward_scorer(
        reward_map,
        backend=reward_backend,
        device=torch_device,
    )
    scores = scorer(media, prompts)
    score_summary = {name: float(value.detach().float().cpu()[0]) for name, value in scores.items()}
    print(f"reward preflight scores: {score_summary}", flush=True)

    del scorer, media, scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("=== DiffusionNFT reward preflight OK ===", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--data-root", type=Path, default=Path("data/diffusion_nft"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/diffusion_nft"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-config-dir", type=Path, default=Path("outputs/diffusion_nft_run_configs"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", default="world-r1-enhanced")
    parser.add_argument("--reward", default="videoalign")
    parser.add_argument("--max-prompts", default="quarter")
    parser.add_argument("--num-frames", type=int, default=77)
    parser.add_argument("--num-latent-t", type=int, default=0)
    parser.add_argument("--num-height", type=int, default=448)
    parser.add_argument("--num-width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-batch-size", type=int, default=128)
    parser.add_argument("--preprocess-num-gpus", type=int, default=1)
    parser.add_argument("--preprocess-master-port", type=int, default=29541)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--samples-per-file", type=int, default=1024)
    parser.add_argument("--flush-frequency", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--hsdp-replicate-dim", type=int, default=1)
    parser.add_argument("--hsdp-shard-dim", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-samples-per-prompt", type=int, default=24)
    parser.add_argument("--collection-batch-size", type=int, default=6)
    parser.add_argument("--inner-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--sample-num-steps", type=int)
    parser.add_argument("--sample-flow-shift", type=float)
    parser.add_argument("--sample-guidance-scale", type=float)
    parser.add_argument("--log-sample-max-videos", type=int, default=2)
    parser.add_argument("--project-name", default="diffusion_nft_wan")
    parser.add_argument("--run-name")
    parser.add_argument("--diffusion-nft-root", type=Path)
    parser.add_argument("--videoalign-checkpoint-path", type=Path)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--check-rewards", action="store_true")
    parser.add_argument("--reward-device", default="auto", help="Device for --check-rewards: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary as the final line.")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.config = (args.repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.cache_root = ((args.repo_root / args.cache_root).resolve()
                       if not args.cache_root.is_absolute() else args.cache_root)
    args.output_dir = ((args.repo_root / args.output_dir).resolve()
                       if not args.output_dir.is_absolute() else args.output_dir)
    args.run_config_dir = ((args.repo_root / args.run_config_dir).resolve()
                           if not args.run_config_dir.is_absolute() else args.run_config_dir)
    args.diffusion_nft_root = args.diffusion_nft_root or (args.cache_root / "DiffusionNFT")
    args.videoalign_checkpoint_path = args.videoalign_checkpoint_path or (args.cache_root / "VideoReward")
    return args


def main() -> None:
    args = parse_args()
    requested_num_latent_t = int(args.num_latent_t)
    derived_num_latent_t = derive_wan_num_latent_t(args.num_frames)
    if requested_num_latent_t <= 0:
        num_latent_t = derived_num_latent_t
    elif requested_num_latent_t != derived_num_latent_t:
        raise ValueError(f"For Wan num_frames={args.num_frames} implies num_latent_t={derived_num_latent_t}, "
                         f"but got num_latent_t={requested_num_latent_t}.")
    else:
        num_latent_t = requested_num_latent_t

    reward_map, reward_backend = resolve_reward_map(args.reward)
    if args.dataset.strip().lower() not in {
            "world-r1",
            "world-r1-final",
            "world_r1",
            "world_r1_final",
            "world-r1-dynamic",
            "world_r1_dynamic",
            "world-r1-final-dynamic",
            "world_r1_final_dynamic",
            "world-r1-enhanced",
            "world_r1_enhanced",
            "world-r1-enhanced-dynamic",
            "world_r1_enhanced_dynamic",
    } or reward_backend == "diffusion_nft":
        ensure_diffusion_nft_repo(args.diffusion_nft_root)
        os.environ["DIFFUSION_NFT_ROOT"] = str(args.diffusion_nft_root)

    if any(name.startswith("videoalign_") for name in reward_map):
        ensure_videoalign_checkpoint(args.videoalign_checkpoint_path)
        os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = str(args.videoalign_checkpoint_path)

    if args.check_rewards:
        check_reward_runtime(
            reward_map,
            reward_backend=reward_backend,
            device=args.reward_device,
        )

    prompt_file, parquet_dir, prompt_count, source_label = prepare_prompt_file(args, num_latent_t=num_latent_t)
    if has_parquet(parquet_dir):
        row_count = verify_text_only_dataset(parquet_dir, prompt_count)
    else:
        if args.skip_preprocess:
            raise RuntimeError(f"No parquet files found under {parquet_dir} and --skip-preprocess was set.")
        run_text_only_preprocess(
            args,
            prompt_file=prompt_file,
            dataset_root=parquet_dir.parent,
            num_latent_t=num_latent_t,
        )
        row_count = verify_text_only_dataset(parquet_dir, prompt_count)

    run_config_path = write_run_config(
        args,
        parquet_dir=parquet_dir,
        num_latent_t=num_latent_t,
        reward_map=reward_map,
        reward_backend=reward_backend,
    )
    summary: dict[str, Any] = {
        "prompt_file": str(prompt_file),
        "prompt_source": source_label,
        "prompt_count": prompt_count,
        "parquet_dir": str(parquet_dir),
        "parquet_rows": row_count,
        "run_config": str(run_config_path),
        "output_dir": str(args.output_dir),
        "num_frames": args.num_frames,
        "num_latent_t": num_latent_t,
        "reward_backend": reward_backend,
        "reward_map": reward_map,
    }
    print("Prepared DiffusionNFT assets:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
