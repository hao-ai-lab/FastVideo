# SPDX-License-Identifier: Apache-2.0
"""Prepare GenRL prompt and reward assets for example training runs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
from pathlib import Path

MIN_TRAIN_PROMPTS = 4
GENRL_REPO = "https://github.com/ModelTC/GenRL.git"
GENRL_PROMPT_FILES = ("train.json", "test.json")
VIDEOREWARD_REPO = "KwaiVGI/VideoReward"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _ensure_genrl_sparse_checkout(genrl_cache_dir: Path) -> Path:
    """Fetch only the GenRL prompt JSONL files into an ignored cache checkout."""
    if not genrl_cache_dir.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                GENRL_REPO,
                str(genrl_cache_dir),
            ]
        )
    elif not (genrl_cache_dir / ".git").exists():
        if _is_nonempty_dir(genrl_cache_dir):
            raise RuntimeError(
                f"{genrl_cache_dir} exists but is not a git checkout. "
                "Pass --genrl-cache-dir to use a different cache directory."
            )
        genrl_cache_dir.rmdir()
        return _ensure_genrl_sparse_checkout(genrl_cache_dir)

    _run(
        ["git", "sparse-checkout", "set", "datasets/filtered_prompts"],
        cwd=genrl_cache_dir,
    )
    _run(["git", "lfs", "install"], cwd=genrl_cache_dir)
    _run(
        [
            "git",
            "lfs",
            "pull",
            "-I",
            "datasets/filtered_prompts/*",
        ],
        cwd=genrl_cache_dir,
    )
    return genrl_cache_dir / "datasets" / "filtered_prompts"


def prepare_genrl_prompts(prompt_dir: Path, genrl_cache_dir: Path) -> Path:
    source_prompt_dir = _ensure_genrl_sparse_checkout(genrl_cache_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for filename in GENRL_PROMPT_FILES:
        shutil.copy2(source_prompt_dir / filename, prompt_dir / filename)

    validate_prompt_file(
        prompt_dir / "train.json",
        min_prompts=MIN_TRAIN_PROMPTS,
    )
    validate_prompt_file(prompt_dir / "test.json", min_prompts=1)
    return prompt_dir


def validate_prompt_file(path: Path, min_prompts: int) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected GenRL filtered_prompts JSONL files. "
            "Run `python examples/train/prepare_genrl_assets.py`."
        )

    prompt_count = 0
    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if (
                prompt_count == 0
                and line_no == 1
                and line.startswith("version https://git-lfs.github.com")
            ):
                raise RuntimeError(
                    f"{path} is a Git LFS pointer, not real prompt JSON. "
                    "Install git-lfs and rerun this script."
                )
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if item.get("prompt"):
                prompt_count += 1

    if prompt_count < min_prompts:
        raise RuntimeError(
            f"{path} has {prompt_count} usable prompts; expected at least "
            f"{min_prompts}."
        )


def prepare_video_reward(videoalign_dir: Path) -> Path:
    if has_video_reward_checkpoint(videoalign_dir):
        return videoalign_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            f"huggingface_hub is required to download {VIDEOREWARD_REPO}. "
            "Install FastVideo dependencies, then rerun this script."
        ) from exc

    snapshot_download(
        repo_id=VIDEOREWARD_REPO,
        repo_type="model",
        local_dir=str(videoalign_dir),
        local_dir_use_symlinks=False,
    )
    if not has_video_reward_checkpoint(videoalign_dir):
        raise RuntimeError(
            f"Downloaded {VIDEOREWARD_REPO}, but no VideoReward checkpoint "
            f"was found under {videoalign_dir}."
        )
    return videoalign_dir


def has_video_reward_checkpoint(root: Path) -> bool:
    model_config = root / "model_config.json"
    if not model_config.exists():
        return False
    for checkpoint in root.glob("checkpoint-*"):
        if (checkpoint / "model.pth").exists():
            return True
        if (
            (checkpoint / "adapter_model.safetensors").exists()
            and (checkpoint / "non_lora_state_dict.pth").exists()
        ):
            return True
    return False


def check_reward_runtime(device: str = "auto") -> None:
    """Run the same lightweight reward-model preflight used by Modal."""
    import numpy as np
    import torch

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=== GenRL reward preflight ===", flush=True)
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

    from fastvideo.train.methods.rl.reward.hpsv3 import (
        _HPSV3_INFERENCERS,
        hpsv3_general_score,
        hpsv3_percentile_score,
        set_hpsv3_device,
    )

    torch_device = torch.device(selected_device)
    dummy_video = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)
    for name, factory in (
        ("HPSv3-general", hpsv3_general_score),
        ("HPSv3-percentile", hpsv3_percentile_score),
    ):
        reward = factory(torch_device)
        scores, _ = reward(dummy_video, ["preflight prompt"], {})
        value = float(scores["avg"].detach().cpu()[0])
        print(f"{name} preflight score: {value:.4f}", flush=True)

    set_hpsv3_device("cpu")
    _HPSV3_INFERENCERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        reward = factory(torch_device)
        scores, _ = reward(dummy_video, ["preflight prompt"], {})
        value = float(scores["avg"].detach().cpu()[0])
        print(f"{name} preflight score: {value:.4f}", flush=True)

    set_videoalign_device("cpu")
    _VIDEOALIGN_INFERENCERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("=== GenRL reward preflight OK ===", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare assets for GenRL HPSv3 + VideoAlign training."
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path(".cache/genrl_filtered_prompts"),
        help="Directory that will contain train.json and test.json.",
    )
    parser.add_argument(
        "--genrl-cache-dir",
        type=Path,
        default=Path(".cache/GenRL"),
        help="Ignored sparse checkout cache used to fetch only GenRL filtered prompts.",
    )
    parser.add_argument(
        "--videoalign-dir",
        type=Path,
        default=Path(".cache/VideoReward"),
        help=f"Directory containing or receiving {VIDEOREWARD_REPO}.",
    )
    parser.add_argument(
        "--check-rewards",
        action="store_true",
        help="After preparing assets, load HPSv3 and VideoAlign on a dummy video.",
    )
    parser.add_argument(
        "--reward-device",
        default="auto",
        help="Device for --check-rewards: auto, cpu, cuda, or cuda:<index>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_dir = prepare_genrl_prompts(args.prompt_dir, args.genrl_cache_dir)
    videoalign_dir = prepare_video_reward(args.videoalign_dir)
    os.environ.setdefault("VIDEOALIGN_CHECKPOINT_PATH", str(videoalign_dir))
    print("GenRL assets ready.")
    print(f"PROMPT_DATASET_PATH={prompt_dir}")
    print(f"VIDEOALIGN_CHECKPOINT_PATH={videoalign_dir}")
    if args.check_rewards:
        check_reward_runtime(args.reward_device)


if __name__ == "__main__":
    main()
