# SPDX-License-Identifier: Apache-2.0
"""Prepare GenRL prompt and reward assets for example training runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MIN_TRAIN_PROMPTS = 4
GENRL_REPO = "https://github.com/ModelTC/GenRL.git"
VIDEOREWARD_REPO = "KwaiVGI/VideoReward"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def prepare_genrl_prompts(genrl_dir: Path) -> Path:
    if not genrl_dir.exists():
        _run(["git", "clone", GENRL_REPO, str(genrl_dir)])
    elif not (genrl_dir / ".git").exists() and not _is_nonempty_dir(
        genrl_dir
    ):
        genrl_dir.rmdir()
        _run(["git", "clone", GENRL_REPO, str(genrl_dir)])

    if (genrl_dir / ".git").exists():
        _run(["git", "lfs", "install"], cwd=genrl_dir)
        _run(
            [
                "git",
                "lfs",
                "pull",
                "-I",
                "datasets/filtered_prompts/*",
            ],
            cwd=genrl_dir,
        )

    prompt_dir = genrl_dir / "datasets" / "filtered_prompts"
    validate_prompt_file(
        prompt_dir / "train.json",
        min_prompts=MIN_TRAIN_PROMPTS,
    )
    validate_prompt_file(prompt_dir / "test.json", min_prompts=1)
    return prompt_dir


def validate_prompt_file(path: Path, min_prompts: int) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected GenRL filtered_prompts JSONL files."
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
            item = json.loads(line)
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
            "huggingface_hub is required to download KwaiVGI/VideoReward. "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare assets for GenRL HPSv3 + VideoAlign training."
    )
    parser.add_argument(
        "--genrl-dir",
        type=Path,
        default=Path("GenRL"),
        help="Directory containing or receiving the ModelTC/GenRL checkout.",
    )
    parser.add_argument(
        "--videoalign-dir",
        type=Path,
        default=Path(".cache/VideoReward"),
        help="Directory containing or receiving KwaiVGI/VideoReward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_dir = prepare_genrl_prompts(args.genrl_dir)
    videoalign_dir = prepare_video_reward(args.videoalign_dir)
    print("GenRL assets ready.")
    print(f"PROMPT_DATASET_PATH={prompt_dir}")
    print(f"VIDEOALIGN_CHECKPOINT_PATH={videoalign_dir}")


if __name__ == "__main__":
    main()
