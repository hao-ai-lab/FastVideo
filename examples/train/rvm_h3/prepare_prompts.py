#!/usr/bin/env python3
"""Download/split DanceGRPO VidProM prompts and wrap them as H3 documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://raw.githubusercontent.com/XueZeyue/DanceGRPO/"
    "15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/assets/video_prompts.txt"
)


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def h3_document(prompt: str) -> str:
    return (
        f"integrated_multimodal_description: {prompt} "
        "overall_soundscape: N/A "
        "non_diegetic_music: N/A"
    )


def read_source(path: Path | None, url: str) -> list[str]:
    if path is None:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    seen: set[str] = set()
    prompts: list[str] = []
    for line in raw.splitlines():
        prompt = normalize_prompt(line)
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    if not prompts:
        raise RuntimeError("No nonempty prompts were found")
    return prompts


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-prompts", type=int, default=100)
    parser.add_argument("--smoke-prompts", type=int, default=16)
    parser.add_argument("--max-train-prompts", type=int, default=None)
    args = parser.parse_args()

    prompts = read_source(args.source, args.url)
    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    eval_count = min(max(1, args.eval_prompts), len(prompts) - 1)
    eval_prompts = prompts[:eval_count]
    train_prompts = prompts[eval_count:]
    if args.max_train_prompts is not None:
        train_prompts = train_prompts[: max(1, args.max_train_prompts)]
    smoke_prompts = train_prompts[: min(max(1, args.smoke_prompts), len(train_prompts))]

    output_dir = args.output_dir
    write_lines(output_dir / "source_train.txt", train_prompts)
    write_lines(output_dir / "source_eval.txt", eval_prompts)
    write_lines(output_dir / "train_h3.txt", [h3_document(prompt) for prompt in train_prompts])
    write_lines(output_dir / "eval_h3.txt", [h3_document(prompt) for prompt in eval_prompts])
    write_lines(output_dir / "smoke_h3.txt", [h3_document(prompt) for prompt in smoke_prompts])

    digest = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
    metadata = {
        "source_url": args.url if args.source is None else None,
        "source_file": str(args.source) if args.source is not None else None,
        "source_sha256": digest,
        "seed": args.seed,
        "unique_source_prompts": len(prompts),
        "train_prompts": len(train_prompts),
        "eval_prompts": len(eval_prompts),
        "smoke_prompts": len(smoke_prompts),
        "audio_fields": "N/A; audio is preserved by the function-space anchor, not optimized by reward",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
