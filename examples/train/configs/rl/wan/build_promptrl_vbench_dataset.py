#!/usr/bin/env python3
"""Build the PromptRL training prompt pool from the pinned VBench suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_VBENCH_ROOT = Path(
    "fastvideo/third_party/eval/vbench/prompts")
DEFAULT_SOURCES = ("all_dimension.txt", "all_category.txt")


def build_rows(vbench_root: Path) -> list[dict[str, str]]:
    """Return deterministic, de-duplicated VBench prompt records."""
    prompts: dict[str, str] = {}
    for source_name in DEFAULT_SOURCES:
        source_path = vbench_root / source_name
        if not source_path.is_file():
            raise FileNotFoundError(
                f"VBench prompt source not found: {source_path}. "
                "Initialize the pinned VBench git submodule first.")
        for raw_prompt in source_path.read_text(encoding="utf-8").splitlines():
            prompt = raw_prompt.strip()
            if prompt:
                prompts.setdefault(prompt, source_name)

    return [{
        "id": f"vbench-{index:04d}",
        "prompt": prompt,
        "reward_tag": "videoscore2",
        "source": source,
    } for index, (prompt, source) in enumerate(prompts.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench-root",
                        type=Path,
                        default=DEFAULT_VBENCH_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.vbench_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"PROMPTRL_DATASET_READY rows={len(rows)} path={args.output}")


if __name__ == "__main__":
    main()
