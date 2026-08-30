#!/usr/bin/env python3
"""Collect local RVM run metadata/validation directories into one JSON index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/rvm_h3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rvm_h3/results_index.json"))
    args = parser.parse_args()
    runs = []
    if args.root.is_dir():
        for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
            checkpoints = sorted(path.name for path in run_dir.glob("checkpoint-*") if path.is_dir())
            validations = sorted(path.name for path in (run_dir / "validation").glob("step-*") if path.is_dir())
            runs.append({"run": run_dir.name, "checkpoints": checkpoints, "validation_steps": validations})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
