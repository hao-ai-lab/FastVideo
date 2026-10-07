# SPDX-License-Identifier: Apache-2.0
"""Run the exact external av-benchmark API used by official MMAudio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-length", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from av_bench.evaluate import evaluate
        from av_bench.extract import extract
    except ImportError as error:
        raise ImportError("The exact official backend is optional. Install hkchengrex/"
                          "av-benchmark in a separate environment before running this file.") from error

    args.prediction_cache.mkdir(parents=True, exist_ok=True)
    extract(
        audio_path=args.audio_dir,
        output_path=args.prediction_cache,
        device=args.device,
        batch_size=args.batch_size,
        audio_length=args.audio_length,
    )
    metrics = evaluate(
        gt_audio_cache=args.gt_cache,
        pred_audio_cache=args.prediction_cache,
    )
    payload = {
        "backend": "hkchengrex/av-benchmark",
        "audio_dir": str(args.audio_dir.resolve()),
        "gt_cache": str(args.gt_cache.resolve()),
        "prediction_cache": str(args.prediction_cache.resolve()),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
