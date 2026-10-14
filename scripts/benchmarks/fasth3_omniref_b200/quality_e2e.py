"""Run raw OmniRef cases end to end and dump each clip's final latents and text conditioning (quality arms).

Same generator build as ``bench_e2e.py`` (preset, ``--nvfp4-dit``, ``--env``); every case runs once at its own
geometry and seed (+ ``--seed-offset``) with ``FASTVIDEO_H3_DUMP_DIR=<out>``, which saves a seed and request
digest filename with the target video/audio rows and the Qwen3-VL embeddings. The digest includes the output
geometry and ordered reference contents, so equal prompts with different references keep separate dumps.

  python quality_e2e.py --model-path <dir> --cases cases.json --preset lossless --nvfp4-dit --out <dump dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import bench_e2e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--preset", choices=("default", "lossless"), default="lossless")
    parser.add_argument("--nvfp4-dit", action="store_true")
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["FASTVIDEO_H3_DUMP_DIR"] = str(Path(args.out).resolve())
    os.environ.setdefault("FASTVIDEO_H3_REF2VA_MEMO_ENTRIES", "0")
    for item in args.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
    example = bench_e2e._example()
    if args.preset == "lossless":
        example.apply_lossless_accel_env()
    model_dir = Path(args.model_path)
    contract = example.load_contract(model_dir)
    cases = json.loads(Path(args.cases).read_text())
    if args.limit:
        cases = cases[:args.limit]

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_config(
        bench_e2e.build_config(example, model_dir, args.num_gpus, args.preset, args.nvfp4_dit))
    log = []
    try:
        for case in cases:
            seeded = {**case, "seed": int(case["seed"]) + args.seed_offset}
            references = bench_e2e.load_references(seeded)
            shape = (int(case["width"]), int(case["height"]), int(case["num_frames"]))
            started = time.perf_counter()
            bench_e2e.generate_once(generator, contract, seeded, references, shape, Path(args.out) / "clip.mp4")
            record = {"id": case["id"], "seed": seeded["seed"], "seconds": round(time.perf_counter() - started, 2)}
            print("QUALITY", json.dumps(record), flush=True)
            log.append(record)
    finally:
        generator.shutdown()
    (Path(args.out) / "cases_run.json").write_text(json.dumps({"args": vars(args), "runs": log}, indent=1))


if __name__ == "__main__":
    main()
