# SPDX-License-Identifier: Apache-2.0
"""Stage 0b: generate a synthetic (video, prompt) dataset with Wan2.2-T2V-A14B.

Text prompts -> .mp4 videos + a FastVideo-compatible manifest, so the existing
preprocess pipeline can ingest the result directly. CoTracker point extraction is a
separate stage (``extract_tracks.py``).

Design (see notes/DECISIONS.md):
- Generate at the *training* fps/length (default 16 fps, 81 frames ~= 5 s) so per-frame
  point tracks align 1:1 with frames and no resampling is needed downstream.
- T2V only. The eventual I2V+points model uses frame 0 as the conditioning image at
  training time, so no input image is needed here.
- Idempotent/resumable: each finished video is appended to ``manifest.jsonl`` and skipped
  on re-run.

Run on a GPU node (never the login node), e.g.:

    srun --jobid=<shao_wm jobid> --overlap --ntasks=1 \
        .venv/bin/python data_pipeline/generate_videos.py \
        --prompts examples/dataset/vidprom/prompts/vidprom_filtered_extended.txt \
        --output-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_t2v_720p \
        --num-videos 50 --num-gpus 8
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import warnings
from pathlib import Path

DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, required=True, help="Text file, one prompt per line.")
    p.add_argument("--output-dir", type=Path, required=True, help="Dataset root (videos/, manifest, ...).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--num-videos", type=int, default=50, help="How many prompts to generate (after --start).")
    p.add_argument("--start", type=int, default=0, help="Offset into the prompt list (for sharding).")
    p.add_argument("--shuffle", action="store_true", help="Deterministically shuffle prompts before slicing.")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=1024, help="Base seed; per-video seed = seed + global index.")
    p.add_argument("--num-inference-steps", type=int, default=None, help="Override model default if set.")
    p.add_argument("--negative-prompt", type=str, default=None)
    # Offload controls. Default OFF: Wan2.2-A14B is ~56GB bf16 and fits on a single H200 (143GB),
    # so offloading (esp. layerwise) only makes generation ~10x slower. Enable on small GPUs.
    p.add_argument("--dit-cpu-offload", action="store_true", help="Offload DiT to CPU (slow).")
    p.add_argument("--dit-layerwise-offload", action="store_true",
                   help="Stream DiT layers from CPU per step (very slow; only for tiny GPUs).")
    p.add_argument("--text-encoder-cpu-offload", action="store_true", help="Offload text encoder to CPU.")
    p.add_argument("--vae-cpu-offload", action="store_true", help="Offload VAE to CPU.")
    return p.parse_args()


def load_prompts(path: Path, start: int, num: int, shuffle: bool, seed: int) -> list[tuple[int, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {path}\n"
            "Download it first (login node): cd examples/dataset/vidprom && ./download_dataset.sh"
        )
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    indexed = list(enumerate(lines))  # global index is stable w.r.t. the raw file order
    if shuffle:
        random.Random(seed).shuffle(indexed)
    return indexed[start:start + num]


def read_done_indices(manifest_jsonl: Path) -> set[int]:
    done: set[int] = set()
    if manifest_jsonl.exists():
        for ln in manifest_jsonl.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                done.add(int(json.loads(ln)["idx"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def rebuild_manifest(manifest_jsonl: Path, videos_dir: Path, json_path: Path, merge_path: Path) -> int:
    """Compile manifest.jsonl -> videos2caption.json + merge.txt (FastVideo format)."""
    records: dict[int, dict] = {}
    if manifest_jsonl.exists():
        for ln in manifest_jsonl.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            records[int(rec["idx"])] = rec
    ordered = [records[k] for k in sorted(records)]

    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ordered, indent=2))
    tmp.replace(json_path)
    merge_path.write_text(f"{videos_dir.resolve()},{json_path.resolve()}\n")
    return len(ordered)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastvideo.*")

    output_dir = args.output_dir
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = output_dir / "manifest.jsonl"
    json_path = output_dir / "videos2caption.json"
    merge_path = output_dir / "merge.txt"
    fail_log = output_dir / "failures.log"

    selected = load_prompts(args.prompts, args.start, args.num_videos, args.shuffle, args.seed)
    done = read_done_indices(manifest_jsonl)
    todo = [(i, pr) for (i, pr) in selected if i not in done]
    print(f"[gen] {len(selected)} selected, {len(done)} already done, {len(todo)} to generate", flush=True)

    if not todo:
        n = rebuild_manifest(manifest_jsonl, videos_dir, json_path, merge_path)
        print(f"[gen] nothing to do; manifest has {n} entries -> {json_path}", flush=True)
        return

    # Import here so --help works without loading torch/fastvideo.
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=args.dit_cpu_offload,
        dit_layerwise_offload=args.dit_layerwise_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        pin_cpu_memory=True,
    )

    extra: dict = {}
    if args.num_inference_steps is not None:
        extra["num_inference_steps"] = args.num_inference_steps
    if args.negative_prompt is not None:
        extra["negative_prompt"] = args.negative_prompt

    duration = float(args.num_frames) / float(args.fps)
    for n_done, (idx, prompt) in enumerate(todo, 1):
        final_path = videos_dir / f"vid_{idx:06d}.mp4"
        if final_path.exists():  # belt-and-suspenders vs manifest
            continue
        tmp_dir = videos_dir / f".tmp_{idx:06d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            generator.generate_video(
                prompt,
                output_path=str(tmp_dir),
                save_video=True,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                seed=args.seed + idx,
                **extra,
            )
            produced = sorted(tmp_dir.glob("*.mp4"))
            if not produced:
                raise RuntimeError("no .mp4 produced by generate_video")
            shutil.move(str(produced[0]), str(final_path))
        except Exception as e:  # noqa: BLE001 - keep the batch alive, log and move on
            with fail_log.open("a") as f:
                f.write(json.dumps({"idx": idx, "error": repr(e), "prompt": prompt}) + "\n")
            print(f"[gen] FAILED idx={idx}: {e!r}", flush=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        shutil.rmtree(tmp_dir, ignore_errors=True)

        record = {
            "idx": idx,
            "path": final_path.name,  # basename; folder in merge.txt is videos_dir
            "cap": [prompt],
            "fps": float(args.fps),
            "duration": duration,
            "num_frames": int(args.num_frames),
            "resolution": {"width": args.width, "height": args.height},
        }
        with manifest_jsonl.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[gen] [{n_done}/{len(todo)}] idx={idx} {time.time()-t0:.1f}s -> {final_path.name}", flush=True)

    n = rebuild_manifest(manifest_jsonl, videos_dir, json_path, merge_path)
    print(f"[gen] done; manifest has {n} entries -> {json_path}", flush=True)


if __name__ == "__main__":
    main()
