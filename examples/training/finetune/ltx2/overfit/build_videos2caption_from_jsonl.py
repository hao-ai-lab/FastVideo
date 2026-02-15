# SPDX-License-Identifier: Apache-2.0
# TODO: Add a docstring explaining what this script is used for.
import argparse
import json
import os
import re
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torchvision
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _numeric_suffix(name: str) -> int:
    match = re.search(r"_(\d+)$", name)
    if not match:
        return -1
    return int(match.group(1))


def _get_video_info(args):
    video_path, prompt = args
    video_tensor, _, info = torchvision.io.read_video(
        str(video_path), output_format="TCHW", pts_unit="sec")

    num_frames = int(video_tensor.shape[0])
    height = int(video_tensor.shape[2])
    width = int(video_tensor.shape[3])
    fps = float(info.get("video_fps", 0.0) or 0.0)
    duration = num_frames / fps if fps > 0 else 0.0

    return {
        "path": video_path.name,
        "resolution": {
            "width": width,
            "height": height
        },
        "size": os.path.getsize(video_path),
        "fps": fps,
        "duration": duration,
        "num_frames": num_frames,
        "cap": [prompt],
    }


def _load_prompt_map(jsonl_path: Path, id_key: str, prompt_key: str) -> dict[str, str]:
    prompt_map: dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get(id_key)
            prompt = record.get(prompt_key)
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"Invalid id at line {line_no}: expected non-empty string '{id_key}'."
                )
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"Invalid prompt at line {line_no}: expected non-empty string '{prompt_key}'."
                )
            prompt_map[sample_id] = prompt
    return prompt_map


def build_videos2caption(
    videos_dir: Path,
    prompts_jsonl: Path,
    output_json: Path,
    workers: int,
    id_key: str,
    prompt_key: str,
) -> None:
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"Prompts jsonl not found: {prompts_jsonl}")

    prompt_map = _load_prompt_map(prompts_jsonl, id_key=id_key, prompt_key=prompt_key)
    print(f"Loaded {len(prompt_map)} prompts from {prompts_jsonl}")

    video_files = [
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    video_files.sort(key=lambda p: (_numeric_suffix(p.stem), p.name))
    print(f"Found {len(video_files)} videos in {videos_dir}")

    process_args = []
    videos_missing_prompt = []
    for video in video_files:
        prompt = prompt_map.get(video.stem)
        if prompt is None:
            videos_missing_prompt.append(video.name)
            continue
        process_args.append((video, prompt))

    if videos_missing_prompt:
        print(
            f"Skipping {len(videos_missing_prompt)} videos without prompt match (by stem==id)."
        )

    if not process_args:
        raise RuntimeError("No videos matched prompt IDs. Nothing to write.")

    workers = max(1, workers)
    start = time.time()
    if workers == 1:
        results = [
            _get_video_info(arg)
            for arg in tqdm(
                process_args,
                total=len(process_args),
                desc="Building videos2caption",
                unit="video",
            )
        ]
    else:
        try:
            with Pool(workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_get_video_info, process_args),
                        total=len(process_args),
                        desc="Building videos2caption",
                        unit="video",
                    ))
        except PermissionError:
            print(
                "Multiprocessing is unavailable in this environment; falling back to workers=1."
            )
            results = [
                _get_video_info(arg)
                for arg in tqdm(
                    process_args,
                    total=len(process_args),
                    desc="Building videos2caption",
                    unit="video",
                )
            ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"Wrote {len(results)} entries to {output_json}")
    print(f"Elapsed: {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build videos2caption.json from a prompt JSONL and video directory."
    )
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--id-key", type=str, default="id")
    parser.add_argument("--prompt-key", type=str, default="video_prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_videos2caption(
        videos_dir=args.videos_dir,
        prompts_jsonl=args.prompts_jsonl,
        output_json=args.output_json,
        workers=args.workers,
        id_key=args.id_key,
        prompt_key=args.prompt_key,
    )


if __name__ == "__main__":
    main()
