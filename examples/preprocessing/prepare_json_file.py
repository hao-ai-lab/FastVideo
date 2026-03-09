# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torchvision
from tqdm import tqdm


def get_video_info(video_path):
    """Get video information using torchvision."""
    # Read video tensor (T, C, H, W)
    video_tensor, _, info = torchvision.io.read_video(str(video_path),
                                                      output_format="TCHW",
                                                      pts_unit="sec")

    num_frames = video_tensor.shape[0]
    height = video_tensor.shape[2]
    width = video_tensor.shape[3]
    fps = info.get("video_fps", 0)
    duration = num_frames / fps if fps > 0 else 0

    # Extract name
    _, _, videos_dir, video_name = str(video_path).split("/")

    return {
        "path": str(video_name),
        "resolution": {
            "width": width,
            "height": height
        },
        "size": os.path.getsize(video_path),
        "fps": fps,
        "duration": duration,
        "num_frames": num_frames
    }


def prepare_dataset_json(folder_path,
                         output_name="videos2caption.json",
                         num_workers=None) -> None:
    """Prepare dataset information from a folder containing videos and prompt.txt."""
    folder_path = Path(folder_path)

    # Read prompt file
    prompt_file = folder_path / "prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt.txt not found in {folder_path}")

    with open(prompt_file) as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Read videos file
    videos_file = folder_path / "videos.txt"
    if not videos_file.exists():
        raise FileNotFoundError(f"videos.txt not found in {folder_path}")

    with open(videos_file) as f:
        video_paths = [line.strip() for line in f.readlines() if line.strip()]

    if len(prompts) != len(video_paths):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) does not match number of videos ({len(video_paths)})"
        )

    # Prepare arguments for multiprocessing
    process_args = [folder_path / video_path for video_path in video_paths]

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free

    # Process videos in parallel
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(get_video_info, process_args),
                 total=len(process_args),
                 desc="Processing videos",
                 unit="video"))

    # Combine results with prompts
    dataset_info = []
    for result, prompt in zip(results, prompts):
        result["cap"] = [prompt]
        dataset_info.append(result)

    # Calculate total processing time
    total_time = time.time() - start_time
    total_videos = len(dataset_info)
    avg_time_per_video = total_time / total_videos if total_videos > 0 else 0

    print("\nProcessing completed:")
    print(f"Total videos processed: {total_videos}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per video: {avg_time_per_video:.2f} seconds")

    # Save to JSON file
    output_file = folder_path / output_name
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    # Create merge.txt
    merge_file = folder_path / "merge.txt"
    with open(merge_file, 'w') as f:
        f.write(f"{folder_path}/videos,{output_file}\n")

    print(f"Dataset information saved to {output_file}")
    print(f"Merge file created at {merge_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare video dataset information in JSON format')
    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Path to the folder containing videos and prompt.txt')
    parser.add_argument(
        '--output',
        type=str,
        default='videos2caption.json',
        help='Name of the output JSON file (default: videos2caption.json)')
    parser.add_argument('--workers',
                        type=int,
                        default=32,
                        help='Number of worker processes (default: 16)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset_json(args.data_folder, args.output, args.workers)
