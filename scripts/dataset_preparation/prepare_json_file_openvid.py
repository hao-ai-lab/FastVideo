# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import torchvision
from tqdm import tqdm


def get_video_info(video_path):
    """Get video information using torchvision."""
    try:
        # Read video tensor (T, C, H, W)
        video_tensor, _, info = torchvision.io.read_video(
            str(video_path),
            output_format="TCHW",
            pts_unit="sec"
        )

        num_frames = video_tensor.shape[0]
        height = video_tensor.shape[2]
        width = video_tensor.shape[3]
        fps = info.get("video_fps", 0)
        duration = num_frames / fps if fps > 0 else 0
    except Exception as e:
        print(f"Warning: Failed to read video {video_path}: {e}")
        # Provide fallback values
        num_frames = 0
        height = 0
        width = 0
        fps = 0
        duration = 0

    # Extract only the filename (not full path) for "path" field
    video_name = video_path.name

    return {
        "path": str(video_name),
        "resolution": {
            "width": width,
            "height": height
        },
        "size": os.path.getsize(video_path) if video_path.exists() else 0,
        "fps": fps,
        "duration": duration,
        "num_frames": num_frames
    }


def prepare_dataset_json(folder_path,
                         output_name="videos2caption.json",
                         num_workers=None) -> None:
    """Prepare dataset information from a folder containing videos and annotations.csv."""
    folder_path = Path(folder_path)

    # Read CSV annotation file (assume it's named 'annotations.csv' or 'captions.csv')
    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {folder_path}")
    
    # Use the first CSV file (you can adjust this if needed)
    csv_file = csv_files[0]
    print(f"Loading annotations from: {csv_file}")

    # Read CSV: expect columns 'video' and 'caption'
    df = pd.read_csv(csv_file)

    # Check required columns
    if 'video' not in df.columns or 'caption' not in df.columns:
        raise ValueError(f"CSV must contain 'video' and 'caption' columns. Found: {list(df.columns)}")

    # Clean and prepare
    df = df[['video', 'caption']].dropna()
    video_names = df['video'].tolist()
    prompts = df['caption'].tolist()

    if len(video_names) != len(prompts):
        raise ValueError("Mismatch between video and caption counts after cleaning.")

    # Build full video paths
    video_paths = [folder_path / vid for vid in video_names]

    # Verify all videos exist (optional but recommended)
    missing = [vp for vp in video_paths if not Path(vp).exists()]
    if missing:
        print(f"Warning: {len(missing)} videos not found. Example: {missing[0] if missing else ''}")
        # Optionally skip missing videos
        valid_indices = [i for i, vp in enumerate(video_paths) if Path(vp).exists()]
        video_paths = [video_paths[i] for i in valid_indices]
        prompts = [prompts[i] for i in valid_indices]

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Process videos in parallel
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(get_video_info, video_paths),
                 total=len(video_paths),
                 desc="Processing videos",
                 unit="video"))

    # Combine results with prompts
    dataset_info = []
    for result, prompt in zip(results, prompts):
        result["cap"] = [prompt]  # Keep as list to match original format
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
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)

    # Create merge.txt
    merge_file = folder_path / "merge.txt"
    with open(merge_file, 'w') as f:
        f.write(f"{folder_path}/video_sample,{output_file}\n")

    print(f"Dataset information saved to {output_file}")
    print(f"Merge file created at {merge_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare video dataset information in JSON format from OpenVidHD-style CSV')
    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Path to the folder containing videos and a CSV annotation file (with "video" and "caption" columns)')
    parser.add_argument(
        '--output',
        type=str,
        default='videos2caption.json',
        help='Name of the output JSON file (default: videos2caption.json)')
    parser.add_argument('--workers',
                        type=int,
                        default=32,
                        help='Number of worker processes (default: 32)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset_json(args.data_folder, args.output, args.workers)
