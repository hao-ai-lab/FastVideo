#!/usr/bin/env python3
"""
For each data dir (from finetune_wangame.slurm), randomly pick 10 samples (mp4 + action.npy),
copy to to_shao/<short_name>/ as 01.mp4, 01_action.npy, ..., 10.mp4, 10_action.npy,
and extract first frame as 01.jpg, ..., 10.jpg.
"""
import os
import random
import shutil

import cv2

# Data dirs from finetune_wangame.slurm (paths with "preprocessed"; we use "video" or "videos" for mp4/npy)
DATA_DIRS = [
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0204_2130/preprocessed",
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_wasd_only/preprocessed",
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/wasdonly_alpha1/preprocessed",
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/camera/preprocessed",
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/camera4hold_alpha1/preprocessed",
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed",
]

OUT_ROOT = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao"
NUM_SAMPLES = 10


def get_video_dir(preprocessed_path: str) -> str | None:
    """Replace 'preprocessed' with 'video' (or 'videos') to get the dir containing mp4/npy."""
    video_path = preprocessed_path.replace("preprocessed", "video")
    if os.path.isdir(video_path):
        return video_path
    videos_path = preprocessed_path.replace("preprocessed", "videos")
    if os.path.isdir(videos_path):
        return videos_path
    return None


# Override short name for specific data dirs (e.g. traindata_0204_2130 -> fully_random)
SHORT_NAME_OVERRIDES: dict[str, str] = {
    "traindata_0204_2130": "fully_random",
}


def get_short_name(preprocessed_path: str) -> str:
    """Short name = parent folder of the preprocessed dir, e.g. 1_wasd_only."""
    name = os.path.basename(os.path.normpath(os.path.dirname(preprocessed_path)))
    return SHORT_NAME_OVERRIDES.get(name, name)


def find_samples(video_dir: str) -> list[str]:
    """Return list of base names (no extension) that have both xxxxxx.mp4 and xxxxxx_action.npy."""
    samples = []
    for f in os.listdir(video_dir):
        if f.endswith(".mp4"):
            base = f[:-4]
            action_path = os.path.join(video_dir, f"{base}_action.npy")
            if os.path.isfile(action_path):
                samples.append(base)
    return samples


def extract_first_frame(mp4_path: str, jpg_path: str) -> None:
    cap = cv2.VideoCapture(mp4_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(jpg_path, frame)


def main() -> None:
    random.seed(42)
    os.makedirs(OUT_ROOT, exist_ok=True)
    total_dir = os.path.join(OUT_ROOT, "total")
    os.makedirs(total_dir, exist_ok=True)
    total_idx = 0

    for preprocessed_path in DATA_DIRS:
        video_dir = get_video_dir(preprocessed_path)
        if video_dir is None:
            print(f"Skip (video dir not found): {preprocessed_path}")
            continue

        short_name = get_short_name(preprocessed_path)
        samples = find_samples(video_dir)
        if len(samples) < NUM_SAMPLES:
            print(f"Skip {short_name}: only {len(samples)} samples (need {NUM_SAMPLES})")
            continue

        chosen = random.sample(samples, NUM_SAMPLES)
        out_dir = os.path.join(OUT_ROOT, short_name)
        os.makedirs(out_dir, exist_ok=True)

        for i, base in enumerate(chosen, start=1):
            num_str = f"{i:02d}"
            src_mp4 = os.path.join(video_dir, f"{base}.mp4")
            src_npy = os.path.join(video_dir, f"{base}_action.npy")
            dst_mp4 = os.path.join(out_dir, f"{num_str}.mp4")
            dst_npy = os.path.join(out_dir, f"{num_str}_action.npy")
            dst_jpg = os.path.join(out_dir, f"{num_str}.jpg")

            shutil.copy2(src_mp4, dst_mp4)
            shutil.copy2(src_npy, dst_npy)
            extract_first_frame(dst_mp4, dst_jpg)

            # Copy into total/ with global numbering
            total_idx += 1
            t_str = f"{total_idx:02d}"
            shutil.copy2(dst_mp4, os.path.join(total_dir, f"{t_str}.mp4"))
            shutil.copy2(dst_npy, os.path.join(total_dir, f"{t_str}_action.npy"))
            shutil.copy2(dst_jpg, os.path.join(total_dir, f"{t_str}.jpg"))

        print(f"Done: {short_name} -> {out_dir} ({NUM_SAMPLES} samples)")

    print(f"Done: total -> {total_dir} ({total_idx} samples)")


if __name__ == "__main__":
    main()
