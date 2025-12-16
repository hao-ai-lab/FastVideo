#!/usr/bin/env python3
import sys
from pathlib import Path
import shutil
import random
from fvd import compute_fvd_with_config, FVDConfig

script_path = Path(__file__).resolve()
fastvideo_root = script_path.parent.parent.parent
sys.path.insert(0, str(fastvideo_root))


def split_videos(video_dir: Path, n_per_subset: int = 128, seed: int = 42):
    subset_a = video_dir.parent / 'bair_full_subset_A'
    subset_b = video_dir.parent / 'bair_full_subset_B'

    if subset_a.exists():
        shutil.rmtree(subset_a)
    if subset_b.exists():
        shutil.rmtree(subset_b)

    subset_a.mkdir(parents=True)
    subset_b.mkdir(parents=True)

    videos = sorted(video_dir.glob('*.mp4'))

    random.seed(seed)
    shuffled = list(videos)
    random.shuffle(shuffled)

    needed = n_per_subset * 2
    if len(shuffled) > needed:
        shuffled = shuffled[:needed]

    mid = len(shuffled) // 2

    print(f"\nSplitting {len(shuffled)} BAIR FULL videos:")
    print(f"  Subset A: {mid} videos")
    print(f"  Subset B: {len(shuffled) - mid} videos")

    for v in shuffled[:mid]:
        shutil.copy2(v, subset_a / v.name)

    for v in shuffled[mid:]:
        shutil.copy2(v, subset_b / v.name)

    return subset_a, subset_b, mid


def validate_fvd(subset_a: Path, subset_b: Path, num_videos: int):
    config = FVDConfig(num_videos=num_videos,
                       num_frames_per_clip=16,
                       clip_strategy='beginning',
                       batch_size=8,
                       device='cuda',
                       seed=42)

    print("\n" + "=" * 70)
    print("TEST 1: Identity Test")
    print("=" * 70)

    result1 = compute_fvd_with_config(real_videos=str(subset_a),
                                      gen_videos=str(subset_a),
                                      config=config,
                                      verbose=False)
    fvd_identity = result1['fvd']
    print(f"\nIdentity FVD: {fvd_identity:.2f}")

    print("\n" + "=" * 70)
    print("TEST 2: Real vs Real")
    print("=" * 70)

    result2 = compute_fvd_with_config(real_videos=str(subset_a),
                                      gen_videos=str(subset_b),
                                      config=config,
                                      verbose=False)
    fvd_real = result2['fvd']
    print(f"\nReal vs Real FVD: {fvd_real:.2f}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Identity:        {fvd_identity:.2f}")
    print(f"Real vs Real:    {fvd_real:.2f}")


def main() -> None:
    bair_dir = Path('benchmarks/data/bair_full_videos')

    subset_a, subset_b, count = split_videos(bair_dir,
                                             n_per_subset=128,
                                             seed=42)
    validate_fvd(subset_a, subset_b, count)


if __name__ == '__main__':
    main()
