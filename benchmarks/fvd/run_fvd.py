import sys
from pathlib import Path
from benchmarks.fvd.fvd import FVDConfig, compute_fvd_with_config

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))


def main() -> None:
    # Get script directory
    script_dir = Path(__file__).parent.resolve()

    clip_strategy = 'beginning'  # Options: 'uniform', 'random', 'beginning', 'end', 'all'
    cfg = FVDConfig(
        num_videos=650,
        num_frames_per_clip=16,
        num_clips_per_video=1,
        clip_strategy=clip_strategy,
        frame_stride=1,
        batch_size=32,
        device='cuda',
        seed=42,
        cache_real_features=str(script_dir / f'fvd-cache/{clip_strategy}'),
    )

    real_dir = "benchmarks/data/real_videos"
    gen_dir = "benchmarks/data/generated_videos"

    results = compute_fvd_with_config(real_dir, gen_dir, cfg, verbose=True)
    print(f"FVD = {results['fvd']:.2f}")


if __name__ == '__main__':
    main()
