import argparse
import json
import sys
from pathlib import Path

from fvd import compute_fvd_with_config, FVDConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Compute Fréchet Video Distance (FVD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard FVD2048_16f protocol
  python -m fastvideo.benchmarks.fvd.cli \\
      --real-path data/real/ \\
      --gen-path outputs/gen/ \\
      --protocol fvd2048_16f

  # Custom configuration
  python -m fastvideo.benchmarks.fvd.cli \\
      --real-path data/real/ \\
      --gen-path outputs/gen/ \\
      --num-videos 1024 \\
      --num-frames 32 \\
      --clip-strategy random \\
      --frame-stride 2
        """)

    # Required arguments
    parser.add_argument('--real-path',
                        type=str,
                        required=True,
                        help='Path to real videos directory')
    parser.add_argument('--gen-path',
                        type=str,
                        required=True,
                        help='Path to generated videos directory')

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (np.random, random, torch)')

    # Protocol presets
    parser.add_argument('--protocol',
                        type=str,
                        default=None,
                        choices=[
                            'fvd2048_16f', 'fvd2048_128f',
                            'fvd2048_128f_subsample8', 'quick_test'
                        ],
                        help='Use standard protocol (overrides other settings)')

    # Video selection
    parser.add_argument('--num-videos',
                        type=int,
                        default=2048,
                        help='Number of videos to use (default: 2048)')

    # Clip sampling
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames per clip (default: 16)')
    parser.add_argument('--num-clips',
                        type=int,
                        default=1,
                        help='Number of clips per video (default: 1)')
    parser.add_argument(
        '--clip-strategy',
        type=str,
        default='beginning',
        choices=['beginning', 'random', 'uniform', 'middle', 'sliding', 'all'],
        help='Clip sampling strategy (default: beginning)')
    parser.add_argument(
        '--frame-stride',
        type=int,
        default=1,
        help='Frame stride for FPS subsampling (default: 1, no subsampling)')
    parser.add_argument('--temporal-stride',
                        type=int,
                        default=1,
                        help='Temporal stride for sliding window (default: 1)')

    # Data processing
    parser.add_argument('--no-frame-dirs',
                        action='store_true',
                        help='Disable frame directory support')

    # Computation
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Batch size for feature extraction (default: 32)')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')

    # Caching
    parser.add_argument('--cache-real-features',
                        type=str,
                        default=None,
                        help='Path to cache real video features')
    parser.add_argument('--i3d-model-path',
                        type=str,
                        default=None,
                        help='Custom cache path for I3D model')

    # Output
    parser.add_argument('--output',
                        type=str,
                        default='fvd_results.json',
                        help='Output JSON file (default: fvd_results.json)')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Create config
    if args.protocol:
        protocol_map = {
            'fvd2048_16f': FVDConfig.fvd2048_16f,
            'fvd2048_128f': FVDConfig.fvd2048_128f,
            'fvd2048_128f_subsample8': FVDConfig.fvd2048_128f_subsample8,
            'quick_test': FVDConfig.quick_test,
        }
        config = protocol_map[args.protocol]()

        # Override device and caching from args
        config.device = args.device
        config.cache_real_features = args.cache_real_features
        config.i3d_model_path = args.i3d_model_path
        config.batch_size = args.batch_size
        config.seed = args.seed
    else:
        # Custom config from args
        config = FVDConfig(num_videos=args.num_videos,
                           num_frames_per_clip=args.num_frames,
                           num_clips_per_video=args.num_clips,
                           clip_strategy=args.clip_strategy,
                           frame_stride=args.frame_stride,
                           temporal_stride=args.temporal_stride,
                           support_frame_dirs=not args.no_frame_dirs,
                           batch_size=args.batch_size,
                           device=args.device,
                           cache_real_features=args.cache_real_features,
                           i3d_model_path=args.i3d_model_path,
                           seed=args.seed)

    # Compute FVD
    try:
        results = compute_fvd_with_config(real_videos=args.real_path,
                                          gen_videos=args.gen_path,
                                          config=config,
                                          verbose=not args.quiet)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"FVD: {results['fvd']:.2f}")
        print(f"Protocol: {results['protocol']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
