import argparse
import sys
import traceback
from .fvd import compute_fvd_with_config, FVDConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Compute Fréchet Video Distance (FVD)')

    # Required arguments
    parser.add_argument('--real-path',
                        type=str,
                        required=True,
                        help='Path to real videos')
    parser.add_argument('--gen-path',
                        type=str,
                        required=True,
                        help='Path to generated videos')

    # Extractor selection
    parser.add_argument('--extractor',
                        type=str,
                        default='i3d',
                        choices=['i3d', 'clip', 'videomae'],
                        help='Feature extractor model to use (default: i3d)')

    # Standard args
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--protocol',
                        type=str,
                        default=None,
                        choices=['fvd2048_16f', 'fvd2048_128f', 'quick_test'],
                        help='Use standard protocol (overrides other settings)')
    parser.add_argument('--num-videos',
                        type=int,
                        default=2048,
                        help='Number of videos to use')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames per clip')
    parser.add_argument('--clip-strategy',
                        type=str,
                        default='beginning',
                        help='Clip sampling strategy')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--cache-real-features',
                        type=str,
                        default=None,
                        help='Path to cache real video features')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Create config
    if args.protocol:
        protocol_map = {
            'fvd2048_16f': FVDConfig.fvd2048_16f,
            'fvd2048_128f': FVDConfig.fvd2048_128f,
            'quick_test': FVDConfig.quick_test,
        }
        config = protocol_map[args.protocol]()
        # Apply overrides
        config.device = args.device
        config.cache_real_features = args.cache_real_features
        config.extractor_model = args.extractor  # Apply extractor arg
    else:
        config = FVDConfig(
            num_videos=args.num_videos,
            num_frames_per_clip=args.num_frames,
            extractor_model=args.extractor,  # Apply extractor arg
            clip_strategy=args.clip_strategy,
            batch_size=args.batch_size,
            device=args.device,
            cache_real_features=args.cache_real_features,
            seed=args.seed)

    try:
        _ = compute_fvd_with_config(
            args.real_path,  # noqa: F841
            args.gen_path,
            config,
            verbose=not args.quiet)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
