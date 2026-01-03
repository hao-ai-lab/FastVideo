import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from benchmarks.fvd.fvd import FVDConfig, compute_fvd_with_config  # noqa: E402


def main() -> None:
    script_dir = Path(__file__).parent.resolve()

    # Define directories
    real_dir = "benchmarks/data/real_videos"
    gen_dir = "benchmarks/data/generated_videos"

    # Compare all 3 models
    models_to_test = ['i3d', 'clip', 'videomae']

    print(f"\n{'='*60}")
    print("STARTING COMPARISON BENCHMARK")
    print(f"{'='*60}")

    for model_name in models_to_test:
        print(f"\n>>> Running evaluation with {model_name.upper()}...")

        try:
            cfg = FVDConfig(
                num_videos=650,
                num_frames_per_clip=16,
                extractor_model=model_name,
                clip_strategy='beginning',
                device='cuda',
                seed=42,
                # Use separate cache folders for each model to avoid conflicts
                cache_real_features=str(script_dir / f'fvd-cache/{model_name}'),
            )

            results = compute_fvd_with_config(real_dir,
                                              gen_dir,
                                              cfg,
                                              verbose=False)
            print(f"FVD: {results['fvd']}\nModel: {results['model']}")

        except Exception as e:
            print(f"{model_name.upper()} Failed: {e}")

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
