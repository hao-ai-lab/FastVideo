# FVD (Fréchet Video Distance) Benchmark

Evaluate generated video quality using FVD with the I3D feature extractor.

## Quick Start

**Run the benchmark:**

```bash
bash benchmarks/scripts/run.sh
```

That's it! The script auto-installs dependencies and runs the benchmark.

**To customize:** Edit `benchmarks/fvd/run_fvd.py` to change:
- Video paths (`real_dir`, `gen_dir`)
- Number of videos, frames, sampling strategy
- Device, batch size, caching, etc.

## Advanced Usage (CLI)

For more control without editing Python files, use the CLI.

**First-time setup** (one-time per pod/environment):

```bash
bash benchmarks/scripts/setup_fvd.sh
```

Then run any configuration you want:

```bash
# Custom configuration
python -m benchmarks.fvd.cli \
    --real-path data/real/ \
    --gen-path outputs/gen/ \
    --num-videos 1024 \
    --num-frames 32 \
    --clip-strategy random \
    --batch-size 32 \
    --seed 42 \
    --extractor clip
```

**Standard protocols:**

```bash
# Use predefined protocols
python -m benchmarks.fvd.cli \
    --real-path data/real/ \
    --gen-path outputs/gen/ \
    --protocol fvd2048_16f  # or fvd2048_128f, quick_test, etc.
```

This would use i3d model by default as the feature extractor

**Feature caching** (speed up repeated evaluations):

```bash
python -m benchmarks.fvd.cli \
    --real-path data/real/ \
    --gen-path outputs/gen/ \
    --protocol fvd2048_16f \
    --cache-real-features fvd-cache/extractor_name  # Directory path (will save/load fvd-cache/extractor_name/extractor-name_real_features.pkl)
```

Run `python -m benchmarks.fvd.cli --help` for all options.

## Available Protocols

- `fvd2048_16f` - Standard (2048 videos, 16 frames)
- `fvd2048_128f` - Long videos (128 frames)
- `fvd2048_128f_subsample8` - Subsampled long videos
- `quick_test` - Fast testing (10 videos)

## Configuration Options

Key options in `FVDConfig`:

```python
num_videos=2048,              # Videos to evaluate
num_frames_per_clip=16,       # Frames per clip
clip_strategy='beginning',    # beginning|random|uniform|middle|sliding
frame_stride=1,               # Frame subsampling
batch_size=32,                # GPU batch size
device='cuda',                # cuda|cpu
cache_real_features=None,     # Cache path for speed
seed=42,                      # Reproducibility
extractor='i3d',              # i3d|clip|videomae
```

## Programmatic Usage

```python
from benchmarks.fvd import compute_fvd_with_config, FVDConfig

config = FVDConfig.fvd2048_16f()  # or custom config
results = compute_fvd_with_config('data/real/', 'outputs/gen/', config)
print(f"FVD: {results['fvd']:.2f}")
```

## Notes

- Requires minimum 10 frames per clip
- Supports both video files (.mp4, .avi, etc.) and frame directories
- `--cache-real-features` expects a **directory path** (e.g., `cache/real`), it will automatically create/load `real_features.pkl` inside that directory
