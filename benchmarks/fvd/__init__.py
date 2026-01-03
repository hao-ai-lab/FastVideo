"""
FastVideo Frechet Video Distance (FVD) Benchmark Module.
    >>> from fastvideo.benchmarks.fvd import compute_fvd_with_config, FVDConfig
    >>> config = FVDConfig.fvd2048_16f()  # Standard protocol
    >>> results = compute_fvd_with_config('data/real/', 'outputs/gen/', config)
    >>> print(f"FVD: {results['fvd']:.2f}")
"""

from .fvd import (
    compute_fvd,
    compute_fvd_with_config,
    compute_frechet_distance,
    compute_statistics,
    FVDConfig,
)
from .feature_extractors import (BaseFeatureExtractor, I3DFeatureExtractor,
                                 load_extractor)
from .video_utils import (
    load_video_auto,
    sample_clips_from_video,
    load_video_clips_streaming,
    ClipSamplingStrategy,
)

__all__ = [
    'compute_fvd',
    'compute_fvd_with_config',
    'compute_frechet_distance',
    'compute_statistics',
    'FVDConfig',
    'BaseFeatureExtractor',
    'I3DFeatureExtractor',
    'load_extractor',
    'load_video_auto',
    'sample_clips_from_video',
    'load_video_clips_streaming',
    'ClipSamplingStrategy',
]
