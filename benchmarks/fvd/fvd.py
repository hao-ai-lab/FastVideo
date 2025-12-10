import numpy as np
import scipy.linalg
import torch
from pathlib import Path
from collections.abc import Iterator
import pickle
from dataclasses import dataclass, field
from .feature_extractors import BaseFeatureExtractor, load_extractor
from .video_utils import ClipSamplingStrategy, load_video_clips_streaming


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_frechet_distance(mu1: np.ndarray,
                             sigma1: np.ndarray,
                             mu2: np.ndarray,
                             sigma2: np.ndarray,
                             eps: float = 1e-6) -> float:
    """
    Compute Fréchet distance between two Gaussians.
    """
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    diff = mu1 - mu2
    mean_distance = np.sum(diff**2)

    trace_sum = np.trace(sigma1 + sigma2)

    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(
                f"Warning: Imaginary component: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    trace_product = np.trace(covmean)

    fvd = mean_distance + trace_sum - 2 * trace_product

    return float(fvd)


@dataclass
class FVDConfig:
    # default configuration for FVD computation:

    # Video selection
    num_videos: int = 2048

    # Feature Extractor Selection
    extractor_model: str = 'i3d'  # Options: 'i3d', 'clip', 'videomae'

    # Clip sampling
    num_frames_per_clip: int = 16
    num_clips_per_video: int = 1
    clip_strategy: str | ClipSamplingStrategy = 'beginning'

    # Temporal subsampling
    frame_stride: int = 1  # 1=no subsampling, 2=every 2nd, 8=every 8th
    temporal_stride: int = 1  # For sliding window clips

    # Data processing
    video_extensions: list[str] = field(
        default_factory=lambda: ['.mp4', '.avi', '.mov', '.mkv'])
    support_frame_dirs: bool = True

    # Computation
    batch_size: int = 32
    device: str = 'cuda'

    use_streaming: bool = True
    resize_before_extraction: bool = True

    # Caching
    cache_real_features: str | None = None
    i3d_model_path: str | None = None

    # Reproducibility
    seed: int | None = None

    @classmethod
    def fvd2048_16f(cls) -> 'FVDConfig':
        """Standard FVD protocol: 2048 videos, 16 frames, beginning clip."""
        return cls(num_videos=2048,
                   num_frames_per_clip=16,
                   clip_strategy='beginning',
                   use_streaming=True)

    @classmethod
    def fvd2048_128f(cls) -> 'FVDConfig':
        """Long video protocol: 2048 videos, 128 frames."""
        return cls(num_videos=2048,
                   num_frames_per_clip=128,
                   clip_strategy='beginning',
                   use_streaming=True)

    @classmethod
    def quick_test(cls) -> 'FVDConfig':
        """Quick test config: 100 videos, 16 frames."""
        return cls(num_videos=100,
                   num_frames_per_clip=16,
                   clip_strategy='beginning')

    def to_dict(self) -> dict:
        """Export config to dict for logging"""
        d = self.__dict__.copy()
        d['clip_strategy'] = str(self.clip_strategy)
        return d

    def __str__(self) -> str:
        """Human-readable protocol name"""
        desc = f"FVD_{self.extractor_model.upper()}_{self.num_videos}_{self.num_frames_per_clip}f"
        if self.frame_stride > 1:
            desc += f"_subsample{self.frame_stride}"
        if self.num_clips_per_video > 1:
            desc += f"_{self.num_clips_per_video}clips"
        if self.clip_strategy != 'beginning':
            desc += f"_{self.clip_strategy}"
        return desc


def extract_features_streaming(video_generator: Iterator[torch.Tensor],
                               extractor: BaseFeatureExtractor,
                               batch_size: int = 32,
                               max_clips: int | None = None,
                               verbose: bool = True) -> np.ndarray:
    """
    Extract features from a video clip generator using streaming.
    """
    all_features = []
    batch = []

    if verbose:
        print(f"Extracting features with batch_size={batch_size}...")

    with torch.no_grad():
        for clip_count, clip in enumerate(video_generator):
            batch.append(clip)

            # Process batch when full
            if len(batch) == batch_size:
                batch_tensor = torch.stack(batch).to(extractor.device)
                features = extractor.extract_features_batch(batch_tensor)

                all_features.append(features.detach().cpu().numpy())
                batch = []

                if verbose and clip_count % (batch_size * 10) == 0:
                    print(f"Processed {clip_count} clips...")

            if max_clips is not None and clip_count >= max_clips:
                break

        # Process remaining clips
        if len(batch) > 0:
            batch_tensor = torch.stack(batch).to(extractor.device)
            features = extractor.extract_features_batch(batch_tensor)
            all_features.append(features.detach().cpu().numpy())

    if len(all_features) == 0:
        raise RuntimeError("No features extracted - check video loading")

    features = np.concatenate(all_features, axis=0)

    if verbose:
        print(f"Extracted {len(features)} feature vectors")

    return features


def load_or_compute_features(videos: str | Path | torch.Tensor,
                             extractor: BaseFeatureExtractor,
                             config: FVDConfig,
                             cache_path: str | None = None,
                             cache_name: str = "real_features") -> np.ndarray:
    """Load features from cache or compute (with streaming support)"""

    if cache_path is not None:
        script_dir = Path(__file__).parent
        cache_dir = script_dir / cache_path
        cache_file = cache_dir / f"{config.extractor_model}_{cache_name}.pkl"

        if cache_file.exists():
            print(f"Loading cached features from {cache_file}")
            with open(cache_file, 'rb') as f:
                features = pickle.load(f)

            # Validate and limit based on config
            max_features = config.num_videos * config.num_clips_per_video

            if len(features) < max_features:
                print(
                    f"WARNING: Cache has {len(features)} features but need {max_features}"
                )
                print("Cached features insufficient - will recompute...")
            elif len(features) > max_features:
                print(
                    f"Using {max_features} features from cache (truncated from {len(features)})"
                )
                features = features[:max_features]
                return features
            else:
                print(f"Using all {len(features)} cached features")
                return features

    print("Computing features from scratch...")

    if isinstance(videos, (str | Path)):
        target_size = (224, 224) if config.resize_before_extraction else None

        video_generator = load_video_clips_streaming(
            videos,
            num_frames=config.num_frames_per_clip,
            max_videos=config.num_videos,
            clip_strategy=config.clip_strategy,
            frame_stride=config.frame_stride,
            num_clips_per_video=config.num_clips_per_video,
            video_extensions=config.video_extensions,
            support_frame_dirs=config.support_frame_dirs,
            target_size=target_size,
            verbose=True)

        max_clips = config.num_videos * config.num_clips_per_video
        features = extract_features_streaming(video_generator,
                                              extractor,
                                              batch_size=config.batch_size,
                                              max_clips=max_clips,
                                              verbose=True)
    else:
        print(f"Extracting features from {len(videos)} video tensors...")
        features = extractor.extract_features(videos,
                                              batch_size=config.batch_size,
                                              verbose=True)
        features = features.numpy()

    # Validate feature count
    expected_count = config.num_videos * config.num_clips_per_video
    if len(features) < expected_count:
        raise ValueError(
            f"ERROR: Only extracted {len(features)} features, but need {expected_count}!\n"
            f"Found fewer videos than expected. Check your video directory.")
    elif len(features) > expected_count:
        print(f"Truncating {len(features)} features to {expected_count}")
        features = features[:expected_count]

    # Cache features if requested
    if cache_path is not None:
        script_dir = Path(__file__).parent
        cache_dir = script_dir / cache_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{config.extractor_model}_{cache_name}.pkl"
        print(f"Caching features to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)

    return features


def compute_fvd_with_config(real_videos: str | Path | torch.Tensor,
                            gen_videos: str | Path | torch.Tensor,
                            config: FVDConfig,
                            verbose: bool = True) -> dict:
    """
    Compute FVD using a standardized configuration.
    """
    # Seed for reproducibility
    if config.seed is not None:
        import random as _rnd
        _rnd.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    if verbose:
        print("=" * 70)
        print(f"Computing FVD with protocol: {config}")
        print(f"Model: {config.extractor_model.upper()}")
        print("=" * 70)
        print("\nConfiguration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        print()

    # Initialize Extractor using Factory
    if verbose:
        print(
            f"\nInitializing {config.extractor_model.upper()} model on {config.device}..."
        )

    extractor = load_extractor(config.extractor_model, device=config.device)

    # Extract features
    if verbose:
        print(f"\n{'='*70}")
        print("Extracting REAL video features...")
        print(f"{'='*70}")

    real_features = load_or_compute_features(
        videos=real_videos,
        extractor=extractor,
        config=config,
        cache_path=config.cache_real_features,
        cache_name="real_features")

    if verbose:
        print(f"\n{'='*70}")
        print("Extracting GENERATED video features...")
        print(f"{'='*70}")

    gen_features = load_or_compute_features(videos=gen_videos,
                                            extractor=extractor,
                                            config=config,
                                            cache_path=None,
                                            cache_name="gen_features")

    if verbose:
        print(f"\nReal videos/clips: {len(real_features)}")
        print(f"Generated videos/clips: {len(gen_features)}")
        print(f"\n{'='*70}")
        print("Computing statistics...")
        print(f"{'='*70}")

    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(gen_features)

    if verbose:
        print(f"\n{'='*70}")
        print("Computing Fréchet distance...")
        print(f"{'='*70}")

    fvd = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FVD Score ({config.extractor_model.upper()}): {fvd:.4f}")
        print(f"Protocol: {config}")
        print(f"{'='*70}\n")

    results = {
        'fvd': fvd,
        'protocol': str(config),
        'model': config.extractor_model,
        'config': config.to_dict(),
    }

    return results


def compute_fvd(real_videos: str | Path | torch.Tensor,
                gen_videos: str | Path | torch.Tensor,
                num_frames: int = 16,
                batch_size: int = 32,
                device: str = 'cuda',
                num_videos: int | None = 2048,
                cache_real_features: str | None = None,
                i3d_model_path: str | None = None,
                seed: int | None = None,
                verbose: bool = True) -> float:
    """
    Backward compatibility wrapper for computing FVD (defaults to I3D).
    """
    num_videos = num_videos if num_videos is not None else 2048

    config = FVDConfig(
        num_videos=num_videos,
        num_frames_per_clip=num_frames,
        extractor_model='i3d',  # Default to I3D
        batch_size=batch_size,
        device=device,
        cache_real_features=cache_real_features,
        i3d_model_path=i3d_model_path,
        seed=seed,
    )

    result = compute_fvd_with_config(real_videos, gen_videos, config, verbose)
    return result['fvd']
