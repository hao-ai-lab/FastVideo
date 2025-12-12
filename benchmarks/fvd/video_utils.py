import torch
import cv2
import numpy as np
from pathlib import Path
from collections.abc import Iterator
from tqdm import tqdm
from enum import Enum


class ClipSamplingStrategy(Enum):
    """Clip sampling strategies for FVD evaluation."""
    BEGINNING = 'beginning'  # Take first N frames (most common)
    RANDOM = 'random'  # Random N consecutive frames
    UNIFORM = 'uniform'  # Uniformly spaced frames across video
    MIDDLE = 'middle'  # Middle N frames
    SLIDING = 'sliding'  # Multiple sliding windows
    ALL = 'all'  # All possible clips


def _load_video_cv2(video_path: str | Path,
                    num_frames: int | None = 16,
                    sample_strategy: str = 'uniform') -> torch.Tensor:
    """
    Load video from video file using OpenCV.
    
    Args:
        video_path: Path to video file (MP4, AVI, MOV, MKV)
        num_frames: Number of frames to extract
        sample_strategy: 'uniform' or 'random'
    
    Returns:
        video: [T, C, H, W]
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames is None:
        # Read all available frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"Video has 0 frames: {video_path}")

        frames = np.stack(frames)  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(0, 3, 1,
                                                  2).float()  # [T, C, H, W]
        return frames

    if total_frames == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    # Determine frame indices for sampling
    if total_frames < num_frames:
        frame_indices = list(range(
            total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    elif sample_strategy == 'uniform':
        frame_indices = np.linspace(0, total_frames - 1, num_frames,
                                    dtype=int).tolist()
    elif sample_strategy == 'random':
        frame_indices = sorted(
            np.random.choice(total_frames, num_frames, replace=False))
    else:
        raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    # Extract frames
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    frames = np.stack(frames)  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(0, 3, 1,
                                              2).float()  # [T, C, H, W]

    return frames


def _load_video_from_frames(
        frame_dir: str | Path,
        num_frames: int | None = 16,
        sample_strategy: str = 'uniform',
        frame_extensions: list[str] | None = None) -> torch.Tensor:
    """
    Load video from directory of frame images.
    
    Args:
        frame_dir: Directory containing frames
        num_frames: Number of frames to sample
        sample_strategy: 'uniform' or 'random'
        frame_extensions: Image file extensions to look for
    
    Returns:
        video: [T, C, H, W]
    """
    if frame_extensions is None:
        frame_extensions = ['.jpg', '.png', '.jpeg', '.bmp']

    frame_dir = Path(frame_dir)

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    # Find all frames
    frame_files: list[Path] = []
    for ext in frame_extensions:
        frame_files.extend(frame_dir.glob(f"*{ext}"))

    if len(frame_files) == 0:
        raise ValueError(
            f"No frames found in {frame_dir} with extensions {frame_extensions}"
        )

    frame_files = sorted(frame_files, key=lambda x: x.name)
    total_frames = len(frame_files)

    # Determine frame indices
    if num_frames is None:
        frame_indices = list(range(total_frames))
    else:
        if total_frames < num_frames:
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (
                num_frames - total_frames)
        elif sample_strategy == 'uniform':
            frame_indices = np.linspace(0,
                                        total_frames - 1,
                                        num_frames,
                                        dtype=int).tolist()
        elif sample_strategy == 'random':
            frame_indices = sorted(
                np.random.choice(total_frames, num_frames, replace=False))
        else:
            raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    # Load frames
    frames = []
    for idx in frame_indices:
        frame_path = frame_files[idx]
        frame = cv2.imread(str(frame_path))

        if frame is None:
            raise RuntimeError(f"Failed to load frame: {frame_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Stack and convert to tensor
    frames = np.stack(frames)  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(0, 3, 1,
                                              2).float()  # [T, C, H, W]

    return frames


def _detect_video_format(path: str | Path) -> str:
    """
    Detect if path is a video file or frame directory.
    
    Returns:
        'video_file', 'frame_directory', or 'unknown'
    """
    path = Path(path)

    if path.is_file():
        return 'video_file'
    elif path.is_dir():
        # Check if contains image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            if list(path.glob(f"*{ext}")):
                return 'frame_directory'
        return 'unknown'
    else:
        raise ValueError(f"Path does not exist: {path}")


def load_video_auto(video_path: str | Path,
                    num_frames: int | None = 16,
                    sample_strategy: str = 'uniform') -> torch.Tensor:
    """
    Automatically detect format and load video.
    
    Supports:
    - Video files (MP4, AVI, MOV, MKV)
    - Frame directories (JPG, PNG)
    
    Args:
        video_path: Path to video file or frame directory
        num_frames: Number of frames to extract
        sample_strategy: 'uniform' or 'random'
    
    Returns:
        video: [T, C, H, W]
    """
    format_type = _detect_video_format(video_path)

    if format_type == 'video_file':
        return _load_video_cv2(video_path, num_frames, sample_strategy)
    elif format_type == 'frame_directory':
        return _load_video_from_frames(video_path, num_frames, sample_strategy)
    else:
        raise ValueError(f"Unknown video format at {video_path}")


def sample_clips_from_video(
        video: torch.Tensor,
        num_frames_per_clip: int = 16,
        num_clips: int = 1,
        strategy: str | ClipSamplingStrategy = ClipSamplingStrategy.BEGINNING,
        frame_stride: int = 1,
        temporal_stride: int = 1) -> list[torch.Tensor]:
    """
    Sample clips from a video with various strategies.
        
    Args:
        video: [T, C, H, W] full video
        num_frames_per_clip: Frames per clip
        num_clips: Number of clips to extract
        strategy: ClipSamplingStrategy or string ('beginning', 'random', etc.)
        frame_stride: Skip frames (FPS control: 1=all, 2=every 2nd, 8=every 8th)
        temporal_stride: Stride between clips for sliding window
    
    Returns:
        List of clips, each [num_frames_per_clip, C, H, W]
    
    Examples:
        >>> # Beginning clip (most common for FVD)
        >>> clips = sample_clips_from_video(video, 16, strategy='beginning')
        
        >>> # Multiple random clips
        >>> clips = sample_clips_from_video(video, 16, num_clips=4, strategy='random')
        
        >>> # Subsample FPS by 2x (every 2nd frame)
        >>> clips = sample_clips_from_video(video, 16, frame_stride=2)
        
        >>> # Sliding window with overlap
        >>> clips = sample_clips_from_video(video, 16, strategy='sliding', temporal_stride=8)
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        strategy = ClipSamplingStrategy(strategy)

    T, C, H, W = video.shape

    # Apply frame stride (FPS subsampling)
    if frame_stride > 1:
        video = video[::frame_stride]
        T = len(video)

    effective_clip_length = num_frames_per_clip

    # Handle videos shorter than clip length
    if effective_clip_length > T:
        pad_length = effective_clip_length - T
        last_frame = video[-1:].repeat(pad_length, 1, 1, 1)
        video = torch.cat([video, last_frame], dim=0)
        T = len(video)

    clips = []

    if strategy == ClipSamplingStrategy.BEGINNING:
        # Take first clip (most common for FVD evaluation)
        clip = video[:effective_clip_length]
        clips.append(clip)

    elif strategy == ClipSamplingStrategy.MIDDLE:
        # Take middle clip
        start = (T - effective_clip_length) // 2
        clip = video[start:start + effective_clip_length]
        clips.append(clip)

    elif strategy == ClipSamplingStrategy.RANDOM:
        # Sample N random clips
        for _ in range(num_clips):
            if effective_clip_length == T:
                start = 0
            else:
                start = np.random.randint(0, T - effective_clip_length + 1)
            clip = video[start:start + effective_clip_length]
            clips.append(clip)

    elif strategy == ClipSamplingStrategy.UNIFORM:
        # Uniformly spaced clips
        if num_clips == 1:
            # Single clip from middle
            start = (T - effective_clip_length) // 2
            clip = video[start:start + effective_clip_length]
            clips.append(clip)
        else:
            # Multiple uniformly spaced clips
            step = (T - effective_clip_length) / (num_clips -
                                                  1) if num_clips > 1 else 0
            for i in range(num_clips):
                start = int(i * step)
                start = min(start, T - effective_clip_length)
                clip = video[start:start + effective_clip_length]
                clips.append(clip)

    elif strategy == ClipSamplingStrategy.SLIDING:
        # Sliding window with stride
        for start in range(0, T - effective_clip_length + 1, temporal_stride):
            clip = video[start:start + effective_clip_length]
            clips.append(clip)
            if len(clips) >= num_clips:
                break

    elif strategy == ClipSamplingStrategy.ALL:
        # All possible clips (overlapping)
        for start in range(T - effective_clip_length + 1):
            clip = video[start:start + effective_clip_length]
            clips.append(clip)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return clips


def load_video_clips_streaming(directory: str | Path,
                               num_frames: int = 16,
                               max_videos: int | None = None,
                               clip_strategy: str
                               | ClipSamplingStrategy = 'beginning',
                               frame_stride: int = 1,
                               num_clips_per_video: int = 1,
                               video_extensions: list[str] | None = None,
                               support_frame_dirs: bool = True,
                               target_size: tuple[int, int] | None = (224, 224),
                               verbose: bool = True) -> Iterator[torch.Tensor]:
    """
    This generator yields clips one-by-one instead of loading all videos into RAM.
    Perfect for large datasets where memory is limited.
    
    Args:
        directory: Path to directory with videos
        num_frames: Frames per clip
        max_videos: Max videos to load
        clip_strategy: 'beginning', 'random', 'uniform', etc.
        frame_stride: Frame skip (1=all, 2=every 2nd, 8=every 8th)
        num_clips_per_video: Number of clips per video
        video_extensions: Video file extensions
        support_frame_dirs: Also load frame directories
        target_size: Resize clips to (H, W). If None, keep original size.
        verbose: Show progress
    
    Yields:
        clip: [T, C, H, W] individual clips
    
    Example:
        >>> for clip in load_video_clips_streaming('data/videos/', num_frames=16):
        >>>     features = model.extract_features(clip.unsqueeze(0))
        >>>     # Process one clip at a time - low memory usage!
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find video paths
    video_paths: list[Path] = []

    # Find video files
    for ext in video_extensions:
        video_paths.extend(directory.glob(f"**/*{ext}"))

    # Find frame directories if enabled
    if support_frame_dirs:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                # Check if it contains frames
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                for ext in image_extensions:
                    if list(subdir.glob(f"*{ext}")):
                        video_paths.append(subdir)
                        break

    if len(video_paths) == 0:
        raise ValueError(f"No videos found in {directory}")

    video_paths = sorted(video_paths)

    if max_videos is not None:
        video_paths = video_paths[:max_videos]

    if verbose:
        print(f"Found {len(video_paths)} videos in {directory}")
        if num_clips_per_video > 1:
            print(f"Extracting {num_clips_per_video} clips per video...")
        if frame_stride > 1:
            print(f"Subsampling frames with stride {frame_stride}...")
        if target_size:
            print(f"Resizing clips to {target_size}...")

    # Track statistics
    failed_count = 0
    total_clips = 0

    iterator = tqdm(video_paths,
                    desc="Loading videos") if verbose else video_paths

    for video_path in iterator:
        try:
            # Load full video
            video = load_video_auto(video_path,
                                    num_frames=None,
                                    sample_strategy='uniform')

            # Sample clips from video
            clips = sample_clips_from_video(video,
                                            num_frames_per_clip=num_frames,
                                            num_clips=num_clips_per_video,
                                            strategy=clip_strategy,
                                            frame_stride=frame_stride)

            if target_size is not None:
                resized_clips = []
                for clip in clips:
                    T, C, H, W = clip.shape
                    if target_size != (H, W):
                        # Resize to target size
                        clip = clip.contiguous(
                        )  # Fix non-contiguous tensors first
                        clip_flat = clip.view(T * C, H,
                                              W).unsqueeze(0)  # [1, T*C, H, W]
                        clip_resized = torch.nn.functional.interpolate(
                            clip_flat,
                            size=target_size,
                            mode='bilinear',
                            align_corners=False)
                        clip = clip_resized.squeeze(0).view(
                            T, C, target_size[0],
                            target_size[1])  # Back to [T, C, H, W]
                    resized_clips.append(clip)
                clips = resized_clips

            # Yield clips one by one
            for clip in clips:
                yield clip
                total_clips += 1

            # Free memory
            del video, clips

        except Exception as e:
            failed_count += 1
            if verbose:
                print(f"\nWarning: Failed to load {video_path}: {e}")
            continue

    # Validate
    if total_clips == 0:
        raise RuntimeError(f"Failed to load any videos from {directory}")

    failure_rate = failed_count / len(video_paths)
    if failure_rate > 0.1:  # More than 10% failed
        print(
            f"\nWARNING: {failure_rate:.1%} of videos failed to load ({failed_count}/{len(video_paths)})"
        )

    if verbose:
        print(
            f"\nSuccessfully loaded {total_clips} clips from {len(video_paths) - failed_count} videos"
        )
