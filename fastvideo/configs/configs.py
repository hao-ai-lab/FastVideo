import dataclasses
from enum import Enum

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class DatasetType(str, Enum):
    """
    Enumeration for different dataset types.
    """
    HF = "hf"
    MERGED = "merged"

    @classmethod
    def from_string(cls, value: str) -> "DatasetType":
        """Convert string to DatasetType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid dataset type: {value}. Must be one of: {', '.join([m.value for m in cls])}") from None


class VideoLoaderType(str, Enum):
    """
    Enumeration for different video loaders.
    """
    TORCHCODEC = "torchcodec"
    TORCHVISION = "torchvision"

    @classmethod
    def from_string(cls, value: str) -> "VideoLoaderType":
        """Convert string to VideoLoader enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid video loader: {value}. Must be one of: {', '.join([m.value for m in cls])}") from None


@dataclasses.dataclass
class PreprocessConfig:
    """Configuration for preprocessing operations."""

    # Model and dataset configuration
    model_path: str = ""
    dataset_path: str = ""
    dataset_type: DatasetType = DatasetType.HF
    dataset_output_dir: str = "./output"

    # Dataloader configuration
    dataloader_num_workers: int = 1
    preprocess_video_batch_size: int = 2

    # Saver configuration
    samples_per_file: int = 64
    flush_frequency: int = 256

    # Video processing parameters
    video_loader_type: VideoLoaderType = VideoLoaderType.TORCHCODEC
    max_height: int = 480
    max_width: int = 848
    num_frames: int = 163
    video_length_tolerance_range: float = 2.0
    train_fps: int = 30
    speed_factor: float = 1.0
    drop_short_ratio: float = 1.0
    do_temporal_sample: bool = False

    # Model configuration
    training_cfg_rate: float = 0.0
    with_audio: bool = False

    # framework configuration
    seed: int = 42

    # Settings of the per-task preprocessing pipelines (fastvideo/pipelines/preprocess/v1_preprocess.py). They read
    # a merged caption file instead of dataset_path and write to dataset_output_dir.
    data_merge_path: str = ""
    preprocess_task: str = "t2v"
    num_latent_t: int = 28
    text_max_length: int = 256
    cache_dir: str = "./cache_dir"
