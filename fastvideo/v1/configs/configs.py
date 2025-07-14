import dataclasses
from typing import Any
import argparse

from fastvideo.v1.utils import FlexibleArgumentParser, StoreBoolean
from fastvideo.v1.configs.utils import update_config_from_args

@dataclasses.dataclass
class PreprocessConfig:
    """Configuration for preprocessing operations."""
    
    # Model and dataset configuration
    model_path: str = "data/mochi"
    model_type: str = "mochi"
    training_dataset_path: str = ""
    validation_dataset_path: str = ""

    num_frames: int = 163
    
    # Dataloader configuration
    dataloader_num_workers: int = 1
    preprocess_video_batch_size: int = 2
    samples_per_file: int = 64
    flush_frequency: int = 256
    
    # Video processing parameters
    num_latent_t: int = 28
    max_height: int = 480
    max_width: int = 848
    video_length_tolerance_range: float = 2.0
    group_frame: bool = False
    group_resolution: bool = False
    preprocess_task: str = "t2v"
    train_fps: int = 30
    use_image_num: int = 0
    text_max_length: int = 256
    speed_factor: float = 1.0
    drop_short_ratio: float = 1.0
    do_temporal_sample: bool = False
    
    # Text encoder and model configuration
    text_encoder_name: str = "google/t5-v1_1-xxl"
    cache_dir: str = "./cache_dir"
    training_cfg_rate: float = 0.0
    dataset_output_dir: str = "./output"

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser, prefix: str = "preprocess") -> FlexibleArgumentParser:
        """Add preprocessing configuration arguments to the parser."""
        prefix_with_dot = f"{prefix}." if (prefix.strip() != "") else ""
        
        # Dataset & dataloader
        parser.add_argument(f"--{prefix_with_dot}model-path", type=str, default=PreprocessConfig.model_path,
                          help="Path to the model for preprocessing")
        parser.add_argument(f"--{prefix_with_dot}model-type", type=str, default=PreprocessConfig.model_type,
                          help="Type of the model for preprocessing")
        parser.add_argument(f"--{prefix_with_dot}training-dataset-path", type=str, required=True,
                          help="Path to the training dataset directory for preprocessing")
        parser.add_argument(f"--{prefix_with_dot}validation-dataset-path", type=str,
                          help="Path to the validation dataset file/directory for preprocessing")
        parser.add_argument(f"--{prefix_with_dot}num-frames", type=int, default=PreprocessConfig.num_frames,
                          help="Number of frames to process")
        parser.add_argument(f"--{prefix_with_dot}dataloader-num-workers", type=int, 
                          default=PreprocessConfig.dataloader_num_workers,
                          help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
        parser.add_argument(f"--{prefix_with_dot}preprocess-video-batch-size", type=int, 
                          default=PreprocessConfig.preprocess_video_batch_size,
                          help="Batch size (per device) for the training dataloader.")
        parser.add_argument(f"--{prefix_with_dot}samples-per-file", type=int, default=PreprocessConfig.samples_per_file,
                          help="Number of samples per output file")
        parser.add_argument(f"--{prefix_with_dot}flush-frequency", type=int, default=PreprocessConfig.flush_frequency,
                          help="How often to save to parquet files")
        
        # Video processing parameters
        parser.add_argument(f"--{prefix_with_dot}num-latent-t", type=int, default=PreprocessConfig.num_latent_t,
                          help="Number of latent timesteps.")
        parser.add_argument(f"--{prefix_with_dot}max-height", type=int, default=PreprocessConfig.max_height,
                          help="Maximum height for video processing")
        parser.add_argument(f"--{prefix_with_dot}max-width", type=int, default=PreprocessConfig.max_width,
                          help="Maximum width for video processing")
        parser.add_argument(f"--{prefix_with_dot}video-length-tolerance-range", type=float, 
                          default=PreprocessConfig.video_length_tolerance_range,
                          help="Video length tolerance range")
        parser.add_argument(f"--{prefix_with_dot}group-frame", action=StoreBoolean, default=PreprocessConfig.group_frame,
                          help="Whether to group frames during processing")
        parser.add_argument(f"--{prefix_with_dot}group-resolution", action=StoreBoolean, default=PreprocessConfig.group_resolution,
                          help="Whether to group resolutions during processing")
        parser.add_argument(f"--{prefix_with_dot}train-fps", type=int, default=PreprocessConfig.train_fps,
                          help="Training FPS")
        parser.add_argument(f"--{prefix_with_dot}use-image-num", type=int, default=PreprocessConfig.use_image_num,
                          help="Number of images to use")
        parser.add_argument(f"--{prefix_with_dot}text-max-length", type=int, default=PreprocessConfig.text_max_length,
                          help="Maximum length for text processing")
        parser.add_argument(f"--{prefix_with_dot}speed-factor", type=float, default=PreprocessConfig.speed_factor,
                          help="Speed factor for video processing")
        parser.add_argument(f"--{prefix_with_dot}drop-short-ratio", type=float, default=PreprocessConfig.drop_short_ratio,
                          help="Ratio for dropping short videos")
        parser.add_argument(f"--{prefix_with_dot}do-temporal-sample", action=StoreBoolean, default=PreprocessConfig.do_temporal_sample,
                          help="Whether to do temporal sampling")
        
        # Text encoder & model configuration
        parser.add_argument(f"--{prefix_with_dot}text-encoder-name", type=str, default=PreprocessConfig.text_encoder_name,
                          help="Name of the text encoder")
        parser.add_argument(f"--{prefix_with_dot}cache-dir", type=str, default=PreprocessConfig.cache_dir,
                          help="Directory for caching")
        parser.add_argument(f"--{prefix_with_dot}training-cfg-rate", type=float, default=PreprocessConfig.training_cfg_rate,
                          help="Training CFG rate")
        parser.add_argument(f"--{prefix_with_dot}dataset-output-dir", type=str, default=PreprocessConfig.dataset_output_dir,
                          help="The output directory where the dataset will be written.")
        
        return parser

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "PreprocessConfig":
        """Create PreprocessConfig from keyword arguments."""
        preprocess_config = cls()
        if not preprocess_config.update_config_from_dict(kwargs, prefix="preprocess"):
            return None
        return preprocess_config

    def update_config_from_dict(self, args: dict[str, Any], prefix: str = "preprocess") -> bool:
        """Update configuration from a dictionary."""
        prefix_with_dot = f"{prefix}." if (prefix.strip() != "") else ""
        return update_config_from_args(self, args, prefix_with_dot, pop_args=True)

