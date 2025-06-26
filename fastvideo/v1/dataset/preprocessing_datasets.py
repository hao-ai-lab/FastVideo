# SPDX-License-Identifier: Apache-2.0
import json
import math
import os
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from os.path import join as opj
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from transformers import AutoTokenizer

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PreprocessBatch:
    """
    Batch information for dataset processing stages.
    
    This class holds all the information about a video-caption or image-caption pair
    as it moves through the processing pipeline. Fields are populated by different stages.
    """
    # Raw metadata
    path: str
    cap: Union[str, List[str]]
    resolution: Optional[Dict] = None
    fps: Optional[float] = None
    duration: Optional[float] = None

    # Processed metadata
    num_frames: Optional[int] = None
    sample_frame_index: Optional[List[int]] = None
    sample_num_frames: Optional[int] = None

    # Processed data
    pixel_values: Optional[torch.Tensor] = None
    text: Optional[str] = None
    input_ids: Optional[torch.Tensor] = None
    cond_mask: Optional[torch.Tensor] = None

    @property
    def is_video(self) -> bool:
        """Check if this is a video item."""
        return self.path.endswith(".mp4")

    @property
    def is_image(self) -> bool:
        """Check if this is an image item."""
        return self.path.endswith(".jpg")


class DatasetStage(ABC):
    """
    Abstract base class for dataset processing stages.
    
    Similar to PipelineStage but designed for dataset preprocessing operations.
    """

    @abstractmethod
    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """
        Process the dataset batch.
        
        Args:
            batch: Dataset batch to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed batch
        """
        raise NotImplementedError


class DatasetFilterStage(ABC):
    """
    Abstract base class for dataset filtering stages.
    
    These stages can filter out items during metadata processing.
    """

    @abstractmethod
    def should_keep(self, batch: PreprocessBatch, **kwargs) -> bool:
        """
        Check if batch should be kept.
        
        Args:
            batch: Dataset batch to check
            **kwargs: Additional parameters
            
        Returns:
            True if batch should be kept, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """
        Process the dataset batch (for non-filtering operations).
        
        Args:
            batch: Dataset batch to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed batch
        """
        raise NotImplementedError


class DataValidationStage(DatasetFilterStage):
    """Stage for validating data items."""

    def should_keep(self, batch: PreprocessBatch, **kwargs) -> bool:
        """
        Validate data item.
        
        Args:
            batch: Dataset batch to validate
            
        Returns:
            True if valid, False if invalid
        """
        # Check for caption
        if batch.cap is None:
            return False

        if batch.is_video:
            # Validate video-specific fields
            if batch.duration is None or batch.fps is None:
                return False
        elif not batch.is_image:
            return False

        return True

    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """Process does nothing for validation - filtering is handled by should_keep."""
        return batch


class ResolutionFilterStage(DatasetFilterStage):
    """Stage for filtering data items based on resolution constraints."""

    def __init__(self,
                 max_h_div_w_ratio: float = 17 / 16,
                 min_h_div_w_ratio: float = 8 / 16,
                 max_height: int = 1024,
                 max_width: int = 1024):
        self.max_h_div_w_ratio = max_h_div_w_ratio
        self.min_h_div_w_ratio = min_h_div_w_ratio
        self.max_height = max_height
        self.max_width = max_width

    def should_keep(self, batch: PreprocessBatch, **kwargs) -> bool:
        """
        Check if data item passes resolution filtering.
        
        Args:
            batch: Dataset batch with resolution information
            
        Returns:
            True if passes filter, False otherwise
        """
        # Only apply to videos
        if not batch.is_video:
            return True

        if batch.resolution is None:
            return False

        height = batch.resolution.get("height", None)
        width = batch.resolution.get("width", None)
        if height is None or width is None:
            return False

        # Check aspect ratio
        aspect = self.max_height / self.max_width
        hw_aspect_thr = 1.5

        return self.filter_resolution(
            height,
            width,
            max_h_div_w_ratio=hw_aspect_thr * aspect,
            min_h_div_w_ratio=1 / hw_aspect_thr * aspect,
        )

    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """Process does nothing for resolution filtering - filtering is handled by should_keep."""
        return batch

    def filter_resolution(self, h: int, w: int, max_h_div_w_ratio: float,
                          min_h_div_w_ratio: float) -> bool:
        """Filter based on height/width ratio."""
        return h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio


class FrameSamplingStage(DatasetFilterStage):
    """Stage for temporal frame sampling and indexing."""

    def __init__(self,
                 num_frames: int,
                 train_fps: int,
                 speed_factor: int = 1,
                 video_length_tolerance_range: float = 5.0,
                 drop_short_ratio: float = 0.0):
        self.num_frames = num_frames
        self.train_fps = train_fps
        self.speed_factor = speed_factor
        self.video_length_tolerance_range = video_length_tolerance_range
        self.drop_short_ratio = drop_short_ratio

    def should_keep(self, batch: PreprocessBatch, **kwargs) -> bool:
        """
        Check if video should be kept based on length constraints.
        
        Args:
            batch: Dataset batch
            
        Returns:
            True if should be kept, False otherwise
        """
        if batch.is_image:
            return True

        if batch.duration is None or batch.fps is None:
            return False

        num_frames = math.ceil(batch.fps * batch.duration)

        # Check if video is too long
        if (num_frames / batch.fps > self.video_length_tolerance_range *
            (self.num_frames / self.train_fps * self.speed_factor)):
            return False

        # Resample frame indices to check length
        frame_interval = batch.fps / self.train_fps
        start_frame_idx = 0
        frame_indices = np.arange(start_frame_idx, num_frames,
                                  frame_interval).astype(int)

        # Filter short videos
        return not (len(frame_indices) < self.num_frames
                    and random.random() < self.drop_short_ratio)

    def process(self,
                batch: PreprocessBatch,
                temporal_sample_fn=None,
                **kwargs) -> PreprocessBatch:
        """
        Process frame sampling for video data items.
        
        Args:
            batch: Dataset batch
            temporal_sample_fn: Function for temporal sampling
            
        Returns:
            Updated batch with frame sampling info
        """
        if batch.is_image:
            # For images, just add sample info
            batch.sample_frame_index = [0]
            batch.sample_num_frames = 1
            return batch

        assert batch.duration is not None and batch.fps is not None
        batch.num_frames = math.ceil(batch.fps * batch.duration)

        # Resample frame indices
        frame_interval = batch.fps / self.train_fps
        start_frame_idx = 0
        frame_indices = np.arange(start_frame_idx, batch.num_frames,
                                  frame_interval).astype(int)

        # Temporal crop if too long
        if len(frame_indices
               ) > self.num_frames and temporal_sample_fn is not None:
            begin_index, end_index = temporal_sample_fn(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        batch.sample_frame_index = frame_indices.tolist()
        batch.sample_num_frames = len(frame_indices)

        return batch


class VideoTransformStage(DatasetStage):
    """Stage for video data transformation."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """
        Transform video data.
        
        Args:
            batch: Dataset batch with video information
            
        Returns:
            Batch with transformed video tensor
        """
        if not batch.is_video:
            return batch

        assert os.path.exists(batch.path), f"file {batch.path} do not exist!"
        assert batch.sample_frame_index is not None, "Frame indices must be set before transformation"

        torchvision_video, _, metadata = torchvision.io.read_video(
            batch.path, output_format="TCHW")
        video = torchvision_video[batch.sample_frame_index]
        if self.transform is not None:
            video = self.transform(video)
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)

        h, w = video.shape[-2:]
        assert (
            h / w <= 17 / 16 and h / w >= 8 / 16
        ), f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({batch.path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}"

        video = video.float() / 127.5 - 1.0
        batch.pixel_values = video
        return batch


class ImageTransformStage(DatasetStage):
    """Stage for image data transformation."""

    def __init__(self, transform, transform_topcrop) -> None:
        self.transform = transform
        self.transform_topcrop = transform_topcrop

    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """
        Transform image data.
        
        Args:
            batch: Dataset batch with image information
            
        Returns:
            Batch with transformed image tensor
        """
        if not batch.is_image:
            return batch

        image = Image.open(batch.path).convert("RGB")
        image = torch.from_numpy(np.array(image))
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)

        if self.transform_topcrop is not None:
            image = self.transform_topcrop(image)
        elif self.transform is not None:
            image = self.transform(image)

        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]
        image = image.float() / 127.5 - 1.0
        batch.pixel_values = image
        return batch


class TextEncodingStage(DatasetStage):
    """Stage for text tokenization and encoding."""

    def __init__(self, tokenizer, text_max_length: int, cfg_rate: float = 0.0):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.cfg_rate = cfg_rate

    def process(self, batch: PreprocessBatch, **kwargs) -> PreprocessBatch:
        """
        Process text data.
        
        Args:
            batch: Dataset batch with caption information
            
        Returns:
            Batch with encoded text information
        """
        text = batch.cap
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text[0] if random.random() > self.cfg_rate else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        batch.text = text
        batch.input_ids = text_tokens_and_mask["input_ids"]
        batch.cond_mask = text_tokens_and_mask["attention_mask"]
        return batch


class VideoCaptionMergedDataset(torch.utils.data.IterableDataset,
                                torch.distributed.checkpoint.stateful.Stateful):
    """
    Merged dataset for video and caption data with stage-based processing.
    Assumes that data_merge_path is a txt file with the following format:
    <folder_path>,<json_file_path>

        The folder should contain videos.

        The json file should be a list of dictionaries with the following format:
        [
        {
            "path": "1gGQy4nxyUo-Scene-016.mp4",
            "resolution": {
            "width": 1920,
            "height": 1080
            },
            "size": 2439112,
            "fps": 25.0,
            "duration": 6.88,
            "num_frames": 172,
            "cap": [
            "A watermelon wearing a helmet is crushed by a hydraulic press, causing it to flatten and burst open."
            ]
        },
        ...
        ]
    
    This dataset processes video and image data through a series of stages:
    - Data validation
    - Resolution filtering  
    - Frame sampling
    - Transformation
    - Text encoding
    """

    def __init__(self,
                 data_merge_path: str,
                 args,
                 transform,
                 temporal_sample,
                 transform_topcrop,
                 start_idx: int = 0):
        self.data_merge_path = data_merge_path
        self.start_idx = start_idx
        self.args = args
        self.temporal_sample = temporal_sample

        # Initialize tokenizer
        tokenizer_path = os.path.join(args.model_path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                  cache_dir=args.cache_dir)

        # Initialize processing stages
        self._init_stages(args, transform, transform_topcrop, tokenizer)

        # Process metadata
        self.processed_batches = self._process_metadata()

    def _init_stages(self, args, transform, transform_topcrop,
                     tokenizer) -> None:
        """Initialize all processing stages."""
        self.validation_stage = DataValidationStage()
        self.resolution_filter_stage = ResolutionFilterStage(
            max_height=args.max_height, max_width=args.max_width)
        self.frame_sampling_stage = FrameSamplingStage(
            num_frames=args.num_frames,
            train_fps=args.train_fps,
            speed_factor=args.speed_factor,
            video_length_tolerance_range=args.video_length_tolerance_range,
            drop_short_ratio=args.drop_short_ratio)
        self.video_transform_stage = VideoTransformStage(transform)
        self.image_transform_stage = ImageTransformStage(
            transform, transform_topcrop)
        self.text_encoding_stage = TextEncodingStage(
            tokenizer=tokenizer,
            text_max_length=args.text_max_length,
            cfg_rate=args.training_cfg_rate)

    def _load_raw_data(self) -> List[Dict]:
        """Load raw data from JSON files."""
        # Read folder-annotation pairs
        with open(self.data_merge_path) as f:
            folder_anno_pairs = [
                line.strip().split(",") for line in f if line.strip()
            ]
        assert len(
            folder_anno_pairs) == 1, "Only support one folder-annotation pair"
        assert len(folder_anno_pairs[0]
                   ) == 2, "Folder-annotation pair should have two elements"
        folder, annotation_file = folder_anno_pairs[0]

        data_items: List[Dict] = []
        with open(annotation_file) as f:
            data_items = json.load(f)

        # Update paths with folder prefix
        for item in data_items:
            item["path"] = opj(folder, item["path"])

        return data_items

    def _process_metadata(self) -> List[PreprocessBatch]:
        """Process the raw metadata through all filtering stages."""
        raw_data = self._load_raw_data()
        processed_batches = []

        # Initialize counters
        filter_counts = {
            "validation_failed": 0,
            "resolution_failed": 0,
            "frame_sampling_failed": 0
        }
        sample_num_frames: List[int] = []

        for item in raw_data:
            batch = PreprocessBatch(path=item["path"],
                                    cap=item["cap"],
                                    resolution=item.get("resolution"),
                                    fps=item.get("fps"),
                                    duration=item.get("duration"))

            # Apply filtering stages
            if not self._apply_filter_stages(batch, filter_counts):
                continue

            # Apply frame sampling processing
            batch = self.frame_sampling_stage.process(
                batch, temporal_sample_fn=self.temporal_sample)

            processed_batches.append(batch)
            assert batch.sample_num_frames is not None
            sample_num_frames.append(batch.sample_num_frames)

        self._log_filtering_stats(filter_counts, sample_num_frames,
                                  len(raw_data), len(processed_batches))
        return processed_batches

    def _apply_filter_stages(self, batch: PreprocessBatch,
                             filter_counts: Dict[str, int]) -> bool:
        """Apply all filter stages and update counters. Returns True if batch should be kept."""
        if not self.validation_stage.should_keep(batch):
            filter_counts["validation_failed"] += 1
            return False

        if not self.resolution_filter_stage.should_keep(batch):
            filter_counts["resolution_failed"] += 1
            return False

        if not self.frame_sampling_stage.should_keep(batch):
            filter_counts["frame_sampling_failed"] += 1
            return False

        return True

    def _log_filtering_stats(self, filter_counts: Dict[str, int],
                             sample_num_frames: List[int], before_count: int,
                             after_count: int):
        """Log filtering statistics."""
        logger.info(
            "validation_failed: %d, resolution_failed: %d, frame_sampling_failed: %d, "
            "Counter(sample_num_frames): %s, before filter: %d, after filter: %d",
            filter_counts['validation_failed'],
            filter_counts['resolution_failed'],
            filter_counts['frame_sampling_failed'], Counter(sample_num_frames),
            before_count, after_count)

    def __iter__(self):
        """Iterate through processed data items."""
        for idx in range(len(self.processed_batches)):
            yield self._get_item(idx)

    def __len__(self):
        return len(self.processed_batches)

    def _get_item(self, idx: int) -> Dict:
        """Get a single processed data item."""
        batch = self.processed_batches[idx]

        # Apply transformation stages
        batch = self.video_transform_stage.process(batch)
        batch = self.image_transform_stage.process(batch)
        batch = self.text_encoding_stage.process(batch)

        # Build result dictionary
        result = {
            "pixel_values": batch.pixel_values,
            "text": batch.text,
            "input_ids": batch.input_ids,
            "cond_mask": batch.cond_mask,
            "path": batch.path,
        }

        # Add video-specific fields
        if batch.is_video:
            result.update({"fps": batch.fps, "duration": batch.duration})

        return result

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {"processed_batches": self.processed_batches}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.processed_batches = state_dict["processed_batches"]
