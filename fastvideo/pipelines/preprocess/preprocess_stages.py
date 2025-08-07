from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from einops import rearrange
from torchvision import transforms

from fastvideo.dataset.transform import (CenterCropResizeVideo,
                                         TemporalRandomCrop)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import (ForwardBatch,
                                                     PreprocessBatch)
from fastvideo.pipelines.stages.base import PipelineStage


class VideoTransformStage(PipelineStage):
    """
    Crop a video in temporal dimension.
    """

    def __init__(self, train_fps: int, num_frames: int, max_height: int,
                 max_width: int, do_temporal_sample: bool) -> None:
        self.train_fps = train_fps
        self.num_frames = num_frames
        if do_temporal_sample:
            self.temporal_sample_fn: Callable | None = TemporalRandomCrop(
                num_frames)
        else:
            self.temporal_sample_fn = None

        self.video_transform = transforms.Compose([
            CenterCropResizeVideo((max_height, max_width)),
        ])

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        batch = cast(PreprocessBatch, batch)
        assert isinstance(batch.fps, list)
        assert isinstance(batch.num_frames, list)

        if batch.data_type != "video":
            return batch

        if len(batch.video_loader) == 0:
            raise ValueError("Video loader is not set")

        video_pixel_batch = []

        for i in range(len(batch.video_loader)):
            frame_interval = batch.fps[i] / self.train_fps
            start_frame_idx = 0
            frame_indices = np.arange(start_frame_idx, batch.num_frames[i],
                                      frame_interval).astype(int)
            if len(frame_indices) > self.num_frames:
                if self.temporal_sample_fn is not None:
                    begin_index, end_index = self.temporal_sample_fn(
                        len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                else:
                    frame_indices = frame_indices[:self.num_frames]

            video = batch.video_loader[i].get_frames_at(frame_indices).data
            video = self.video_transform(video)
            video_pixel_batch.append(video)

        video_pixel_values = torch.stack(video_pixel_batch)
        video_pixel_values = rearrange(video_pixel_values,
                                       "b t c h w -> b c t h w")
        video_pixel_values = video_pixel_values.to(torch.uint8)
        video_pixel_values = video_pixel_values.float() / 255.0
        batch.latents = video_pixel_values
        return cast(ForwardBatch, batch)
