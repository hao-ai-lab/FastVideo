from einops import rearrange
import numpy as np
import torch
from torchvision import transforms

from fastvideo.v1.dataset.transform import CenterCropResizeVideo, TemporalRandomCrop
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch, PreprocessBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

class VideoTransformStage(PipelineStage):
    """
    Crop a video in temporal dimension.
    """
    def __init__(self, train_fps: int, num_frames: int, max_height: int, max_width: int, do_temporal_sample: bool) -> None:
        self.train_fps = train_fps
        self.num_frames = num_frames
        if do_temporal_sample:
            self.temporal_sample_fn = TemporalRandomCrop(num_frames)
        else:
            self.temporal_sample_fn = None
        
        self.video_transform = transforms.Compose([
            CenterCropResizeVideo((max_height, max_width)),
        ])

    def forward(self, batch: PreprocessBatch) -> PreprocessBatch:
        if batch.data_type != "video":
            return batch
        
        if len(batch.video_loader) == 0:
            raise ValueError("Video loader is not set")
        
        frame_interval = batch.fps / self.train_fps
        start_frame_idx = 0
        frame_indices = np.arange(start_frame_idx, batch.num_frames,
                                  frame_interval).astype(int)
        if len(frame_indices) > self.num_frames:
            if self.temporal_sample_fn is not None:
                begin_index, end_index = self.temporal_sample_fn(len(frame_indices))
                frame_indices = frame_indices[begin_index:end_index]
            else:
                frame_indices = frame_indices[:self.num_frames]

        video_pixel_batch = []
        for video_loader in batch.video_loader:
            video = video_loader.get_frames_at(frame_indices).data
            video = self.video_transform(video)
            video_pixel_batch.append(video)

        video_pixel_values = torch.stack(video_pixel_batch)
        video_pixel_values = rearrange(video_pixel_values, "b t c h w -> b c t h w")
        video_pixel_values = video_pixel_values.to(torch.uint8)
        video_pixel_values = video_pixel_values.float() / 255.0
        batch.latents = video_pixel_values
        return batch


