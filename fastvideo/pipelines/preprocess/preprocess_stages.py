import random
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from torchvision import transforms

from fastvideo.configs.configs import VideoLoaderType
from fastvideo.dataset.transform import (CenterCropResizeVideo,
                                         TemporalRandomCrop, best_output_size)
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.logger import init_logger

logger = init_logger(__name__)

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
        assert fastvideo_args.preprocess_config is not None

        if batch.data_type != "video":
            return batch

        if len(batch.video_loader) == 0:
            raise ValueError("Video loader is not set")

        video_pixel_batch = []
        pil_image_batch = []

        enable_smart_resize = fastvideo_args.preprocess_config.enable_smart_resize
        smart_resize_max_area = fastvideo_args.preprocess_config.smart_resize_max_area
        if smart_resize_max_area is None:
            smart_resize_max_area = 480 * 832

        calculated_size = None

        for i in range(len(batch.video_loader)):
            # logger.info(f"Processing video {i+1}/{len(batch.video_loader)}")
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

            logger.info(
                f"Frame indices selected (count={len(frame_indices)}): [{frame_indices[0]}, ..., {frame_indices[-1]}]"
            )

            if fastvideo_args.preprocess_config.video_loader_type == VideoLoaderType.TORCHCODEC:
                try:
                    video = batch.video_loader[i].get_frames_at(
                        frame_indices).data
                except Exception as e:
                    # Try to get filename if available in PreprocessBatch
                    video_path = "unknown"
                    print(f"batch: {batch}")
                    if isinstance(batch, PreprocessBatch) and hasattr(
                            batch, 'video_file_name') and i < len(
                                batch.video_file_name):
                        video_path = batch.video_file_name[i]

                    logger.error(
                        f"Failed to load frames for video {video_path}: {e}")
                    logger.error(
                        f"Attempting to load frame indices: {frame_indices}")
                    raise e
            elif fastvideo_args.preprocess_config.video_loader_type == VideoLoaderType.TORCHVISION:
                video, _, _ = torchvision.io.read_video(batch.video_loader[i],
                                                        output_format="TCHW")
                video = video[frame_indices]
            else:
                raise ValueError(
                    f"Invalid video loader type: {fastvideo_args.preprocess_config.video_loader_type}"
                )

            logger.info(f"Video tensor shape after loading: {video.shape}")

            if enable_smart_resize:
                if calculated_size is None:
                    _, _, h_in, w_in = video.shape
                    # Get config values
                    patch_size = fastvideo_args.pipeline_config.dit_config.arch_config.patch_size
                    vae_stride = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
                    dh, dw = patch_size[1] * vae_stride, patch_size[
                        2] * vae_stride

                    ow, oh = best_output_size(w_in, h_in, dw, dh,
                                              smart_resize_max_area)
                    calculated_size = (oh, ow)
                    logger.info(
                        f"Smart resize: input=({h_in}, {w_in}), output=({oh}, {ow})"
                    )

                # Resize video frames using CenterCropResizeVideo (efficient)
                processed_video = CenterCropResizeVideo(calculated_size)(video)
                logger.info(
                    f"Processed video shape after resize: {processed_video.shape}"
                )
                video_pixel_batch.append(processed_video)

                # Process pil_image (condition) with high quality Lanczos if I2V
                if fastvideo_args.workload_type == WorkloadType.I2V:
                    # Extract first frame
                    img_tensor = video[0]  # C, H, W
                    img = TF.to_pil_image(img_tensor)
                    iw, ih = img.width, img.height
                    ow, oh = calculated_size[1], calculated_size[0]

                    # Smart Resize logic for PIL image
                    scale = max(ow / iw, oh / ih)
                    resampling = Image.Resampling.LANCZOS if hasattr(
                        Image, 'Resampling') else Image.LANCZOS
                    img = img.resize((round(iw * scale), round(ih * scale)),
                                     resampling)

                    # center-crop
                    x1 = (img.width - ow) // 2
                    y1 = (img.height - oh) // 2
                    img = img.crop((x1, y1, x1 + ow, y1 + oh))

                    # to tensor [0, 255] uint8
                    img_t = torch.from_numpy(np.array(img)).permute(
                        2, 0, 1).unsqueeze(0)
                    pil_image_batch.append(img_t)

            else:
                video = self.video_transform(video)
                video_pixel_batch.append(video)

        video_pixel_values = torch.stack(video_pixel_batch)
        logger.info(
            f"Final stacked video batch shape: {video_pixel_values.shape}")
        video_pixel_values = rearrange(video_pixel_values,
                                       "b t c h w -> b c t h w")
        video_pixel_values = video_pixel_values.to(torch.uint8)

        if fastvideo_args.workload_type == WorkloadType.I2V:
            if enable_smart_resize and len(pil_image_batch) > 0:
                batch.pil_image = torch.cat(
                    pil_image_batch, dim=0).to(self.device if hasattr(
                        self, 'device') else video_pixel_values.device)
            else:
                batch.pil_image = video_pixel_values[:, :, 0, :, :]

        video_pixel_values = video_pixel_values.float() / 255.0
        batch.latents = video_pixel_values
        batch.num_frames = [video_pixel_values.shape[2]] * len(
            batch.video_loader)
        batch.height = [video_pixel_values.shape[3]] * len(batch.video_loader)
        batch.width = [video_pixel_values.shape[4]] * len(batch.video_loader)
        return cast(ForwardBatch, batch)


class TextTransformStage(PipelineStage):
    """
    Process text data according to the cfg rate.
    """

    def __init__(self, cfg_uncondition_drop_rate: float, seed: int) -> None:
        self.cfg_rate = cfg_uncondition_drop_rate
        self.rng = random.Random(seed)

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        batch = cast(PreprocessBatch, batch)

        prompts = []
        for prompt in batch.prompt:
            if not isinstance(prompt, list):
                prompt = [prompt]
            prompt = self.rng.choice(prompt)
            prompt = prompt if self.rng.random() > self.cfg_rate else ""
            prompts.append(prompt)

        batch.prompt = prompts
        return cast(ForwardBatch, batch)
