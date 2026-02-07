# SPDX-License-Identifier: Apache-2.0
"""Utilities for upscaling videos with LTX-2 spatial upsampler."""

from __future__ import annotations

from pathlib import Path

import av
import imageio
import numpy as np
import torch

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import UpsamplerLoader, VAELoader
from fastvideo.models.upsamplers import upsample_video
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _read_video(path: str | Path,
                max_frames: int | None = None) -> tuple[torch.Tensor, float]:
    """Read video frames via PyAV.

    Returns a tensor of shape [F, C, H, W] in [0, 1] and the fps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)
        for frame in container.decode(video=0):
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise ValueError(f"No frames decoded from {path}")

    frames_np = np.stack(frames, axis=0)
    video = torch.from_numpy(frames_np).float().div(255.0)
    return video.permute(0, 3, 1, 2), fps


def _write_video(frames: torch.Tensor, output_path: str | Path,
                 fps: float) -> None:
    """Write frames [F, C, H, W] in [0, 1] to a video file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = frames.clamp(0, 1)
    frames = (frames * 255.0).to(torch.uint8)
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
    imageio.mimsave(str(output_path), list(frames_np), fps=fps, format="mp4")


def _prepare_video(
    video: torch.Tensor,
    *,
    trim_frames: bool,
    pad_frames: bool,
    crop_multiple: int,
) -> torch.Tensor:
    """Ensure frames count and resolution satisfy LTX-2 VAE constraints."""
    frames, _, height, width = video.shape

    if trim_frames and pad_frames:
        raise ValueError(
            "Only one of trim_frames or pad_frames can be enabled.")

    if trim_frames and ((frames - 1) % 8) != 0:
        valid_frames = 1 + 8 * ((frames - 1) // 8)
        if valid_frames < 1:
            raise ValueError("Video must have at least 1 frame.")
        if valid_frames != frames:
            logger.warning(
                "Trimming frames from %d to %d to satisfy 1+8k requirement.",
                frames,
                valid_frames,
            )
            video = video[:valid_frames]
            frames = valid_frames
    elif pad_frames and ((frames - 1) % 8) != 0:
        valid_frames = 1 + 8 * (((frames - 1) + 7) // 8)
        pad_count = valid_frames - frames
        if pad_count > 0:
            logger.warning(
                "Padding frames from %d to %d to satisfy 1+8k requirement.",
                frames,
                valid_frames,
            )
            pad = video[-1:].repeat(pad_count, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
            frames = valid_frames

    if crop_multiple > 0:
        new_height = height - (height % crop_multiple)
        new_width = width - (width % crop_multiple)
        if new_height != height or new_width != width:
            top = max((height - new_height) // 2, 0)
            left = max((width - new_width) // 2, 0)
            logger.warning(
                "Center-cropping from %dx%d to %dx%d to be divisible by %d.",
                height,
                width,
                new_height,
                new_width,
                crop_multiple,
            )
            video = video[:, :, top:top + new_height, left:left + new_width]

    return video


def upscale_video_file(
    *,
    input_video: str | Path,
    output_video: str | Path,
    vae_path: str | Path,
    upsampler_path: str | Path,
    precision: str = "bf16",
    device: str | None = None,
    max_frames: int | None = None,
    trim_frames: bool = True,
    pad_frames: bool = False,
    crop_multiple: int = 32,
    output_fps: float | None = None,
) -> None:
    """Upscale an existing video using the LTX-2 spatial upsampler."""
    input_video = str(input_video)
    output_video = str(output_video)
    vae_path = str(vae_path)
    upsampler_path = str(upsampler_path)

    video, fps = _read_video(input_video, max_frames=max_frames)
    original_frames = video.shape[0]
    video = _prepare_video(
        video,
        trim_frames=trim_frames,
        pad_frames=pad_frames,
        crop_multiple=crop_multiple,
    )
    final_frames = video.shape[0]

    target_device = torch.device(device) if device else (torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu"))

    precision = precision.lower()
    dtype = PRECISION_TO_TYPE.get(precision, torch.bfloat16)
    if target_device.type == "cpu" and dtype != torch.float32:
        logger.warning("CPU device selected; overriding precision to fp32.")
        dtype = torch.float32
        precision = "fp32"

    args = FastVideoArgs(
        model_path=vae_path,
        pipeline_config=PipelineConfig(vae_precision=precision),
        vae_cpu_offload=False,
    )

    vae_loader = VAELoader()
    upsampler_loader = UpsamplerLoader()

    vae = vae_loader.load(vae_path, args).to(device=target_device, dtype=dtype)
    upsampler = upsampler_loader.load(upsampler_path,
                                      args).to(device=target_device,
                                               dtype=dtype)

    if hasattr(vae.decoder, "decode_noise_scale"):
        vae.decoder.decode_noise_scale = 0.0

    # [F, C, H, W] -> [B, C, F, H, W]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=target_device,
                                                         dtype=dtype)

    with torch.no_grad():
        latents = vae.encoder(video)
        up_latents = upsample_video(latents, vae.encoder,
                                    getattr(upsampler, "model", upsampler))

        timestep_value = getattr(vae.decoder, "decode_timestep", 0.05)
        timestep = torch.full((video.shape[0], ),
                              float(timestep_value),
                              device=target_device,
                              dtype=dtype)
        decoded = vae.decoder(up_latents, timestep=timestep)

    # [B, C, F, H, W] -> [F, C, H, W]
    decoded = decoded[0].permute(1, 0, 2, 3).detach().cpu()
    if pad_frames and final_frames != original_frames:
        decoded = decoded[:original_frames]
    final_fps = output_fps or fps
    _write_video(decoded, output_video, final_fps)
    logger.info("Upscaled video saved to %s", output_video)


__all__ = ["upscale_video_file"]
