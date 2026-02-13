# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Stable Audio text-to-audio generation."""
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class StableAudioSamplingParam(SamplingParam):
    """Default sampling parameters for Stable Audio Open text-to-audio.

    Matches stable-audio-tools defaults: 44.1kHz, ~47.5s max duration,
    250 steps, cfg_scale=6.
    """

    data_type: str = "audio"

    # Audio-specific
    sample_rate: int = 44100
    duration_seconds: float = 10.0  # Output duration in seconds
    seconds_start: float = 0.0  # Conditioning: start offset (seconds)
    seconds_total: float = 10.0  # Conditioning: total duration (seconds)

    # Override video defaults for audio
    num_frames: int = 1
    height: int = 1
    width: int = 1
    output_video_name: str | None = None  # For audio, use output_audio_name
    negative_prompt: str = ""

    # Denoising (stable-audio-tools demo: 250 steps, cfg 3/6/9)
    num_inference_steps: int = 250
    guidance_scale: float = 6.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.data_type = "audio"

    @property
    def sample_size(self) -> int:
        """Audio samples (time dimension). For stereo, shape is (B, 2, sample_size)."""
        return int(self.duration_seconds * self.sample_rate)
