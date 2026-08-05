# SPDX-License-Identifier: Apache-2.0
"""Audio conditioning stage for speech-driven video (Wan2.2-S2V).

Text has a tokenizer because language is discrete -- a finite vocabulary you can
look words up in. Audio has none: a waveform is a continuous stream, so the
"tokens" are manufactured by slicing time and learning features, not by lookup.
This stage does that: load -> resample to wav2vec2's 16kHz -> encode -> resample
the feature stream to the video frame rate -> bucket into a per-frame window.

The resampling is the load-bearing part. If the audio feature stream and the
video latents disagree on frame count, nothing crashes -- the lips just drift
out of sync. ``verify_output`` pins the frame count for that reason.
"""
import os
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)

WAV2VEC_SAMPLE_RATE = 16000
WAV2VEC_FEATURE_RATE = 50  # wav2vec2 emits one frame per 20ms
REFERENCE_VIDEO_RATE = 30  # the rate Wan-S2V's audio features are aligned to


def _resample_features(features: torch.Tensor,
                       input_rate: int,
                       output_rate: int,
                       output_len: int | None = None) -> torch.Tensor:
    """Linearly resample a [layers, T, C] feature stream between frame rates."""
    features = features.transpose(1, 2)
    if output_len is None:
        output_len = int(features.shape[2] / float(input_rate) * output_rate)
    return F.interpolate(features, size=output_len, align_corners=True, mode="linear").transpose(1, 2)


class AudioEncodingStage(PipelineStage):
    """Waveform -> per-frame audio embeddings for the DiT's audio injector.

    Writes ``batch.audio_embeds`` with shape ``[B, num_layers, C_a, num_frames]``.
    All wav2vec2 hidden states are kept, not just the last: the model learns its
    own weighting over encoder depth (``casual_audio_encoder.weights``).
    """

    def __init__(self, audio_encoder, audio_processor) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_processor = audio_processor

    def _load_waveform(self, audio_path: str) -> np.ndarray:
        try:
            import librosa
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Speech-to-video needs librosa to decode audio. Install it with "
                              "`uv pip install librosa` (or the `eval-audio` extra).") from exc

        # librosa reads local files only, but image_path already accepts URLs
        # (load_image in input_validation), so audio should behave the same way.
        if urlparse(audio_path).scheme in ("http", "https"):
            suffix = Path(urlparse(audio_path).path).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                with urllib.request.urlopen(audio_path) as response:
                    tmp.write(response.read())
                local_path = tmp.name
            try:
                waveform, _ = librosa.load(local_path, sr=WAV2VEC_SAMPLE_RATE)
            finally:
                os.unlink(local_path)
            return waveform

        waveform, _ = librosa.load(audio_path, sr=WAV2VEC_SAMPLE_RATE)
        return waveform

    def _bucket_to_frames(self, features: torch.Tensor, num_frames: int, fps: int, window: int = 0) -> torch.Tensor:
        """Pick the audio feature window belonging to each video frame.

        ``features`` is [num_layers, T, C] at REFERENCE_VIDEO_RATE. Frame i takes
        the features around its own timestamp, so frame i attends to the sound
        that happens at frame i -- the alignment the whole model depends on.
        """
        num_layers, total, dim = features.shape
        # Keep the ratio fractional. Video frame i happens at i/fps seconds, which
        # is feature index i * (30 / fps). Rounding that ratio to an integer (30/16
        # -> 1) makes a 5s video read only 2.6s of audio: the video finishes while
        # the speech is half-done, and nothing errors -- the lips just drift.
        step = REFERENCE_VIDEO_RATE / float(fps)
        out = []
        for i in range(num_frames):
            centre = min(int(round(i * step)), max(total - 1, 0))
            span = max(1, int(round(step)))
            idx = [min(max(centre + offset * span, 0), total - 1) for offset in range(-window, window + 1)]
            out.append(features[:, idx].flatten(start_dim=-2))
        return torch.stack(out, dim=0).permute(1, 2, 0)  # [num_layers, C, num_frames]

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = self.audio_encoder.device
        waveform = self._load_waveform(batch.audio_path)
        inputs = self.audio_processor(waveform, sampling_rate=WAV2VEC_SAMPLE_RATE, return_tensors="pt")

        with torch.no_grad():
            outputs = self.audio_encoder(inputs.input_values.to(device), output_hidden_states=True)
        features = torch.stack(outputs.hidden_states).squeeze(1)  # [num_layers, T, C]
        features = _resample_features(features, WAV2VEC_FEATURE_RATE, REFERENCE_VIDEO_RATE)

        assert batch.num_frames is not None and batch.fps is not None
        batch.audio_embeds = self._bucket_to_frames(features, batch.num_frames, batch.fps).unsqueeze(0)
        logger.info("Encoded %s -> audio embeds %s", batch.audio_path, tuple(batch.audio_embeds.shape))
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio_path", batch.audio_path, lambda v: isinstance(v, str) and bool(v))
        result.add_check("num_frames", batch.num_frames, lambda v: isinstance(v, int) and v > 0)
        result.add_check("fps", batch.fps, lambda v: isinstance(v, int) and v > 0)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # Frame count must match what was requested, or audio and video drift
        # apart with no error anywhere downstream.
        result.add_check("audio_embeds", batch.audio_embeds,
                         lambda v: v is not None and v.shape[-1] == batch.num_frames)
        return result
