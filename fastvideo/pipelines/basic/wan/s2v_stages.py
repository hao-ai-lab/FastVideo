# SPDX-License-Identifier: Apache-2.0
"""Wan-S2V-specific pipeline stages and the clip plan the pipeline loops over.

These replicate the steps of the official runner (``wan/speech2video.py``) that
the shared stages do differently:

* The reference image is VAE-encoded **alone** as one frame. The shared
  ``ImageVAEEncodingStage`` instead builds an I2V-style zero-padded video and
  encodes all of it, which is a different conditioning format entirely.
* Decoding prepends temporal context so the causal VAE does not start cold:
  the reference latent on the first clip, the previous clip's motion latents
  afterwards. Only the generated span is kept, and the first clip also drops
  ``WARMUP_FRAMES`` frames that decode with too little context.
* Long audio is covered by generating several clips back to back. Each clip
  re-encodes the last ``motion_frames`` pixels of the video so far as its
  motion history (``plan_clips`` decides how many clips a request needs).
"""
from dataclasses import dataclass

import PIL.Image
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.audio_encoding import EXTRA_AUDIO_FRAMES
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

# The official runner drops the first 3 decoded frames of the first clip: the
# causal VAE decodes them with only the reference latent as context.
WARMUP_FRAMES = 3

# ``ForwardBatch.extra`` keys the pipeline uses to hand the plan to its stages.
EXTRA_INFER_FRAMES = "s2v_infer_frames"  # frames the DiT generates in the current clip
EXTRA_MOTION_LATENTS = "motion_latents"  # previous clip's history; None on the first clip
# EXTRA_AUDIO_FRAMES (video frames' worth of audio to bucket) is owned by the
# audio stage and re-exported here for the pipeline.
__all__ = [
    "EXTRA_AUDIO_FRAMES", "EXTRA_INFER_FRAMES", "EXTRA_MOTION_LATENTS", "WARMUP_FRAMES", "S2VClipPlan",
    "S2VDecodingStage", "S2VRefImageEncodingStage", "plan_clips"
]


@dataclass(frozen=True)
class S2VClipPlan:
    """How one request splits into clips.

    ``num_frames`` is what the caller asked for and what they get back: the
    usual FastVideo ``4k+1`` count. ``infer_frames`` is what the transformer
    generates per clip (``4n``, the official runner's ``infer_frames``). The
    first clip shows ``infer_frames - WARMUP_FRAMES`` of those, later clips all
    of them, and the concatenation is cut down to ``num_frames``.
    """
    num_frames: int
    infer_frames: int
    num_clips: int

    @property
    def audio_frames(self) -> int:
        return self.num_clips * self.infer_frames

    def visible_frames(self, clip_index: int) -> int:
        return self.infer_frames - WARMUP_FRAMES if clip_index == 0 else self.infer_frames


def plan_clips(num_frames: int, clip_frames: int) -> S2VClipPlan:
    """Split ``num_frames`` output frames into clips of at most ``clip_frames``.

    A single clip covers ``clip_frames - WARMUP_FRAMES`` visible frames, so the
    default 84-frame clip yields exactly the 81-frame default request. Shorter
    requests shrink the clip instead of generating frames that get thrown away.
    """
    if num_frames < 1 or (num_frames - 1) % 4 != 0:
        raise ValueError(f"Wan-S2V needs num_frames = 4k+1 (e.g. 81), got {num_frames}. The VAE turns "
                         "4 pixel frames into 1 latent frame, plus one for the first frame.")
    if clip_frames < 4 or clip_frames % 4 != 0:
        raise ValueError(f"clip_frames must be a positive multiple of 4, got {clip_frames}")
    infer_frames = min(clip_frames, num_frames + WARMUP_FRAMES)
    remaining = max(0, num_frames - (infer_frames - WARMUP_FRAMES))
    num_clips = 1 + -(-remaining // infer_frames)  # ceil division
    return S2VClipPlan(num_frames=num_frames, infer_frames=infer_frames, num_clips=num_clips)


class S2VRefImageEncodingStage(ImageVAEEncodingStage):
    """VAE-encode the reference image as a single latent frame.

    Writes ``batch.image_latent`` with shape [B, C, 1, h, w]. Deterministic
    (distribution mode, not a sample): the reference is ground truth to
    preserve, and the official runner's native VAE encode is deterministic too.
    ``encode_pixels`` is shared with the pipeline's motion-history encode so
    both conditioning latents go through exactly the same normalisation.
    """

    def encode_pixels(self, pixels: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """[B, 3, T, H, W] pixels in [-1, 1] -> normalised latents [B, C, t, h, w]."""
        self.vae.to(get_local_torch_device())  # Module.to is in-place; no rebind, keeps mypy able to type self.vae
        pixels = pixels.to(get_local_torch_device(), dtype=torch.float32)

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
            if not vae_autocast_enabled:
                pixels = pixels.to(vae_dtype)
            latent = self.retrieve_latents(self.vae.encode(pixels), generator=None, sample_mode="argmax")

        # Same normalisation the other latents go through.
        if getattr(self.vae, "shift_factor", None) is not None:
            shift = self.vae.shift_factor
            latent = latent - (shift.to(latent.device, latent.dtype) if isinstance(shift, torch.Tensor) else shift)
        scale = self.vae.scaling_factor
        latent = latent * (scale.to(latent.device, latent.dtype) if isinstance(scale, torch.Tensor) else scale)

        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return latent

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.pil_image is not None and isinstance(batch.pil_image, PIL.Image.Image)
        assert batch.height is not None and batch.width is not None

        image = self.preprocess(batch.pil_image,
                                vae_scale_factor=self.vae.spatial_compression_ratio,
                                height=batch.height,
                                width=batch.width)
        batch.image_latent = self.encode_pixels(image.unsqueeze(2), fastvideo_args)  # [B, C, 1, H, W] in
        return batch

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # One latent frame exactly: the DiT parks these tokens at a dedicated
        # time index and its RoPE grid claims precisely one frame.
        result.add_check("image_latent", batch.image_latent,
                         lambda v: v is not None and v.dim() == 5 and v.shape[2] == 1)
        return result


class S2VDecodingStage(DecodingStage):
    """Decode with temporal context prepended, official-runner style.

    The Wan VAE is causal in time: the first frames decode with less context
    and come out degraded. The official runner therefore decodes
    ``[context | generated latents]`` and keeps the trailing ``infer_frames``
    pixels. The context is the previous clip's motion latents when there are
    any, else the reference latent -- and in that first-clip case 3 more
    warm-up frames are dropped.
    """

    @staticmethod
    def trim(frames: torch.Tensor, infer_frames: int, first_clip: bool) -> torch.Tensor:
        """Keep the generated span of a decoded ``[B, C, T, H, W]`` tensor."""
        frames = frames[:, :, -infer_frames:]
        return frames[:, :, WARMUP_FRAMES:] if first_clip else frames

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None
        motion_latents = batch.extra.get(EXTRA_MOTION_LATENTS)
        first_clip = motion_latents is None
        context = batch.image_latent if first_clip else motion_latents
        assert context is not None, "S2V decode needs the reference latent for temporal context"
        # Without a plan (stages run outside the pipeline loop) the request is
        # one clip whose visible frames are ``num_frames``.
        infer_frames = batch.extra.get(EXTRA_INFER_FRAMES)
        if infer_frames is None:
            assert batch.num_frames is not None
            infer_frames = batch.num_frames + WARMUP_FRAMES

        batch.latents = torch.cat([context.to(batch.latents.device, batch.latents.dtype), batch.latents], dim=2)
        batch = super().forward(batch, fastvideo_args)
        if fastvideo_args.output_type != "latent" and batch.output is not None:
            batch.output = self.trim(batch.output, infer_frames, first_clip)
        return batch
