# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-specific pipeline stages.

These replicate the conditioning assembly of the official runner
(``wan/animate.py``) and diffusers' ``WanAnimatePipeline``, which the shared
stages do differently:

* The denoised sequence carries an extra leading latent frame (the reference
  slot), so latent preparation allocates ``T_lat + 1`` frames.
* The conditional latent ``y`` is 20 channels -- a 4-channel folded I2V mask
  plus the 16-channel VAE encoding of [reference | zeros-or-background] --
  temporally concatenated as ``[ref (1 frame) | target (T_lat frames)]`` and
  channel-concatenated with the noise by the shared DenoisingStage
  (channel order: noise 16 | mask 4 | cond 16 = the DiT's in_dim 36).
* The pose skeleton video is VAE-encoded onto the same latent grid.
* The face video stays in pixel space (the DiT's motion encoder consumes raw
  512x512 crops).
* Decoding drops the reference slot (``vae.decode(latents[:, :, 1:])``).

Every numeric convention here (mask folding, mask inversion, zeros-video
encoding, argmax sampling, reflect padding) is transcribed from
``pipeline_wan_animate.py`` -- none of it crashes when wrong, it just produces
subtly broken video, so deviations are bugs even when output "looks fine".

v1 scope: a single 77-frame segment. Multi-segment chaining (temporal-guidance
frames from the previous segment) is pipeline-loop work on top of these stages.
"""
import PIL.Image
import torch
import torch.nn.functional as F

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_video, normalize, numpy_to_pt, pil_to_numpy, resize
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

ANIMATE_MODES = ("animation", "replace")


def _pad_frames(frames: list, target: int) -> list:
    """Reflect-pad (or truncate) a frame list to exactly ``target`` frames.

    Transcribed from diffusers' ``pad_video_frames``:
    [1, 2, 3] -> [1, 2, 3, 2, 1, 2, ...]. Truncation covers the common case of
    a driving video longer than the requested clip.
    """
    if not frames:
        raise ValueError(f"empty driving video: need at least one frame to pad to {target}")
    if len(frames) >= target:
        return list(frames[:target])
    if len(frames) == 1:
        return [frames[0]] * target  # reflecting a single frame is repetition
    out: list = []
    idx, flip = 0, False
    while len(out) < target:
        out.append(frames[idx])
        idx = idx - 1 if flip else idx + 1
        if idx == 0 or idx == len(frames) - 1:
            flip = not flip
    return out


def _frames_to_tensor(frames: list[PIL.Image.Image], height: int, width: int, grayscale: bool = False) -> torch.Tensor:
    """PIL frames -> [1, C, T, H, W] float tensor.

    RGB frames are normalised to [-1, 1] (the VAE/motion-encoder input range);
    grayscale masks stay in [0, 1] (they are selection weights, not imagery).
    """
    frames = [resize(f.convert("L" if grayscale else "RGB"), height, width) for f in frames]
    video = numpy_to_pt(pil_to_numpy(frames))  # [T, C, H, W] in [0, 1]
    video = video if grayscale else normalize(video)
    return video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]


def _fold_i2v_mask(mask_pixel: torch.Tensor | None, latent_t: int, latent_h: int, latent_w: int, mask_len: int,
                   temporal_ratio: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fold a per-pixel-frame binary mask into Wan-I2V's 4-channel latent mask.

    Convention: 1 = preserved/clean, 0 = generate. The first pixel frame is
    repeated ``temporal_ratio`` times, then every group of ``temporal_ratio``
    pixel frames folds into the channel axis -> [1, 4, T_lat, h, w].
    """
    if mask_pixel is None:
        mask = torch.zeros(1, 1, (latent_t - 1) * temporal_ratio + 1, latent_h, latent_w, dtype=dtype, device=device)
    else:
        mask = mask_pixel.clone().to(device=device, dtype=dtype)
    mask[:, :, :mask_len] = 1
    first = mask[:, :, 0:1].repeat_interleave(temporal_ratio, dim=2)
    mask = torch.cat([first, mask[:, :, 1:]], dim=2)
    return mask.view(1, -1, temporal_ratio, latent_h, latent_w).transpose(1, 2)


def _encode_normalized(stage: ImageVAEEncodingStage, video: torch.Tensor,
                       fastvideo_args: FastVideoArgs) -> torch.Tensor:
    """Deterministic (argmax) VAE encode + the standard latent normalisation.

    Conditioning is ground truth to preserve, not a sample -- and the official
    pipelines encode all of it with the distribution mode.
    """
    vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
    autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast
    with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=autocast_enabled):
        if fastvideo_args.pipeline_config.vae_tiling:
            stage.vae.enable_tiling()
        if not autocast_enabled:
            video = video.to(vae_dtype)
        latent = stage.retrieve_latents(stage.vae.encode(video), generator=None, sample_mode="argmax")
    shift = getattr(stage.vae, "shift_factor", None)
    if shift is not None:
        latent = latent - (shift.to(latent.device, latent.dtype) if isinstance(shift, torch.Tensor) else shift)
    scale = stage.vae.scaling_factor
    return latent * (scale.to(latent.device, latent.dtype) if isinstance(scale, torch.Tensor) else scale)


class AnimateLatentPreparationStage(LatentPreparationStage):
    """Allocate noise for ``T_lat + 1`` latent frames: the extra slot is the
    reference frame, denoised alongside the video and dropped at decode."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        # The parent derives its latent frame count from batch.num_frames and
        # nothing else, so inflating by one pixel group buys exactly one extra
        # latent frame -- and the inflated count also lands in raw_latent_shape,
        # where the attention-metadata builders need the true T_lat + 1.
        # (num_frames + ratio - 1) // ratio + 1 == (num_frames - 1) // ratio + 2.
        batch.num_frames += ratio
        try:
            return super().forward(batch, fastvideo_args)
        finally:
            batch.num_frames -= ratio


class AnimateConditioningLatentsStage(ImageVAEEncodingStage):
    """Assemble the 20-channel conditional latent ``y`` into ``batch.image_latent``.

    Layout (channel-first inside each frame group, ref frame first in time):

        [ mask 4ch | cond latent 16ch ] x [ ref (1 frame) | target (T_lat) ]

    Animation mode: the target's cond video is a **black video encoded through
    the VAE** (VAE(zeros-pixels) != zero latents -- encoding it is load-bearing)
    and the target mask is all zeros ("generate everything").
    Replace mode: the target's cond video is the background video and the mask
    is the *inverted* character mask (input convention: white = generate),
    nearest-downsampled to the latent grid -- 1 on preserved background, 0 in
    the person-shaped hole.
    """

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.pil_image is not None and isinstance(batch.pil_image, PIL.Image.Image)
        assert batch.height is not None and batch.width is not None and batch.num_frames is not None
        mode = batch.animate_mode or "animation"
        if mode not in ANIMATE_MODES:
            raise ValueError(f"animate_mode must be one of {ANIMATE_MODES}, got {mode!r} "
                             "(diffusers spells animation mode 'animate'; FastVideo uses 'animation')")
        device = get_local_torch_device()
        self.vae.to(device)

        height, width, num_frames = batch.height, batch.width, batch.num_frames
        spatial = self.vae.spatial_compression_ratio
        temporal = self.vae.temporal_compression_ratio
        latent_h, latent_w = height // spatial, width // spatial
        latent_t = (num_frames - 1) // temporal + 1

        # --- reference slot: the character image, encoded alone, mask all-ones ---
        # Diffusers letterboxes the reference (resize_mode="fill", black pad,
        # bilinear); vision_utils.resize has no "fill" mode, so this stretches
        # instead. Identical for the documented input (the official src_ref.png,
        # already at height x width).
        image = self.preprocess(batch.pil_image, vae_scale_factor=spatial, height=height,
                                width=width).to(device, dtype=torch.float32).unsqueeze(2)
        ref_latent = _encode_normalized(self, image, fastvideo_args)
        ref_mask = _fold_i2v_mask(None,
                                  1,
                                  latent_h,
                                  latent_w,
                                  mask_len=1,
                                  temporal_ratio=temporal,
                                  device=ref_latent.device,
                                  dtype=ref_latent.dtype)
        ref_part = torch.cat([ref_mask, ref_latent], dim=1)

        # --- target frames: zeros video (animation) or background video (replace) ---
        if mode == "replace":
            # Diffusers splits the background into [carried frames | rest]
            # and rejoins it -- the identity for a single segment; the split only
            # matters once the head frames come from the previous segment.
            bg_frames = _pad_frames(load_video(batch.background_video_path), num_frames)
            cond_video = _frames_to_tensor(bg_frames, height, width).to(device, torch.float32)
            mask_frames = _pad_frames(load_video(batch.mask_video_path), num_frames)
            mask_pixel = _frames_to_tensor(mask_frames, height, width, grayscale=True).to(device)
            # Input convention: white (1) = generate. Fold wants 1 = preserve.
            mask_pixel = 1.0 - mask_pixel
            # F.interpolate's spatial path needs 4-D input, so fold T into the
            # batch slot (valid because B == 1 here) and unfold afterwards.
            mask_pixel = F.interpolate(mask_pixel.squeeze(0).transpose(0, 1), size=(latent_h, latent_w),
                                       mode="nearest").transpose(0, 1).unsqueeze(0)
        else:
            cond_video = torch.zeros(1, 3, num_frames, height, width, device=device, dtype=torch.float32)
            mask_pixel = None
        cond_latents = _encode_normalized(self, cond_video, fastvideo_args)
        # Single segment: no temporal-guidance frames yet, so mask_len=0 (the
        # official runner's `mask_len = refert_num if start_frame > 0 else 0`).
        cond_mask = _fold_i2v_mask(mask_pixel,
                                   latent_t,
                                   latent_h,
                                   latent_w,
                                   mask_len=0,
                                   temporal_ratio=temporal,
                                   device=cond_latents.device,
                                   dtype=cond_latents.dtype)
        target_part = torch.cat([cond_mask, cond_latents], dim=1)

        batch.image_latent = torch.cat([ref_part, target_part], dim=2)
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # None normalises to "animation" in forward (the API layer's default);
        # other unknown values raise there with the legal set.
        result.add_check("animate_mode", batch.animate_mode, lambda v: v is None or v in ANIMATE_MODES)
        # The reference raises on non-multiples of 16; smaller multiples of 8
        # would silently truncate a latent row/column at the patchifier.
        result.add_check("height", batch.height, [V.positive_int, V.divisible(16)])
        result.add_check("width", batch.width, [V.positive_int, V.divisible(16)])
        # Segment length must be 4k + 1 or the mask folding above misaligns.
        result.add_check("num_frames", batch.num_frames, [V.positive_int, lambda v: (v - 1) % 4 == 0])
        if batch.animate_mode == "replace":
            result.add_check("background_video_path", batch.background_video_path,
                             lambda v: isinstance(v, str) and bool(v))
            result.add_check("mask_video_path", batch.mask_video_path, lambda v: isinstance(v, str) and bool(v))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        latent_channels = fastvideo_args.pipeline_config.dit_config.latent_channels
        ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent, [
            V.is_tensor,
            V.with_dims(5),
            lambda v: v.shape[1] == latent_channels + 4,
            lambda v: v.shape[2] == (batch.num_frames - 1) // ratio + 2,
        ])
        return result


class AnimatePoseVideoEncodingStage(ImageVAEEncodingStage):
    """VAE-encode the preprocessed skeleton video onto the target latent grid.

    Writes ``batch.pose_latents`` with T_lat frames -- one fewer than the
    denoised sequence, because the reference slot carries no pose (the DiT
    adds pose tokens to frames 1..T only).
    """

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.height is not None and batch.width is not None and batch.num_frames is not None
        device = get_local_torch_device()
        self.vae.to(device)
        frames = _pad_frames(load_video(batch.pose_video_path), batch.num_frames)
        pose_video = _frames_to_tensor(frames, batch.height, batch.width).to(device, torch.float32)
        batch.pose_latents = _encode_normalized(self, pose_video, fastvideo_args)
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("pose_video_path", batch.pose_video_path, lambda v: isinstance(v, str) and bool(v))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        result = VerificationResult()
        result.add_check("pose_latents", batch.pose_latents, [
            V.is_tensor,
            V.with_dims(5),
            lambda v: v.shape[2] == (batch.num_frames - 1) // ratio + 1,
        ])
        return result


class AnimateFaceVideoStage(PipelineStage):
    """Load the preprocessed face-crop video as raw pixels.

    The DiT's motion encoder consumes pixel crops directly (no VAE): resized to
    ``motion_encoder_size`` (512), normalised to [-1, 1]. ``num_frames`` pixel
    frames yield exactly one motion-vector group per latent frame after the
    model's causal 4x funnel (num_frames = 4 * T_lat - 3).
    """

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.num_frames is not None
        size = fastvideo_args.pipeline_config.dit_config.motion_encoder_size
        frames = _pad_frames(load_video(batch.face_video_path), batch.num_frames)
        batch.face_pixel_values = _frames_to_tensor(frames, size, size).to(get_local_torch_device(), torch.float32)
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("face_video_path", batch.face_video_path, lambda v: isinstance(v, str) and bool(v))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("face_pixel_values", batch.face_pixel_values, [
            V.is_tensor,
            V.with_dims(5),
            lambda v: v.shape[1] == 3,
            lambda v: v.shape[2] == batch.num_frames,
        ])
        return result


class AnimateDecodingStage(DecodingStage):
    """Decode without the reference slot.

    The denoiser emits ``T_lat + 1`` latent frames; the leading one is the
    reference-image slot, generated only as conditioning context. The official
    runner and diffusers both decode ``latents[:, :, 1:]``.
    """

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None and batch.latents.shape[2] >= 2, \
            "Animate decode expects the reference slot plus at least one video frame"
        batch.latents = batch.latents[:, :, 1:]
        return super().forward(batch, fastvideo_args)
