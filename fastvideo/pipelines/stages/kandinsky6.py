# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import contextlib
from copy import deepcopy
import math
from typing import Any

import PIL
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from fastvideo.attention.backends.nabla import NablaAttentionMetadataBuilder
from fastvideo.configs.pipelines.kandinsky6 import (
    KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX, )
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader, VAELoader
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vision_utils import normalize, numpy_to_pt, pil_to_numpy
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE, get_mixed_precision_state

logger = init_logger(__name__)

# batch.extra keys used to pass state between the Kandinsky6 stages -- the
# same "stash it on batch.extra, downstream stages check for it" pattern
# fastvideo/pipelines/basic/ltx2 uses for its audio latents.
_TAIL_COND_KEY = "kandinsky6_tail_cond_active"
_TOKEN_TYPE_IDS_KEY = "kandinsky6_visual_token_type_ids"
_AUDIO_SAMPLE_RATE_DEFAULT = 44100


def audio_latent_duration(video_latent_frames: int, *, fps: float, audio_sample_rate: int,
                          audio_downsample_factor: int) -> int:
    """Audio latent length matching the diffusers Kandinsky6 T2VA reference:
    ``ceil(pixel_frames / fps * audio_sample_rate / audio_downsample_factor)``,
    where ``pixel_frames = (video_latent_frames - 1) * 4 + 1`` is the causal
    video VAE's temporal-compression convention (matches
    HunyuanVAEConfig.temporal_compression_ratio == 4).
    """
    pixel_frames = (video_latent_frames - 1) * 4 + 1
    return int(math.ceil(pixel_frames / fps * audio_sample_rate / audio_downsample_factor))


class Kandinsky6CFGResolutionStage(PipelineStage):
    """Recomputes ``batch.do_classifier_free_guidance`` before text encoding using Kandinsky6's CFG
    contract. The diffusers reference (``pipeline_kandinsky6_ti2va.py``)'s ``do_classifier_free_guidance``
    property gates on the standard ``guidance_scale > 1.0`` (any guidance_scale <= 1.0, including below 1,
    runs cond-only with no negative-prompt encoding); this stage mirrors that. A PiFlow (distilled)
    scheduler never uses CFG regardless of the requested guidance_scale (its cond-only contract, and the
    guidance==1.0 requirement, are enforced by Kandinsky6DenoisingStage instead), so this also
    short-circuits to False for it and skips encoding a negative prompt that would otherwise go unused.
    """

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if bool(getattr(self.scheduler, "is_piflow", False)):
            batch.do_classifier_free_guidance = False
        else:
            batch.do_classifier_free_guidance = batch.guidance_scale > 1.0
        return batch


class Kandinsky6TextEncodingStage(TextEncodingStage):
    """Maps the request's ``max_sequence_length`` onto the Qwen encoder only.

    Matches the diffusers reference, whose Qwen ``max_length = max_sequence_length + 129`` (the
    prompt-template prefix crop, ``ENCODE_START_IDX``). The CLIP pooled encoder always keeps its fixed
    77-token tokenizer config regardless of the request: applying the same override to it (the
    generic ``TextEncodingStage``'s default behavior, one ``max_length`` for every encoder) makes a
    CLIP tokenizer with only 77 positions try to pad to whatever length was requested for Qwen and
    crash.
    """

    def _resolve_max_length(self, batch: ForwardBatch,
                            fastvideo_args: FastVideoArgs) -> int | Sequence[int | None] | None:
        if batch.max_sequence_length is None:
            return None
        return [KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX + batch.max_sequence_length, None]


class Kandinsky6LatentPreparationStage(PipelineStage):
    """Draw initial video *and* audio noise latents.

    Runs for both T2VA and IT2VA calls -- the conditioning image (if any) is
    applied afterward by Kandinsky6ImageEncodingStage, which appends an extra
    reference frame rather than overwriting one drawn here, so the RNG order
    here does not depend on whether an image was supplied.
    """

    def __init__(self, scheduler, transformer) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError("height and width must be provided for Kandinsky6.")
        height = int(batch.height)
        width = int(batch.width)
        num_frames = int(batch.num_frames)

        pipeline_config = fastvideo_args.pipeline_config
        temporal_ratio = pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = pipeline_config.vae_config.arch_config.spatial_compression_ratio
        patch_size = pipeline_config.dit_config.arch_config.patch_size

        # Round to the nearest k*temporal_ratio+1, matching the diffusers reference's
        # `check_inputs`/`__call__` num_frames adjustment (`num_frames // temporal_ratio *
        # temporal_ratio + 1`, warning first): this floors for a non-exact multiple (119 -> 117) but
        # rounds *up* for an exact multiple of temporal_ratio (120 -> 121, 96 -> 97), unlike a pure
        # floor formula, which would floor every case (120 -> 117, 96 -> 93); 121 -> 121 unchanged
        # either way.
        if num_frames % temporal_ratio != 1:
            rounded_num_frames = max(num_frames // temporal_ratio * temporal_ratio + 1, 1)
            logger.warning("Kandinsky6: num_frames=%d is not of the form k*%d+1; rounding to %d.", num_frames,
                           temporal_ratio, rounded_num_frames)
            num_frames = rounded_num_frames
            batch.num_frames = num_frames

        required_divisor_h = spatial_ratio * patch_size[1]
        required_divisor_w = spatial_ratio * patch_size[2]
        if height % required_divisor_h != 0 or width % required_divisor_w != 0:
            raise ValueError(f"Kandinsky6 height must be divisible by {required_divisor_h} and width by "
                             f"{required_divisor_w}; got height={height}, width={width}.")

        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]
        batch_size *= batch.num_videos_per_prompt

        dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]
        device = get_local_torch_device()
        num_latent_frames = (num_frames - 1) // temporal_ratio + 1
        num_channels = getattr(self.transformer, "in_visual_dim", pipeline_config.dit_config.arch_config.in_visual_dim)
        video_shape = (batch_size, num_latent_frames, height // spatial_ratio, width // spatial_ratio, num_channels)

        if batch.latents is None:
            video = randn_tensor(video_shape, generator=batch.generator, device=device, dtype=dtype)
            if hasattr(self.scheduler, "init_noise_sigma"):
                video = video * self.scheduler.init_noise_sigma
        else:
            video = batch.latents.to(device=device, dtype=dtype)
            if tuple(video.shape) != video_shape:
                raise ValueError(f"Provided latents shape {list(video.shape)} does not match expected "
                                 f"Kandinsky6 video latent shape {list(video_shape)}.")

        visual_cond = getattr(self.transformer, "visual_cond", False)
        if visual_cond:
            cond = torch.zeros_like(video)
            mask = torch.zeros((*video.shape[:-1], 1), device=video.device, dtype=video.dtype)
            video = torch.cat([video, cond, mask], dim=-1)

        num_audio_channels = getattr(self.transformer, "in_audio_dim",
                                     pipeline_config.dit_config.arch_config.in_audio_dim)
        # Matches the diffusers reference's `sample_fps` request field, which
        # drives the audio latent length directly (not just the saved video's
        # playback rate): a request fps of 30 with 121 frames yields 174
        # audio latents there, not the 218 a fixed 24 fps would give.
        requested_fps = batch.fps
        if isinstance(requested_fps, list):
            requested_fps = requested_fps[0] if requested_fps else None
        fps = float(requested_fps) if requested_fps else pipeline_config.sample_fps
        num_audio_frames = audio_latent_duration(
            num_latent_frames,
            fps=fps,
            audio_sample_rate=pipeline_config.audio_sample_rate,
            audio_downsample_factor=pipeline_config.audio_downsample_factor,
        )
        audio_shape = (batch_size, num_audio_frames, num_audio_channels)
        if batch.audio_latents is None:
            # Drawn from the same generator right after the video noise, so it
            # is automatically an independent sample (no manual seed offset
            # needed -- unlike the diffusers reference, which threads a bare
            # int seed through its packed-latent-prep helpers and has to
            # offset it by hand).
            audio = randn_tensor(audio_shape, generator=batch.generator, device=device, dtype=dtype)
        else:
            audio = batch.audio_latents.to(device=device, dtype=dtype)
            if tuple(audio.shape) != audio_shape:
                raise ValueError(f"Provided audio_latents shape {list(audio.shape)} does not match expected "
                                 f"Kandinsky6 audio latent shape {list(audio_shape)}.")

        batch.latents = video
        batch.audio_latents = audio
        batch.raw_latent_shape = (batch_size, num_channels, num_latent_frames, height // spatial_ratio,
                                  width // spatial_ratio)
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("audio_latents", batch.audio_latents, [V.is_tensor, V.with_dims(3)])
        return result


class Kandinsky6ImageEncodingStage(PipelineStage):
    """Optional IT2VA conditioning: no-op for a pure T2VA call.

    When ``batch.pil_image`` is supplied, encodes it through the shared video
    VAE and appends it as one extra "clean" reference frame at the end of the
    video latent sequence (the diffusers reference's default
    ``tail_cond_first_frame`` scheme), tagged via a token-type id so the
    transformer's ``visual_token_type_embeddings`` can distinguish it from
    generated frames. Kandinsky6DenoisingStage re-pins this frame every step
    and strips it back out after the loop.
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        self.vae = vae

    @staticmethod
    def _cover_resize_dims(src_h: int, src_w: int, height: int, width: int) -> tuple[int, int]:
        """Resize target for a subsequent centre crop to (height, width), matching the diffusers
        reference's ``encode_i2va_first_frame``: ``scale = min(src_h/h, src_w/w)`` so the resized
        image covers the target box on both axes (never smaller than it), and the excess is
        centre-cropped afterward -- unlike squeezing the whole source image to fit, which distorts
        its aspect ratio.
        """
        scale = min(src_h / height, src_w / width)
        # `int(src / scale)` matches the diffusers reference formula, but on the constraining axis
        # (the one that produced `scale`) it can land a hair below the target -- e.g. 831 for 832 --
        # from float error in the scale round-trip, which would give a negative centre-crop offset.
        # Clamp each axis to its target as a floor; this is a no-op whenever `int()` already reaches
        # the target, so it doesn't change the (non-constraining-axis) diffusers-parity values.
        new_h = max(height, int(src_h / scale))
        new_w = max(width, int(src_w / scale))
        return new_h, new_w

    @classmethod
    def _preprocess(cls, image, height: int, width: int) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            src_w, src_h = image.size
            new_h, new_w = cls._cover_resize_dims(src_h, src_w, height, width)
            image = image.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
            top, left = (new_h - height) // 2, (new_w - width) // 2
            image = image.crop((left, top, left + width, top + height))
            image = numpy_to_pt(pil_to_numpy(image))
            return normalize(image)
        if image.min() >= 0:
            logger.warning("Kandinsky6 conditioning image tensor has no negative values; "
                           "assuming range [0, 1] and normalizing to [-1, 1]. "
                           "Pass a [-1, 1] tensor with negative values to skip normalization.")
            image = normalize(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        src_h, src_w = image.shape[-2], image.shape[-1]
        if (src_h, src_w) != (height, width):
            new_h, new_w = cls._cover_resize_dims(src_h, src_w, height, width)
            image = torch.nn.functional.interpolate(image.float(), size=(new_h, new_w), mode="bilinear", antialias=True)
            top, left = (new_h - height) // 2, (new_w - width) // 2
            image = image[..., top:top + height, left:left + width]
        return image

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.pil_image is None:
            return batch

        if not fastvideo_args.model_loaded["vae"]:
            vae = getattr(self, "vae", None)
            if vae is None:
                loader = VAELoader()
                vae = loader.load(fastvideo_args.model_paths["vae"], fastvideo_args)
                self.vae = vae
            fastvideo_args.model_loaded["vae"] = True

        device = get_local_torch_device()
        vae = self.vae.to(device)
        self.vae = vae
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = vae_dtype != torch.float32 and not fastvideo_args.disable_autocast

        image = self._preprocess(batch.pil_image, int(batch.height), int(batch.width))
        image = image.to(device=device, dtype=torch.float32).unsqueeze(2)  # [B,C,1,H,W]

        prev_use_tiling = vae.use_tiling
        vae.use_tiling = False
        try:
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if not vae_autocast_enabled:
                    image = image.to(vae_dtype)
                generator = batch.generator
                if isinstance(generator, list) and len(generator) != image.shape[0]:
                    generator = generator[0]
                image_latent = vae.encode(image).sample(generator=generator)
        finally:
            vae.use_tiling = prev_use_tiling

        image_latent = image_latent * vae.scaling_factor
        # [B,C,1,H,W] -> [B,1,H,W,C] channel-last, matching the video latent.
        image_latent = image_latent.permute(0, 2, 3, 4, 1).contiguous()
        batch.image_latent = image_latent

        latents = batch.latents
        image_latent = image_latent.to(device=latents.device, dtype=latents.dtype)
        num_channels = image_latent.shape[-1]

        ref_frame = image_latent
        if latents.shape[-1] > num_channels:
            # visual_cond channel layout: [real, cond, mask]. Per the
            # diffusers reference's `_build_video_input` (pipeline_kandinsky6_
            # ti2va.py), the tail_cond_first_frame scheme leaves the cond block
            # at zero and only writes the real channel block (done above via
            # `ref_frame = image_latent`) and mask=1 -- unlike Kandinsky5's
            # I2V "pretrain"/"i2v" schemes (fastvideo/pipelines/stages/
            # kandinsky5.py), which duplicate the image latent into cond.
            # Different scheme, don't carry that convention over.
            cond_block = torch.zeros_like(image_latent)
            mask_block = torch.ones_like(image_latent[..., :1])
            ref_frame = torch.cat([image_latent, cond_block, mask_block], dim=-1)

        if ref_frame.shape[0] != latents.shape[0]:
            # One image is encoded per request regardless of
            # num_videos_per_prompt (there is only ever one conditioning
            # image); broadcast it to every video in the batch instead of
            # letting the concatenation below fail with a tensor-size
            # mismatch.
            if ref_frame.shape[0] != 1:
                raise ValueError(f"Kandinsky6 image conditioning batch size {ref_frame.shape[0]} does not match "
                                 f"the video latent batch size {latents.shape[0]} and is not broadcastable.")
            ref_frame = ref_frame.expand(latents.shape[0], *ref_frame.shape[1:])

        batch.latents = torch.cat([latents, ref_frame], dim=1)
        batch.extra[_TAIL_COND_KEY] = True

        num_video_frames = latents.shape[1]
        token_type_ids = torch.zeros((latents.shape[0], num_video_frames + 1), dtype=torch.long, device=device)
        token_type_ids[:, -1] = 1
        batch.extra[_TOKEN_TYPE_IDS_KEY] = token_type_ids

        if fastvideo_args.vae_cpu_offload:
            vae.to("cpu")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result


class Kandinsky6DenoisingStage(PipelineStage):
    """Joint video/audio denoising with independent scheduler state per modality.

    Mirrors the diffusers reference's ``denoise_loop`` (pipeline_kandinsky6_
    ti2va.py): one joint transformer call per step for both modalities, CFG
    via ``uncond + w*(cond-uncond)``, video advanced through the shared
    scheduler's ``step()`` (which owns the sigma index), audio advanced with
    a manual Euler update using the *same* per-step sigma delta so the
    scheduler's step index is never double-advanced.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @staticmethod
    def _text_rope_pos(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        seq_len = int(mask.sum(1).max().item())
        return torch.arange(seq_len, device=device)

    def _resolve_target_dtype(self, fastvideo_args: FastVideoArgs) -> torch.dtype:
        """See Kandinsky5DenoisingStage._resolve_target_dtype: trusts
        ``dit_precision`` for a plain load, but FSDP2 always computes in the
        policy's ``param_dtype`` regardless of parameter storage dtype."""
        declared_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        try:
            from torch.distributed.fsdp import FSDPModule
        except Exception:  # pragma: no cover - FSDP not always available
            return declared_dtype
        if not isinstance(self.transformer, FSDPModule):
            return declared_dtype
        try:
            policy_dtype = get_mixed_precision_state().param_dtype
        except ValueError:
            policy_dtype = None
        return policy_dtype if policy_dtype is not None else torch.bfloat16

    @staticmethod
    def fast_sta_nabla(T: int, H: int, W: int, wT: int, wH: int, wW: int, device: torch.device | str) -> torch.Tensor:
        max_extent = int(torch.tensor([T, H, W], device=device).amax().item())
        r = torch.arange(0, max_extent, 1, dtype=torch.int16, device=device)
        mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
        sta_t = (mat[:T, :T].flatten() <= wT // 2)
        sta_h = (mat[:H, :H].flatten() <= wH // 2)
        sta_w = (mat[:W, :W].flatten() <= wW // 2)
        sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
        sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
        return sta.reshape(T * H * W, T * H * W)

    def get_sparse_params(self, video: torch.Tensor, device: torch.device) -> dict[str, Any] | None:
        cfg = self.transformer.config
        if cfg.attention_engine != "nabla":
            return None
        assert cfg.patch_size[0] == 1
        _, T, H, W, _ = video.shape
        T, H, W = T // cfg.patch_size[0], H // cfg.patch_size[1], W // cfg.patch_size[2]
        sta_mask = self.fast_sta_nabla(T,
                                       H // 8,
                                       W // 8,
                                       cfg.attention_wT,
                                       cfg.attention_wH,
                                       cfg.attention_wW,
                                       device=device)
        return {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": cfg.attention_engine,
            "to_fractal": True,
            "P": cfg.attention_P,
            "wT": cfg.attention_wT,
            "wW": cfg.attention_wW,
            "wH": cfg.attention_wH,
            "add_sta": cfg.attention_add_sta,
            "visual_shape": (T, H, W),
            "method": cfg.attention_method,
        }

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        use_piflow = bool(getattr(self.scheduler, "is_piflow", False))
        if use_piflow and (not math.isfinite(batch.guidance_scale) or abs(batch.guidance_scale - 1.0) > 1e-6):
            nfe = getattr(self.scheduler.config, "nfe", None)
            raise ValueError(
                f"Kandinsky6 PiFlow (distilled) checkpoints need guidance_scale=1.0 (got {batch.guidance_scale}); "
                "pi-Flow runs without classifier-free guidance, so the official Diffusers pipeline rejects any "
                f"other value too. num_inference_steps is not constrained by the checkpoint -- {nfe} is only the "
                "scheduler's nfe, the value this checkpoint was distilled for and the default of the "
                "'kandinsky6_ti2va_distilled' preset (itself the default for "
                "'kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers' and local copies named like it, a "
                "Kandinsky-6 name containing 'distill'). Pass guidance_scale=1.0 explicitly otherwise.")
        audio_scheduler = deepcopy(self.scheduler) if use_piflow else None
        if batch.timesteps is None:
            raise ValueError("timesteps must be prepared before Kandinsky6 denoising.")
        if batch.latents is None or batch.audio_latents is None:
            raise ValueError("video and audio latents must be prepared before Kandinsky6 denoising.")
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(fastvideo_args.model_paths["transformer"], fastvideo_args)
            fastvideo_args.model_loaded["transformer"] = True

        device = get_local_torch_device()
        target_dtype = self._resolve_target_dtype(fastvideo_args)
        autocast_enabled = target_dtype != torch.float32 and not fastvideo_args.disable_autocast
        video = batch.latents
        audio = batch.audio_latents
        num_channels = getattr(self.transformer, "in_visual_dim",
                               fastvideo_args.pipeline_config.dit_config.arch_config.in_visual_dim)

        tail_cond = bool(batch.extra.get(_TAIL_COND_KEY, False))
        visual_token_type_ids = batch.extra.get(_TOKEN_TYPE_IDS_KEY)

        prompt_embeds = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        pooled = batch.prompt_embeds[1].to(device=device, dtype=target_dtype)
        if not batch.prompt_attention_mask:
            raise ValueError("Kandinsky6 requires Qwen prompt attention masks.")
        text_rope_pos = self._text_rope_pos(batch.prompt_attention_mask[0].to(device), device)

        neg_prompt_embeds = None
        neg_pooled = None
        negative_text_rope_pos = None
        if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
            neg_prompt_embeds = batch.negative_prompt_embeds[0].to(device=device, dtype=target_dtype)
            neg_pooled = batch.negative_prompt_embeds[1].to(device=device, dtype=target_dtype)
            if not batch.negative_attention_mask:
                raise ValueError("Kandinsky6 requires Qwen negative attention masks for CFG.")
            negative_text_rope_pos = self._text_rope_pos(batch.negative_attention_mask[0].to(device), device)

        height = int(batch.height)
        width = int(batch.width)
        pipeline_config = fastvideo_args.pipeline_config
        spatial_ratio = pipeline_config.vae_config.arch_config.spatial_compression_ratio
        patch_size = pipeline_config.dit_config.arch_config.patch_size

        num_video_frames = video.shape[1] - (1 if tail_cond else 0)
        t_positions = torch.arange(num_video_frames, device=device)
        if tail_cond:
            # The appended reference frame reuses T-position 0's rope row
            # (matches the diffusers reference's `torch.cat([rope, rope[:1]])`
            # duplication -- equivalent since RoPE3D is a pure position ->
            # table lookup).
            t_positions = torch.cat([t_positions, t_positions.new_zeros(1)])
        visual_rope_pos = [
            t_positions,
            torch.arange(height // spatial_ratio // patch_size[1], device=device),
            torch.arange(width // spatial_ratio // patch_size[2], device=device),
        ]
        # Read straight off the loaded transformer (falls back to the pipeline's static arch config
        # for a not-yet-loaded model, e.g. a verify_input probe): both official checkpoints declare
        # (1.0, 2.0, 2.0) in transformer/config.json. Diffusers uses this value unconditionally, not
        # a height/width-dependent heuristic.
        scale_factor = getattr(self.transformer, "scale_factor",
                               tuple(float(v) for v in pipeline_config.dit_config.arch_config.scale_factor))
        sparse_params = self.get_sparse_params(video, device)

        # PiFlow's noisy latent state is tracked in fp32 across steps here, separately from the
        # `video` buffer's storage dtype -- matches the diffusers reference, whose `PiflowScheduler.
        # step` returns fp32 and whose pipeline rebinds the video state to that fp32 result every
        # step rather than writing it back into a lower-precision buffer (`scheduling_piflow.py`,
        # `pipeline_kandinsky6_ti2va.py`'s `denoise_loop`). flow-Euler does not need this: the
        # diffusers reference also downcasts its state to the parameter dtype every step there.
        # Always a Tensor (not `Tensor | None`) even on the flow-Euler path, where it is simply
        # unused: the initial cast is a single cheap op, and keeping it non-Optional avoids
        # re-narrowing `video_state` after every `if use_piflow:` branch re-entry in the loop below.
        video_state = video[..., :num_channels].to(torch.float32)

        with tqdm(total=batch.num_inference_steps, desc="Kandinsky6 Denoising") as progress_bar:
            for i, timestep in enumerate(batch.timesteps):
                if hasattr(self, "interrupt") and self.interrupt:
                    break

                # The scheduler timestep is fp32 in the diffusers reference (`t.unsqueeze(0).
                # expand(bs)`, embedded via `time.float()`); Kandinsky6TimeEmbeddings' sinusoid
                # table (`torch.outer(time, freqs)`) needs that precision directly; rounding the
                # timestep itself to bf16 first (as opposed to computing the embedding in bf16 from
                # an already-fp32 timestep) is the largest single numeric gap against the reference.
                t_expand = timestep.unsqueeze(0).repeat(video.shape[0]).to(device=device, dtype=torch.float32)
                if use_piflow:
                    video_input = video_state.to(dtype=target_dtype)
                    if video.shape[-1] > num_channels:
                        video_input = torch.cat([video_input, video[..., num_channels:].to(dtype=target_dtype)], dim=-1)
                else:
                    video_input = video.to(dtype=target_dtype)
                attn_metadata = None
                if sparse_params is not None:
                    attn_metadata = NablaAttentionMetadataBuilder().build(
                        current_timestep=i,
                        sta_mask=sparse_params["sta_mask"],
                        P=sparse_params["P"],
                        visual_shape=sparse_params["visual_shape"],
                    )
                autocast_ctx = (torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled)
                                if device.type == "cuda" else contextlib.nullcontext())
                with set_forward_context(current_timestep=i, attn_metadata=attn_metadata,
                                         forward_batch=batch), autocast_ctx:
                    video_vel, audio_vel = self.transformer(
                        hidden_states=video_input,
                        hidden_states_audio=audio.to(dtype=target_dtype),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        timestep=t_expand,
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=text_rope_pos,
                        scale_factor=scale_factor,
                        sparse_params=sparse_params,
                        visual_token_type_ids=visual_token_type_ids,
                        return_dict=True,
                    ).sample

                    if neg_prompt_embeds is not None and neg_pooled is not None:
                        uncond_video_vel, uncond_audio_vel = self.transformer(
                            hidden_states=video_input,
                            hidden_states_audio=audio.to(dtype=target_dtype),
                            encoder_hidden_states=neg_prompt_embeds,
                            pooled_projections=neg_pooled,
                            timestep=t_expand,
                            visual_rope_pos=visual_rope_pos,
                            text_rope_pos=negative_text_rope_pos,
                            scale_factor=scale_factor,
                            sparse_params=sparse_params,
                            visual_token_type_ids=visual_token_type_ids,
                            return_dict=True,
                        ).sample
                        video_vel = uncond_video_vel + batch.guidance_scale * (video_vel - uncond_video_vel)
                        audio_vel = uncond_audio_vel + batch.guidance_scale * (audio_vel - uncond_audio_vel)

                if use_piflow:
                    video_state = self.scheduler.step(video_vel, timestep, video_state, return_dict=False)[0]
                    video[..., :num_channels] = video_state.to(video.dtype)
                    audio = audio_scheduler.step(audio_vel, timestep, audio, return_dict=False)[0]
                else:
                    video[..., :num_channels] = self.scheduler.step(video_vel,
                                                                    timestep,
                                                                    video[..., :num_channels],
                                                                    return_dict=False)[0]
                    # Manual Euler update for audio: one `scheduler.step()` call
                    # per iteration total (against video, above) so its internal
                    # sigma index never double-advances; audio uses the same
                    # sigma delta directly, matching the diffusers reference.
                    step_size = self.scheduler.sigmas[i + 1] - self.scheduler.sigmas[i]
                    audio = audio + step_size.to(device=audio.device, dtype=audio.dtype) * audio_vel
                if tail_cond:
                    ref_frame = batch.image_latent.to(device=video.device, dtype=video.dtype)
                    video[:, -1:, :, :, :num_channels] = ref_frame
                    if use_piflow:
                        video_state[:, -1:] = ref_frame.to(torch.float32)

                if i == len(batch.timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        video = video[..., :num_channels]
        if tail_cond:
            video = video[:, :-1]

        batch.latents = video
        batch.audio_latents = audio
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("audio_latents", batch.audio_latents, [V.is_tensor, V.with_dims(3)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.min_list_length(2))
        return result


class Kandinsky6DecodingStage(DecodingStage):
    """Channel-last [B,T,H,W,C] -> channel-first [B,C,T,H,W] then the
    generic VAE decode, matching Kandinsky5DecodingStage."""

    def __init__(self, vae: ParallelTiledVAE, pipeline=None) -> None:
        super().__init__(vae=vae, pipeline=pipeline)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("latents must be available before Kandinsky6 decoding.")
        batch.latents = batch.latents.permute(0, 4, 1, 2, 3).contiguous()
        return super().forward(batch, fastvideo_args)


class Kandinsky6AudioDecodingStage(PipelineStage):
    """Decode Kandinsky6 audio latents into a waveform via two separate
    components -- ``audio_vae`` (mel-VAE decoder) and ``vocoder``
    (BigVGAN-v2, mel->waveform) -- like LTX-2's audio_vae/vocoder pair.
    Writes ``batch.extra["audio"]`` / ``["audio_sample_rate"]``, which
    ``VideoGenerator`` already knows how to mux into the saved video file.
    """

    def __init__(self, audio_vae, vocoder) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.vocoder = vocoder

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.audio_latents is None:
            return batch
        if not fastvideo_args.is_output_rank:
            # Non-output SPMD ranks never return audio. The audio VAE and the
            # vocoder have no collectives, so skip them and the waveform host
            # copy instead of discarding the result later.
            batch.extra.pop("audio", None)
            batch.extra.pop("audio_sample_rate", None)
            batch.audio_latents = None
            return batch

        device = get_local_torch_device()
        self.audio_vae = self.audio_vae.to(device)
        self.vocoder = self.vocoder.to(device)

        # Cast to the module's own loaded weight dtype directly (no autocast
        # / precision-config knob) -- matches the existing MMAudio audio
        # decoding stage (fastvideo/pipelines/basic/mmaudio/stages.py).
        decoder_dtype = next(self.audio_vae.parameters()).dtype

        # Diffusion-latent-level denorm (matches the diffusers reference's
        # postprocess_audio: `audio / scaling_factor`). Read from the loaded
        # module itself -- this value lives in the checkpoint's own
        # audio_vae/config.json, not a pipeline-level default.
        scaling_factor = getattr(self.audio_vae, "scaling_factor", 1.0)
        latents = batch.audio_latents / scaling_factor

        # [B, A, D] -> [B, D, A] to match the audio VAE's 1D-conv (channel,
        # length) convention.
        latents = latents.to(device=device, dtype=decoder_dtype).transpose(1, 2)
        mel = self.audio_vae.decode(latents)  # [B, num_mels, T]
        waveform = self.vocoder(mel.to(next(self.vocoder.parameters()).dtype))  # [B, 1, samples]

        # Batch size is 1 in practice (Kandinsky6, like Kandinsky5, is a
        # single-prompt pipeline); VideoGenerator's audio mux path
        # (`_audio_to_int16`) expects one plain [-1,1] float array/tensor per
        # call, not a per-batch-item list, and does its own int16 conversion.
        batch.extra["audio"] = waveform[0, 0].clamp(-1.0, 1.0).float().cpu()
        batch.extra["audio_sample_rate"] = _resolve_audio_sample_rate(self.audio_vae, fastvideo_args.pipeline_config)

        if fastvideo_args.vae_cpu_offload:
            self.audio_vae.to("cpu")
            self.vocoder.to("cpu")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        return result


def _resolve_audio_sample_rate(vocoder, pipeline_config) -> int:
    model = getattr(vocoder, "module", vocoder)
    for attr in ("output_sampling_rate", "output_sample_rate", "sample_rate"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    return getattr(pipeline_config, "audio_sample_rate", _AUDIO_SAMPLE_RATE_DEFAULT)
