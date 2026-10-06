# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""End-to-end MiniMax-H3 joint audio-video generation on Apple Silicon MLX.

Phased runtime that keeps one heavyweight component resident at a time:

1. **Condition** — streamed Qwen3-VL text encoding (or a verified prompt
   embedding cache);
2. **Denoise** — pre-quantized H3 DiT (one of int8/int6/int4 resident),
   dual rectified-flow schedulers (video shift 12 / audio shift 3), served
   from the persisted AdaLN ladder;
3. **Decode** — MLX H3 video VAE and audio VAE, sequentially;
4. **Mux** — H.264 24 fps + AAC 32 kHz stereo MP4 via ffmpeg.

MLX-native memory cleanup (`mx.clear_cache`) runs between every phase.
The MLX path itself does not call PyTorch.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.frame_upsample import (
    DEFAULT_PIXEL_UPSAMPLE_MODE,
    PIXEL_UPSAMPLE_MODES,
    upsample_frames,
)
from fastvideo.mlx_runtime.minimax_h3 import (
    H3_MANIFEST_FILENAME,
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_ALIGNED_FRAMES,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_ALIGNED_FRAMES,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3SchedulerState,
    _eval_value,
    adaln_timestep_union,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_layout,
    build_row_timesteps,
    dense_only_vsa_error,
    load_mlx_h3_checkpoint,
    mlx_h3_checkpoint_vsa_capable,
    resolve_h3_denoise_schedule,
    temporal_position_grid,
    unpatchify_video_tokens,
    unpack_audio_tokens,
    video_latent_num_frames,
)
from fastvideo.mlx_runtime.minimax_h3_vsa import MiniMaxH3VSAConfig
from fastvideo.mlx_runtime.prompt_cache import fingerprint_digest, text_encoder_fingerprint

logger = init_logger(__name__)


@dataclass
class GenerationResult:
    video_path: str | None
    frames: np.ndarray | None
    waveform: np.ndarray | None
    sample_rate: int
    timings: dict[str, float] = field(default_factory=dict)
    peak_memory_gib: dict[str, float] = field(default_factory=dict)
    vsa: dict[str, Any] = field(default_factory=dict)
    video_decode_backend: str = "h3-vae"


@dataclass(frozen=True)
class FastTemporalPlan:
    """Sparse video geometry for RIFE fast mode with full-duration audio."""

    target_frames: int
    source_frames: int
    factor: int
    video_temporal_scale: float


def plan_fast_temporal(target_frames: int, factor: int = 2) -> FastTemporalPlan:
    """Choose the smallest H3-valid source sequence that covers the target timeline."""
    target_frames = align_num_frames(target_frames)
    if factor < 2:
        raise ValueError(f"fast factor must be at least 2, got {factor}.")
    ideal_source_frames = math.ceil((target_frames - 1) / factor) + 1
    source_frames = align_num_frames(ideal_source_frames)
    if source_frames >= target_frames:
        raise ValueError(f"fast factor {factor} does not reduce the H3-aligned target of {target_frames} frames.")

    source_grid = temporal_position_grid(video_latent_num_frames(source_frames), 0.0)
    target_grid = temporal_position_grid(video_latent_num_frames(target_frames), 0.0)
    video_temporal_scale = float(target_grid[-1] / source_grid[-1])
    return FastTemporalPlan(
        target_frames=target_frames,
        source_frames=source_frames,
        factor=factor,
        video_temporal_scale=video_temporal_scale,
    )


# Resampling from a smaller decode softens output the same way on every
# runtime; 0.4 matches the tuned Wan default without the halos that show
# up by ~0.8.
DEFAULT_FAST_SPATIAL_SHARPEN = 0.4


@dataclass(frozen=True)
class FastSpatialPlan:
    """Reduced-canvas geometry for spatial fast mode (RIFE's spatial twin)."""

    target_height: int
    target_width: int
    stage1_height: int
    stage1_width: int
    canvas_height: int
    canvas_width: int
    scale: int
    upsample_mode: str
    sharpen: float


def plan_fast_spatial(
    height: int,
    width: int,
    *,
    scale: int = 2,
    upsample_mode: str = DEFAULT_PIXEL_UPSAMPLE_MODE,
    sharpen: float = DEFAULT_FAST_SPATIAL_SHARPEN,
) -> FastSpatialPlan:
    """Choose the smallest H3-valid canvas that covers ``target / scale``.

    H3 geometry rounds *up* to the 32px model grid and center-crops after
    decode — the same convention plain 720p generation uses via
    ``_model_canvas_size`` — so no size the full-resolution path accepts is
    rejected here. The return trip to the target size runs in pixel space
    after the VAE decode, never on latents; see
    :mod:`fastvideo.mlx_runtime.frame_upsample` for why.
    """
    if scale < 2:
        raise ValueError(f"fast-spatial scale must be at least 2, got {scale}.")
    if upsample_mode not in PIXEL_UPSAMPLE_MODES:
        raise ValueError(f"Unsupported upsample mode: {upsample_mode!r} "
                         f"(expected one of {', '.join(PIXEL_UPSAMPLE_MODES)})")
    if sharpen < 0:
        raise ValueError(f"fast_spatial_sharpen must be non-negative, got {sharpen}.")
    target_canvas_height, target_canvas_width = _model_canvas_size(height, width)
    stage1_height = math.ceil(height / scale)
    stage1_width = math.ceil(width / scale)
    canvas_height, canvas_width = _model_canvas_size(stage1_height, stage1_width)
    if canvas_height * canvas_width >= target_canvas_height * target_canvas_width:
        raise ValueError(
            f"fast-spatial scale {scale} does not reduce the H3 canvas for {height}x{width} "
            f"(stage-1 canvas {canvas_width}x{canvas_height} vs {target_canvas_width}x{target_canvas_height}).")
    return FastSpatialPlan(
        target_height=height,
        target_width=width,
        stage1_height=stage1_height,
        stage1_width=stage1_width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        scale=scale,
        upsample_mode=upsample_mode,
        sharpen=sharpen,
    )


def _model_canvas_size(height: int, width: int) -> tuple[int, int]:
    """Round an exact output size up to H3's 32-pixel model grid."""
    if height <= 0 or width <= 0:
        raise ValueError(f"H3 output size must be positive, got {height}x{width}.")
    multiple = 32
    return math.ceil(height / multiple) * multiple, math.ceil(width / multiple) * multiple


def _center_crop_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    frame_height, frame_width = frames.shape[1:3]
    if height > frame_height or width > frame_width:
        raise ValueError(f"cannot crop {frame_width}x{frame_height} frames to {width}x{height}.")
    top = (frame_height - height) // 2
    left = (frame_width - width) // 2
    return np.ascontiguousarray(frames[:, top:top + height, left:left + width])


def _sharpen_frames(frames: list[np.ndarray], amount: float) -> list[np.ndarray]:
    if amount <= 0:
        return frames
    import cv2

    sharpened = []
    for frame in frames:
        blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
        sharpened.append(cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0))
    return sharpened


def _peak_memory_gib() -> float:
    import mlx.core as mx

    getter = getattr(mx, "get_peak_memory", None)
    return 0.0 if getter is None else float(getter()) / 2**30


def _reset_peak_memory() -> None:
    import mlx.core as mx

    reset = getattr(mx, "reset_peak_memory", None)
    if reset is not None:
        reset()


def _cleanup_mlx() -> None:
    import mlx.core as mx

    gc.collect()
    clear = getattr(mx, "clear_cache", None)
    if clear is not None:
        clear()


def _default_metal_wired_limit_gib(mx) -> float:
    """Legacy helper for allocator capacity, not the wired-residency setting."""
    metal = getattr(mx, "metal", None)
    if metal is None:
        return 30.0
    try:
        total_bytes = int(metal.device_info().get("memory_size", 0))
    except (AttributeError, TypeError, ValueError):
        return 30.0
    if total_bytes <= 0:
        return 30.0
    return min(30.0, 0.84 * total_bytes / 2**30)


def _configure_metal_memory_limits(mx,
                                   wired_limit_gib: float | None,
                                   *,
                                   resident: bool = False) -> tuple[int | None, int | None]:
    """Set Metal limits and return their previous process-wide values."""
    if wired_limit_gib is not None and (not math.isfinite(wired_limit_gib) or wired_limit_gib <= 0):
        raise ValueError("metal_wired_limit_gib must be finite and positive")
    set_wired = getattr(mx, "set_wired_limit", None)
    if set_wired is None and hasattr(mx, "metal"):
        set_wired = getattr(mx.metal, "set_wired_limit", None)
    if wired_limit_gib is not None and set_wired is None:
        raise RuntimeError("This MLX build cannot set the requested wired-memory limit")

    previous_memory = None
    set_memory = getattr(mx, "set_memory_limit", None)
    if set_memory is None and hasattr(mx, "metal"):
        set_memory = getattr(mx.metal, "set_memory_limit", None)
    if set_memory is not None and not resident:
        # An explicit wired request must fit under the allocator cap, or it pins memory MLX cannot allocate.
        memory_limit_gib = max(_default_metal_wired_limit_gib(mx), wired_limit_gib or 0.0)
        try:
            previous_memory = int(set_memory(int(memory_limit_gib * 2**30)))
        except Exception as error:  # noqa: BLE001 - older MLX best effort
            logger.info("Could not set the Metal allocation limit: %s", error)
    if wired_limit_gib is None:
        return previous_memory, None
    # Explicit requests must succeed; do not silently benchmark an unwired model.
    assert set_wired is not None
    try:
        previous = set_wired(int(wired_limit_gib * 2**30))
    except Exception:
        if previous_memory is not None and set_memory is not None:
            set_memory(previous_memory)
        raise
    logger.info("MLX wired limit %.2f GiB (previous %.2f GiB)", wired_limit_gib, previous / 2**30)
    return previous_memory, int(previous)


def _restore_metal_wired_limit(mx, previous_bytes: int | None) -> None:
    if previous_bytes is None:
        return
    set_wired = getattr(mx, "set_wired_limit", None)
    if set_wired is None and hasattr(mx, "metal"):
        set_wired = getattr(mx.metal, "set_wired_limit", None)
    if set_wired is not None:
        set_wired(previous_bytes)


def _restore_metal_memory_limit(mx, previous_bytes: int | None) -> None:
    if previous_bytes is None:
        return
    set_memory = getattr(mx, "set_memory_limit", None)
    if set_memory is None and hasattr(mx, "metal"):
        set_memory = getattr(mx.metal, "set_memory_limit", None)
    if set_memory is not None:
        set_memory(previous_bytes)


MINIMAX_H3_PROMPT_CACHE_VERSION = "v2-attention-layout"


def prompt_cache_path(cache_dir: str | Path, model_root: str | Path, prompt: str) -> Path:
    digest = hashlib.sha256(
        f"{MINIMAX_H3_PROMPT_CACHE_VERSION}::{Path(model_root)}::{prompt}".encode()).hexdigest()[:24]
    return Path(cache_dir) / f"prompt_embeds_{digest}.npz"


def _audio_sample_count(num_frames: int, fps: int = MINIMAX_H3_FPS, sample_rate: int = 32000) -> int:
    return math.ceil(num_frames / fps * sample_rate)


def _adaln_schedule_union(num_steps: int) -> np.ndarray:
    video = MiniMaxH3SchedulerState.create(MINIMAX_H3_VIDEO_SHIFT, num_steps)
    audio = MiniMaxH3SchedulerState.create(MINIMAX_H3_AUDIO_SHIFT, num_steps)
    return adaln_timestep_union(video, audio)


def _cached_adaln_timesteps(checkpoint_dir: str | Path) -> np.ndarray | None:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / H3_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing MLX H3 checkpoint manifest: {manifest_path}")
    cache_info = json.loads(manifest_path.read_text()).get("adaln_cache")
    if cache_info is None:
        return None
    return np.asarray(cache_info["timesteps"], dtype=np.float32)


def _load_resident_dit(checkpoint_dir: str | Path):
    """Load an H3 DiT and materialize every weight before timed requests."""
    import mlx.core as mx

    dit = load_mlx_h3_checkpoint(checkpoint_dir)
    for group in [dit.weights, *dit.blocks, *dit.refiner]:
        for value in group.values():
            # With the converter's AdaLN cache, dropped AdaLN projection weights are None.
            if value is not None:
                _eval_value(value)
    cache = dit._adaln_cache
    if cache is not None:
        mx.eval(cache.block_tables, cache.norm_out_shift, cache.norm_out_scale)
    return dit


def _adaln_weights_dropped(dit: Any) -> bool:
    key = "adaln_proj.linear.weight"
    return any(key in block and block[key] is None for block in getattr(dit, "blocks", ()))


def _validate_checkpoint_step_ladder(checkpoint_dir: str | Path,
                                     num_steps: int,
                                     *,
                                     model_root: str | Path | None = None) -> None:
    """Reject a schedule that a fixed, AdaLN-dropped checkpoint cannot serve."""
    resolve_h3_denoise_schedule(
        model_root,
        num_steps,
        cached_timesteps=_cached_adaln_timesteps(checkpoint_dir),
    )


def _require_positive_vae_tiles(vae_tile_height: int, vae_tile_width: int) -> None:
    if type(vae_tile_height) is not int or type(
            vae_tile_width) is not int or vae_tile_height <= 0 or vae_tile_width <= 0:
        raise ValueError("VAE tile dimensions must be positive integers, "
                         f"got height={vae_tile_height!r}, width={vae_tile_width!r}.")


def _preflight_media_dependencies(*,
                                  fast: bool,
                                  fast_sharpen: float,
                                  rife_weights_dir: str | Path | None,
                                  fast_spatial: bool = False) -> None:
    """Fail before conditioning when required output dependencies are unavailable."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for MP4 muxing; install it before generation.")
    if fast_spatial and importlib.util.find_spec("cv2") is None:
        raise RuntimeError("OpenCV is required for --fast-spatial resampling.")
    if not fast:
        return
    if fast_sharpen > 0 and importlib.util.find_spec("cv2") is None:
        raise RuntimeError("OpenCV is required when --fast-sharpen is greater than zero.")
    from fastvideo.mlx_runtime.rife_interp import ensure_weights_available

    ensure_weights_available(weights_dir=str(rife_weights_dir) if rife_weights_dir is not None else None)


class MiniMaxH3MLXPipeline:
    """Text-to-video-with-audio generation through the native MLX runtime."""

    # Defaults for pipelines built without __init__ (unit tests use __new__).
    resident = False
    conditioner_mode = "auto"

    def __init__(
        self,
        *,
        model_root: str | Path,
        mlx_dit_checkpoint: str | Path,
        vae_dtype: str = "fp32",
        prompt_cache_dir: str | Path | None = None,
        conditioner_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        metal_wired_limit_gib: float | None = None,
        video_decode_backend: str = "h3-vae",
        taeh3_checkpoint: str | Path | None = None,
        taeh3_chunk_size: int = 5,
        conditioner_mode: str = "auto",
        resident: bool = False,
    ) -> None:
        import mlx.core as mx

        self.model_root = Path(model_root)
        if conditioner_mode not in ("auto", "streamed", "nvfp4"):
            raise ValueError(f"Unknown H3 conditioner mode: {conditioner_mode}")
        if resident and video_decode_backend != "h3-vae":
            raise ValueError("Resident H3 generation requires the H3 video VAE.")
        self.conditioner_mode = conditioner_mode
        self.resident = resident
        self._resident_components: dict[str, Any] = {}
        self.dit_checkpoint = Path(mlx_dit_checkpoint)
        self.vae_dtype = vae_dtype
        if video_decode_backend not in ("h3-vae", "taeh3"):
            raise ValueError(f"Unknown H3 video decoder: {video_decode_backend}")
        if taeh3_chunk_size < 1:
            raise ValueError("taeh3_chunk_size must be positive.")
        if taeh3_checkpoint is not None and video_decode_backend != "taeh3":
            raise ValueError("taeh3_checkpoint requires video_decode_backend='taeh3'.")
        self.video_decode_backend = video_decode_backend
        self.taeh3_checkpoint = taeh3_checkpoint
        self.taeh3_chunk_size = taeh3_chunk_size
        self.prompt_cache_dir = Path(prompt_cache_dir) if prompt_cache_dir else None
        self.conditioner_dir = Path(conditioner_dir) if conditioner_dir else self.model_root / "text_encoder"
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else self.model_root / "tokenizer"
        self._validate_inputs_before_loading()
        manifest = json.loads((self.dit_checkpoint / H3_MANIFEST_FILENAME).read_text())
        dit_config = manifest["config"]
        patch_size = dit_config["patch_size"]
        if len(patch_size) != 3:
            raise ValueError(f"H3 DiT patch_size must have three dimensions, got {patch_size}.")
        self._dit_patch_size = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
        self._dit_in_channels = int(dit_config["in_channels"])
        self.last_dit_forward_s = 0.0
        self.last_vsa_stats: dict[str, Any] | None = None
        self._previous_memory_limit, self._previous_wired_limit = _configure_metal_memory_limits(mx,
                                                                                                 metal_wired_limit_gib,
                                                                                                 resident=resident)

    # -- input validation (before anything heavy loads) -------------------

    def _validate_inputs_before_loading(self) -> bool:
        missing = []
        if not self.dit_checkpoint.exists():
            missing.append(str(self.dit_checkpoint))
        vae_dir = self.model_root / "vae"
        audio_dir = self.model_root / "audio_vae"
        if self.video_decode_backend == "h3-vae" and not (vae_dir.exists() and any(vae_dir.glob("*.safetensors"))):
            missing.append(str(vae_dir))
        if not (audio_dir.exists() and any(audio_dir.glob("*.safetensors"))):
            missing.append(str(audio_dir))
        if missing:
            raise FileNotFoundError(f"Missing required H3 components: {missing}")
        return True

    @staticmethod
    def resolve_geometry(
        height: int,
        width: int,
        num_frames: int,
        *,
        enforce_duration: bool = True,
    ) -> dict[str, int]:
        """Explicit canvases pass through (positive multiples of 32); the
        aspect-ratio resolver only applies when dimensions are omitted."""
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(f"H3 canvas must be positive multiples of 32, got {height}x{width}.")
        aligned_frames = align_num_frames(num_frames)
        latent_frames = video_latent_num_frames(aligned_frames)
        duration = aligned_frames / MINIMAX_H3_FPS
        if enforce_duration and not MINIMAX_H3_MIN_ALIGNED_FRAMES <= aligned_frames <= MINIMAX_H3_MAX_ALIGNED_FRAMES:
            raise ValueError(f"H3 generates {MINIMAX_H3_MIN_DURATION:g}-{MINIMAX_H3_MAX_DURATION:g} s at "
                             f"{MINIMAX_H3_FPS} fps; {aligned_frames} frames is {duration:.2f} s, outside the "
                             f"accepted {MINIMAX_H3_MIN_ALIGNED_FRAMES}-{MINIMAX_H3_MAX_ALIGNED_FRAMES} frame range.")
        return {
            "height": height,
            "width": width,
            "num_frames": aligned_frames,
            "latent_frame_count": latent_frames,
            "latent_height": height // 16,
            "latent_width": width // 16,
        }

    # -- phase 1: conditioning -------------------------------------------

    def prepare_resident(self) -> None:
        """Load the encoder, DiT, and both decoders once, before timed requests."""
        if not self.resident or self._resident_components:
            return
        from fastvideo.mlx_runtime.minimax_h3_conditioner import ResidentNVFP4MiniMaxH3TextConditioner
        from fastvideo.mlx_runtime.minimax_h3_audio_vae import mlx_h3_audio_vae_from_dir
        from fastvideo.mlx_runtime.minimax_h3_video_vae import mlx_h3_video_vae_from_dir

        import mlx.core as mx

        try:
            conditioner = self._load_conditioner()
            if not isinstance(conditioner, ResidentNVFP4MiniMaxH3TextConditioner):
                conditioner.close()
                raise ValueError("All-resident generation requires the packed NVFP4 text encoder.")
            self._resident_components["conditioner"] = conditioner
            self._resident_components["dit"] = _load_resident_dit(self.dit_checkpoint)
            self._resident_components["video_vae"] = mlx_h3_video_vae_from_dir(self.model_root / "vae",
                                                                               include_encoder=False,
                                                                               storage_dtype=self.vae_dtype)
            self._resident_components["audio_vae"] = mlx_h3_audio_vae_from_dir(self.model_root / "audio_vae",
                                                                               include_encoder=False)
            mx.eval(list(self._resident_components["audio_vae"].weights.values()))
            logger.info("H3 components resident: %.2f GiB active MLX memory", mx.get_active_memory() / 2**30)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        conditioner = self._resident_components.get("conditioner")
        if conditioner is not None:
            conditioner.close()
        self._resident_components.clear()
        _cleanup_mlx()
        previous = getattr(self, "_previous_wired_limit", None)
        if previous is not None:
            import mlx.core as mx

            _restore_metal_wired_limit(mx, previous)
            self._previous_wired_limit = None
        previous_memory = getattr(self, "_previous_memory_limit", None)
        if previous_memory is not None:
            import mlx.core as mx

            _restore_metal_memory_limit(mx, previous_memory)
            self._previous_memory_limit = None

    def encode_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (hidden states (S, hidden), token tags). Uses the cache or
        the streamed conditioner."""
        cache_key = self._prompt_cache_key(prompt)
        if cache_key is not None and cache_key.exists():
            data = np.load(cache_key)
            logger.info("Loaded prompt embeddings from cache %s", cache_key)
            return data["hidden_states"], data["token_tags"]

        if self.resident:
            self.prepare_resident()
            conditioner = self._resident_components["conditioner"]
        else:
            conditioner = self._load_conditioner()
        hidden, tags = conditioner.encode_prompt(prompt)
        if not self.resident:
            conditioner.close()
        _cleanup_mlx()
        if cache_key is not None:
            cache_key.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache = cache_key.with_name(f".{cache_key.name}.tmp")
            try:
                with tmp_cache.open("wb") as handle:
                    np.savez(handle, hidden_states=hidden, token_tags=tags)
                tmp_cache.replace(cache_key)
            finally:
                tmp_cache.unlink(missing_ok=True)
        return hidden, tags

    def _prompt_cache_key(self, prompt: str) -> Path | None:
        """Cache path bound to the effective encoder and its files; None skips the cache."""
        if self.prompt_cache_dir is None:
            return None
        fingerprint = {
            "conditioner": self._conditioner_kind(),
            "text_encoder": text_encoder_fingerprint(self.conditioner_dir),
            "tokenizer": text_encoder_fingerprint(self.tokenizer_dir),
        }
        if not (fingerprint["text_encoder"]["complete"] and fingerprint["tokenizer"]["complete"]):
            return None
        return prompt_cache_path(self.prompt_cache_dir, fingerprint_digest(fingerprint), prompt)

    def load_prompt_cache(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        return data["hidden_states"], data["token_tags"]

    def has_conditioner_weights(self) -> bool:
        marker = self.conditioner_dir / "model.safetensors.index.json"
        single = self.conditioner_dir / "model.safetensors"
        return marker.exists() or single.exists()

    def _load_conditioner(self):
        from fastvideo.mlx_runtime.minimax_h3_conditioner import (
            ResidentNVFP4MiniMaxH3TextConditioner,
            StreamedMiniMaxH3TextConditioner,
        )

        if self._conditioner_kind() == "nvfp4":
            return ResidentNVFP4MiniMaxH3TextConditioner(self.conditioner_dir, self.tokenizer_dir)
        return StreamedMiniMaxH3TextConditioner(self.conditioner_dir, self.tokenizer_dir)

    def _conditioner_kind(self) -> str:
        """The encoder that conditioner_mode selects for these weights: 'nvfp4' or 'streamed'."""
        config = json.loads((self.conditioner_dir / "config.json").read_text())
        packed = str(config.get("quantization_config", {}).get("quant_method", "")).lower() == "nvfp4"
        if self.conditioner_mode == "nvfp4" or (self.conditioner_mode == "auto" and packed):
            return "nvfp4"
        if packed:
            raise ValueError("The streamed conditioner requires BF16 weights; use conditioner_mode='nvfp4'.")
        return "streamed"

    # -- phase 2: denoise --------------------------------------------------

    def denoise(
        self,
        text_rows: np.ndarray,
        token_tags: np.ndarray,
        *,
        height: int,
        width: int,
        num_frames: int,
        audio_num_frames: int | None = None,
        video_temporal_scale: float = 1.0,
        seed: int,
        num_steps: int = 4,
        dit: Any | None = None,
        vsa_config: MiniMaxH3VSAConfig | None = None,
        inter_step_cooldown_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Denoise joint latents; returns (normalized video rows, audio rows)."""
        import mlx.core as mx

        geometry = self.resolve_geometry(height, width, num_frames, enforce_duration=audio_num_frames is None)
        audio_frames = geometry["num_frames"] if audio_num_frames is None else align_num_frames(audio_num_frames)

        resident_dit = dit is None and self.resident
        if resident_dit:
            _validate_checkpoint_step_ladder(self.dit_checkpoint, num_steps, model_root=self.model_root)
            self.prepare_resident()
            dit = self._resident_components["dit"]
        owned_dit = dit is None
        if owned_dit:
            _validate_checkpoint_step_ladder(self.dit_checkpoint, num_steps, model_root=self.model_root)
            t0 = time.perf_counter()
            dit = load_mlx_h3_checkpoint(self.dit_checkpoint)
            logger.info("Loaded MLX H3 DiT from %s in %.1fs", self.dit_checkpoint, time.perf_counter() - t0)

        schedule = resolve_h3_denoise_schedule(self.model_root, num_steps)
        video_scheduler = schedule.video
        audio_scheduler = schedule.audio
        union = schedule.adaln_timesteps
        num_steps = schedule.num_steps
        # The keyframe-noise timestep (0.999) is only exercised by FL2VA/Ref2VA
        # conditioning rows; those modes recompute the ladder before denoise.

        cache = getattr(dit, "_adaln_cache", None)
        if cache is None or not np.array_equal(cache.timesteps.astype(np.float32), union):
            if cache is not None:
                extra = np.setdiff1d(union, cache.timesteps)
                logger.info("Recomputing AdaLN cache for %d-step ladder (extra timesteps %s).", num_steps, extra)
            if _adaln_weights_dropped(dit):
                # An earlier request on this DiT already released the AdaLN projections.
                if not resident_dit:
                    raise ValueError("This H3 DiT dropped its AdaLN weights for another step ladder; "
                                     "pass a freshly loaded DiT for a different num_steps.")
                logger.info("Reloading the resident H3 DiT for the %d-step ladder.", num_steps)
                cache = dit = None
                self._resident_components.pop("dit", None)
                _cleanup_mlx()
                dit = self._resident_components["dit"] = _load_resident_dit(self.dit_checkpoint)
            dit.precompute_adaln(union, drop_weights=True)

        if resident_dit:
            # The resident DiT keeps the previous request's VSA mode unless reset to the default.
            vsa_config = vsa_config or MiniMaxH3VSAConfig()
        if vsa_config is not None:
            dit.configure_vsa(vsa_config)
        if hasattr(dit, "reset_vsa_stats"):
            dit.reset_vsa_stats()

        layout = build_packed_layout(
            len(token_tags),
            geometry["latent_frame_count"],
            geometry["latent_height"],
            geometry["latent_width"],
            audio_latent_num_frames(audio_frames),
            patch_size=dit.patch_size,
            text_token_tags=np.asarray(token_tags, dtype=np.int64),
            video_temporal_scale=video_temporal_scale,
        )
        if getattr(dit, "vsa_config", None) is not None and dit.vsa_config.enabled:
            dit.prepare_vsa_geometry(layout)

        video_key, audio_key = mx.random.split(mx.random.key(seed))
        target_video_rows = int(layout.video_indices.shape[0] - layout.num_condition_video_rows)
        target_audio_rows = int(layout.audio_indices.shape[0] - layout.num_condition_audio_rows)
        x_v = mx.random.normal((target_video_rows, dit.patch_dim), key=video_key)
        x_a = mx.random.normal((target_audio_rows, dit.audio_in_channels), key=audio_key)
        text = mx.array(text_rows.astype(np.float32))
        dit_forward_s = 0.0

        for step_index in range(num_steps):
            video_t = float(video_scheduler.timesteps[step_index])
            audio_t = float(audio_scheduler.timesteps[step_index])
            unique, inverse = build_row_timesteps(
                layout,
                video_timestep=video_t,
                audio_timestep=audio_t,
                condition_video_timestep=max(video_t, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                condition_audio_timestep=1.0,
            )
            step_started = time.perf_counter()
            video_velocity, audio_velocity = dit.forward_with_cache(
                x_v,
                x_a,
                text,
                layout=layout,
                step_timesteps=unique,
                row_timestep_inverse=inverse,
                step_index=step_index,
            )
            # Only target rows are being denoised (no conditions in T2VA).
            video_velocity = video_velocity[layout.num_condition_video_rows:]
            audio_velocity = audio_velocity[layout.num_condition_audio_rows:]
            x_v = video_scheduler.step(video_velocity, step_index, x_v)
            x_a = audio_scheduler.step(audio_velocity, step_index, x_a)
            mx.eval(x_v, x_a)
            step_s = time.perf_counter() - step_started
            dit_forward_s += step_s
            logger.info("H3 denoise step %d/%d (sigma_video=%.4f, sigma_audio=%.4f) in %.2fs", step_index + 1,
                        num_steps, 1.0 - video_t, 1.0 - audio_t, step_s)
            del video_velocity, audio_velocity
            _cleanup_mlx()
            if inter_step_cooldown_s > 0 and step_index < num_steps - 1:
                logger.info("Thermal cooldown %.1fs before step %d...", inter_step_cooldown_s, step_index + 2)
                time.sleep(inter_step_cooldown_s)

        self.last_dit_forward_s = dit_forward_s
        stats = getattr(dit, "last_vsa_stats", None)
        self.last_vsa_stats = None if stats is None else {
            "enabled": bool(getattr(dit.vsa_config, "enabled", False)),
            "configured_sparsity": stats.configured_sparsity,
            "layer_sparsity": stats.layer_sparsity,
            "achieved_sparsity": stats.achieved_sparsity,
            "tile_size": stats.tile_size,
            "prefix_mode": stats.prefix_mode,
            "impl": stats.impl,
            "num_prefix_tiles": stats.num_prefix_tiles,
            "num_video_tiles": stats.num_video_tiles,
            "video_keep": stats.video_keep,
            "dense_fallback_reason": stats.dense_fallback_reason,
            "attention_calls": stats.attention_calls,
            "sparse_calls": stats.sparse_calls,
            "impl_counts": stats.impl_counts,
            "fallback_reasons": stats.fallback_reasons,
            "checkpoint_vsa_capable": bool(getattr(dit, "vsa_capable", False)),
        }

        video_rows = np.asarray(x_v, dtype=np.float32)
        audio_rows = np.asarray(x_a, dtype=np.float32)
        if owned_dit:
            del dit
            _cleanup_mlx()
        return video_rows, audio_rows

    # -- phase 3a: video decode -------------------------------------------

    def decode_video(self,
                     video_rows: np.ndarray,
                     *,
                     height: int,
                     width: int,
                     num_frames: int,
                     tiled: bool = True,
                     vae_tile_height: int = 256,
                     vae_tile_width: int = 256) -> np.ndarray:
        """Normalized packed rows -> (T, H, W, 3) uint8 frames."""
        _require_positive_vae_tiles(vae_tile_height, vae_tile_width)
        import mlx.core as mx

        from fastvideo.mlx_runtime.minimax_h3_video_vae import mlx_h3_video_vae_from_dir

        geometry = self.resolve_geometry(height, width, num_frames, enforce_duration=False)
        if self.video_decode_backend == "taeh3":
            from fastvideo.mlx_runtime.minimax_h3_taeh3 import decode_latents_taeh3_mlx

            if self._dit_in_channels != 24:
                raise ValueError("TAEH3 requires a 24-channel H3 checkpoint.")
            latents = unpatchify_video_tokens(video_rows, geometry["latent_frame_count"], geometry["latent_height"],
                                              geometry["latent_width"], self._dit_in_channels, self._dit_patch_size)
            pixels = decode_latents_taeh3_mlx(latents,
                                              checkpoint_path=self.taeh3_checkpoint,
                                              dtype=self.vae_dtype,
                                              chunk_size=self.taeh3_chunk_size)
            frames = (pixels[0] * 255.0).astype(np.uint8)
            if frames.shape != (geometry["num_frames"], height, width, 3):
                raise RuntimeError(f"TAEH3 produced unexpected frame shape: {frames.shape}")
            _cleanup_mlx()
            return frames
        if self.resident:
            self.prepare_resident()
            vae = self._resident_components["video_vae"]
        else:
            vae = mlx_h3_video_vae_from_dir(self.model_root / "vae",
                                            include_encoder=False,
                                            storage_dtype=self.vae_dtype)
        expected_height = height // vae.spatial_compression_ratio
        expected_width = width // vae.spatial_compression_ratio
        if (geometry["latent_height"], geometry["latent_width"]) != (expected_height, expected_width):
            raise RuntimeError("H3 pipeline/VAE spatial compression mismatch: "
                               f"pipeline={(geometry['latent_height'], geometry['latent_width'])}, "
                               f"VAE={(expected_height, expected_width)}.")
        if vae.latent_channels != self._dit_in_channels:
            raise RuntimeError(
                f"H3 DiT/VAE latent-channel mismatch: DiT={self._dit_in_channels}, VAE={vae.latent_channels}.")
        latents = unpatchify_video_tokens(
            video_rows,
            geometry["latent_frame_count"],
            geometry["latent_height"],
            geometry["latent_width"],
            vae.latent_channels,
            self._dit_patch_size,
        )
        z = mx.array(latents)
        z = vae.denormalize_latents(z)
        decoded = vae.decode(z,
                             tiled=tiled,
                             tile_sample_min_height=min(geometry["height"], vae_tile_height),
                             tile_sample_min_width=min(geometry["width"], vae_tile_width))
        pixels = np.clip(np.asarray(vae.denormalize_pixels(decoded)), 0.0, 1.0)
        del vae, decoded, z
        _cleanup_mlx()
        frames = (pixels[0].transpose(1, 2, 3, 0) * 255.0).astype(np.uint8)  # (T, H, W, C)
        if frames.shape[0] != geometry["num_frames"]:
            raise RuntimeError(f"decoded {frames.shape[0]} frames, expected {geometry['num_frames']}")
        return frames

    # -- phase 3b: audio decode --------------------------------------------

    def decode_audio(self, audio_rows: np.ndarray, *, num_frames: int) -> np.ndarray:
        """Normalized packed audio rows -> stereo waveform (2, S) fp32 in [-1, 1]."""
        import mlx.core as mx

        from fastvideo.mlx_runtime.minimax_h3_audio_vae import mlx_h3_audio_vae_from_dir

        num_audio_latents = audio_latent_num_frames(align_num_frames(num_frames))
        latents = unpack_audio_tokens(audio_rows, num_audio_latents)
        if self.resident:
            self.prepare_resident()
            vae = self._resident_components["audio_vae"]
        else:
            vae = mlx_h3_audio_vae_from_dir(self.model_root / "audio_vae", include_encoder=False)
        z = vae.denormalize_latents(mx.array(latents))
        waveform = np.asarray(vae.decode(z))[:, 0, :]  # (B, 1, S) -> (B, S)
        del vae, z
        _cleanup_mlx()
        # Keep audio at least as long as the final video packet. Rounding down
        # by a fractional sample makes ffmpeg's ``-shortest`` drop frame 124.
        expected_samples = _audio_sample_count(align_num_frames(num_frames))
        if waveform.shape[-1] < expected_samples:
            waveform = np.pad(waveform, ((0, 0), (0, expected_samples - waveform.shape[-1])))
        return np.clip(waveform[:, :expected_samples], -1.0, 1.0)

    # -- phase 4: mux --------------------------------------------------------

    def mux(self,
            frames: np.ndarray,
            waveform: np.ndarray,
            output_path: str | Path,
            fps: int = MINIMAX_H3_FPS,
            sample_rate: int = 32000) -> Path:
        """H.264 video + AAC stereo audio, A/V durations within one frame."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_video = output_path.with_suffix(".tmp.mp4")
        tmp_audio = output_path.with_suffix(".tmp.wav")

        pcm = (np.clip(waveform.T, -1.0, 1.0) * 32767.0).astype("<i2")  # (S, 2)
        import wave

        try:
            with wave.open(str(tmp_audio), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm.tobytes())

            height, width = frames.shape[1:3]
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required for MP4 muxing.")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-i",
                    str(tmp_audio),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(tmp_video),
                ],
                input=frames.tobytes(),
                check=True,
            )
            tmp_video.replace(output_path)
        finally:
            tmp_audio.unlink(missing_ok=True)
            tmp_video.unlink(missing_ok=True)
        return output_path

    # -- end-to-end ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        output_path: str | Path,
        height: int = 480,
        width: int = 832,
        num_frames: int = 124,
        seed: int = 0,
        num_steps: int = 4,
        save_frames: bool = False,
        tiled_video_decode: bool = True,
        vae_tile_height: int = 256,
        vae_tile_width: int = 256,
        fast: bool = False,
        fast_factor: int = 2,
        fast_sharpen: float = 0.6,
        rife_weights_dir: str | Path | None = None,
        fast_spatial: bool = False,
        fast_spatial_scale: int = 2,
        fast_spatial_upsample_mode: str = DEFAULT_PIXEL_UPSAMPLE_MODE,
        fast_spatial_sharpen: float = DEFAULT_FAST_SPATIAL_SHARPEN,
        vsa: bool = False,
        vsa_sparsity: float = 0.9,
        vsa_tile_size: int = 64,
        vsa_prefix_mode: str = "exempt",
        vsa_dense_first_n_steps: int = 0,
        vsa_dense_layers: tuple[int, ...] = (),
        vsa_impl: str = "auto",
        inter_step_cooldown_s: float = 0.0,
    ) -> GenerationResult:
        timings: dict[str, float] = {}
        peaks: dict[str, float] = {}

        if fast_sharpen < 0:
            raise ValueError(f"fast_sharpen must be non-negative, got {fast_sharpen}.")
        if inter_step_cooldown_s < 0:
            raise ValueError(f"inter_step_cooldown_s must be non-negative, got {inter_step_cooldown_s}.")
        _require_positive_vae_tiles(vae_tile_height, vae_tile_width)
        _validate_checkpoint_step_ladder(self.dit_checkpoint, num_steps, model_root=self.model_root)
        vsa_config = MiniMaxH3VSAConfig(
            enabled=vsa,
            sparsity=vsa_sparsity,
            tile_size=vsa_tile_size,
            prefix_mode=vsa_prefix_mode,  # type: ignore[arg-type]
            dense_first_n_steps=vsa_dense_first_n_steps,
            dense_layers=vsa_dense_layers,
            impl=vsa_impl,  # type: ignore[arg-type]
        ) if vsa else MiniMaxH3VSAConfig()
        if vsa_config.enabled and not mlx_h3_checkpoint_vsa_capable(self.dit_checkpoint):
            raise dense_only_vsa_error(self.dit_checkpoint)
        spatial_plan = plan_fast_spatial(
            height,
            width,
            scale=fast_spatial_scale,
            upsample_mode=fast_spatial_upsample_mode,
            sharpen=fast_spatial_sharpen,
        ) if fast_spatial else None
        _preflight_media_dependencies(
            fast=fast,
            fast_sharpen=fast_sharpen,
            rife_weights_dir=rife_weights_dir,
            fast_spatial=fast_spatial,
        )
        if spatial_plan is not None:
            canvas_height, canvas_width = spatial_plan.canvas_height, spatial_plan.canvas_width
        else:
            canvas_height, canvas_width = _model_canvas_size(height, width)
        target_geometry = self.resolve_geometry(canvas_height, canvas_width, num_frames)
        fast_plan = plan_fast_temporal(target_geometry["num_frames"], fast_factor) if fast else None
        video_num_frames = fast_plan.source_frames if fast_plan is not None else target_geometry["num_frames"]
        video_temporal_scale = fast_plan.video_temporal_scale if fast_plan is not None else 1.0
        video_geometry = self.resolve_geometry(
            canvas_height,
            canvas_width,
            video_num_frames,
            enforce_duration=fast_plan is None,
        )
        logger.info(
            "Geometry: output=%dx%dx%d model=%dx%dx%d audio_frames=%d fast=%s fast_spatial=%s",
            width,
            height,
            target_geometry["num_frames"],
            canvas_width,
            canvas_height,
            video_geometry["num_frames"],
            target_geometry["num_frames"],
            fast_plan,
            spatial_plan,
        )

        if self.video_decode_backend == "taeh3":
            from fastvideo.mlx_runtime.minimax_h3_taeh3 import ensure_taeh3_checkpoint

            started = time.perf_counter()
            ensure_taeh3_checkpoint(self.taeh3_checkpoint)
            timings["decoder_prepare_s"] = time.perf_counter() - started
            logger.warning("TAEH3 is an approximate preview decoder; reconstruction differs from the full H3 VAE.")

        _reset_peak_memory()
        started = time.perf_counter()
        text_rows, token_tags = self.encode_prompt(prompt)
        timings["condition_s"] = time.perf_counter() - started
        peaks["condition_gib"] = _peak_memory_gib()
        _cleanup_mlx()

        _reset_peak_memory()
        started = time.perf_counter()
        video_rows, audio_rows = self.denoise(
            text_rows,
            token_tags,
            height=video_geometry["height"],
            width=video_geometry["width"],
            num_frames=video_geometry["num_frames"],
            audio_num_frames=target_geometry["num_frames"] if fast_plan is not None else None,
            video_temporal_scale=video_temporal_scale,
            seed=seed,
            num_steps=num_steps,
            vsa_config=vsa_config,
            inter_step_cooldown_s=inter_step_cooldown_s,
        )
        timings["denoise_s"] = time.perf_counter() - started
        timings["dit_forward_s"] = float(getattr(self, "last_dit_forward_s", 0.0))
        peaks["denoise_gib"] = _peak_memory_gib()
        del text_rows
        _cleanup_mlx()

        _reset_peak_memory()
        started = time.perf_counter()
        frames = self.decode_video(
            video_rows,
            height=video_geometry["height"],
            width=video_geometry["width"],
            num_frames=video_geometry["num_frames"],
            tiled=tiled_video_decode,
            vae_tile_height=vae_tile_height,
            vae_tile_width=vae_tile_width,
        )
        if spatial_plan is not None:
            frames = _center_crop_frames(frames, spatial_plan.stage1_height, spatial_plan.stage1_width)
        else:
            frames = _center_crop_frames(frames, height, width)
        timings["video_decode_s"] = time.perf_counter() - started
        peaks["video_decode_gib"] = _peak_memory_gib()
        _cleanup_mlx()

        if fast_plan is not None:
            from fastvideo.mlx_runtime.rife_interp import interpolate_to_frame_count, load_model

            _reset_peak_memory()
            started = time.perf_counter()
            model = load_model(weights_dir=str(rife_weights_dir) if rife_weights_dir is not None else None)
            try:
                interpolated = interpolate_to_frame_count(
                    frames,
                    target_geometry["num_frames"],
                    model=model,
                )
                if spatial_plan is None:
                    interpolated = _sharpen_frames(interpolated, fast_sharpen)
                frames = np.stack(interpolated)
                if frames.shape[0] != target_geometry["num_frames"]:
                    raise RuntimeError(
                        f"RIFE produced {frames.shape[0]} frames, expected {target_geometry['num_frames']}.")
                del interpolated
                timings["rife_s"] = time.perf_counter() - started
                peaks["rife_gib"] = _peak_memory_gib()
            finally:
                load_model.cache_clear()
                del model
                _cleanup_mlx()

        if spatial_plan is not None:
            started = time.perf_counter()
            # One sharpen pass, at full resolution: RIFE and the resample soften
            # for the same reason, so the stronger requested amount is applied
            # once instead of stacking two unsharp masks.
            sharpen = spatial_plan.sharpen if fast_plan is None else max(spatial_plan.sharpen, fast_sharpen)
            frames = np.stack(
                upsample_frames(
                    frames,
                    width=spatial_plan.target_width,
                    height=spatial_plan.target_height,
                    mode=spatial_plan.upsample_mode,
                    sharpen=sharpen,
                ))
            timings["spatial_upsample_s"] = time.perf_counter() - started

        _reset_peak_memory()
        started = time.perf_counter()
        waveform = self.decode_audio(audio_rows, num_frames=target_geometry["num_frames"])
        timings["audio_decode_s"] = time.perf_counter() - started
        peaks["audio_decode_gib"] = _peak_memory_gib()
        _cleanup_mlx()

        started = time.perf_counter()
        video_path = self.mux(frames, waveform, output_path)
        timings["mux_s"] = time.perf_counter() - started
        timings["generate_s"] = sum(
            timings.get(key, 0.0) for key in ("decoder_prepare_s", "condition_s", "denoise_s", "video_decode_s",
                                              "rife_s", "spatial_upsample_s", "audio_decode_s", "mux_s"))

        result = GenerationResult(
            video_path=str(video_path),
            frames=frames if save_frames else None,
            waveform=waveform,
            sample_rate=32000,
            timings=timings,
            peak_memory_gib=peaks,
            video_decode_backend=self.video_decode_backend,
            vsa=getattr(self, "last_vsa_stats", None) or {
                "enabled": vsa_config.enabled,
                "checkpoint_vsa_capable": mlx_h3_checkpoint_vsa_capable(self.dit_checkpoint),
            },
        )
        logger.info("Generation complete: %s | timings=%s peaks=%s", video_path, timings, peaks)
        return result
