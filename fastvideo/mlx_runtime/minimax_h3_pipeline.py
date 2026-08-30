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
from fastvideo.mlx_runtime.minimax_h3 import (
    H3_MANIFEST_FILENAME,
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3SchedulerState,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_layout,
    build_row_timesteps,
    dense_only_vsa_error,
    load_mlx_h3_checkpoint,
    mlx_h3_checkpoint_vsa_capable,
    temporal_position_grid,
    unpatchify_video_tokens,
    unpack_audio_tokens,
    video_latent_num_frames,
)
from fastvideo.mlx_runtime.minimax_h3_vsa import MiniMaxH3VSAConfig

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
    """Keep the default below both physical memory and the tested 30 GiB cap."""
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
    return np.unique(np.concatenate([video.timesteps, audio.timesteps, [1.0]]).astype(np.float32))


def _validate_checkpoint_step_ladder(checkpoint_dir: str | Path, num_steps: int) -> None:
    """Reject a schedule that a fixed, AdaLN-dropped checkpoint cannot serve."""
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / H3_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing MLX H3 checkpoint manifest: {manifest_path}")
    cache_info = json.loads(manifest_path.read_text()).get("adaln_cache")
    if cache_info is None:
        return
    cached = np.asarray(cache_info["timesteps"], dtype=np.float32)
    requested = _adaln_schedule_union(num_steps)
    if not np.array_equal(cached, requested):
        raise ValueError(
            f"MLX H3 checkpoint {checkpoint_dir} has a fixed AdaLN ladder that does not support --steps "
            f"{num_steps}. Use the step count used during conversion (normally 4), or re-export the checkpoint.")


def _preflight_media_dependencies(*, fast: bool, fast_sharpen: float, rife_weights_dir: str | Path | None) -> None:
    """Fail before conditioning when required output dependencies are unavailable."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for MP4 muxing; install it before generation.")
    if not fast:
        return
    if fast_sharpen > 0 and importlib.util.find_spec("cv2") is None:
        raise RuntimeError("OpenCV is required when --fast-sharpen is greater than zero.")
    from fastvideo.mlx_runtime.rife_interp import ensure_weights_available

    ensure_weights_available(weights_dir=str(rife_weights_dir) if rife_weights_dir is not None else None)


class MiniMaxH3MLXPipeline:
    """Text-to-video-with-audio generation through the native MLX runtime."""

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
    ) -> None:
        import mlx.core as mx

        set_limit = getattr(mx, "set_memory_limit", None)
        if set_limit is None and hasattr(mx, "metal"):
            set_limit = getattr(mx.metal, "set_memory_limit", None)
        if set_limit is not None:
            # Keep large resident models inside a predictable wired budget.
            try:
                if metal_wired_limit_gib is None:
                    metal_wired_limit_gib = _default_metal_wired_limit_gib(mx)
                set_limit(int(metal_wired_limit_gib * 2**30))
            except Exception as error:  # noqa: BLE001 - best effort on older MLX
                logger.info("Could not raise the Metal wired limit: %s", error)
        self.model_root = Path(model_root)
        self.dit_checkpoint = Path(mlx_dit_checkpoint)
        self.vae_dtype = vae_dtype
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

    # -- input validation (before anything heavy loads) -------------------

    def _validate_inputs_before_loading(self) -> bool:
        missing = []
        if not self.dit_checkpoint.exists():
            missing.append(str(self.dit_checkpoint))
        vae_dir = self.model_root / "vae"
        audio_dir = self.model_root / "audio_vae"
        if not (vae_dir.exists() and any(vae_dir.glob("*.safetensors"))):
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
        if enforce_duration and not 5.0 <= duration <= 15.0:
            raise ValueError(f"H3 generates 5-15 s at {MINIMAX_H3_FPS} fps; {aligned_frames} frames "
                             f"is {duration:.2f} s.")
        return {
            "height": height,
            "width": width,
            "num_frames": aligned_frames,
            "latent_frame_count": latent_frames,
            "latent_height": height // 16,
            "latent_width": width // 16,
        }

    # -- phase 1: conditioning -------------------------------------------

    def encode_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (hidden states (S, hidden), token tags). Uses the cache or
        the streamed conditioner."""
        cache_key = None
        if self.prompt_cache_dir is not None:
            cache_key = prompt_cache_path(self.prompt_cache_dir, self.model_root, prompt)
            if cache_key.exists():
                data = np.load(cache_key)
                logger.info("Loaded prompt embeddings from cache %s", cache_key)
                return data["hidden_states"], data["token_tags"]

        conditioner = self._load_conditioner()
        hidden, tags = conditioner.encode_prompt(prompt)
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

    def load_prompt_cache(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        return data["hidden_states"], data["token_tags"]

    def has_conditioner_weights(self) -> bool:
        marker = self.conditioner_dir / "model.safetensors.index.json"
        single = self.conditioner_dir / "model.safetensors"
        return marker.exists() or single.exists()

    def _load_conditioner(self):
        from fastvideo.mlx_runtime.minimax_h3_conditioner import StreamedMiniMaxH3TextConditioner

        return StreamedMiniMaxH3TextConditioner(self.conditioner_dir, self.tokenizer_dir)

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Denoise joint latents; returns (normalized video rows, audio rows)."""
        import mlx.core as mx

        geometry = self.resolve_geometry(height, width, num_frames, enforce_duration=audio_num_frames is None)
        audio_frames = geometry["num_frames"] if audio_num_frames is None else align_num_frames(audio_num_frames)

        owned_dit = dit is None
        if owned_dit:
            _validate_checkpoint_step_ladder(self.dit_checkpoint, num_steps)
            t0 = time.perf_counter()
            dit = load_mlx_h3_checkpoint(self.dit_checkpoint)
            logger.info("Loaded MLX H3 DiT from %s in %.1fs", self.dit_checkpoint, time.perf_counter() - t0)
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

        video_scheduler = MiniMaxH3SchedulerState.create(MINIMAX_H3_VIDEO_SHIFT, num_steps)
        audio_scheduler = MiniMaxH3SchedulerState.create(MINIMAX_H3_AUDIO_SHIFT, num_steps)
        # The released artifacts persist the converter grid: video ∪ audio ∪ {1.0}.
        union = _adaln_schedule_union(num_steps)
        # The keyframe-noise timestep (0.999) is only exercised by FL2VA/Ref2VA
        # conditioning rows; those modes recompute the ladder before denoise.

        cache = getattr(dit, "_adaln_cache", None)
        if cache is None:
            dit.precompute_adaln(union, drop_weights=True)
        elif not np.array_equal(cache.timesteps.astype(np.float32), union):
            extra = np.setdiff1d(union, cache.timesteps)
            logger.info("Recomputing AdaLN cache for %d-step ladder (extra timesteps %s).", num_steps, extra)
            dit.precompute_adaln(union, drop_weights=True)

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
            dit_forward_s += time.perf_counter() - step_started

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
                     tiled: bool = True) -> np.ndarray:
        """Normalized packed rows -> (T, H, W, 3) uint8 frames."""
        import mlx.core as mx

        from fastvideo.mlx_runtime.minimax_h3_video_vae import mlx_h3_video_vae_from_dir

        geometry = self.resolve_geometry(height, width, num_frames, enforce_duration=False)
        vae = mlx_h3_video_vae_from_dir(self.model_root / "vae", include_encoder=False, storage_dtype=self.vae_dtype)
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
                             tile_sample_min_height=min(geometry["height"], 256),
                             tile_sample_min_width=min(geometry["width"], 256))
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
            fast: bool = False,
            fast_factor: int = 2,
            fast_sharpen: float = 0.6,
            rife_weights_dir: str | Path | None = None,
            vsa: bool = False,
            vsa_sparsity: float = 0.9,
            vsa_tile_size: int = 64,
            vsa_prefix_mode: str = "exempt",
            vsa_dense_first_n_steps: int = 0,
            vsa_dense_layers: tuple[int, ...] = (),
            vsa_impl: str = "auto",
    ) -> GenerationResult:
        timings: dict[str, float] = {}
        peaks: dict[str, float] = {}

        if fast_sharpen < 0:
            raise ValueError(f"fast_sharpen must be non-negative, got {fast_sharpen}.")
        _validate_checkpoint_step_ladder(self.dit_checkpoint, num_steps)
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
        _preflight_media_dependencies(
            fast=fast,
            fast_sharpen=fast_sharpen,
            rife_weights_dir=rife_weights_dir,
        )
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
            "Geometry: output=%dx%dx%d model=%dx%dx%d audio_frames=%d fast=%s",
            width,
            height,
            target_geometry["num_frames"],
            canvas_width,
            canvas_height,
            video_geometry["num_frames"],
            target_geometry["num_frames"],
            fast_plan,
        )

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
        )
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
            timings.get(key, 0.0)
            for key in ("condition_s", "denoise_s", "video_decode_s", "rife_s", "audio_decode_s", "mux_s"))

        result = GenerationResult(
            video_path=str(video_path),
            frames=frames if save_frames else None,
            waveform=waveform,
            sample_rate=32000,
            timings=timings,
            peak_memory_gib=peaks,
            vsa=getattr(self, "last_vsa_stats", None) or {
                "enabled": vsa_config.enabled,
                "checkpoint_vsa_capable": mlx_h3_checkpoint_vsa_capable(self.dit_checkpoint),
            },
        )
        logger.info("Generation complete: %s | timings=%s peaks=%s", video_path, timings, peaks)
        return result
