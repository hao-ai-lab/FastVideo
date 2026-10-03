# SPDX-License-Identifier: Apache-2.0
"""Text-to-video generation for Wan2.1 and Wan2.2-TI2V (FastMetal) through the
native MLX runtime.

Scoped to each family's validated cookbook path only: text-to-video,
DMD-distilled denoising, a packed MLX DiT checkpoint, and TAEHV decode.
Refine, fast-spatial, RIFE fast mode, and prompt enrichment stay in the CLI
scripts (examples/inference/basic/mlx_wan_prompt_to_video.py and
mlx_wan22_generate.py) -- this module holds only what a resident server needs
to call repeatedly.

Prompt encoding and rotary tables are shared with the CLI scripts through
wan_helpers; checkpoint loading, DMD scheduling, and decode use the same native
runtime helpers. MLXWanPipeline (Wan2.1: 1.3B/14B) and MLXWan22Pipeline (Wan2.2:
5B) share the UMT5 prompt encoder and rotary-embedding builder, since
that piece is identical across both families; everything DiT/VAE-shaped is
not, because the two are genuinely different architectures.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from fastvideo.mlx_runtime.checkpoint_compat import (
    UnsupportedMLXCheckpointError,
    raise_if_unsupported_mlx_checkpoint,
)
from fastvideo.mlx_runtime.memory import cleanup_mlx, cleanup_torch_mps
from fastvideo.mlx_runtime.generation_result import GenerationResult
from fastvideo.mlx_runtime.wan_helpers import (
    WAN_DMD_STEPS,
    WAN21_CHANNELS,
    WAN22_CHANNELS,
    encode_wan_prompt as _encode_wan_prompt,
    make_wan_rotary_embeddings as _make_wan_rotary_embeddings,
    plan_wan_generation,
)
from fastvideo.utils import init_logger

logger = init_logger(__name__)


def _peak_memory_gib() -> float:
    """Read MLX's peak-memory counter in GiB."""
    import mlx.core as mx

    getter = getattr(mx, "get_peak_memory", None)
    return 0.0 if getter is None else float(getter()) / 2**30


def _packed_dit_channels(mlx_checkpoint: Path) -> int | None:
    """Read in_channels from a packed mlx_dit.json, or None if unreadable.

    A best-effort check: an unpacked/diffusers-style or missing checkpoint is
    left for generate() to fail on when it actually loads the weights.
    """
    manifest_path = mlx_checkpoint / "mlx_dit.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(manifest, dict):
        return None
    config = manifest.get("config", manifest)
    if not isinstance(config, dict):
        return None
    channels = config.get("in_channels")
    if channels is None:
        return None
    try:
        return int(channels)
    except (TypeError, ValueError):
        return None


class MLXWanPipeline:
    """Text-to-video generation through the native MLX runtime (Wan2.1/FastMetal)."""

    def __init__(self,
                 *,
                 model_root: str | Path,
                 mlx_checkpoint: str | Path,
                 prompt_cache_dir: str | Path | None = "outputs/wan_prompt_cache") -> None:
        self.model_root = Path(model_root)
        self.mlx_checkpoint = Path(mlx_checkpoint)
        self.prompt_cache_dir = Path(prompt_cache_dir).expanduser() if prompt_cache_dir is not None else None
        try:
            raise_if_unsupported_mlx_checkpoint(self.mlx_checkpoint)
        except UnsupportedMLXCheckpointError as error:
            raise ValueError(str(error)) from error
        if not (self.model_root / "tokenizer").exists() or not (self.model_root / "text_encoder").exists():
            raise FileNotFoundError(f"Missing tokenizer/ or text_encoder/ under {self.model_root}.")
        channels = _packed_dit_channels(self.mlx_checkpoint)
        if channels == WAN22_CHANNELS:
            raise ValueError(f"{self.mlx_checkpoint} is a {channels}-channel Wan2.2-TI2V checkpoint "
                             "(e.g. FastMetal-5B-QAD); MLXWanPipeline only supports Wan2.1's "
                             f"{WAN21_CHANNELS}-channel checkpoints (1.3B/14B). Use MLXWan22Pipeline instead.")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str | Path,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        seed: int = 0,
        dmd_denoising_steps: tuple[int, ...] = WAN_DMD_STEPS,
        flow_shift: float = 8.0,
        fps: int = 16,
        max_sequence_length: int = 512,
    ) -> GenerationResult:
        plan = plan_wan_generation(height=height, width=width, num_frames=num_frames)
        import mlx.core as mx
        import torch

        from fastvideo.mlx_runtime.checkpoint import load_mlx_dit_checkpoint
        from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step
        from fastvideo.mlx_runtime.wan_vae import decode_latents_to_video
        from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

        timings: dict[str, float] = {}
        mx.random.seed(seed)

        started = time.perf_counter()
        prompt_embeds = _encode_wan_prompt(model_root=self.model_root,
                                           prompt=prompt,
                                           max_sequence_length=max_sequence_length,
                                           cache_dir=self.prompt_cache_dir)
        timings["encode_s"] = time.perf_counter() - started

        started = time.perf_counter()
        mx.clear_cache()
        mx.reset_peak_memory()
        dit = load_mlx_dit_checkpoint(self.mlx_checkpoint, compile=True)
        timings["load_s"] = time.perf_counter() - started
        timings["load_peak_gib"] = _peak_memory_gib()

        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        timesteps = torch.tensor(list(dmd_denoising_steps), dtype=torch.long)
        dmd_schedule = MLXDMDSchedule.from_torch_scheduler(scheduler)

        latents_seed = torch.Generator(device="cpu").manual_seed(seed)
        latents_torch = torch.randn(
            (1, int(
                dit.config["in_channels"]), plan.latent_frames, plan.stage1_latent_height, plan.stage1_latent_width),
            generator=latents_seed,
            dtype=torch.float32,
        )
        latents = mx.array(latents_torch.numpy()).astype(mx.float16)
        encoder_hidden_states = mx.array(prompt_embeds.numpy()).astype(mx.float16)
        freqs_cis = _make_wan_rotary_embeddings(
            dit.config,
            latent_frames=plan.latent_frames,
            latent_height=plan.stage1_latent_height,
            latent_width=plan.stage1_latent_width,
        )

        started = time.perf_counter()
        mx.reset_peak_memory()
        for step_index, timestep in enumerate(timesteps):
            timestep_mx = mx.array([float(timestep.item())]).astype(mx.float32)
            noise_pred = dit(latents.astype(mx.float16), encoder_hidden_states, timestep_mx, freqs_cis)

            is_last_step = step_index == len(timesteps) - 1
            next_timestep = None if is_last_step else float(timesteps[step_index + 1].item())
            renoise = None if is_last_step else mx.random.normal(latents.shape).astype(mx.float32)
            latents_f32 = latents.astype(mx.float32)
            latents = dmd_step(
                latents=latents_f32,
                noise_input_latent=latents_f32,
                pred_noise=noise_pred.astype(mx.float32),
                schedule=dmd_schedule,
                timestep=float(timestep.item()),
                next_timestep=next_timestep,
                noise=renoise,
            ).astype(mx.float16)
            mx.eval(latents)
            logger.info("Wan MLX denoise step %d/%d complete", step_index + 1, len(timesteps))
        timings["denoise_s"] = time.perf_counter() - started
        timings["denoise_peak_gib"] = _peak_memory_gib()

        latents_np = np.array(latents.astype(mx.float32))
        # Free the DiT before decode -- this is a resident server, but MLX's
        # unified memory still holds one heavyweight phase at a time, matching
        # the CLI script's proven memory behavior on this hardware class.
        del dit, latents, encoder_hidden_states, freqs_cis
        cleanup_mlx()

        started = time.perf_counter()
        output_path = Path(output_path)
        decode_latents_to_video(
            latents_np,
            output_path,
            fps=fps,
            backend="taehv",
            z_dim=latents_np.shape[1],
            taehv_checkpoint=None,
            torch_device="auto",
        )
        timings["decode_s"] = time.perf_counter() - started
        cleanup_torch_mps()

        return GenerationResult(video_path=str(output_path),
                                video_decode_backend="taehv",
                                timings=timings,
                                peak_memory_gib={
                                    k: v
                                    for k, v in timings.items() if k.endswith("_gib")
                                })


class MLXWan22Pipeline:
    """Text-to-video generation through the native MLX runtime (Wan2.2-TI2V/FastMetal-5B)."""

    def __init__(self,
                 *,
                 model_root: str | Path,
                 mlx_checkpoint: str | Path,
                 prompt_cache_dir: str | Path | None = "outputs/wan_prompt_cache") -> None:
        self.model_root = Path(model_root)
        self.mlx_checkpoint = Path(mlx_checkpoint)
        self.prompt_cache_dir = Path(prompt_cache_dir).expanduser() if prompt_cache_dir is not None else None
        try:
            raise_if_unsupported_mlx_checkpoint(self.mlx_checkpoint)
        except UnsupportedMLXCheckpointError as error:
            raise ValueError(str(error)) from error
        if not (self.model_root / "tokenizer").exists() or not (self.model_root / "text_encoder").exists():
            raise FileNotFoundError(f"Missing tokenizer/ or text_encoder/ under {self.model_root}.")
        channels = _packed_dit_channels(self.mlx_checkpoint)
        if channels is not None and channels != WAN22_CHANNELS:
            raise ValueError(f"{self.mlx_checkpoint} is a {channels}-channel checkpoint; MLXWan22Pipeline only "
                             f"supports Wan2.2-TI2V's {WAN22_CHANNELS}-channel checkpoints (FastMetal-5B-QAD). "
                             "Use MLXWanPipeline for 1.3B/14B.")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str | Path,
        # Defaults match the validated FastMetal-5B-QAD cookbook recipe, not
        # mlx_wan22_generate.py's own argparse defaults (448x832x121), which
        # were never the evidence-backed shape for this checkpoint.
        height: int = 704,
        width: int = 1280,
        num_frames: int = 81,
        seed: int = 1234,
        dmd_denoising_steps: tuple[int, ...] = WAN_DMD_STEPS,
        flow_shift: float = 5.0,
        fps: int = 24,
        max_sequence_length: int = 512,
    ) -> GenerationResult:
        plan = plan_wan_generation(height=height, width=width, num_frames=num_frames, wan22=True)
        import mlx.core as mx
        import torch

        from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_mlx_checkpoint
        from fastvideo.mlx_runtime.wan22_sample import sample_wan22_dmd
        from fastvideo.mlx_runtime.wan_vae import decode_latents_to_video

        timings: dict[str, float] = {}
        mx.random.seed(seed)

        started = time.perf_counter()
        prompt_embeds = _encode_wan_prompt(model_root=self.model_root,
                                           prompt=prompt,
                                           max_sequence_length=max_sequence_length,
                                           device_arg="cpu",
                                           dtype_arg="fp16",
                                           cache_dir=self.prompt_cache_dir)
        timings["encode_s"] = time.perf_counter() - started

        started = time.perf_counter()
        mx.clear_cache()
        mx.reset_peak_memory()
        dit = mlx_wan22_dit_from_mlx_checkpoint(self.mlx_checkpoint, compile=True)
        timings["load_s"] = time.perf_counter() - started
        timings["load_peak_gib"] = _peak_memory_gib()

        latents_seed = torch.Generator(device="cpu").manual_seed(seed)
        latents_torch = torch.randn(
            (1, int(
                dit.config["in_channels"]), plan.latent_frames, plan.stage1_latent_height, plan.stage1_latent_width),
            generator=latents_seed,
            dtype=torch.float32,
        )
        noise = mx.array(latents_torch.numpy()).astype(mx.float16)
        encoder_hidden_states = mx.array(prompt_embeds.numpy()).astype(mx.float16)
        freqs_cis = _make_wan_rotary_embeddings(
            dit.config,
            latent_frames=plan.latent_frames,
            latent_height=plan.stage1_latent_height,
            latent_width=plan.stage1_latent_width,
        )

        started = time.perf_counter()
        mx.reset_peak_memory()
        # sample_wan22_dmd's own re-noise seed defaults to 0 in the CLI script
        # (--renoise-seed), independent of --seed; matched here rather than
        # exposed as a second knob nobody overrides in the validated recipe.
        latents = sample_wan22_dmd(
            dit,
            encoder_hidden_states,
            noise,
            freqs_cis,
            dmd_denoising_steps=list(dmd_denoising_steps),
            flow_shift=flow_shift,
            warp_denoising_step=True,
            seed=0,
        )
        timings["denoise_s"] = time.perf_counter() - started
        timings["denoise_peak_gib"] = _peak_memory_gib()

        latents_np = np.array(latents.astype(mx.float32))
        # Free the DiT before decode, matching the CLI script's phase-memory
        # policy -- the 5B DiT and the decoder are not held resident together.
        del dit, latents, encoder_hidden_states, freqs_cis, noise
        cleanup_mlx()

        started = time.perf_counter()
        output_path = Path(output_path)
        decode_latents_to_video(
            latents_np,
            output_path,
            fps=fps,
            backend="taehv",
            z_dim=latents_np.shape[1],
            taehv_checkpoint=None,
            torch_device="auto",
        )
        timings["decode_s"] = time.perf_counter() - started
        cleanup_torch_mps()

        return GenerationResult(video_path=str(output_path),
                                video_decode_backend="taehv",
                                timings=timings,
                                peak_memory_gib={
                                    k: v
                                    for k, v in timings.items() if k.endswith("_gib")
                                })
