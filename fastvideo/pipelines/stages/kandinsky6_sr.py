# SPDX-License-Identifier: Apache-2.0
"""Stages of the Kandinsky6 video super-resolution (SR) pipeline.

    video encoding -> latent preparation -> denoising -> decoding

The source clip is encoded once.  Its latent is cut into overlapping tiles whose upscaled size is a resolution the SR
DiT was trained on (``basic/kandinsky6_sr/tiling.py``); every tile is upscaled by the latent upscaler, noised, denoised
with the bundle's scheduler (flow-matching Euler or the distilled pi-Flow) and decoded, and the decoded tiles are
blended into the output video.  The denoising loop runs per tile, which is why these stages replace the generic
latent-preparation / denoising / decoding stages.
"""
from __future__ import annotations

import contextlib
from typing import Any

import torch
from tqdm.auto import tqdm

from fastvideo.configs.pipelines.kandinsky6_sr_options import Kandinsky6SROptions
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader, UpsamplerLoader, VAELoader
from fastvideo.pipelines.basic.kandinsky6_sr import sr_io
from fastvideo.pipelines.basic.kandinsky6_sr.tiling import (TileGrid, plan_tiles, pre_upscale, resolve_scale,
                                                            stitch_tiles)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Optional request inputs routed into ``batch.extra`` (``REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS``): a raw KVAE latent
# of the source instead of ``video_path``, and an audio track to mux instead of the source's own.
_LR_LATENT_KEY = "sr_lr_latent"
_AUDIO_KEY = "sr_audio"
_AUDIO_RATE_KEY = "sr_audio_sample_rate"


def _resolve(stage: PipelineStage, name: str, fastvideo_args: FastVideoArgs, loader: Any) -> Any:
    """Reload a component that was released after a previous request."""
    if not fastvideo_args.model_loaded.get(name, True):
        setattr(stage, name, loader.load(fastvideo_args.model_paths[name], fastvideo_args))
        fastvideo_args.model_loaded[name] = True
    return getattr(stage, name)


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _tile_grid(batch: ForwardBatch, transformer: torch.nn.Module, vae: torch.nn.Module) -> tuple[TileGrid, int]:
    """Pixel tile grid of the encoded source (``batch.lq_latents``) and the integer tiling scale."""
    sr_options = Kandinsky6SROptions.from_extra(batch.extra)
    scale, _ = resolve_scale(float(sr_options.sr_resolution_scale))
    spatial_factor = vae.spatial_factor
    _, _, height, width = batch.lq_latents.shape
    grid = plan_tiles(height * spatial_factor, width * spatial_factor, transformer.config.arch_config.visual_size,
                      scale, float(sr_options.sr_tile_min_overlap), spatial_factor)
    return grid, scale


class Kandinsky6SRVideoEncodingStage(PipelineStage):
    """Read the source video (and its audio), apply the x2.25 pre-upscale and KVAE-encode it into ``lq_latents``."""

    def __init__(self, vae: Any) -> None:
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sr_options = Kandinsky6SROptions.from_extra(batch.extra)
        _, pre_upscale_factor = resolve_scale(float(sr_options.sr_resolution_scale))
        sr_io.validate_target_spec(sr_options.sr_target_resolution, sr_options.sr_target_resize_mode)
        if isinstance(batch.video_path, list):
            raise ValueError("Kandinsky6 SR processes one video per request; pass a single video_path.")
        vae = _resolve(self, "vae", fastvideo_args, VAELoader())

        lr_latent = batch.extra.pop(_LR_LATENT_KEY, None)
        if lr_latent is not None:
            if batch.video_path is not None:
                raise ValueError("Kandinsky6 SR: pass either video_path or sr_lr_latent, not both.")
            if pre_upscale_factor != 1.0:
                raise ValueError("Kandinsky6 SR: sr_resolution_scale=2.25 needs a pixel pre-upscale and cannot start "
                                 "from sr_lr_latent; use scale 2 or 4, or pass video_path.")
            if not isinstance(lr_latent, torch.Tensor) or lr_latent.ndim != 4:
                raise ValueError("sr_lr_latent must be a raw (unscaled) KVAE latent tensor [T, C, H, W].")
            batch.lq_latents = lr_latent.float()
            fps = sr_io.TARGET_FPS
        else:
            video, fps = self._read_source(batch, pre_upscale_factor, vae.spatial_factor)
            batch.lq_latents = self._encode(video, vae, fastvideo_args)

        audio = batch.extra.pop(_AUDIO_KEY, None)
        audio_rate = batch.extra.pop(_AUDIO_RATE_KEY, sr_io.AUDIO_SAMPLE_RATE)
        if audio is None and batch.video_path is not None:
            audio = sr_io.read_audio(batch.video_path, int(batch.num_frames), fps)
            audio_rate = sr_io.AUDIO_SAMPLE_RATE
        if audio is not None:
            batch.extra["audio"] = audio
            batch.extra["audio_sample_rate"] = int(audio_rate)
        # ``batch`` is the worker-side copy; the frame rate the mp4 must use travels back through ``extra``.
        batch.fps = fps
        batch.extra["output_fps"] = fps
        return batch

    def _read_source(self, batch: ForwardBatch, factor: float, spatial_factor: int) -> tuple[torch.Tensor, int]:
        sr_options = Kandinsky6SROptions.from_extra(batch.extra)
        if batch.video_path is None:
            if batch.save_video or batch.return_frames:
                raise ValueError("Kandinsky6 SR is a video-to-video pipeline: pass the source clip via video_path.")
            # Warm-up / health-check request: one tile of deterministic noise.
            generator = torch.Generator().manual_seed(0)
            video, fps = torch.randint(0, 256, (9, 3, 256, 384), dtype=torch.uint8, generator=generator), 24
        else:
            video, fps = sr_io.read_video(batch.video_path)
        if factor != 1.0:
            video = pre_upscale(video, factor, spatial_factor)
        batch.num_frames, batch.height, batch.width = video.shape[0], video.shape[2], video.shape[3]
        logger.info("Kandinsky6 SR input: %d frames %dx%d @ %d fps -> x%s", video.shape[0], video.shape[3],
                    video.shape[2], fps, sr_options.sr_resolution_scale)
        return video, fps

    @staticmethod
    def _encode(video: torch.Tensor, vae: torch.nn.Module, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """``[T, C, H, W]`` uint8 -> raw ``[T', C', h, w]`` fp32 latent (the latent upscaler applies the scaling)."""
        device = get_local_torch_device()
        vae.to(device)
        pixels = vae.normalize_data(video.permute(1, 0, 2, 3).unsqueeze(0).to(device).float())
        latent = vae.encode(pixels.to(_module_dtype(vae)))[0]
        if fastvideo_args.vae_cpu_offload:
            vae.to("cpu")
        return latent.squeeze(0).permute(1, 0, 2, 3).float()

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("lq_latents", batch.lq_latents, [V.is_tensor, V.with_dims(4)])
        return result


class Kandinsky6SRLatentPreparationStage(PipelineStage):
    """Upscale every latent tile and build the SR DiT input ``[noised upscaled latent | anchor | anchor mask]``.

    The released checkpoints are trained with an HR anchor channel group; SR of a real low-quality clip has no anchor,
    so it is zero with a zero mask.  The starting point is the upscaled latent mixed with Gaussian noise
    (variance-preserving, ``lq_noise_scale``).  Noise is drawn per group of ``sr_tiles_batch_size`` tiles from a
    generator seeded with ``seed + first tile index``, so results do not depend on how many tile groups ran before.
    Output: ``batch.latents`` ``[num_tiles, T', H, W, 2C + 1]`` fp32.
    """

    def __init__(self, latent_upscaler: Any, vae: Any, transformer: Any) -> None:
        super().__init__()
        self.latent_upscaler = latent_upscaler
        self.vae = vae
        self.transformer = transformer

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sr_options = Kandinsky6SROptions.from_extra(batch.extra)
        upscaler = _resolve(self, "latent_upscaler", fastvideo_args, UpsamplerLoader())
        grid, scale = _tile_grid(batch, self.transformer, self.vae)
        if scale not in upscaler.scales:
            # The diffusers reference falls back to a pixel-space path here (bilinearly upscale the
            # pixel tile to the base resolution, then VAE-encode it directly) when its latent_upscaler
            # doesn't cover the requested scale. FastVideo makes latent_upscaler mandatory instead and
            # fails fast: by this stage the pipeline only carries the already-KVAE-encoded
            # batch.lq_latents (Kandinsky6SRVideoEncodingStage encodes the whole clip upfront, before
            # tiling), not the source pixel video, so matching the reference exactly would mean
            # threading the pixel video through additional pipeline state all the way to this later
            # stage -- a real architectural change for a path no released checkpoint's scales/tile_sizes
            # combination is known to hit (every supported SR bundle ships a complete upscaler bank).
            raise ValueError(f"The latent upscaler has no x{scale} model (available: {upscaler.scales}).")
        arch = self.transformer.config.arch_config
        device = get_local_torch_device()
        upscaler.to(device)
        upscaler_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.upsampler_precision]
        autocast = (torch.autocast("cuda", dtype=upscaler_dtype) if device.type == "cuda" else contextlib.nullcontext())
        scaling_factor = float(self.vae.scaling_factor)

        tiles = grid.to_latent(self.vae.spatial_factor).extract(batch.lq_latents)
        group = int(sr_options.sr_tiles_batch_size)
        if group < 1:
            raise ValueError(f"sr_tiles_batch_size must be >= 1, got {group}")
        prepared = []
        for start in range(0, len(tiles), group):
            upscaled = []
            for tile in tiles[start:start + group]:
                z = tile.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=_module_dtype(upscaler))
                with autocast:
                    out = upscaler(z * scaling_factor, scale)
                upscaled.append(out.squeeze(0).permute(1, 2, 3, 0).float())  # [T', H, W, C]
            prepared.append(self._initial_latent(torch.stack(upscaled), int(batch.seed) + start, arch.lq_noise_scale))
        batch.latents = torch.cat(prepared)
        if fastvideo_args.vae_cpu_offload:
            upscaler.to("cpu")
        return batch

    @staticmethod
    def _initial_latent(upscaled: torch.Tensor, seed: int, noise_scale: float) -> torch.Tensor:
        # One draw for the whole tile group, in the ``[group * T', H, W, C]`` shape the model was trained with.
        flat = upscaled.flatten(0, 1)
        generator = torch.Generator(device=flat.device).manual_seed(seed)
        noise = torch.randn(flat.shape, device=flat.device, dtype=flat.dtype, generator=generator)
        start = (1 - noise_scale**2)**0.5 * flat + noise_scale * noise
        anchor_and_mask = torch.zeros((*flat.shape[:-1], flat.shape[-1] + 1), device=flat.device, dtype=flat.dtype)
        return torch.cat([start, anchor_and_mask], dim=-1).unflatten(0, upscaled.shape[:2])

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result


class Kandinsky6SRDenoisingStage(PipelineStage):
    """Denoise every tile with the bundle's scheduler; ``batch.latents`` becomes ``[num_tiles, T', H, W, C]``.

    ``FlowMatchEulerDiscreteScheduler`` bundles run ``num_inference_steps`` Euler steps over the shifted
    ``linspace(1, 0)`` grid; ``PiflowScheduler`` bundles integrate the distilled policy (the DiT head then holds
    ``n_grid`` predictions per channel).  The latent state stays fp32 across steps.  Only the first ``C`` channels are
    denoised; the anchor channels are fixed conditioning.

    ``num_inference_steps`` is the number of DiT calls per tile for both schedulers, as elsewhere in FastVideo.  The
    upstream Diffusers pipeline counts timestep grid points instead, so its default 5 is 4 steps here.
    """

    def __init__(self, transformer: Any, scheduler: Any) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sr_options = Kandinsky6SROptions.from_extra(batch.extra)
        transformer = _resolve(self, "transformer", fastvideo_args, TransformerLoader())
        arch = transformer.config.arch_config
        device = get_local_torch_device()
        if not fastvideo_args.dit_layerwise_offload:
            transformer.to(device)
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        autocast_enabled = device.type == "cuda" and target_dtype != torch.float32 and not fastvideo_args.disable_autocast
        use_piflow = bool(getattr(self.scheduler, "is_piflow", False))
        channels = arch.in_visual_dim
        num_steps = int(batch.num_inference_steps)
        self._check_scheduler_fits(arch, use_piflow, num_steps)

        latents = batch.latents
        num_tiles, duration, height, width, _ = latents.shape
        rope_pos = [
            torch.arange(duration, device=device),
            torch.arange(height // arch.patch_size[1], device=device),
            torch.arange(width // arch.patch_size[2], device=device),
        ]
        group = int(sr_options.sr_tiles_batch_size)
        denoised = []
        with tqdm(total=num_tiles * num_steps, desc="Kandinsky6 SR denoising [tiles x steps]") as progress_bar:
            for start in range(0, num_tiles, group):
                x = latents[start:start + group].to(device)
                state, cond = x[..., :channels], x[..., channels:]
                self._set_timesteps(num_steps, device, use_piflow)
                for i, t in enumerate(self.scheduler.timesteps):
                    # Stop if interrupted
                    if getattr(self, "interrupt", False):
                        break
                    timestep = t.to(device=device, dtype=torch.float32).expand(x.shape[0])
                    autocast = (torch.autocast("cuda", dtype=target_dtype)
                                if autocast_enabled else contextlib.nullcontext())
                    with set_forward_context(current_timestep=i, attn_metadata=None, forward_batch=batch), autocast:
                        prediction = transformer(hidden_states=torch.cat([state, cond], dim=-1),
                                                 timestep=timestep,
                                                 visual_rope_pos=rope_pos,
                                                 scale_factor=arch.rope_scale_factor)
                    if not use_piflow:
                        # The Euler step casts its result to the prediction dtype; keep the state fp32.
                        prediction = prediction.float()
                    state = self.scheduler.step(prediction, t, state, return_dict=False)[0]
                    progress_bar.update(x.shape[0])
                denoised.append(state.float())
        batch.latents = torch.cat(denoised)
        self._offload(transformer, fastvideo_args)
        return batch

    def _check_scheduler_fits(self, arch: Any, use_piflow: bool, num_steps: int) -> None:
        """Reject a transformer / scheduler pair from different bundles before the first DiT call.

        The flow-matching DiT predicts ``in_visual_dim`` channels; the distilled one ``n_grid * in_visual_dim``, which
        only ``PiflowScheduler`` can integrate.  Mixing them would otherwise fail on a shape error mid-run.
        """
        if num_steps < 1:
            raise ValueError(f"Kandinsky6 SR: num_inference_steps must be >= 1, got {num_steps}")
        scheduler_name = type(self.scheduler).__name__
        expected = arch.in_visual_dim * (int(self.scheduler.n_grid) if use_piflow else 1)
        if arch.out_visual_dim != expected:
            raise ValueError(
                f"Kandinsky6 SR: the transformer head is {arch.out_visual_dim} channels wide, but {scheduler_name} "
                f"needs {expected} (in_visual_dim={arch.in_visual_dim}). The transformer and scheduler come from "
                "different bundles: the flow-matching transformer needs FlowMatchEulerDiscreteScheduler, the distilled "
                "one PiflowScheduler with the n_grid it was trained with.")
        nfe = getattr(self.scheduler.config, "nfe", None) if use_piflow else None
        if nfe is not None and num_steps != int(nfe):
            logger.warning(
                "Kandinsky6 SR: running %d pi-Flow steps per tile; this checkpoint was distilled for nfe=%d.",
                num_steps, int(nfe))

    def _set_timesteps(self, num_steps: int, device: torch.device, use_piflow: bool) -> None:
        if use_piflow:
            self.scheduler.set_timesteps(num_steps, device=device)
        else:
            # Uniform grid from pure noise; the scheduler applies its shift and appends the terminal 0.
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1)[:-1].tolist()
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)

    @staticmethod
    def _offload(transformer: torch.nn.Module, fastvideo_args: FastVideoArgs) -> None:
        if fastvideo_args.dit_layerwise_offload:
            manager = getattr(transformer, "_layerwise_offload_manager", None)
            if manager is not None and getattr(manager, "enabled", False):
                manager.release_all()
        elif fastvideo_args.dit_cpu_offload and not fastvideo_args.use_fsdp_inference:
            transformer.to("cpu")

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result


class Kandinsky6SRDecodingStage(PipelineStage):
    """KVAE-decode every tile, blend the tiles and apply the optional delivery resize.

    Tiles stay float through blending and are quantized to uint8 once, inside ``stitch_tiles`` --
    matching the diffusers reference's ``decode_latents``/``__call__`` (it accumulates
    ``video_acc += tile * window`` in float and only rounds the final ``video_acc / weight_acc``), not
    the lossier quantize-per-tile-then-blend-already-rounded-values approach.  ``batch.output`` stores
    code ``v`` as ``(v + 0.5) / 255`` so the truncating ``* 255 -> uint8`` conversion in
    ``VideoGenerator`` gives back exactly ``v``.
    """

    def __init__(self, vae: Any, transformer: Any) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sr_options = Kandinsky6SROptions.from_extra(batch.extra)
        vae = _resolve(self, "vae", fastvideo_args, VAELoader())
        grid, scale = _tile_grid(batch, self.transformer, vae)
        device = get_local_torch_device()
        vae.to(device)
        scaling_factor = float(vae.scaling_factor)
        tiles = []
        for tile_latent in batch.latents:
            z = (tile_latent.to(device) / scaling_factor).permute(3, 0, 1, 2).unsqueeze(0)
            # Upcast before the pixel-range rescale/clamp (matches the diffusers reference's decode_latents,
            # which does the same right after vae.decode): decoded stays in the VAE's compute dtype otherwise,
            # right before the uint8 quantization below.
            decoded = vae.decode(z.to(_module_dtype(vae))).sample.float()
            pixels = (vae.denormalize_data(decoded) / 255.0).clamp(0.0, 1.0)
            tiles.append((pixels * 255.0)[0].cpu())
        if fastvideo_args.vae_cpu_offload:
            vae.to("cpu")

        _, _, height, width = batch.lq_latents.shape
        spatial_factor = vae.spatial_factor
        video = stitch_tiles(tiles, grid, height * spatial_factor, width * spatial_factor, scale)
        target_hw = sr_io.resolve_target_hw(sr_options.sr_target_resolution, tuple(video.shape[-2:]),
                                            sr_options.sr_target_resize_mode)
        if target_hw is not None:
            video = sr_io.resize_video(video, target_hw)

        batch.output = video.float().unsqueeze(0).add_(0.5).div_(255.0).clamp_(max=1.0)
        batch.num_frames, batch.height, batch.width = video.shape[1], video.shape[2], video.shape[3]
        batch.latents = None
        batch.lq_latents = None
        return batch

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        return result
