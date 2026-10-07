# SPDX-License-Identifier: Apache-2.0
"""Tiny random Kandinsky6 SR components and an official-Diffusers-layout SR repo writer (CPU tests only).

A 5-level KVAE (spatial x16, temporal x4) with 4 latent channels, a 2-block text-free DiT and a multi-scale
latent-upscaler bank, so tiles of 64x64 / 64x96 / 96x64 pixels are 4x4 / 4x6 / 6x4 latents (2x2 / 2x3 / 3x2 DiT tokens).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from k6_sr_release import (DISTILLED_MODEL_INDEX, FLOW_MATCHING_MODEL_INDEX, FLOW_MATCHING_SCHEDULER_CONFIG,
                           FLOW_MATCHING_SR_CONFIG)
from safetensors.torch import save_file

# Same code-path knobs as the released vae/config.json, tiny widths.
TINY_KVAE_ENC = dict(ch=8, ch_mult=[1, 1, 2, 2, 2], num_res_blocks=2, in_channels=3, z_channels=4, double_z=True,
                     temporal_compress_times=4, temporal_compress_start_level=1, norm_type="rms_norm",
                     padding_mode="zeros", downsample_version=2, fix_pxs=True)
TINY_KVAE_DEC = dict(ch=8, out_ch=3, ch_mult=[1, 1, 2, 2, 2], num_res_blocks=2, z_channels=4,
                     temporal_compress_times=4, temporal_compress_start_level=1, norm_type="rms_norm",
                     padding_mode="zeros")
TINY_SCALING_FACTOR = 0.5
TINY_SR_PARAMS = dict(scale_factor={"512": [1.0, 1.0, 1.0]}, visual_size=[512], scheduler_scale=5.0,
                      lq_noise_scale=0.7, lq_noise_type="ddpm", lq_channel_noise_scale=0.0, cap_noise_timestep=False,
                      fps=24)
TINY_DIT_CFG: dict[str, Any] = dict(in_visual_dim=4, in_text_dim=8, in_text_dim2=8, time_dim=16, out_visual_dim=4,
                                    patch_size=[1, 2, 2], model_dim=32, ff_dim=64, num_text_blocks=0,
                                    num_visual_blocks=2, axes_dims=[8, 4, 4], visual_cond=True,
                                    instruct_type="hybrid_anchor", attention_params={"512": {"type": "flash"}},
                                    use_text=False, attribute_overrides={}, sr_params=TINY_SR_PARAMS)
TINY_PIFLOW = dict(nfe=2, num_policy_substeps=8, final_step_size_scale=0.5, shift=5.0, n_grid=3, eps=1e-6)
# The released latent-upscaler architecture at toy widths (stage widths narrow 8 -> 8 -> 4 like the release's
# 2048 -> 1024 -> 512, so the narrowing shortcut convs are exercised).
TINY_LU_MODEL = {"architecture": "multi_scale", "in_channels": 4, "hidden_channels": 8, "stage_channels": [8, 8, 4],
                 "num_pre_blocks": 2, "num_mid_blocks": 1, "num_post_blocks": 2, "expand_ratio": 1, "kernel_size": 3,
                 "dims": 3, "temporal_padding": "replicate", "upsample_mode": "pxs_v2", "upsample_padding_mode": "zeros",
                 "modulated_norm": True, "modulated_output_proj": True, "bare_stem": True, "input_skip": False,
                 "temporal_mix": False, "enable_x2_entry": True, "x2_adapter_blocks": 2, "x2_adapter_sources": [0, 1],
                 "x2_finisher": "pxs_residual", "x2_tail_mode": "private_full"}
TINY_RESOLUTIONS = {512: [(64, 64), (64, 96), (96, 64)]}


def randomize(module: torch.nn.Module, seed: int, std: float = 0.05) -> torch.nn.Module:
    """Deterministic non-trivial weights (many reference inits are zero, which would hide layout bugs)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
    return module


def dit_config_dict(piflow: dict | None = TINY_PIFLOW, dit_cfg: dict | None = None) -> dict[str, Any]:
    """``transformer/config.json`` of the tiny DiT; a pi-Flow head is ``n_grid`` times ``in_visual_dim`` wide."""
    cfg = copy.deepcopy(dit_cfg or TINY_DIT_CFG)
    if piflow is not None:
        cfg["out_visual_dim"] = cfg["in_visual_dim"] * int(piflow["n_grid"])
    return cfg


def build_dit(piflow: dict | None = TINY_PIFLOW, seed: int = 0, dit_cfg: dict | None = None):
    from fastvideo.configs.models.dits.kandinsky6_sr import Kandinsky6SRConfig
    from fastvideo.models.dits.kandinsky6_sr import Kandinsky6SRTransformer3DModel

    cfg = dit_config_dict(piflow, dit_cfg)
    config = Kandinsky6SRConfig()
    config.update_model_arch(cfg)
    return randomize(Kandinsky6SRTransformer3DModel(config, cfg), seed + 2).eval()


def vae_config_dict() -> dict[str, Any]:
    return dict(vae_type="video-kvae", encoder_config=copy.deepcopy(TINY_KVAE_ENC),
                decoder_config=copy.deepcopy(TINY_KVAE_DEC), scaling_factor=TINY_SCALING_FACTOR, spatial_factor=16,
                temporal_factor=4)


def build_vae(seed: int = 0):
    from fastvideo.configs.models.vaes.kandinsky6_sr import Kandinsky6SRVAEConfig
    from fastvideo.models.vaes.kandinsky6_sr import Kandinsky6SRVAE

    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(vae_config_dict())
    return randomize(Kandinsky6SRVAE(config), seed + 1, std=0.1).eval()


def lu_config_dict(scales=("2x", "4x")) -> dict[str, Any]:
    """``latent_upscaler/config.json``. The release config has no ``scales`` field (the bank defaults to (2, 4), which
    fixes the ``_models.<index>`` order); any other bank must declare its scales to match its ``models``."""
    config = {"models": [{"target_scale": s, "model": copy.deepcopy(TINY_LU_MODEL)} for s in scales],
              "scaling_factor": TINY_SCALING_FACTOR}
    if sorted(int(s.removesuffix("x")) for s in scales) != [2, 4]:
        config["scales"] = [int(s.removesuffix("x")) for s in scales]
    return config


def build_lu(scales=("2x", "4x"), seed: int = 0):
    from fastvideo.configs.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerConfig
    from fastvideo.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerBank

    config = Kandinsky6SRLatentUpscalerConfig(**lu_config_dict(scales))
    return randomize(Kandinsky6SRLatentUpscalerBank(config), seed + 10).eval()


def to_official_dit_keys(dit_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """FastVideo DiT keys -> official Diffusers keys: the inverse of ``Kandinsky6SRArchConfig.param_names_mapping``
    (``feed_forward.mlp.fc_in/fc_out`` -> ``feed_forward.net.0.proj/net.2``,
    ``time_embeddings.in/out_layer`` -> ``time_embeddings.timestep_embedder.linear_1/2``)."""
    out = {}
    for key, value in dit_state.items():
        key = key.replace("feed_forward.mlp.fc_in.", "feed_forward.net.0.proj.")
        key = key.replace("feed_forward.mlp.fc_out.", "feed_forward.net.2.")
        key = key.replace("time_embeddings.in_layer.", "time_embeddings.timestep_embedder.linear_1.")
        key = key.replace("time_embeddings.out_layer.", "time_embeddings.timestep_embedder.linear_2.")
        out[key] = value.detach().clone().contiguous()
    return out


def scheduler_config_dict(piflow: dict = TINY_PIFLOW) -> dict[str, Any]:
    """``scheduler/scheduler_config.json`` of a distilled bundle (``PiflowScheduler``)."""
    return {"_class_name": "PiflowScheduler", "_diffusers_version": "0.41.0.dev0", "eps": piflow["eps"],
            "final_step_size_scale": piflow["final_step_size_scale"], "n_grid": piflow["n_grid"], "nfe": piflow["nfe"],
            "num_policy_substeps": piflow["num_policy_substeps"], "num_train_timesteps": 1000,
            "shift": piflow["shift"]}


def flow_matching_scheduler_config_dict() -> dict[str, Any]:
    """``scheduler/scheduler_config.json`` of the flow-matching bundle (the real one: shift 5.0)."""
    return copy.deepcopy(FLOW_MATCHING_SCHEDULER_CONFIG)


def write_official_sr_repo(root: Path, *, piflow: dict | None = TINY_PIFLOW, dit_cfg: dict | None = None,
                           seed: int = 0) -> Path:
    """Write a tiny repo in the layout of an official ``Kandinsky6SRPipeline`` bundle:
    ``transformer/ vae/ latent_upscaler/ scheduler/`` with unprefixed weight keys.

    ``piflow`` picks the bundle: the distilled one (an ``n_grid``-wide head plus a ``PiflowScheduler``) or, for
    ``None``, the flow-matching one (a single-width head plus the released ``FlowMatchEulerDiscreteScheduler`` config
    and the release repo's extra metadata files)."""
    root.mkdir(parents=True, exist_ok=True)
    scheduler = scheduler_config_dict(piflow) if piflow is not None else flow_matching_scheduler_config_dict()
    flow_matching = piflow is None
    cfg = dit_config_dict(piflow, dit_cfg)
    dit = build_dit(piflow, seed, dit_cfg)
    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(json.dumps(cfg))
    save_file(to_official_dit_keys(dit.state_dict()), str(root / "transformer" / "diffusion_pytorch_model.safetensors"))

    vae = build_vae(seed)
    (root / "vae").mkdir(exist_ok=True)
    (root / "vae" / "config.json").write_text(json.dumps(vae_config_dict()))
    save_file({k: v.detach().clone().contiguous() for k, v in vae.state_dict().items()},
              str(root / "vae" / "diffusion_pytorch_model.safetensors"))

    (root / "scheduler").mkdir(exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler))

    index = copy.deepcopy(FLOW_MATCHING_MODEL_INDEX if flow_matching else DISTILLED_MODEL_INDEX)
    if flow_matching:
        (root / "sr_config.json").write_text(json.dumps(FLOW_MATCHING_SR_CONFIG))
    lu = build_lu(seed=seed)
    (root / "latent_upscaler").mkdir(exist_ok=True)
    (root / "latent_upscaler" / "config.json").write_text(json.dumps(lu_config_dict()))
    save_file({k: v.detach().clone().contiguous() for k, v in lu.state_dict().items()},
              str(root / "latent_upscaler" / "diffusion_pytorch_model.safetensors"))
    (root / "model_index.json").write_text(json.dumps(index))
    return root


def write_mp4(path: Path, num_frames: int, fps, size=(64, 96), audio_seconds: float | None = None,
              seed: int = 0) -> None:
    """Write a random-noise mpeg4 clip (+ optional 440 Hz aac track); skips the test if PyAV cannot encode it."""
    from fractions import Fraction

    import numpy as np
    import pytest

    av = pytest.importorskip("av")
    rng = np.random.default_rng(seed)
    rate = Fraction(fps).limit_denominator(1001)
    try:
        with av.open(str(path), "w") as container:
            video = container.add_stream("mpeg4", rate=rate)
            video.width, video.height, video.pix_fmt = size[1], size[0], "yuv420p"
            audio = None
            if audio_seconds is not None:
                audio = container.add_stream("aac", rate=44100)
                audio.layout = "mono"
            for i in range(num_frames):
                frame = av.VideoFrame.from_ndarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8), format="rgb24")
                frame.pts = i
                frame.time_base = Fraction(1) / rate
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            if audio is not None:
                t = np.arange(int(44100 * audio_seconds)) / 44100
                wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :]
                audio_frame = av.AudioFrame.from_ndarray(wave, format="fltp", layout="mono")
                audio_frame.sample_rate = 44100
                audio_frame.pts = 0
                audio_frame.time_base = Fraction(1, 44100)
                for packet in audio.encode(audio_frame):
                    container.mux(packet)
                for packet in audio.encode():
                    container.mux(packet)
    except Exception as exc:  # pragma: no cover - PyAV build without the mpeg4 / aac encoders
        pytest.skip(f"PyAV cannot encode the test clip: {exc}")
