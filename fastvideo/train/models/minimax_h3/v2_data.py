# SPDX-License-Identifier: Apache-2.0
"""Data-free text-to-video rows for FastH3 V2 QAD.

A row is a prompt with a precomputed Qwen3-VL text embedding (``precompute_v2_text.py``) plus a seed. The initial
packed state is pure noise (target video + audio), drawn from one CPU generator so (row, seed) reproduces the same
trajectory everywhere; V2 is a plain 8-step DMD ladder, so there is no PDD start scaling and no conditioning rows.
The result mimics ``omniref_data.PreparedRow`` so the QAD method and metrics run unchanged (``case == "t2v"``
yields no keyframe metrics).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

V2_CASE = "t2v"


@dataclass(frozen=True)
class V2RowSpec:
    id: str
    source: str
    prompt: str
    embedding: str  # path of the .pt written by precompute_v2_text.py
    seed: int
    width: int = 832
    height: int = 480
    num_frames: int = 124
    case: str = V2_CASE

    @property
    def resolution(self) -> str:
        return f"{self.height}p"


def load_row_specs(manifest: str, width: int, height: int, num_frames: int) -> list[V2RowSpec]:
    """Rows of a ``build_v2_prompts.py`` manifest (JSONL: id, source, prompt, embedding, seed)."""
    specs = []
    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        specs.append(
            V2RowSpec(item["id"], item.get("source", ""), item["prompt"], item["embedding"], int(item["seed"]),
                      int(item.get("width", width)), int(item.get("height", height)),
                      int(item.get("num_frames", num_frames))))
    if not specs:
        raise ValueError(f"{manifest} has no rows")
    return specs


def read_embedding(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    embeds, tags = data["embeds"], data["tags"]
    if embeds.ndim != 2 or tags.shape != (embeds.shape[0], ):
        raise ValueError(f"{path}: bad embedding/tag shapes {tuple(embeds.shape)} / {tuple(tags.shape)}")
    return embeds, tags.to(torch.long)


def prepare_row(spec: V2RowSpec, *, patch_size: tuple[int, int, int], latent_channels: int, audio_channels: int,
                device: torch.device) -> Any:
    """Initial packed state of ``spec`` (fp32 noise rows), laid out exactly as ``MiniMaxH3LatentPreparationStage``."""
    from diffusers.utils.torch_utils import randn_tensor

    from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_AUDIO_CHANNELS, audio_latent_num_frames,
                                                              build_packed_sequence, patchify_video_latents,
                                                              video_latent_num_frames)
    from fastvideo.train.models.minimax_h3.omniref_data import PreparedRow

    frames, lat_h, lat_w = video_latent_num_frames(spec.num_frames), spec.height // 16, spec.width // 16
    embeds, tags = read_embedding(spec.embedding)
    num_audio = audio_latent_num_frames(spec.num_frames)
    layout = build_packed_sequence(tags, frames, lat_h, lat_w, num_audio, patch_size)
    generator = torch.Generator("cpu").manual_seed(spec.seed)
    video_noise = randn_tensor((1, latent_channels, frames, lat_h, lat_w),
                               generator=generator,
                               device=device,
                               dtype=torch.float32)
    audio = randn_tensor((num_audio * MINIMAX_H3_AUDIO_CHANNELS, audio_channels),
                         generator=generator,
                         device=device,
                         dtype=torch.float32)
    video = patchify_video_latents(video_noise, patch_size)
    if video.shape[0] != layout.video_indices.numel() or audio.shape[0] != layout.audio_indices.numel():
        raise ValueError(f"{spec.id}: packed rows do not match the T2V layout")
    return PreparedRow(spec, layout,
                       embeds.to(device=device, dtype=torch.bfloat16)[None], video, audio,
                       (latent_channels, frames, lat_h, lat_w), [], torch.empty(0, video.shape[-1]))


__all__ = ["V2RowSpec", "V2_CASE", "load_row_specs", "prepare_row", "read_embedding"]
