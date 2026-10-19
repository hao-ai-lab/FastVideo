# SPDX-License-Identifier: Apache-2.0
"""OmniRef (FastH3 Ref2VA) rows for per-forward distillation of the PDD student.

A row is one OmniRef training parquet row (precomputed Qwen3-VL presentation,
clean reference latent rows and their geometry); nothing is re-encoded. The
initial packed state of a (row, seed) pair is built exactly as
``generate_omniref_latents.py`` and ``MiniMaxH3LatentPreparationStage`` build
it (reference rows noise-augmented to 0.999, target video/audio noise from the
same CPU generator, PDD start scaling), so trajectories, teacher latents and
Stage A's teacher-forced error are all reproducible from the row id and seed.

Row selection reuses the generator's deterministic plan (per (case,
resolution): seeded shuffle, then interleave), so the first rows of each group
are the rows the OmniRef latent generator and Stage A calibration used.
Held-out sources are excluded at both resolutions.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

OMNIREF_CASES = ("first_frame", "first_last_frame", "storyboard", "continue_scene", "continue_shot",
                 "full_scene_audio_image_reference")
RESOLUTIONS = ("480p", "768p")
_BUCKET_FRAMES = re.compile(r"bucket=\d+x\d+-(\d+)f")
ROW_COLUMNS = [
    "id", "vae_latent_shape", "audio_latent_shape", "text_embedding_bytes", "text_embedding_shape",
    "text_token_tags_bytes", "text_token_tags_shape", "reference_video_latent_rows_bytes",
    "reference_video_latent_rows_shape", "reference_audio_latent_rows_bytes", "reference_audio_latent_rows_shape",
    "reference_media_types", "reference_has_audio", "reference_num_latent_frames", "reference_latent_heights",
    "reference_latent_widths", "reference_num_audio_latents", "width", "height", "num_frames"
]


@dataclass(frozen=True)
class RowSpec:
    id: str
    source: str
    case: str
    resolution: str
    parquet: str
    seed: int


def clip_seed(base_seed: int, clip_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{clip_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def load_manifest_groups(paths: list[str], cases: tuple[str, ...],
                         max_frames: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Manifest entries grouped by (case, resolution); several manifests (v4 + v5 additions) merge.

    An entry repeated across manifests is kept once per (id, resolution): the same source at 480p and 768p is
    two rows.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                entry = json.loads(line)
                match = _BUCKET_FRAMES.search(entry["parquet"])
                key = (entry["id"], entry["resolution"])
                if (match is None or entry["case"] not in cases or entry["resolution"] not in RESOLUTIONS or key in seen
                        or int(match.group(1)) > max_frames):
                    continue
                seen.add(key)
                groups[(entry["case"], entry["resolution"])].append(entry)
    return groups


def limit_frames(groups: dict[tuple[str, str], list[dict[str, Any]]],
                 max_frames: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """``groups`` without entries whose bucket is longer than ``max_frames`` (0 = no limit); a new mapping."""
    if not max_frames:
        return dict(groups)

    def frames(entry: dict[str, Any]) -> int:
        match = _BUCKET_FRAMES.search(entry["parquet"])
        return int(match.group(1)) if match else max_frames + 1

    return {key: [entry for entry in entries if frames(entry) <= max_frames] for key, entries in groups.items()}


def select_rows(groups: dict[tuple[str, str], list[dict[str, Any]]],
                seed: int,
                per_group: dict[str, int],
                exclude_sources: frozenset[str] = frozenset()) -> list[RowSpec]:
    """The generator's plan: per group a seeded shuffle, minus excluded sources, then interleaved."""
    picked = []
    for key in sorted(groups):
        entries = sorted(groups[key], key=lambda entry: entry["id"])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(entries)
        entries = [entry for entry in entries if entry["id"].split(":", 1)[-1] not in exclude_sources]
        picked.append(entries[:int(per_group.get(key[1], 0))])
    plan = []
    for index in range(max((len(entries) for entries in picked), default=0)):
        for entries in picked:
            if index < len(entries):
                entry = entries[index]
                source = entry["id"].split(":", 1)[-1]
                clip_id = f"omniref-{entry['case']}-{source}-{entry['resolution']}"
                plan.append(
                    RowSpec(clip_id, source, entry["case"], entry["resolution"], entry["parquet"],
                            clip_seed(seed, clip_id)))
    return plan


def load_eval_rows(manifest_json: str,
                   cases: tuple[str, ...],
                   resolutions: tuple[str, ...],
                   per_group: int | dict[str, int],
                   max_frames: int = 0) -> list[RowSpec]:
    """Held-out rows from a ``generate_omniref_latents.py --eval-dir`` manifest (ids, parquets and seeds)."""
    items = json.loads(Path(manifest_json).read_text())
    counts: dict[tuple[str, str], int] = defaultdict(int)
    rows = []
    for item in sorted(items, key=lambda item: item["id"]):
        key = (item["case"], item["resolution"])
        limit = int(per_group.get(item["resolution"], 0)) if isinstance(per_group, dict) else int(per_group)
        if item["case"] not in cases or item["resolution"] not in resolutions or counts[key] >= limit:
            continue
        if max_frames and int(item.get("num_frames", 0)) > max_frames:
            continue
        counts[key] += 1
        rows.append(
            RowSpec(item["id"], item["source"], item["case"], item["resolution"], item["parquet"], int(item["seed"])))
    return rows


def top_up_eval_rows(eval_rows: list[RowSpec], groups: dict[tuple[str, str], list[dict[str, Any]]], seed: int,
                     per_group: dict[str, int]) -> list[RowSpec]:
    """Fill each (case, resolution) up to ``per_group`` with fresh held-out rows (call before the training plan).

    Extra rows come from the generator's plan rule at ``seed``, skipping sources already in ``eval_rows``;
    the training plan must then exclude every eval source. Result is interleaved across groups.
    """
    have: dict[tuple[str, str], int] = defaultdict(int)
    for row in eval_rows:
        have[(row.case, row.resolution)] += 1
    taken = frozenset(row.source for row in eval_rows)
    extra = []
    for key in sorted(groups):
        need = int(per_group.get(key[1], 0)) - have[key]
        if need > 0:
            extra += select_rows({key: groups[key]}, seed, {key[1]: need}, taken)
    return interleave_by_group(eval_rows + extra)


def interleave_by_group(rows: list[RowSpec]) -> list[RowSpec]:
    """Round-robin over (case, resolution) groups, so any prefix covers every case."""
    buckets: dict[tuple[str, str], list[RowSpec]] = defaultdict(list)
    for row in rows:
        buckets[(row.case, row.resolution)].append(row)
    ordered = [buckets[key] for key in sorted(buckets)]
    return [
        bucket[i] for i in range(max((len(b) for b in ordered), default=0)) for bucket in ordered if i < len(bucket)
    ]


class RowStream:
    """Infinite, deterministic per-data-parallel-rank row stream (every SP rank of a group sees the same row)."""

    def __init__(self, plan: list[RowSpec], *, dp_rank: int, dp_size: int, seed: int) -> None:
        if not plan:
            raise ValueError("the training row plan is empty")
        self.plan = plan
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.seed = seed
        self.position = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            epoch, offset = divmod(self.position * self.dp_size + self.dp_rank, len(self.plan))
            order = list(range(len(self.plan)))
            random.Random(f"{self.seed}:epoch{epoch}").shuffle(order)
            self.position += 1
            yield {"row": self.plan[order[offset]], "epoch": epoch}

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = int(state["position"])


# --------------------------------------------------------------------------- one row -> initial packed state
def _tensor(row: dict[str, Any], name: str) -> torch.Tensor:
    shape = [int(value) for value in row[f"{name}_shape"]]
    array = np.frombuffer(row[f"{name}_bytes"], dtype=np.float32)
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f"{name}: {array.size} values do not fill shape {shape}")
    return torch.from_numpy(array.reshape(shape).copy())


def read_row(parquet: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=ROW_COLUMNS).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"{parquet}: expected one row, got {len(rows)}")
    return rows[0]


def prepared_references(row: dict[str, Any]) -> list[Any]:
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference

    fields = ("reference_media_types", "reference_has_audio", "reference_num_latent_frames", "reference_latent_heights",
              "reference_latent_widths", "reference_num_audio_latents")
    return [
        MiniMaxH3PreparedReference(media_type=media,
                                   has_audio=bool(audio),
                                   num_latent_frames=int(frames),
                                   latent_height=int(height),
                                   latent_width=int(width),
                                   num_audio_latents=int(count))
        for media, audio, frames, height, width, count in zip(*(row[name] for name in fields), strict=True)
    ]


@dataclass
class PreparedRow:
    """Everything one PDD trajectory of a row needs; ``video``/``audio`` are the packed initial rows (fp32)."""
    spec: RowSpec
    layout: Any
    prompt_embeds: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    latent_shape: tuple[int, int, int, int]  # (channels, frames, height, width) of the target video
    references: list[Any]
    reference_video_rows: torch.Tensor  # clean, CPU


def prepare_row(spec: RowSpec, row: dict[str, Any], *, patch_size: tuple[int, int, int], latent_channels: int,
                audio_channels: int, pdd_start_sigmas: tuple[float, float], device: torch.device) -> PreparedRow:
    """Initial packed state, bit-identical to ``OmniRefLatentGenerator._batch`` + ``scale_pdd_initial_noise``."""
    from diffusers.utils.torch_utils import randn_tensor

    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                                              audio_latent_num_frames, build_ref2va_packed_sequence,
                                                              keyframe_condition_noise, patchify_video_latents,
                                                              video_latent_num_frames)

    num_frames, height, width = int(row["num_frames"]), int(row["height"]), int(row["width"])
    frames, lat_h, lat_w = video_latent_num_frames(num_frames), height // 16, width // 16
    expected = tuple(int(value) for value in row["vae_latent_shape"][1:])
    if (frames, lat_h, lat_w) != expected or int(row["vae_latent_shape"][0]) != latent_channels:
        raise ValueError(f"{spec.id}: target geometry {(frames, lat_h, lat_w)} disagrees with {expected}")
    num_audio = audio_latent_num_frames(num_frames)
    references = prepared_references(row)
    tags = _tensor(row, "text_token_tags").round().to(torch.long)
    embeds = _tensor(row, "text_embedding").to(device=device, dtype=torch.bfloat16)[None]
    layout = build_ref2va_packed_sequence(tags, references, frames, lat_h, lat_w, num_audio, patch_size)

    generator = torch.Generator("cpu").manual_seed(spec.seed)
    shapes = tuple(
        (ref.num_latent_frames, ref.latent_height, ref.latent_width) for ref in references if ref.media_type != "audio")
    noise = keyframe_condition_noise(shapes, patch_size, latent_channels, generator=generator, device=device)
    clean = _tensor(row, "reference_video_latent_rows")
    condition_video = MiniMaxH3Scheduler().scale_noise(clean.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
    video_noise = randn_tensor((1, latent_channels, frames, lat_h, lat_w),
                               generator=generator,
                               device=device,
                               dtype=torch.float32)
    audio_rows = randn_tensor((num_audio * MINIMAX_H3_AUDIO_CHANNELS, audio_channels),
                              generator=generator,
                              device=device,
                              dtype=torch.float32)
    condition_audio = _tensor(row, "reference_audio_latent_rows").to(device)
    video = torch.cat((condition_video, patchify_video_latents(video_noise, patch_size)))
    audio = torch.cat((condition_audio, audio_rows))
    if video.shape[0] != layout.video_indices.numel() or audio.shape[0] != layout.audio_indices.numel():
        raise ValueError(f"{spec.id}: packed rows do not match the Ref2VA layout")
    video[layout.num_condition_video_rows:] *= pdd_start_sigmas[0]
    audio[layout.num_condition_audio_rows:] *= pdd_start_sigmas[1]
    return PreparedRow(spec, layout, embeds, video, audio, (latent_channels, frames, lat_h, lat_w), references, clean)


__all__ = [
    "OMNIREF_CASES", "PreparedRow", "RowSpec", "RowStream", "interleave_by_group", "load_eval_rows",
    "load_manifest_groups", "top_up_eval_rows", "prepare_row", "read_row", "select_rows"
]
