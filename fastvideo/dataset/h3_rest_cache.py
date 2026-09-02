# SPDX-License-Identifier: Apache-2.0
"""Immutable tensor-cache dataset for H3 scored-trajectory distillation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler, passthrough
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import canonical_json_hash

logger = init_logger(__name__)

H3_REST_CACHE_SCHEMA_VERSION = 1
_METADATA = "metadata.json"
_MANIFEST = "manifest.jsonl"
_COMPLETE = "COMPLETE"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


@dataclass(frozen=True, slots=True)
class H3RESTCacheSummary:
    cache_dir: str
    fingerprint: str
    num_prompts: int
    num_trajectories: int
    num_segments: int
    num_examples: int
    student_timesteps: tuple[float, ...]
    reward_names: tuple[str, ...]


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Manifest line {line_number} must be a JSON object, got {type(value).__name__}"
                )
            entries.append(value)
    return entries


def _resolve_cache_file(cache_dir: Path, relative_path: Any, *, field: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{field} must be a nonempty relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"{field} must be relative to the cache root, got {relative_path!r}")
    root = cache_dir.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the cache root: {relative_path!r}") from exc
    return resolved


def _require_int(mapping: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"metadata.{key} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _validate_timesteps(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) < 2:
        raise ValueError("metadata.student_timesteps must contain at least two values")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("metadata.student_timesteps must be finite")
    if any(left <= right for left, right in zip(values[:-1], values[1:], strict=True)):
        raise ValueError(
            "metadata.student_timesteps must be strictly descending, got "
            f"{list(values)}"
        )
    return values


def validate_h3_rest_cache(
    cache_dir: str | os.PathLike[str],
    *,
    verify_file_hashes: bool = False,
    expected_student_timesteps: Sequence[int | float] | None = None,
) -> H3RESTCacheSummary:
    """Validate cache completion, provenance fingerprint, manifest, and files."""
    root = Path(cache_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"H3 REST cache directory not found: {root}")
    metadata_path = root / _METADATA
    manifest_path = root / _MANIFEST
    complete_path = root / _COMPLETE
    for required in (metadata_path, manifest_path, complete_path):
        if not required.is_file():
            raise FileNotFoundError(f"Incomplete H3 REST cache: missing {required}")

    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    schema_version = _require_int(metadata, "schema_version", minimum=1)
    if schema_version != H3_REST_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported H3 REST cache schema: "
            f"expected={H3_REST_CACHE_SCHEMA_VERSION}, observed={schema_version}"
        )
    observed_fingerprint = metadata.get("fingerprint")
    if not _is_sha256(observed_fingerprint):
        raise ValueError("metadata.fingerprint must be a SHA-256 hex digest")
    fingerprint_payload = dict(metadata)
    fingerprint_payload.pop("fingerprint", None)
    expected_fingerprint = canonical_json_hash(fingerprint_payload)
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            "H3 REST metadata fingerprint mismatch: "
            f"stored={observed_fingerprint}, recomputed={expected_fingerprint}"
        )
    completion = complete_path.read_text(encoding="utf-8").strip()
    if completion != observed_fingerprint:
        raise ValueError(
            "H3 REST completion marker does not match metadata fingerprint: "
            f"COMPLETE={completion!r}, metadata={observed_fingerprint!r}"
        )

    manifest_sha = metadata.get("manifest_sha256")
    if not _is_sha256(manifest_sha):
        raise ValueError("metadata.manifest_sha256 must be a SHA-256 hex digest")
    actual_manifest_sha = sha256_file(manifest_path)
    if manifest_sha != actual_manifest_sha:
        raise ValueError(
            "H3 REST manifest hash mismatch: "
            f"stored={manifest_sha}, observed={actual_manifest_sha}"
        )

    student_timesteps = _validate_timesteps(metadata.get("student_timesteps"))
    if expected_student_timesteps is not None:
        expected = tuple(float(value) for value in expected_student_timesteps)
        if student_timesteps != expected:
            raise ValueError(
                "Cache/student timestep mismatch: "
                f"cache={list(student_timesteps)}, expected={list(expected)}"
            )
    num_segments = _require_int(metadata, "num_segments", minimum=1)
    if num_segments != len(student_timesteps) - 1:
        raise ValueError(
            "metadata.num_segments must equal len(student_timesteps)-1, got "
            f"{num_segments} vs {len(student_timesteps) - 1}"
        )
    num_prompts = _require_int(metadata, "num_prompts", minimum=1)
    num_trajectories = _require_int(metadata, "num_trajectories", minimum=1)
    samples_per_prompt = _require_int(metadata, "samples_per_prompt", minimum=2)
    if num_trajectories != num_prompts * samples_per_prompt:
        raise ValueError(
            "metadata trajectory count is inconsistent: "
            f"{num_trajectories} != {num_prompts} * {samples_per_prompt}"
        )
    reward_names_raw = metadata.get("reward_names")
    if not isinstance(reward_names_raw, list) or not reward_names_raw:
        raise ValueError("metadata.reward_names must be a nonempty list")
    reward_names = tuple(str(value).strip().lower() for value in reward_names_raw)
    if any(not value for value in reward_names) or len(set(reward_names)) != len(reward_names):
        raise ValueError(f"metadata.reward_names contains empty/duplicate names: {reward_names}")

    entries = _read_manifest(manifest_path)
    if len(entries) != num_trajectories:
        raise ValueError(
            "Manifest trajectory count mismatch: "
            f"metadata={num_trajectories}, manifest={len(entries)}"
        )
    trajectory_ids: set[str] = set()
    trajectory_files: set[str] = set()
    prompt_ids: set[str] = set()
    prompt_counts: dict[str, int] = {}
    prompt_candidates: dict[str, set[int]] = {}
    prompt_files: dict[str, tuple[str, str, int]] = {}
    for index, entry in enumerate(entries):
        trajectory_id = entry.get("trajectory_id")
        prompt_id = entry.get("prompt_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"Manifest entry {index} has invalid trajectory_id")
        if trajectory_id in trajectory_ids:
            raise ValueError(f"Duplicate trajectory_id in manifest: {trajectory_id!r}")
        trajectory_ids.add(trajectory_id)
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Manifest entry {index} has invalid prompt_id")
        prompt_ids.add(prompt_id)
        prompt_counts[prompt_id] = prompt_counts.get(prompt_id, 0) + 1
        candidate_index = entry.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < samples_per_prompt
        ):
            raise ValueError(
                f"Manifest entry {index} has invalid candidate_index={candidate_index!r}"
            )
        candidates = prompt_candidates.setdefault(prompt_id, set())
        if candidate_index in candidates:
            raise ValueError(
                f"Duplicate candidate_index={candidate_index} for prompt {prompt_id!r}"
            )
        candidates.add(candidate_index)

        trajectory_path = _resolve_cache_file(
            root,
            entry.get("trajectory_file"),
            field=f"manifest[{index}].trajectory_file",
        )
        prompt_path = _resolve_cache_file(
            root,
            entry.get("prompt_file"),
            field=f"manifest[{index}].prompt_file",
        )
        for kind, path in (("trajectory", trajectory_path), ("prompt", prompt_path)):
            if not path.is_file():
                raise FileNotFoundError(f"Manifest {kind} file not found: {path}")
        trajectory_key = str(trajectory_path)
        if trajectory_key in trajectory_files:
            raise ValueError(f"Trajectory file reused by multiple entries: {trajectory_path}")
        trajectory_files.add(trajectory_key)
        expected_trajectory_hash = entry.get("trajectory_sha256")
        expected_prompt_hash = entry.get("prompt_sha256")
        if not _is_sha256(expected_trajectory_hash):
            raise ValueError(f"Manifest entry {index} has invalid trajectory_sha256")
        if not _is_sha256(expected_prompt_hash):
            raise ValueError(f"Manifest entry {index} has invalid prompt_sha256")

        reward_scores = entry.get("reward_scores")
        reward_advantages = entry.get("reward_advantages")
        if not isinstance(reward_scores, dict) or set(reward_scores) != set(reward_names):
            raise ValueError(
                f"Manifest entry {index} reward_scores must match {reward_names}, "
                f"got {sorted(reward_scores) if isinstance(reward_scores, dict) else type(reward_scores).__name__}"
            )
        if not isinstance(reward_advantages, dict) or set(reward_advantages) != set(reward_names):
            raise ValueError(
                f"Manifest entry {index} reward_advantages must match {reward_names}"
            )
        numeric_values = [
            *reward_scores.values(),
            *reward_advantages.values(),
            entry.get("mixed_advantage"),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError(f"Manifest entry {index} contains non-finite reward data")

        trajectory_bytes = entry.get("trajectory_bytes")
        if isinstance(trajectory_bytes, bool) or not isinstance(trajectory_bytes, int) or trajectory_bytes <= 0:
            raise ValueError(f"Manifest entry {index} has invalid trajectory_bytes")
        if trajectory_path.stat().st_size != trajectory_bytes:
            raise ValueError(
                f"Trajectory size mismatch for {trajectory_path}: "
                f"manifest={trajectory_bytes}, observed={trajectory_path.stat().st_size}"
            )
        prompt_bytes = entry.get("prompt_bytes")
        if isinstance(prompt_bytes, bool) or not isinstance(prompt_bytes, int) or prompt_bytes <= 0:
            raise ValueError(f"Manifest entry {index} has invalid prompt_bytes")
        if prompt_path.stat().st_size != prompt_bytes:
            raise ValueError(
                f"Prompt size mismatch for {prompt_path}: "
                f"manifest={prompt_bytes}, observed={prompt_path.stat().st_size}"
            )

        prompt_key = str(prompt_path)
        prompt_signature = (prompt_id, str(expected_prompt_hash), prompt_bytes)
        previous = prompt_files.get(prompt_key)
        if previous is None:
            prompt_files[prompt_key] = prompt_signature
        elif previous != prompt_signature:
            raise ValueError(f"Inconsistent prompt metadata for {prompt_path}")
        if verify_file_hashes:
            if sha256_file(trajectory_path) != expected_trajectory_hash:
                raise ValueError(f"Trajectory SHA-256 mismatch: {trajectory_path}")
            if previous is None and sha256_file(prompt_path) != expected_prompt_hash:
                raise ValueError(f"Prompt SHA-256 mismatch: {prompt_path}")

    if len(prompt_ids) != num_prompts:
        raise ValueError(
            f"Manifest prompt count mismatch: metadata={num_prompts}, manifest={len(prompt_ids)}"
        )
    expected_candidates = set(range(samples_per_prompt))
    for prompt_id in sorted(prompt_ids):
        if prompt_counts[prompt_id] != samples_per_prompt:
            raise ValueError(
                f"Prompt {prompt_id!r} has {prompt_counts[prompt_id]} trajectories; "
                f"expected {samples_per_prompt}"
            )
        if prompt_candidates[prompt_id] != expected_candidates:
            raise ValueError(
                f"Prompt {prompt_id!r} candidate set mismatch: "
                f"observed={sorted(prompt_candidates[prompt_id])}, "
                f"expected={sorted(expected_candidates)}"
            )

    return H3RESTCacheSummary(
        cache_dir=str(root),
        fingerprint=observed_fingerprint,
        num_prompts=num_prompts,
        num_trajectories=num_trajectories,
        num_segments=num_segments,
        num_examples=num_trajectories * num_segments,
        student_timesteps=student_timesteps,
        reward_names=reward_names,
    )


class H3RESTCacheDataset(Dataset):
    """Map-style dataset expanding every cached trajectory into its segments."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        batch_size: int = 1,
        seed: int = 0,
        verify_file_hashes: bool = False,
        expected_student_timesteps: Sequence[int | float] | None = None,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError(
                "MiniMax H3 REST currently requires train_batch_size=1; "
                f"got {batch_size}"
            )
        self.root = Path(cache_dir).expanduser().resolve()
        self.summary = validate_h3_rest_cache(
            self.root,
            verify_file_hashes=verify_file_hashes,
            expected_student_timesteps=expected_student_timesteps,
        )
        self.entries = _read_manifest(self.root / _MANIFEST)
        self.num_segments = self.summary.num_segments
        self.batch_size = 1
        self.sampler = DP_SP_BatchSampler(
            batch_size=1,
            dataset_size=len(self),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=True,
            seed=int(seed),
        )
        logger.info(
            "Loaded H3 REST cache %s: %d trajectories x %d segments = %d examples",
            self.root,
            self.summary.num_trajectories,
            self.num_segments,
            len(self),
        )

    def __len__(self) -> int:
        return len(self.entries) * self.num_segments

    @staticmethod
    def _load_tensor_dict(path: Path) -> dict[str, torch.Tensor]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in payload.items()
        ):
            raise ValueError(f"Cache tensor file must be a dict[str, Tensor]: {path}")
        return payload

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        if len(indices) != 1:
            raise ValueError(
                "MiniMax H3 REST expects one cache example per SP group, "
                f"got indices={indices}"
            )
        flat_index = int(indices[0])
        if not 0 <= flat_index < len(self):
            raise IndexError(f"H3 REST cache index out of range: {flat_index}")
        trajectory_index, segment_index = divmod(flat_index, self.num_segments)
        entry = self.entries[trajectory_index]
        prompt_path = _resolve_cache_file(
            self.root,
            entry["prompt_file"],
            field="manifest.prompt_file",
        )
        trajectory_path = _resolve_cache_file(
            self.root,
            entry["trajectory_file"],
            field="manifest.trajectory_file",
        )
        prompt_payload = self._load_tensor_dict(prompt_path)
        trajectory_payload = self._load_tensor_dict(trajectory_path)

        text_embedding = prompt_payload.get("text_embedding")
        text_attention_mask = prompt_payload.get("text_attention_mask")
        anchor_states = trajectory_payload.get("anchor_states")
        anchor_timesteps = trajectory_payload.get("anchor_timesteps")
        if text_embedding is None or text_attention_mask is None:
            raise ValueError(f"Prompt cache missing text tensors: {prompt_path}")
        if anchor_states is None or anchor_timesteps is None:
            raise ValueError(f"Trajectory cache missing anchor tensors: {trajectory_path}")
        if anchor_states.ndim != 2 or anchor_states.shape[0] != self.num_segments + 1:
            raise ValueError(
                f"anchor_states must have shape [{self.num_segments + 1}, N], "
                f"got {tuple(anchor_states.shape)} in {trajectory_path}"
            )
        if anchor_timesteps.ndim != 1 or anchor_timesteps.numel() != self.num_segments + 1:
            raise ValueError(
                f"anchor_timesteps must have shape [{self.num_segments + 1}], "
                f"got {tuple(anchor_timesteps.shape)} in {trajectory_path}"
            )
        observed_timesteps = tuple(float(value) for value in anchor_timesteps.tolist())
        if observed_timesteps != self.summary.student_timesteps:
            raise ValueError(
                "Trajectory/student timestep mismatch in "
                f"{trajectory_path}: {observed_timesteps} != {self.summary.student_timesteps}"
            )
        if text_embedding.ndim < 2 or text_embedding.shape[0] != 1:
            raise ValueError(
                f"text_embedding must retain batch dimension 1, got {tuple(text_embedding.shape)}"
            )
        if text_attention_mask.ndim < 2 or text_attention_mask.shape[0] != 1:
            raise ValueError(
                "text_attention_mask must retain batch dimension 1, got "
                f"{tuple(text_attention_mask.shape)}"
            )

        reward_scores = {
            str(name): float(value)
            for name, value in entry["reward_scores"].items()
        }
        reward_advantages = {
            str(name): float(value)
            for name, value in entry["reward_advantages"].items()
        }
        return {
            "text_embedding": text_embedding,
            "text_attention_mask": text_attention_mask,
            "trajectory_current": anchor_states[segment_index].unsqueeze(0),
            "trajectory_next": anchor_states[segment_index + 1].unsqueeze(0),
            "trajectory_timestep": anchor_timesteps[segment_index].reshape(1),
            "trajectory_next_timestep": anchor_timesteps[segment_index + 1].reshape(1),
            "rest_mixed_advantage": torch.tensor(
                [float(entry["mixed_advantage"])], dtype=torch.float32
            ),
            "rest_segment_index": torch.tensor([segment_index], dtype=torch.long),
            "rest_reward_scores": reward_scores,
            "rest_reward_advantages": reward_advantages,
            "rest_cache_fingerprint": self.summary.fingerprint,
            "info_list": [
                {
                    "prompt": str(entry.get("prompt", "")),
                    "id": str(entry["prompt_id"]),
                    "trajectory_id": str(entry["trajectory_id"]),
                    "candidate_index": int(entry.get("candidate_index", 0)),
                    "seed": int(entry.get("seed", 0)),
                }
            ],
        }


def build_h3_rest_cache_dataloader(
    cache_dir: str | os.PathLike[str],
    *,
    batch_size: int,
    num_data_workers: int,
    seed: int,
    verify_file_hashes: bool = False,
    expected_student_timesteps: Sequence[int | float] | None = None,
) -> tuple[H3RESTCacheDataset, StatefulDataLoader]:
    dataset = H3RESTCacheDataset(
        cache_dir,
        batch_size=batch_size,
        seed=seed,
        verify_file_hashes=verify_file_hashes,
        expected_student_timesteps=expected_student_timesteps,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=int(num_data_workers),
        pin_memory=True,
        persistent_workers=int(num_data_workers) > 0,
    )
    return dataset, loader


__all__ = [
    "H3RESTCacheDataset",
    "H3RESTCacheSummary",
    "H3_REST_CACHE_SCHEMA_VERSION",
    "build_h3_rest_cache_dataloader",
    "sha256_file",
    "validate_h3_rest_cache",
]
