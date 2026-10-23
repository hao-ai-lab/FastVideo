# SPDX-License-Identifier: Apache-2.0
"""Shared sharded feature-cache writer for video-to-audio models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class V2AFeatureShardWriter:
    """Write model-specific feature dictionaries as resumable TensorDict shards.

    Pipelines publish a ``dict[str, Tensor]`` with a common batch dimension;
    this writer handles buffering, per-rank shard names, metadata, and resume
    discovery without knowing what any feature means.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        rank: int,
        samples_per_shard: int,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        self.output_root = Path(output_root).expanduser().resolve()
        self.worker_root = self.output_root / f"worker_{rank:05d}"
        self.worker_root.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = int(samples_per_shard)
        self._features: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._metadata: list[dict[str, Any]] = []
        self.existing_ids = self._read_existing_ids(self.output_root)
        existing_indices = []
        for path in self.worker_root.glob("shard_*"):
            try:
                existing_indices.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
        self._next_shard = max(existing_indices, default=-1) + 1

    @staticmethod
    def _read_existing_ids(root: Path) -> set[str]:
        existing: set[str] = set()
        if not root.is_dir():
            return existing
        for metadata_path in root.rglob("samples.jsonl"):
            with metadata_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sample_id = json.loads(line).get("id")
                    if sample_id:
                        existing.add(str(sample_id))
        return existing

    def contains(self, sample_id: str) -> bool:
        return sample_id in self.existing_ids

    @property
    def buffered_samples(self) -> int:
        return len(self._metadata)

    def append(
        self,
        features: dict[str, torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> None:
        if not features:
            return
        batch_sizes = {int(tensor.shape[0]) for tensor in features.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"V2A feature batch dimensions do not match: {batch_sizes}")
        batch_size = batch_sizes.pop()
        if batch_size != len(metadata):
            raise ValueError(f"V2A feature batch has {batch_size} samples but {len(metadata)} metadata rows")
        if self._features and set(features) != set(self._features):
            raise ValueError(
                f"V2A feature keys changed within one shard: {sorted(self._features)} vs {sorted(features)}")

        for index, row in enumerate(metadata):
            sample_id = str(row["id"])
            if sample_id in self.existing_ids:
                continue
            for name, tensor in features.items():
                self._features[name].append(tensor[index].detach().cpu().contiguous())
            self._metadata.append(dict(row))
            self.existing_ids.add(sample_id)
            if self.buffered_samples >= self.samples_per_shard:
                self.flush()

    def flush(self) -> Path | None:
        if not self._metadata:
            return None
        try:
            from tensordict import TensorDict
        except ImportError as exc:
            raise ImportError("V2A feature caches require TensorDict; install fastvideo[mmaudio-train].") from exc

        shard_path = self.worker_root / f"shard_{self._next_shard:06d}"
        if shard_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V2A cache shard: {shard_path}")
        stacked = {name: torch.stack(values) for name, values in self._features.items()}
        TensorDict(stacked, batch_size=[len(self._metadata)]).memmap_(str(shard_path))
        with (shard_path / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for row in self._metadata:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d V2A feature samples to %s", len(self._metadata), shard_path)
        self._features.clear()
        self._metadata.clear()
        self._next_shard += 1
        return shard_path

    def close(self) -> None:
        self.flush()


__all__ = ["V2AFeatureShardWriter"]
