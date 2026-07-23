# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped precomputed feature dataset for audio generation training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class MMAudioFeatureDataset(Dataset):
    """Read MMAudio-compatible TensorDict memmaps without upstream imports.

    Required tensors are ``mean``, ``std``, and ``text_features``. Video
    caches additionally contain ``clip_features`` and ``sync_features``.
    Audio-only caches omit both video tensors; fixed-size placeholders and a
    false ``video_exists`` flag are returned so video and audio datasets can be
    mixed by the same dataloader.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        latent_seq_len: int,
        latent_dim: int,
        clip_seq_len: int,
        clip_dim: int,
        sync_seq_len: int,
        sync_dim: int,
        text_seq_len: int,
        text_dim: int,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"MMAudio feature cache does not exist: {self.root}")

        try:
            from tensordict import TensorDict
        except ImportError as exc:
            raise ImportError(
                "MMAudio training caches require TensorDict. Install with "
                "`uv pip install -e '.[mmaudio-train]'`."
            ) from exc

        self.tensors = TensorDict.load_memmap(self.root)
        keys = set(self.tensors.keys())
        required = {"mean", "std", "text_features"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"MMAudio feature cache {self.root} is missing tensors: {missing}")

        self.mean = self.tensors["mean"]
        self.std = self.tensors["std"]
        self.text_features = self.tensors["text_features"]
        self.clip_features = self.tensors.get("clip_features")
        self.sync_features = self.tensors.get("sync_features")
        if (self.clip_features is None) != (self.sync_features is None):
            raise ValueError(
                "MMAudio feature cache must contain both clip_features and "
                "sync_features, or neither for audio-only data."
            )

        self._length = int(self.mean.shape[0])
        for name in required:
            if int(self.tensors[name].shape[0]) != self._length:
                raise ValueError(f"MMAudio cache tensor {name!r} has a mismatched sample count")
        if self.clip_features is not None and int(self.clip_features.shape[0]) != self._length:
            raise ValueError("MMAudio cache clip_features has a mismatched sample count")
        if self.sync_features is not None and int(self.sync_features.shape[0]) != self._length:
            raise ValueError("MMAudio cache sync_features has a mismatched sample count")

        self._require_shape("mean", self.mean, (latent_seq_len, latent_dim))
        self._require_shape("std", self.std, (latent_seq_len, latent_dim))
        self._require_shape("text_features", self.text_features, (text_seq_len, text_dim))
        if self.clip_features is not None:
            self._require_shape("clip_features", self.clip_features, (clip_seq_len, clip_dim))
            assert self.sync_features is not None
            self._require_shape("sync_features", self.sync_features, (sync_seq_len, sync_dim))

        feature_dtype = self.text_features.dtype
        self.empty_clip = torch.zeros(clip_seq_len, clip_dim, dtype=feature_dtype)
        self.empty_sync = torch.zeros(sync_seq_len, sync_dim, dtype=feature_dtype)
        self.has_video = self.clip_features is not None
        logger.info("Loaded %d MMAudio feature samples from %s", self._length, self.root)

    @staticmethod
    def _require_shape(
        name: str,
        tensor: torch.Tensor,
        expected: tuple[int, ...],
    ) -> None:
        actual = tuple(tensor.shape[1:])
        if actual != expected:
            raise ValueError(
                f"MMAudio cache tensor {name!r} must have per-sample shape "
                f"{expected}, got {actual}"
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip_features = self.empty_clip
        sync_features = self.empty_sync
        if self.has_video:
            assert self.clip_features is not None and self.sync_features is not None
            clip_features = self.clip_features[index]
            sync_features = self.sync_features[index]
        return {
            "audio_latent_mean": self.mean[index],
            "audio_latent_std": self.std[index],
            "clip_features": clip_features,
            "sync_features": sync_features,
            "text_features": self.text_features[index],
            "video_exists": torch.tensor(self.has_video, dtype=torch.bool),
            "text_exists": torch.tensor(True, dtype=torch.bool),
        }


def _data_specs(data_path: str | Sequence[str] | dict[str, int]) -> list[tuple[str, int]]:
    if isinstance(data_path, dict):
        return [
            (str(path), int(repeat))
            for path, repeat in data_path.items()
            if int(repeat) > 0
        ]
    if isinstance(data_path, Sequence) and not isinstance(data_path, str):
        return [(str(path), 1) for path in data_path if str(path).strip()]
    return [(part.strip(), 1) for part in str(data_path).split(",") if part.strip()]


def _expand_cache_root(root: str | Path) -> list[Path]:
    """Resolve one direct TensorDict cache or a directory of cache shards."""
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"MMAudio feature cache does not exist: {path}")
    if (path / "meta.json").is_file():
        return [path]
    shards = sorted(metadata.parent for metadata in path.rglob("meta.json"))
    if not shards:
        raise FileNotFoundError(
            f"No TensorDict cache shards (meta.json) were found under {path}")
    return shards


def build_mmaudio_feature_dataloader(
    data_path: str | Sequence[str] | dict[str, int],
    *,
    batch_size: int,
    num_data_workers: int,
    seed: int,
    pin_memory: bool,
    feature_shapes: dict[str, int],
) -> StatefulDataLoader:
    """Build a distributed/stateful loader over one or more feature caches."""
    specs = _data_specs(data_path)
    if not specs:
        raise ValueError(
            "training.data.data_path is empty. Set it to one or more "
            "precomputed MMAudio TensorDict mmap directories."
        )

    datasets: list[Dataset] = []
    for root, repeat in specs:
        for shard_root in _expand_cache_root(root):
            dataset = MMAudioFeatureDataset(shard_root, **feature_shapes)
            datasets.extend([dataset] * repeat)
    combined: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    sp_world_size = get_sp_world_size()
    sampler = DP_SP_BatchSampler(
        batch_size=int(batch_size),
        dataset_size=len(combined),
        num_sp_groups=get_world_size() // sp_world_size,
        sp_world_size=sp_world_size,
        global_rank=get_world_rank(),
        drop_last=True,
        seed=int(seed),
    )
    return StatefulDataLoader(
        combined,
        batch_sampler=sampler,
        num_workers=int(num_data_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_data_workers) > 0,
    )


__all__ = ["MMAudioFeatureDataset", "build_mmaudio_feature_dataloader"]
