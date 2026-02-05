# SPDX-License-Identifier: Apache-2.0
# Dataset utilities for loading LTX2 precomputed training artifacts.
#
# Usage:
# - Input root can be either `<data_root>/` or `<data_root>/.precomputed/`.
# - Required sources are `latents/` and `conditions/` with matching `.pt` files.
# - Optional source `audio_latents/` is loaded when provided in `data_sources`.
# - `build_ltx2_precomputed_dataloader(...)` is the intended entrypoint used by
#   `fastvideo/training/ltx2_training_pipeline.py`.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size
from fastvideo.logger import init_logger

logger = init_logger(__name__)

PRECOMPUTED_DIR_NAME = ".precomputed"


class LTX2PrecomputedDataset(Dataset):
    """Dataset for LTX-2 precomputed latents and conditions.

    Expected directory structure (data_root):
      .precomputed/
        latents/*.pt
        conditions/*.pt
        audio_latents/*.pt (optional)
    """

    def __init__(
        self,
        data_root: str,
        data_sources: dict[str, str] | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        data_root_path = Path(data_root).expanduser().resolve()
        if not data_root_path.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {data_root_path}")
        if (data_root_path / PRECOMPUTED_DIR_NAME).exists():
            data_root_path = data_root_path / PRECOMPUTED_DIR_NAME
        return data_root_path

    @staticmethod
    def _normalize_data_sources(
        data_sources: dict[str, str] | list[str] | None,
    ) -> dict[str, str]:
        if data_sources is None:
            return {"latents": "latents", "conditions": "conditions"}
        if isinstance(data_sources, list):
            return {source: source for source in data_sources}
        if isinstance(data_sources, dict):
            return data_sources.copy()
        raise TypeError(
            f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        source_paths: dict[str, Path] = {}
        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Required {dir_name} directory does not exist: {source_path}")
            source_paths[dir_name] = source_path
        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        data_key = ("latents"
                    if "latents" in self.data_sources else next(iter(
                        self.data_sources.keys())))
        data_path = self.source_paths[data_key]
        data_files = list(data_path.glob("**/*.pt"))
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        sample_files = {output_key: [] for output_key in self.data_sources.values()}
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)
        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file,
                                                         rel_path)
            if not expected_path.exists():
                logger.warning(
                    "No matching %s file found for: %s (expected in: %s)",
                    dir_name,
                    data_file.name,
                    expected_path,
                )
                return False
        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path,
                                rel_path: Path) -> Path:
        source_path = self.source_paths[dir_name]
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"
        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path,
                                sample_files: dict[str, list[Path]]) -> None:
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file,
                                                         rel_path)
            sample_files[output_key].append(
                expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        if not self.sample_files:
            raise ValueError(
                "No valid samples found - all data sources must have matching files"
            )
        sample_counts = {
            key: len(files)
            for key, files in self.sample_files.items()
        }
        if len(set(sample_counts.values())) > 1:
            raise ValueError(
                f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result: dict[str, Any] = {}
        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)
                result[output_key] = data
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {output_key} from {file_path}: {e}") from e
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        latents = data["latents"]
        if latents.dim() == 2:
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )
            data = data.copy()
            data["latents"] = latents
        return data


def build_ltx2_precomputed_dataloader(
    path: str,
    batch_size: int,
    num_data_workers: int,
    data_sources: dict[str, str] | list[str] | None = None,
    drop_last: bool = True,
    seed: int = 42,
) -> tuple[LTX2PrecomputedDataset, StatefulDataLoader]:
    dataset = LTX2PrecomputedDataset(path, data_sources=data_sources)
    sampler = DP_SP_BatchSampler(
        batch_size=batch_size,
        dataset_size=len(dataset),
        num_sp_groups=get_world_size() // get_sp_world_size(),
        sp_world_size=get_sp_world_size(),
        global_rank=get_world_rank(),
        drop_last=drop_last,
        drop_first_row=False,
        seed=seed,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=None,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
