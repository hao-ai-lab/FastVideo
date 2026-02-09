# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import pickle
import random
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
# Torch in general
import torch
import tqdm
# Dataset
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from fastvideo.platforms import current_platform

from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import (get_sp_world_size, get_world_group,
                                   get_world_rank, get_world_size)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class DP_SP_BatchSampler(Sampler[list[int]]):
    """
    A simple sequential batch sampler that yields batches of indices.
    """

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        drop_last: bool = True,
        drop_first_row: bool = False,
        reshuffle_each_epoch: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size
        self.drop_first_row = drop_first_row
        self.reshuffle_each_epoch = reshuffle_each_epoch

        self._build_indices(0)  # build indices for epoch 0 to initialize the sampler

    def _build_indices(self, epoch: int) -> None:
        # ── epoch-level RNG ────────────────────────────────────────────────
        rng = torch.Generator().manual_seed(self.seed + epoch)
        # Create a random permutation of all indices
        global_indices = torch.randperm(self.dataset_size, generator=rng)

        dataset_size = self.dataset_size
        if self.drop_first_row:
            # drop 0 in global_indices
            global_indices = global_indices[global_indices != 0]
            dataset_size = dataset_size - 1

        if self.drop_last:
            # For drop_last=True, we:
            # 1. Ensure total samples is divisible by (batch_size * num_sp_groups)
            # 2. This guarantees each SP group gets same number of complete batches
            # 3. Prevents uneven batch sizes across SP groups at end of epoch
            num_batches = dataset_size // self.batch_size
            num_global_batches = num_batches // self.num_sp_groups
            global_indices = global_indices[:num_global_batches *
                                            self.num_sp_groups *
                                            self.batch_size]
        else:
            if dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                # add more indices to make it divisible by (batch_size * num_sp_groups)
                padding_size = self.num_sp_groups * self.batch_size - (
                    dataset_size % (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d",
                            dataset_size, dataset_size + padding_size)
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]])

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.
                                                num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d",
                    len(sp_group_local_indices))

    def set_epoch(self, epoch: int) -> None:
        if not self.reshuffle_each_epoch:
            return
        self._build_indices(epoch)

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


def _parse_data_path_specs(path: str) -> list[tuple[str, int]]:
    """
    Parse data_path into a list of (directory, repeat_count).
    Syntax: comma-separated entries; each entry is "path" (default 1) or "path:N" (N = repeat count).
    N=0 means skip that path (convenience to disable without removing). If no ":" present, default is 1.
    Example: "/dir1:2,/dir2,/dir3:0" -> dir1 2x, dir2 1x, dir3 skipped.
    """
    specs: list[tuple[str, int]] = []
    for part in path.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            p, _, count_str = part.rpartition(":")
            p = p.strip()
            try:
                count = int(count_str.strip())
            except ValueError:
                raise ValueError(
                    f"data_path repeat count must be an integer, got {count_str!r}"
                ) from None
            if count < 0:
                raise ValueError(
                    f"data_path repeat count must be >= 0, got {count}"
                )
            specs.append((p, count))
        else:
            specs.append((part, 1))
    return specs


def _scan_parquet_files_for_path(p: str) -> tuple[list[str], list[int]]:
    """Return (file_paths, row_lengths) for a single directory."""
    file_names: list[str] = []
    for root, _, files in os.walk(p):
        for file in sorted(files):
            if file.endswith(".parquet"):
                file_names.append(os.path.join(root, file))
    lengths = []
    for file_path in tqdm.tqdm(
            file_names, desc="Reading parquet files to get lengths"):
        lengths.append(pq.ParquetFile(file_path).metadata.num_rows)
    logger.info("Found %d parquet files with %d total rows", len(file_names), sum(lengths))
    return file_names, lengths


def get_parquet_files_and_length(path: str):
    """
    Collect parquet file paths and row lengths from one or more directories.
    path: single directory, or comma-separated "path" or "path:N" (N = repeat count).
    E.g. "/dir1:2,/dir2:1" -> dir1's files appear 2x (oversampled), dir2 once.
    """
    path_specs = _parse_data_path_specs(path)
    if not path_specs:
        raise ValueError(
            "data_path must be a non-empty path or comma-separated path specs"
        )
    # Use first path with count > 0 for cache_dir (single-path case only)
    first_path = next(
        (p for p, c in path_specs if c > 0),
        path_specs[0][0],
    )
    is_single_no_repeat = (
        len(path_specs) == 1 and path_specs[0][1] == 1
    )
    effective_path = path.strip()
    # Single path, no repeat: cache under that path (backward compatible).
    # Multi-path or repeat: cache in a neutral dir keyed by hash of full path spec,
    # so we never reuse "first path's" cache and the cached list is the merged list.
    if is_single_no_repeat:
        cache_dir = os.path.join(first_path, "map_style_cache")
        cache_suffix = "file_info.pkl"
    else:
        neutral_root = os.environ.get(
            "FASTVIDEO_MAP_STYLE_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "fastvideo", "map_style_cache"),
        )
        cache_dir = neutral_root
        cache_suffix = (
            "file_info_"
            + hashlib.md5(effective_path.encode()).hexdigest()[:16]
            + ".pkl"
        )
    cache_file = os.path.join(cache_dir, cache_suffix)

    # Only rank 0 checks for cache and scans files if needed
    if get_world_rank() == 0:
        cache_loaded = False
        file_names_sorted = None
        lengths_sorted = None

        # First try to load existing cache
        if os.path.exists(cache_file):
            logger.info("Loading cached file info from %s", cache_file)
            try:
                with open(cache_file, "rb") as f:
                    file_names_sorted, lengths_sorted = pickle.load(f)
                cache_loaded = True
                logger.info("Successfully loaded cached file info")
            except Exception as e:
                logger.error("Error loading cached file info: %s", str(e))
                logger.info("Falling back to scanning files")
                cache_loaded = False

        # If cache not loaded (either doesn't exist or failed to load), scan files
        if not cache_loaded:
            logger.info(
                "Scanning parquet files (path specs: %s)",
                [(p, c) for p, c in path_specs],
            )
            # Build list with repeats; use (path, length, sort_index) for stable order
            # Skip paths with count 0 (no I/O for disabled paths)
            combined: list[tuple[str, int, int]] = []
            sort_index = 0
            for p, count in path_specs:
                if count == 0:
                    continue
                fnames, lens = _scan_parquet_files_for_path(p)
                for _ in range(count):
                    for f, ln in zip(fnames, lens, strict=True):
                        combined.append((f, ln, sort_index))
                        sort_index += 1
            if not combined:
                raise ValueError(
                    "No parquet files found in the dataset (paths: %s)"
                    % [p for p, _ in path_specs]
                )
            combined.sort(key=lambda x: (x[0], x[2]))
            file_names_sorted = tuple(x[0] for x in combined)
            lengths_sorted = tuple(x[1] for x in combined)

            # Save the cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((file_names_sorted, lengths_sorted), f)
            logger.info("Saved file info to %s", cache_file)

    # Wait for rank 0 to finish creating/loading cache
    world_group = get_world_group()
    world_group.barrier()

    # Now all ranks load the cache (it should exist and be valid now)
    logger.info("Loading cached file info from %s after barrier", cache_file)
    with open(cache_file, "rb") as f:
        file_names_sorted, lengths_sorted = pickle.load(f)

    return file_names_sorted, lengths_sorted


def read_row_from_parquet_file(parquet_files: list[str], global_row_idx: int,
                               lengths: list[int]) -> dict[str, Any]:
    '''
    Read a row from a parquet file.
    Args:
        parquet_files: List[str]
        global_row_idx: int
        lengths: List[int]
    Returns:
    '''
    # find the parquet file and local row index
    cumulative = 0
    file_index = 0
    local_row_idx = 0

    for file_index in range(len(lengths)):
        if cumulative + lengths[file_index] > global_row_idx:
            local_row_idx = global_row_idx - cumulative
            break
        cumulative += lengths[file_index]
    else:
        # If we reach here, global_row_idx is out of bounds
        raise IndexError(
            f"global_row_idx {global_row_idx} is out of bounds for dataset")

    parquet_file = pq.ParquetFile(parquet_files[file_index])

    # Calculate the row group to read into memory and the local idx
    # This way we can avoid reading in the entire parquet file
    cumulative = 0
    row_group_index = 0
    local_index = 0

    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if cumulative + num_rows > local_row_idx:
            row_group_index = i
            local_index = local_row_idx - cumulative
            break
        cumulative += num_rows
    else:
        # If we reach here, local_row_idx is out of bounds for this parquet file
        raise IndexError(
            f"local_row_idx {local_row_idx} is out of bounds for parquet file {parquet_files[file_index]}"
        )

    row_group = parquet_file.read_row_group(row_group_index).to_pydict()
    row_dict = {k: v[local_index] for k, v in row_group.items()}
    del row_group

    return row_dict


# ────────────────────────────────────────────────────────────────────────────
# 2.  Dataset with batched __getitems__
# ────────────────────────────────────────────────────────────────────────────
class LatentsParquetMapStyleDataset(Dataset):
    """
    Return latents[B,C,T,H,W] and embeddings[B,L,D] in pinned CPU memory.
    Note: 
    Using parquet for map style dataset is not efficient, we mainly keep it for backward compatibility and debugging.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        parquet_schema: pa.Schema,
        cfg_rate: float = 0.0,
        seed: int = 42,
        drop_last: bool = True,
        drop_first_row: bool = False,
        reshuffle_each_epoch: bool = False,
        text_padding_length: int = 512,
    ):
        super().__init__()
        self.path = path
        self.cfg_rate = cfg_rate
        self.parquet_schema = parquet_schema
        self.seed = seed
        # Create a seeded random generator for deterministic CFG
        self.rng = random.Random(seed)
        logger.info("Initializing LatentsParquetMapStyleDataset with path: %s",
                    path)
        self.parquet_files, self.lengths = get_parquet_files_and_length(path)
        self.batch = batch_size
        self.text_padding_length = text_padding_length
        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=sum(self.lengths),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            reshuffle_each_epoch=reshuffle_each_epoch,
            seed=seed,
        )
        logger.info("Dataset initialized with %d parquet files and %d rows",
                    len(self.parquet_files), sum(self.lengths))

    def get_validation_negative_prompt(
            self) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get the negative prompt for validation. 
        This method ensures the negative prompt is loaded and cached properly.
        Returns the processed negative prompt data (latents, embeddings, masks, info).
        """

        # Read first row from first parquet file
        file_path = self.parquet_files[0]
        row_idx = 0
        # Read the negative prompt data
        row_dict = read_row_from_parquet_file([file_path], row_idx,
                                              [self.lengths[0]])

        batch = collate_rows_from_parquet_schema([row_dict],
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=0.0,
                                                 rng=self.rng)
        negative_prompt = batch['info_list'][0]['prompt']
        negative_prompt_embedding = batch['text_embedding']
        negative_prompt_attention_mask = batch['text_attention_mask']
        if len(negative_prompt_embedding.shape) == 2:
            negative_prompt_embedding = negative_prompt_embedding.unsqueeze(0)
        if len(negative_prompt_attention_mask.shape) == 1:
            negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(
                0).unsqueeze(0)

        return negative_prompt_embedding, negative_prompt_attention_mask, negative_prompt

    # PyTorch calls this ONLY because the batch_sampler yields a list
    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """
        Batch fetch using read_row_from_parquet_file for each index.
        """
        rows = [
            read_row_from_parquet_file(self.parquet_files, idx, self.lengths)
            for idx in indices
        ]

        batch = collate_rows_from_parquet_schema(rows,
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=self.cfg_rate,
                                                 rng=self.rng)
        return batch

    def __len__(self):
        return sum(self.lengths)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Loader helper – everything else stays just like your original trainer
# ────────────────────────────────────────────────────────────────────────────
def passthrough(batch):
    return batch


def build_parquet_map_style_dataloader(
        path,
        batch_size,
        num_data_workers,
        parquet_schema,
        cfg_rate=0.0,
        drop_last=True,
        drop_first_row=False,
        reshuffle_each_epoch=False,
        text_padding_length=512,
        seed=42) -> tuple[LatentsParquetMapStyleDataset, StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(
        path,
        batch_size,
        cfg_rate=cfg_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        reshuffle_each_epoch=reshuffle_each_epoch,
        text_padding_length=text_padding_length,
        parquet_schema=parquet_schema,
        seed=seed)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        pin_memory_device=current_platform.device_name,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
