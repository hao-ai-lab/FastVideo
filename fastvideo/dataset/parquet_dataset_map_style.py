# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import pickle
import random
from collections.abc import Sequence
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

DatasetPath = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
VALID_DATA_SPLITS = frozenset(("all", "train", "validation"))


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
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size

        # ── epoch-level RNG ────────────────────────────────────────────────
        rng = torch.Generator().manual_seed(self.seed)
        # Create a random permutation of all indices
        global_indices = torch.randperm(self.dataset_size, generator=rng)

        if drop_first_row:
            # drop 0 in global_indices
            global_indices = global_indices[global_indices != 0]
            self.dataset_size = self.dataset_size - 1

        if self.drop_last:
            # For drop_last=True, we:
            # 1. Ensure total samples is divisible by (batch_size * num_sp_groups)
            # 2. This guarantees each SP group gets same number of complete batches
            # 3. Prevents uneven batch sizes across SP groups at end of epoch
            num_batches = self.dataset_size // self.batch_size
            num_global_batches = num_batches // self.num_sp_groups
            global_indices = global_indices[:num_global_batches *
                                            self.num_sp_groups *
                                            self.batch_size]
        else:
            if self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                # add more indices to make it divisible by (batch_size * num_sp_groups)
                padding_size = self.num_sp_groups * self.batch_size - (
                    self.dataset_size % (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d",
                            self.dataset_size, self.dataset_size + padding_size)
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]])

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.
                                                num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d",
                    len(sp_group_local_indices))

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


def _normalize_dataset_paths(path: DatasetPath) -> tuple[str, ...]:
    if isinstance(path, (str, os.PathLike)):
        raw_path = os.fspath(path).strip()
        paths = raw_path.split(os.pathsep) if os.pathsep in raw_path else [raw_path]
    elif isinstance(path, Sequence):
        paths = [os.fspath(p).strip() for p in path]
    else:
        raise TypeError(
            "Dataset path must be a path string, an os.PathLike object, "
            "or a sequence of path strings.")

    paths = [p for p in paths if p]
    if not paths:
        raise ValueError("Dataset path must contain at least one non-empty path.")
    return tuple(paths)


def _normalize_data_split(data_split: str) -> str:
    normalized = (data_split or "all").strip().lower()
    if normalized == "val":
        normalized = "validation"
    if normalized not in VALID_DATA_SPLITS:
        valid = ", ".join(sorted(VALID_DATA_SPLITS | {"val"}))
        raise ValueError(
            f"data_split must be one of: {valid}. Got {data_split!r}.")
    return normalized


def _validate_split_ratio(data_split: str,
                          validation_split_ratio: float) -> None:
    if validation_split_ratio < 0.0 or validation_split_ratio >= 1.0:
        raise ValueError(
            "validation_split_ratio must be in the range [0.0, 1.0). "
            f"Got {validation_split_ratio}.")
    if data_split != "all" and validation_split_ratio <= 0.0:
        raise ValueError(
            "validation_split_ratio must be > 0.0 when data_split is "
            f"{data_split!r}.")


def _validation_score(sample_key: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{sample_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big") / float(1 << 64)


def _is_validation_sample(sample_key: str,
                          validation_split_ratio: float,
                          seed: int) -> bool:
    return _validation_score(sample_key, seed) < validation_split_ratio


def _split_cache_file(dataset_root: str,
                      data_split: str,
                      validation_split_ratio: float,
                      seed: int) -> str:
    ratio_token = f"{validation_split_ratio:.8f}".rstrip("0").rstrip(".")
    ratio_token = ratio_token.replace(".", "p") or "0"
    return os.path.join(
        dataset_root,
        "map_style_cache",
        f"split_indices_{data_split}_ratio_{ratio_token}_seed_{seed}.pkl",
    )


def _get_single_parquet_files_and_length(path: str):
    dataset_root = os.path.realpath(os.path.expanduser(path))
    # Check if cached info exists
    cache_dir = os.path.join(dataset_root, "map_style_cache")
    cache_file = os.path.join(cache_dir, "file_info.pkl")

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
                file_names_sorted = tuple(
                    os.path.realpath(
                        os.path.join(os.getcwd(), p)
                        if not os.path.isabs(p) else p)
                    for p in file_names_sorted)
                files_outside_dataset_root = [
                    file_path for file_path in file_names_sorted
                    if os.path.commonpath([dataset_root, file_path
                                           ]) != dataset_root
                ]
                missing_files = [
                    file_path for file_path in file_names_sorted
                    if not os.path.exists(file_path)
                ]
                if files_outside_dataset_root:
                    logger.warning(
                        "Cached parquet file list points outside dataset root "
                        "(%s). Cache will be rebuilt. First out-of-root file: %s",
                        dataset_root,
                        files_outside_dataset_root[0],
                    )
                    cache_loaded = False
                elif missing_files:
                    logger.warning(
                        "Cached parquet file list contains %d missing files. "
                        "Cache will be rebuilt. First missing file: %s",
                        len(missing_files),
                        missing_files[0],
                    )
                    cache_loaded = False
                else:
                    cache_loaded = True
                    logger.info("Successfully loaded cached file info")
            except Exception as e:
                logger.error("Error loading cached file info: %s", str(e))
                logger.info("Falling back to scanning files")
                cache_loaded = False

        # If cache not loaded (either doesn't exist or failed to load), scan files
        if not cache_loaded:
            logger.info("Scanning parquet files to get lengths")
            lengths = []
            file_names = []
            for root, _, files in os.walk(dataset_root):
                for file in sorted(files):
                    if file.endswith('.parquet'):
                        file_path = os.path.realpath(os.path.join(root, file))
                        file_names.append(file_path)
            if len(file_names) == 0:
                raise FileNotFoundError(
                    "No parquet files found under dataset path: "
                    f"{path}. "
                    "Please verify this path points to preprocessed parquet "
                    "data.")
            for file_path in tqdm.tqdm(
                    file_names, desc="Reading parquet files to get lengths"):
                num_rows = pq.ParquetFile(file_path).metadata.num_rows
                lengths.append(num_rows)
            # sort according to file name to ensure all rank has the same order
            file_names_sorted, lengths_sorted = zip(*sorted(zip(file_names,
                                                                lengths,
                                                                strict=True),
                                                            key=lambda x: x[0]),
                                                    strict=True)
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
    if len(file_names_sorted) == 0:
        raise RuntimeError(
            "Cached parquet metadata is empty after synchronization at "
            f"{cache_file}. "
            "Please verify the dataset path and regenerate cache.")
    if len(file_names_sorted) != len(lengths_sorted):
        raise RuntimeError(
            "Cached parquet metadata is corrupted at "
            f"{cache_file}: file count and length count do not match.")

    return file_names_sorted, lengths_sorted


def _select_split_row_indices(
    file_names: Sequence[str],
    lengths: Sequence[int],
    data_split: str,
    validation_split_ratio: float,
    seed: int,
) -> tuple[int, ...]:
    selected_indices: list[int] = []
    global_row_idx = 0

    for file_path, expected_rows in tqdm.tqdm(
            zip(file_names, lengths, strict=True),
            total=len(file_names),
            desc=f"Selecting {data_split} split rows"):
        parquet_file = pq.ParquetFile(file_path)
        num_rows = parquet_file.metadata.num_rows
        if num_rows != expected_rows:
            raise RuntimeError(
                "Parquet row count changed while selecting split rows for "
                f"{file_path}: expected {expected_rows}, found {num_rows}.")

        schema_names = set(parquet_file.schema.names)
        key_columns = [
            column for column in ("id", "file_name")
            if column in schema_names
        ]
        row_data = (parquet_file.read(
            columns=key_columns).to_pydict() if key_columns else {})

        for local_row_idx in range(num_rows):
            sample_key = ""
            for column in key_columns:
                value = row_data[column][local_row_idx]
                if value:
                    sample_key = str(value)
                    break
            if not sample_key:
                sample_key = f"{file_path}:{local_row_idx}"

            in_validation = _is_validation_sample(
                sample_key,
                validation_split_ratio,
                seed,
            )
            if ((data_split == "validation" and in_validation) or
                    (data_split == "train" and not in_validation)):
                selected_indices.append(global_row_idx)

            global_row_idx += 1

    total_rows = sum(lengths)
    if global_row_idx != total_rows:
        raise RuntimeError(
            "Split row selection saw a different number of rows than the "
            f"cached metadata: saw {global_row_idx}, expected {total_rows}.")
    if not selected_indices:
        logger.warning(
            "Selected %s split is empty for ratio=%s and seed=%s",
            data_split,
            validation_split_ratio,
            seed,
        )
    return tuple(selected_indices)


def _get_single_split_row_indices(path: str,
                                  file_names: Sequence[str],
                                  lengths: Sequence[int],
                                  data_split: str,
                                  validation_split_ratio: float,
                                  seed: int) -> tuple[int, ...]:
    dataset_root = os.path.realpath(os.path.expanduser(path))
    cache_file = _split_cache_file(
        dataset_root,
        data_split,
        validation_split_ratio,
        seed,
    )

    if get_world_rank() == 0:
        cache_loaded = False
        split_indices: tuple[int, ...] | None = None
        if os.path.exists(cache_file):
            logger.info("Loading cached split indices from %s", cache_file)
            try:
                with open(cache_file, "rb") as f:
                    cached_file_names, cached_lengths, split_indices = pickle.load(
                        f)
                cache_loaded = (
                    tuple(cached_file_names) == tuple(file_names) and
                    tuple(cached_lengths) == tuple(lengths))
                if not cache_loaded:
                    logger.warning(
                        "Cached split indices at %s do not match parquet "
                        "metadata. Cache will be rebuilt.",
                        cache_file,
                    )
            except Exception as e:
                logger.error("Error loading split cache: %s", str(e))
                cache_loaded = False

        if not cache_loaded:
            logger.info(
                "Building %s split indices with validation ratio=%s seed=%s",
                data_split,
                validation_split_ratio,
                seed,
            )
            split_indices = _select_split_row_indices(
                file_names,
                lengths,
                data_split,
                validation_split_ratio,
                seed,
            )
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((tuple(file_names), tuple(lengths), split_indices),
                            f)
            logger.info("Saved split indices to %s", cache_file)

    world_group = get_world_group()
    world_group.barrier()

    with open(cache_file, "rb") as f:
        _, _, split_indices = pickle.load(f)
    return tuple(split_indices)


def get_parquet_files_and_length(path: DatasetPath):
    dataset_roots = _normalize_dataset_paths(path)
    if len(dataset_roots) == 1:
        return _get_single_parquet_files_and_length(dataset_roots[0])

    all_file_names = []
    all_lengths = []
    for dataset_root in dataset_roots:
        file_names, lengths = _get_single_parquet_files_and_length(
            dataset_root)
        all_file_names.extend(file_names)
        all_lengths.extend(lengths)

    return tuple(all_file_names), tuple(all_lengths)


def get_parquet_split_indices(path: DatasetPath,
                              data_split: str,
                              validation_split_ratio: float,
                              seed: int) -> tuple[int, ...]:
    data_split = _normalize_data_split(data_split)
    _validate_split_ratio(data_split, validation_split_ratio)

    all_indices: list[int] = []
    offset = 0
    for dataset_root in _normalize_dataset_paths(path):
        file_names, lengths = _get_single_parquet_files_and_length(
            dataset_root)
        root_length = sum(lengths)
        if data_split == "all":
            root_indices = range(root_length)
        else:
            root_indices = _get_single_split_row_indices(
                dataset_root,
                file_names,
                lengths,
                data_split,
                validation_split_ratio,
                seed,
            )
        all_indices.extend(offset + idx for idx in root_indices)
        offset += root_length

    return tuple(all_indices)


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
        path: DatasetPath,
        batch_size: int,
        parquet_schema: pa.Schema,
        cfg_rate: float = 0.0,
        seed: int = 42,
        drop_last: bool = True,
        drop_first_row: bool = False,
        text_padding_length: int = 512,
        data_split: str = "all",
        validation_split_ratio: float = 0.0,
    ):
        super().__init__()
        self.path = path
        self.cfg_rate = cfg_rate
        self.parquet_schema = parquet_schema
        self.seed = seed
        self.data_split = _normalize_data_split(data_split)
        self.validation_split_ratio = float(validation_split_ratio)
        _validate_split_ratio(self.data_split, self.validation_split_ratio)
        # Create a seeded random generator for deterministic CFG
        self.rng = random.Random(seed)
        logger.info("Initializing LatentsParquetMapStyleDataset with path(s): %s",
                    _normalize_dataset_paths(path))
        self.parquet_files, self.lengths = get_parquet_files_and_length(path)
        if self.data_split == "all":
            self.sample_indices = None
            dataset_size = sum(self.lengths)
        else:
            self.sample_indices = get_parquet_split_indices(
                path,
                self.data_split,
                self.validation_split_ratio,
                seed,
            )
            dataset_size = len(self.sample_indices)
        self.batch = batch_size
        self.text_padding_length = text_padding_length
        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=dataset_size,
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )
        logger.info(
            "Dataset initialized with %d parquet files, %d source rows, "
            "%d selected rows for data_split=%s",
            len(self.parquet_files),
            sum(self.lengths),
            dataset_size,
            self.data_split,
        )

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
        row_indices = [
            self.sample_indices[idx]
            if self.sample_indices is not None else idx
            for idx in indices
        ]
        rows = [
            read_row_from_parquet_file(self.parquet_files, idx, self.lengths)
            for idx in row_indices
        ]

        # Inject sample indices for deterministic CFG dropout
        # that is reproducible across checkpoint resume.
        for row, idx in zip(rows, row_indices):
            row["_sample_index"] = idx

        batch = collate_rows_from_parquet_schema(rows,
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=self.cfg_rate,
                                                 seed=self.seed)
        return batch

    def __len__(self):
        if self.sample_indices is not None:
            return len(self.sample_indices)
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
        text_padding_length=512,
        seed=42,
        data_split="all",
        validation_split_ratio=0.0) -> tuple[LatentsParquetMapStyleDataset, StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(
        path,
        batch_size,
        cfg_rate=cfg_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        text_padding_length=text_padding_length,
        parquet_schema=parquet_schema,
        seed=seed,
        data_split=data_split,
        validation_split_ratio=validation_split_ratio)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
