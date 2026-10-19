# SPDX-License-Identifier: Apache-2.0
import os
import pickle
import random
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from numbers import Integral
from pathlib import Path
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

from fastvideo.dataset.shape_bucket import parse_video_shape_bucket_id
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import (get_sp_world_size, get_world_group, get_world_rank, get_world_size)
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
        seed: int = 0,
        sample_bucket_ids: Sequence[str] | None = None,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size

        # ── epoch-level RNG ────────────────────────────────────────────────
        if batch_size <= 0 or num_sp_groups <= 0 or sp_world_size <= 0:
            raise ValueError("batch_size, num_sp_groups, and sp_world_size must be positive")

        rng = torch.Generator().manual_seed(self.seed)
        if sample_bucket_ids is None:
            # Legacy behavior: one permutation over the complete dataset.
            global_indices = torch.randperm(self.dataset_size, generator=rng)
            if drop_first_row:
                global_indices = global_indices[global_indices != 0]
                self.dataset_size -= 1

            if self.drop_last:
                num_batches = self.dataset_size // self.batch_size
                num_global_batches = num_batches // self.num_sp_groups
                global_indices = global_indices[:num_global_batches * self.num_sp_groups * self.batch_size]
            elif self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                padding_size = self.num_sp_groups * self.batch_size - (self.dataset_size %
                                                                       (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d", self.dataset_size, self.dataset_size + padding_size)
                global_indices = torch.cat([global_indices, global_indices[:padding_size]])
            self.bucket_schedule: tuple[str, ...] | None = None
            self.bucket_padding: dict[str, int] | None = None
            self.num_padded_samples = 0
        else:
            if len(sample_bucket_ids) != self.dataset_size:
                raise ValueError("sample_bucket_ids must contain one identifier per dataset row, got "
                                 f"{len(sample_bucket_ids)} for dataset_size={self.dataset_size}")
            global_indices, self.bucket_schedule = self._bucketed_global_schedule(
                sample_bucket_ids,
                rng=rng,
                drop_first_row=drop_first_row,
            )
            if drop_first_row:
                self.dataset_size -= 1

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d", len(sp_group_local_indices))

    def _bucketed_global_schedule(
        self,
        sample_bucket_ids: Sequence[str],
        *,
        rng: torch.Generator,
        drop_first_row: bool,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Build same-shape rounds shared by every data-parallel group.

        One round contains ``num_sp_groups * batch_size`` rows from exactly
        one bucket. Strided DP sharding below gives every group a distinct
        local batch while preserving the same bucket at that microstep. SP
        ranks map to the same group and therefore receive identical indices.
        """
        by_bucket: dict[str, list[int]] = defaultdict(list)
        for index, bucket_id in enumerate(sample_bucket_ids):
            if drop_first_row and index == 0:
                continue
            parse_video_shape_bucket_id(bucket_id)
            by_bucket[bucket_id].append(index)

        samples_per_round = self.num_sp_groups * self.batch_size
        rounds: list[torch.Tensor] = []
        round_bucket_ids: list[str] = []
        bucket_padding: dict[str, int] = {}
        padded = 0
        for bucket_id in sorted(by_bucket):
            bucket_indices = torch.tensor(by_bucket[bucket_id], dtype=torch.long)
            bucket_indices = bucket_indices[torch.randperm(len(bucket_indices), generator=rng)]
            remainder = len(bucket_indices) % samples_per_round
            if remainder:
                # Native bucketing must retain every frozen row, including a
                # rare bucket smaller than one global microbatch. Repeat only
                # within that bucket; legacy drop_last behavior remains in the
                # unbucketed branch above.
                padding_size = samples_per_round - remainder
                repeats = (padding_size + len(bucket_indices) - 1) // len(bucket_indices)
                padding = bucket_indices.repeat(repeats)[:padding_size]
                bucket_indices = torch.cat((bucket_indices, padding))
                padded += padding_size
                bucket_padding[bucket_id] = padding_size
                logger.info(
                    "Exact-shape bucket %s has %d row(s); repeated %d row(s) to fill global microbatches of %d",
                    bucket_id,
                    len(by_bucket[bucket_id]),
                    padding_size,
                    samples_per_round,
                )
            bucket_rounds = list(bucket_indices.reshape(-1, samples_per_round).unbind(0))
            rounds.extend(bucket_rounds)
            round_bucket_ids.extend([bucket_id] * len(bucket_rounds))

        if not rounds:
            raise ValueError("Exact-shape bucketing requires at least one dataset row")
        round_order = torch.randperm(len(rounds), generator=rng).tolist()
        self.bucket_padding = bucket_padding
        self.num_padded_samples = padded
        if padded:
            logger.info("Exact-shape bucketing repeated %d row(s) to fill bucket-local global microbatches", padded)
        return (
            torch.cat([rounds[index] for index in round_order]),
            tuple(round_bucket_ids[index] for index in round_order),
        )

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


def _shape_bucket_id_from_parquet_path(file_path: str) -> str:
    """Return and validate the sole ``bucket=...`` ancestor of a parquet."""
    matches = [part for part in Path(file_path).parts if part.startswith("bucket=")]
    if len(matches) != 1:
        raise ValueError(
            "Native-shape parquet paths must have exactly one ancestor named "
            "'bucket=<width>x<height>-<num_frames>f', got "
            f"{file_path!r} with bucket ancestors {matches}"
        )
    bucket_id = matches[0]
    parse_video_shape_bucket_id(bucket_id)
    return bucket_id


def shape_bucket_ids_from_parquet_files(
    parquet_files: Sequence[str],
    lengths: Sequence[int],
) -> list[str]:
    """Expand canonical path bucket IDs to one identifier per dataset row."""
    if len(parquet_files) != len(lengths):
        raise ValueError("parquet_files and lengths must have matching lengths")
    sample_bucket_ids: list[str] = []
    for file_path, length in zip(parquet_files, lengths, strict=True):
        if int(length) < 0:
            raise ValueError(f"Parquet row counts must be non-negative, got {length}")
        bucket_id = _shape_bucket_id_from_parquet_path(str(file_path))
        sample_bucket_ids.extend([bucket_id] * int(length))
    return sample_bucket_ids


def _parse_data_path_specs(path: str | Sequence[str] | dict[str, int]) -> list[tuple[str, int]]:
    """Parse one or more dataset roots with old-framework repeat counts."""
    if isinstance(path, dict):
        return [(str(root), int(repeat)) for root, repeat in path.items() if int(repeat) > 0]
    if isinstance(path, Sequence) and not isinstance(path, str):
        return [(str(root), 1) for root in path]

    specs: list[tuple[str, int]] = []
    for part in str(path).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            dir_path, count_str = part.rsplit(":", 1)
            count = int(count_str)
        else:
            dir_path, count = part, 1
        if count > 0:
            specs.append((dir_path.strip(), count))
    return specs


def _resolve_cached_parquet_file(file_path: str) -> tuple[str, bool]:
    resolved = os.path.realpath(file_path, strict=True)
    return resolved, os.path.isfile(resolved)


def _resolve_cached_parquet_files(file_names: Sequence[str], *, num_workers: int) -> Iterator[tuple[str, bool]]:
    """Yield (realpath, is_file) in input order; a bounded window overlaps per-file filesystem latency."""
    names = iter(file_names)
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="parquet-index") as pool:
        pending = deque(pool.submit(_resolve_cached_parquet_file, name) for name in islice(names, 2 * num_workers))
        while pending:
            result = pending.popleft().result()
            if (name := next(names, None)) is not None:
                pending.append(pool.submit(_resolve_cached_parquet_file, name))
            yield result


def _read_required_parquet_cache(dataset_root: str, cache_file: str):
    """Validate the published index without walking, repairing or reordering it."""
    resolved_cache = os.path.realpath(cache_file, strict=True)
    if os.path.commonpath([dataset_root, resolved_cache]) != dataset_root:
        raise ValueError(f"Parquet cache points outside dataset root: {cache_file}")
    with open(cache_file, "rb") as f:
        cached = pickle.load(f)
        if f.read(1):
            raise ValueError("Parquet cache contains trailing data")
    if not isinstance(cached, (list, tuple)) or len(cached) != 2:
        raise ValueError("Parquet cache must contain file names and row counts")
    file_names, lengths = cached
    if not isinstance(file_names, (list, tuple)) or not isinstance(lengths, (list, tuple)):
        raise ValueError("Parquet cache file names and row counts must be sequences")
    if not file_names or len(file_names) != len(lengths):
        raise ValueError("Parquet cache is empty or file count and length count do not match")
    for file_path, length in zip(file_names, lengths, strict=True):
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(f"Invalid cached parquet path: {file_path!r}")
        if isinstance(length, bool) or not isinstance(length, Integral) or length < 0:
            raise ValueError(f"Invalid cached parquet row count for {file_path!r}: {length!r}")
    declared_lengths: dict[str, int] = {}
    resolved_files = _resolve_cached_parquet_files(file_names, num_workers=32)
    for file_path, length, (resolved, is_file) in zip(file_names, lengths, resolved_files, strict=True):
        if os.path.commonpath([dataset_root, resolved]) != dataset_root:
            raise ValueError(f"Cached parquet path points outside dataset root: {file_path}")
        if not is_file:
            raise ValueError(f"Cached parquet path is not a file: {file_path}")
        if resolved in declared_lengths and declared_lengths[resolved] != length:
            raise ValueError(f"Repeated cached parquet path has inconsistent row counts: {file_path}")
        declared_lengths[resolved] = length
    if not any(lengths):
        raise ValueError("Parquet cache contains no rows")
    # Preserve the exact cached path spellings, order, repeats and row counts.
    return file_names, lengths


def get_parquet_files_and_length(path: str | Sequence[str] | dict[str, int], *,
                                  require_existing_parquet_cache: bool = False):
    if type(require_existing_parquet_cache) is not bool:
        raise ValueError("require_existing_parquet_cache must be a boolean")
    specs = _parse_data_path_specs(path)
    if len(specs) != 1 or specs[0][1] != 1:
        all_file_names: list[str] = []
        all_lengths: list[int] = []
        for root, repeat in specs:
            file_names, lengths = get_parquet_files_and_length(
                str(root), require_existing_parquet_cache=require_existing_parquet_cache)
            for _ in range(repeat):
                all_file_names.extend(file_names)
                all_lengths.extend(lengths)
        if not all_file_names:
            raise FileNotFoundError("No parquet files found under dataset paths: "
                                    f"{path}. "
                                    "Please verify these paths point to preprocessed parquet data.")
        file_lengths = sorted(
            zip(all_file_names, all_lengths, strict=True),
            key=lambda x: x[0],
        )
        file_names_sorted, lengths_sorted = zip(*file_lengths, strict=True)
        return file_names_sorted, lengths_sorted

    dataset_root = os.path.realpath(os.path.expanduser(specs[0][0]))
    # Check if cached info exists
    cache_dir = os.path.join(dataset_root, "map_style_cache")
    cache_file = os.path.join(cache_dir, "file_info.pkl")

    if require_existing_parquet_cache:
        cached, error = None, None
        if get_world_rank() == 0:
            try:
                cached = _read_required_parquet_cache(dataset_root, cache_file)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        # Publish both success and failure; no peer waits at the fallback
        # barrier or rereads a potentially changed index after rank zero.
        cached, error = get_world_group().broadcast_object((cached, error), src=0)
        if error is not None:
            raise RuntimeError(f"Required parquet cache failed validation at {cache_file}: {error}")
        if cached is None:
            raise RuntimeError(f"Rank zero did not publish required parquet cache: {cache_file}")
        return cached

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
                    os.path.realpath(os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p)
                    for p in file_names_sorted)
                files_outside_dataset_root = [
                    file_path for file_path in file_names_sorted
                    if os.path.commonpath([dataset_root, file_path]) != dataset_root
                ]
                missing_files = [file_path for file_path in file_names_sorted if not os.path.exists(file_path)]
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
                raise FileNotFoundError("No parquet files found under dataset path: "
                                        f"{path}. "
                                        "Please verify this path points to preprocessed parquet "
                                        "data.")
            for file_path in tqdm.tqdm(file_names, desc="Reading parquet files to get lengths"):
                num_rows = pq.ParquetFile(file_path).metadata.num_rows
                lengths.append(num_rows)
            # sort according to file name to ensure all rank has the same order
            file_names_sorted, lengths_sorted = zip(*sorted(zip(file_names, lengths, strict=True), key=lambda x: x[0]),
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
        raise RuntimeError("Cached parquet metadata is empty after synchronization at "
                           f"{cache_file}. "
                           "Please verify the dataset path and regenerate cache.")
    if len(file_names_sorted) != len(lengths_sorted):
        raise RuntimeError("Cached parquet metadata is corrupted at "
                           f"{cache_file}: file count and length count do not match.")

    return file_names_sorted, lengths_sorted


def read_row_from_parquet_file(
    parquet_files: list[str],
    global_row_idx: int,
    lengths: list[int],
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
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
        raise IndexError(f"global_row_idx {global_row_idx} is out of bounds for dataset")

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
        raise IndexError(f"local_row_idx {local_row_idx} is out of bounds for parquet file {parquet_files[file_index]}")

    # Project at the Parquet reader boundary. This is especially important for
    # data-free training over a T2VA superset: the text-only schema must not
    # pull hundreds of MiB of unused video/audio latent bytes into host memory.
    row_group = parquet_file.read_row_group(row_group_index, columns=columns).to_pydict()
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
        path: str | Sequence[str] | dict[str, int],
        batch_size: int,
        parquet_schema: pa.Schema,
        cfg_rate: float = 0.0,
        seed: int = 42,
        drop_last: bool = True,
        drop_first_row: bool = False,
        text_padding_length: int | None = 512,
        native_shape_bucketing: bool = False,
        max_presentation_tokens: int | None = None,
        require_existing_parquet_cache: bool = False,
    ):
        super().__init__()
        self.path = path
        self.cfg_rate = cfg_rate
        self.parquet_schema = parquet_schema
        self.seed = seed
        # Create a seeded random generator for deterministic CFG
        self.rng = random.Random(seed)
        logger.info("Initializing LatentsParquetMapStyleDataset with path: %s", path)
        self.parquet_files, self.lengths = get_parquet_files_and_length(
            path, require_existing_parquet_cache=require_existing_parquet_cache)
        self.batch = batch_size
        self.text_padding_length = text_padding_length
        self.max_presentation_tokens = max_presentation_tokens
        self.sample_bucket_ids = (
            shape_bucket_ids_from_parquet_files(self.parquet_files, self.lengths) if native_shape_bucketing else None
        )
        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=sum(self.lengths),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
            sample_bucket_ids=self.sample_bucket_ids,
        )
        logger.info("Dataset initialized with %d parquet files and %d rows", len(self.parquet_files), sum(self.lengths))

    def get_validation_negative_prompt(self) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get the negative prompt for validation. 
        This method ensures the negative prompt is loaded and cached properly.
        Returns the processed negative prompt data (latents, embeddings, masks, info).
        """

        # Read first row from first parquet file
        file_path = self.parquet_files[0]
        row_idx = 0
        # Read the negative prompt data
        row_dict = read_row_from_parquet_file(
            [file_path],
            row_idx,
            [self.lengths[0]],
            columns=self.parquet_schema.names,
        )

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
            negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(0).unsqueeze(0)

        return negative_prompt_embedding, negative_prompt_attention_mask, negative_prompt

    # PyTorch calls this ONLY because the batch_sampler yields a list
    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """
        Batch fetch using read_row_from_parquet_file for each index.
        """
        rows = [
            read_row_from_parquet_file(
                self.parquet_files,
                idx,
                self.lengths,
                columns=self.parquet_schema.names,
            ) for idx in indices
        ]

        # Inject sample indices for deterministic CFG dropout
        # that is reproducible across checkpoint resume.
        for row, idx in zip(rows, indices):
            row["_sample_index"] = idx

        batch = collate_rows_from_parquet_schema(rows,
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=self.cfg_rate,
                                                 seed=self.seed,
                                                 max_presentation_tokens=self.max_presentation_tokens)
        if self.sample_bucket_ids is not None:
            bucket_ids = {self.sample_bucket_ids[index] for index in indices}
            if len(bucket_ids) != 1:
                raise RuntimeError(f"Exact-shape sampler emitted a mixed bucket batch: {sorted(bucket_ids)}")
            batch["_shape_bucket_id"] = bucket_ids.pop()
        return batch

    def __len__(self):
        return sum(self.lengths)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Loader helper – everything else stays just like your original trainer
# ────────────────────────────────────────────────────────────────────────────
def passthrough(batch):
    return batch


def build_parquet_map_style_dataloader(path,
                                       batch_size,
                                       num_data_workers,
                                       parquet_schema,
                                       cfg_rate=0.0,
                                       drop_last=True,
                                       drop_first_row=False,
                                       text_padding_length=512,
                                       seed=42,
                                       native_shape_bucketing=False,
                                       max_presentation_tokens=None,
                                       require_existing_parquet_cache=False) -> tuple[LatentsParquetMapStyleDataset,
                                                                            StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(path,
                                            batch_size,
                                            cfg_rate=cfg_rate,
                                            drop_last=drop_last,
                                            drop_first_row=drop_first_row,
                                            text_padding_length=text_padding_length,
                                            parquet_schema=parquet_schema,
                                            seed=seed,
                                            native_shape_bucketing=native_shape_bucketing,
                                            max_presentation_tokens=max_presentation_tokens,
                                            require_existing_parquet_cache=require_existing_parquet_cache)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
