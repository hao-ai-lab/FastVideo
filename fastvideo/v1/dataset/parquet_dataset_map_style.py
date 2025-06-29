# SPDX-License-Identifier: Apache-2.0
import os
import pickle
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
# Torch in general
import torch
import tqdm
# Dataset
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.v1.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.v1.distributed import (get_sp_world_size, get_world_rank,
                                      get_world_size)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class DP_SP_BatchSampler(Sampler[List[int]]):
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


def get_parquet_files_and_length(path: str):
    # Check if cached info exists
    cache_dir = os.path.join(path, "map_style_cache")
    cache_file = os.path.join(cache_dir, "file_info.pkl")

    if os.path.exists(cache_file):
        logger.info("Loading cached file info from %s", cache_file)
        try:
            with open(cache_file, "rb") as f:
                file_names_sorted, lengths_sorted = pickle.load(f)
            return file_names_sorted, lengths_sorted
        except Exception as e:
            logger.error("Error loading cached file info: %s", str(e))
            logger.info("Falling back to scanning files")

    # If no cache exists or loading failed, scan files
    if get_world_rank() == 0:
        lengths = []
        file_names = []
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    file_names.append(file_path)
        for file_path in tqdm.tqdm(file_names,
                                   desc="Reading parquet files to get lengths"):
            num_rows = pq.ParquetFile(file_path).metadata.num_rows
            lengths.append(num_rows)
        # sort according to file name to ensure all rank has the same order (in case os.walk is not sorted)
        file_names_sorted, lengths_sorted = zip(
            *sorted(zip(file_names, lengths), key=lambda x: x[0]))
        assert len(
            file_names_sorted) != 0, "No parquet files found in the dataset"

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((file_names_sorted, lengths_sorted), f)
        logger.info("Saved file info to %s", cache_file)

    # Wait for rank 0 to finish saving
    if get_world_size() > 1:
        torch.distributed.barrier()

    return get_parquet_files_and_length(path)


def read_row_from_parquet_file(parquet_files: List[str], global_row_idx: int,
                               lengths: List[int]) -> Dict[str, Any]:
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
    for file_index in range(len(lengths)):
        if cumulative + lengths[file_index] > global_row_idx:
            local_row_idx = global_row_idx - cumulative
            break
        cumulative += lengths[file_index]

    parquet_file = pq.ParquetFile(parquet_files[file_index])

    # Calculate the row group to read into memory and the local idx
    # This way we can avoid reading in the entire parquet file
    cumulative = 0
    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if cumulative + num_rows > local_row_idx:
            row_group_index = i
            local_index = local_row_idx - cumulative
            break
        cumulative += num_rows

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
        text_padding_length: int = 512,
    ):
        super().__init__()
        self.path = path
        self.cfg_rate = cfg_rate
        self.parquet_schema = parquet_schema
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
                                                 cfg_rate=0.0)
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
    def __getitems__(self, indices: List[int]) -> Dict[str, Any]:
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
                                                 cfg_rate=self.cfg_rate)
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
        text_padding_length=512,
        seed=42) -> Tuple[LatentsParquetMapStyleDataset, StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(
        path,
        batch_size,
        cfg_rate=cfg_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        text_padding_length=text_padding_length,
        parquet_schema=parquet_schema,
        seed=seed)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
