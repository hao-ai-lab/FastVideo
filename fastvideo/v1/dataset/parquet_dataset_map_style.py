# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
# Torch in general
import torch
# Dataset
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

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
            # add more indices to make it divisible by (batch_size * num_sp_groups)
            padding_size = self.num_sp_groups * self.batch_size - (
                self.dataset_size % (self.num_sp_groups * self.batch_size))
            global_indices = torch.cat(
                [global_indices, global_indices[:padding_size]])

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.
                                                num_sp_groups]

        self.sp_group_local_indices = sp_group_local_indices
        logger.info("sp_group_local_indices: %d", len(sp_group_local_indices))

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


def get_parquet_files_and_length(path: str):
    lengths = []
    file_names = []
    for root, _, files in os.walk(path):
        for file in sorted(files):
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                num_rows = pq.ParquetFile(file_path).metadata.num_rows
                lengths.append(num_rows)
                file_names.append(file_path)
    # sort according to file name to ensure all rank has the same order (in case os.walk is not sorted)
    file_names_sorted, lengths_sorted = zip(
        *sorted(zip(file_names, lengths), key=lambda x: x[0]))
    assert len(file_names_sorted) != 0, "No parquet files found in the dataset"
    return file_names_sorted, lengths_sorted


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
    # Modify this in the future if we want to add more keys, for example, in image to video.
    keys = ["vae_latent", "text_embedding"]

    def __init__(
        self,
        path: str,
        batch_size: int,
        cfg_rate: float = 0.0,
        seed: int = 42,
        drop_last: bool = True,
        text_padding_length: int = 512,
    ):
        super().__init__()
        self.path = path
        self.cfg_rate = cfg_rate
        if cfg_rate > 0.0:
            raise ValueError(
                "cfg_rate > 0.0 is not supported for now because it will trigger bug when num_data_workers > 0"
            )
        logger.info("Initializing LatentsParquetMapStyleDataset with path: %s",
                    path)
        self.parquet_files, self.lengths = get_parquet_files_and_length(path)
        self.batch = batch_size
        self.text_padding_length = text_padding_length
        self._cols = [
            "vae_latent_bytes",
            "vae_latent_shape",
            "text_embedding_bytes",
            "text_embedding_shape",
            "text_embedding_dtype",
            "height",
            "width",
        ]
        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=sum(self.lengths),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            seed=seed,
        )
        logger.info("Dataset initialized with %d parquet files and %d rows",
                    len(self.parquet_files), sum(self.lengths))

    def _get_torch_tensors_from_row_dict(
            self, row_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Get the latents and prompts from a row dictionary.
        """
        return_dict = {}
        for key in self.keys:
            shape = row_dict[f"{key}_shape"]
            bytes = row_dict[f"{key}_bytes"]
            # TODO (peiyuan): read precision
            data = np.frombuffer(bytes, dtype=np.float32).reshape(shape).copy()
            data = torch.from_numpy(data)
            return_dict[key] = data
        return return_dict

    def get_validation_negative_prompt(self) -> tuple[Any, Any, Any, Any]:
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

        # Get tensors using the existing helper method
        data = self._get_torch_tensors_from_row_dict(row_dict)
        emb = data["text_embedding"]

        # Pad the embedding and get mask
        padded_emb, mask = self._pad(emb, self.text_padding_length)

        # Pin memory for faster transfer to GPU
        padded_emb = padded_emb
        mask = mask

        return None, padded_emb, mask, None

    def _pad(self, t: torch.Tensor, padding_length: int) -> torch.Tensor:
        """
        Pad or crop an embedding [L, D] to exactly padding_length tokens.
        Return:
        - [L, D] tensor in pinned CPU memory
        - [L] attention mask in pinned CPU memory
        """
        L, D = t.shape
        if padding_length > L:  # pad
            pad = torch.zeros(padding_length - L,
                              D,
                              dtype=t.dtype,
                              device=t.device)
            return torch.cat([t, pad], 0), torch.cat(
                [torch.ones(L), torch.zeros(padding_length - L)], 0)
        else:  # crop
            return t[:padding_length], torch.ones(padding_length)

    # PyTorch calls this ONLY because the batch_sampler yields a list
    def __getitems__(self, indices: List[int]):
        """
        Batch fetch using read_row_from_parquet_file for each index.
        """
        rows = [
            read_row_from_parquet_file(self.parquet_files, idx, self.lengths)
            for idx in indices
        ]

        # Initialize tensors to hold padded embeddings and masks
        all_latents = []
        all_embs = []
        all_masks = []

        # Process each row individually
        for i, row in enumerate(rows):
            # Get tensors from row
            data = self._get_torch_tensors_from_row_dict(row)
            latents, emb = data["vae_latent"], data["text_embedding"]

            padded_emb, mask = self._pad(emb, self.text_padding_length)
            # Store in batch tensors
            all_latents.append(latents)
            all_embs.append(padded_emb)
            all_masks.append(mask)

        # Pin memory for faster transfer to GPU
        all_latents = torch.stack(all_latents)
        all_embs = torch.stack(all_embs)
        all_masks = torch.stack(all_masks)

        return all_latents, all_embs, all_masks, indices

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
        cfg_rate=0.0,
        drop_last=True,
        text_padding_length=512,
        seed=42) -> Tuple[LatentsParquetMapStyleDataset, StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(
        path,
        batch_size,
        cfg_rate=cfg_rate,
        drop_last=drop_last,
        text_padding_length=text_padding_length,
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
