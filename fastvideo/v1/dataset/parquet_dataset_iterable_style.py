import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.v1.dataset.utils import collate_latents_embs_masks
from fastvideo.v1.distributed import (get_sp_world_size, get_world_rank,
                                      get_world_size)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class BatchIterator:
    # TODO: Implement state_dict and load_state_dict to support resume.
    def __init__(self, files, batch_size, text_padding_length, keys,
                 worker_num_samples, read_batch_size):
        self.files = files
        self.batch_size = batch_size
        self.text_padding_length = text_padding_length
        self.keys = keys
        self.worker_num_samples = worker_num_samples
        self.processed_samples = 0
        self.buffer = []
        self.read_batch_size = read_batch_size

    def __iter__(self):
        for file in self.files:
            if self.processed_samples >= self.worker_num_samples:
                return

            reader = pq.ParquetFile(file)
            for batch in reader.iter_batches(batch_size=self.read_batch_size):
                if self.processed_samples >= self.worker_num_samples:
                    return

                self.buffer.extend(batch.to_pylist())

                while len(self.buffer) >= self.batch_size:
                    if self.processed_samples >= self.worker_num_samples:
                        return

                    batch_to_process = self.buffer[:self.batch_size]
                    self.buffer = self.buffer[self.batch_size:]

                    all_latents, all_embs, all_masks, caption_text = collate_latents_embs_masks(
                        batch_to_process, self.text_padding_length, self.keys)
                    self.processed_samples += self.batch_size
                    yield all_latents, all_embs, all_masks, caption_text


class LatentsParquetIterStyleDataset(IterableDataset):
    """Efficient loader for video-text data from a directory of Parquet files."""

    # Modify this in the future if we want to add more keys, for example, in image to video.
    keys = [("vae_latent", "latent"), ("text_embedding")]

    def __init__(self,
                 path: str,
                 batch_size: int = 1024,
                 cfg_rate: float = 0.1,
                 num_workers: int = 1,
                 drop_last: bool = True,
                 text_padding_length: int = 512,
                 seed: int = 42,
                 read_batch_size: int = 32,
                 parquet_schema: pa.Schema = None):
        super().__init__()
        self.path = str(path)
        self.batch_size = batch_size
        self.parquet_schema = parquet_schema
        self.cfg_rate = cfg_rate
        self.text_padding_length = text_padding_length
        self.seed = seed
        self.read_batch_size = read_batch_size
        # Get distributed training info
        self.global_rank = get_world_rank()
        self.world_size = get_world_size()
        self.sp_world_size = get_sp_world_size()
        self.num_sp_groups = self.world_size // self.sp_world_size
        num_workers = 1 if num_workers == 0 else num_workers
        # Get sharding info
        shard_parquet_files, shard_total_samples, shard_parquet_lengths = shard_parquet_files_across_sp_groups_and_workers(
            self.path, self.num_sp_groups, num_workers, seed)

        if drop_last:
            self.worker_num_samples = min(
                shard_total_samples) // batch_size * batch_size
            # Assign files to current rank's SP group
            ith_sp_group = self.global_rank // self.sp_world_size
            self.sp_group_parquet_files = shard_parquet_files[ith_sp_group::self
                                                              .num_sp_groups]
            self.sp_group_parquet_lengths = shard_parquet_lengths[
                ith_sp_group::self.num_sp_groups]
            self.sp_group_num_samples = shard_total_samples[ith_sp_group::self.
                                                            num_sp_groups]
            logger.info(
                "In total %d parquet files, %d samples, after sharding we retain %d samples due to drop_last",
                sum([len(shard) for shard in shard_parquet_files]),
                sum(shard_total_samples),
                self.worker_num_samples * self.num_sp_groups * num_workers)
        else:
            raise ValueError("drop_last must be True")
        logger.info("Each dataloader worker will load %d samples",
                    self.worker_num_samples)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 1

        worker_files = self.sp_group_parquet_files[worker_id]

        batch_iterator = BatchIterator(
            files=worker_files,
            batch_size=self.batch_size,
            text_padding_length=self.text_padding_length,
            keys=self.keys,
            worker_num_samples=self.worker_num_samples,
            read_batch_size=self.read_batch_size)  # type: ignore

        yield from batch_iterator

        if batch_iterator.processed_samples != self.worker_num_samples:
            raise ValueError(
                "Rank %d, Worker %d: Not enough samples to process, this should not happen",
                self.global_rank, worker_id)


def shard_parquet_files_across_sp_groups_and_workers(
    path: str,
    num_sp_groups: int,
    num_workers: int,
    seed: int = 42,
) -> Tuple[List[List[str]], List[int], List[Dict[str, int]]]:
    """
    Shard parquet files across SP groups and workers in a balanced way.
    
    Args:
        path: Directory containing parquet files
        num_sp_groups: Number of SP groups to shard across
        num_workers: Number of workers per SP group
        seed: Random seed for shuffling
        
    Returns:
        Tuple containing:
        - List of lists of parquet files for each shard
        - List of total samples per shard
        - List of dictionaries mapping file paths to their lengths
    """
    # Check if sharding plan already exists
    sharding_info_dir = os.path.join(
        path, f"sharding_info_{num_sp_groups}_sp_groups_{num_workers}_workers")
    if os.path.exists(sharding_info_dir):
        logger.info("Sharding plan already exists")
        logger.info("Loading sharding plan from %s", sharding_info_dir)
        try:
            with open(
                    os.path.join(sharding_info_dir, "shard_parquet_files.pkl"),
                    "rb") as f:
                shard_parquet_files = pickle.load(f)
            with open(
                    os.path.join(sharding_info_dir, "shard_total_samples.pkl"),
                    "rb") as f:
                shard_total_samples = pickle.load(f)
            with open(
                    os.path.join(sharding_info_dir,
                                 "shard_parquet_lengths.pkl"), "rb") as f:
                shard_parquet_lengths = pickle.load(f)
            return shard_parquet_files, shard_total_samples, shard_parquet_lengths
        except Exception as e:
            logger.error("Error loading sharding plan: %s", str(e))
            logger.info("Falling back to creating new sharding plan")

    if get_world_rank() == 0:
        logger.info("Scanning for parquet files in %s", path)

        # Find all parquet files
        parquet_files = []

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))

        if not parquet_files:
            raise ValueError("No parquet files found in %s", path)

        # Calculate file lengths efficiently using a single pass
        logger.info("Calculating file lengths...")
        lengths = []
        for file in tqdm.tqdm(parquet_files, desc="Reading parquet files"):
            lengths.append(pq.ParquetFile(file).metadata.num_rows)

        total_samples = sum(lengths)
        logger.info("Found %d files with %d total samples", len(parquet_files),
                    total_samples)

        # Sort files by length for better balancing
        sorted_indices = np.argsort(lengths)
        sorted_files = [parquet_files[i] for i in sorted_indices]
        sorted_lengths = [lengths[i] for i in sorted_indices]

        # Create shards
        num_shards = num_sp_groups * num_workers
        shard_parquet_files = [[] for _ in range(num_shards)]
        shard_total_samples = [0] * num_shards
        shard_parquet_lengths = [{} for _ in range(num_shards)]

        # Distribute files to shards using a greedy approach
        logger.info("Distributing files to shards...")
        for file, length in zip(reversed(sorted_files),
                                reversed(sorted_lengths)):
            # Find shard with minimum current length
            target_shard = np.argmin(shard_total_samples)
            shard_parquet_files[target_shard].append(file)
            shard_total_samples[target_shard] += length
            shard_parquet_lengths[target_shard][file] = length
        #randomize each shard
        for shard in shard_parquet_files:
            random.seed(seed)
            random.shuffle(shard)

        save_dir = os.path.join(
            path,
            f"sharding_info_{num_sp_groups}_sp_groups_{num_workers}_workers")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "shard_parquet_files.pkl"), "wb") as f:
            pickle.dump(shard_parquet_files, f)
        with open(os.path.join(save_dir, "shard_total_samples.pkl"), "wb") as f:
            pickle.dump(shard_total_samples, f)
        with open(os.path.join(save_dir, "shard_parquet_lengths.pkl"),
                  "wb") as f:
            pickle.dump(shard_parquet_lengths, f)
        logger.info("Saved sharding info to %s", save_dir)

    # wait for all ranks to finish
    torch.distributed.barrier()
    # recursive call
    return shard_parquet_files_across_sp_groups_and_workers(
        path, num_sp_groups, num_workers, seed)


def build_parquet_iterable_style_dataloader(
    path: str,
    batch_size: int,
    num_data_workers: int,
    cfg_rate: float = 0.0,
    drop_last: bool = True,
    text_padding_length: int = 512,
    seed: int = 42,
    read_batch_size: int = 32
) -> Tuple[LatentsParquetIterStyleDataset, StatefulDataLoader]:
    """Build a dataloader for the LatentsParquetIterStyleDataset."""
    dataset = LatentsParquetIterStyleDataset(
        path=path,
        batch_size=batch_size,
        cfg_rate=cfg_rate,
        num_workers=num_data_workers,
        drop_last=drop_last,
        text_padding_length=text_padding_length,
        seed=seed,
        read_batch_size=read_batch_size)

    loader = StatefulDataLoader(
        dataset,
        batch_size=1,
        num_workers=num_data_workers,
        pin_memory=True,
    )
    return dataset, loader
