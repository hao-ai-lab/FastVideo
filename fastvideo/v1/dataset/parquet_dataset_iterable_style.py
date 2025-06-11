import torch
import pyarrow.parquet as pq
import numpy as np
from torchdata.stateful_dataloader import StatefulDataLoader
import tqdm
import random
import os
import pyarrow.parquet as pq
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from typing import  List, Tuple, Dict, Any

from fastvideo.v1.distributed import (get_sp_world_size, get_world_rank,
                                      get_world_size)
from fastvideo.v1.logger import init_logger
import pickle

logger = init_logger(__name__)



class LatentsParquetIterStyleDataset(IterableDataset):
    """Efficient loader for video-text data from a directory of Parquet files."""
    
    # Modify this in the future if we want to add more keys, for example, in image to video.
    keys = ["vae_latent", "text_embedding"]
    
    def __init__(
        self,
        path: str,
        batch_size: int = 1024,
        cfg_rate: float = 0.1,
        frame_len: int = 8,
        num_workers: int = 1,
        drop_last: bool = True,
        text_padding_length: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        self.path = str(path)
        self.batch_size = batch_size
        self.cfg_rate = cfg_rate
        self.frame_len = frame_len
        self.text_padding_length = text_padding_length
        self.seed = seed
        
        # Get distributed training info
        self.global_rank = get_world_rank()
        self.world_size = get_world_size()
        self.sp_world_size = get_sp_world_size()
        self.num_sp_groups = self.world_size // self.sp_world_size
        
        # Get sharding info
        shard_parquet_files, shard_total_samples, shard_parquet_lengths = shard_parquet_files_across_sp_groups_and_workers(
            self.path,
            self.num_sp_groups,
            num_workers,
            seed
        )
        
        if drop_last:
            self.worker_num_samples = min(shard_total_samples) // batch_size * batch_size
            # Assign files to current rank's SP group
            ith_sp_group = self.global_rank // self.sp_world_size
            self.sp_group_parquet_files = shard_parquet_files[ith_sp_group::self.num_sp_groups]
            self.sp_group_parquet_lengths = shard_parquet_lengths[ith_sp_group::self.num_sp_groups]
            self.sp_group_num_samples = shard_total_samples[ith_sp_group::self.num_sp_groups]
            
            logger.info(f"In total {sum([len(shard) for shard in shard_parquet_files])} parquet files, {sum(shard_total_samples)} samples, after sharding we retain {self.worker_num_samples*self.num_sp_groups*num_workers} samples due to drop_last")
        else:
            raise ValueError("drop_last must be True")
        logger.info(f"Each dataloader worker will load {self.worker_num_samples} samples")
        



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

    def __iter__(self):
        processed_samples = 0
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        logger.info(f"Rank {self.global_rank}, Worker {worker_id}: Starting iteration")
        worker_files = self.sp_group_parquet_files[worker_id::num_workers]
        
        # Buffer to accumulate samples until we have a full batch
        buffer = []
        
        for file in worker_files:
            reader = pq.ParquetFile(file)
            batches = reader.iter_batches(batch_size=self.batch_size)
            
            for batch in batches:
                batch_dict = batch.to_pydict()
                
                buffer.extend(batch_dict)
                
                # Process full batches from buffer
                while len(buffer["vae_latent_bytes"]) >= self.batch_size:
                    # Extract a full batch
                    batch_to_process = buffer[:self.batch_size]
                    # Remove processed samples from buffer
                    buffer = buffer[self.batch_size:]
                    
                    # Initialize tensors to hold padded embeddings and masks
                    all_latents = []
                    all_embs = []
                    all_masks = []

                    # Process each row individually
                    for i, row in enumerate(batch_to_process):
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

                    yield all_latents, all_embs, all_masks
                    processed_samples += self.batch_size
                    
                    if processed_samples >= self.worker_num_samples:
                        return
        raise ValueError(f"Rank {self.global_rank}, Worker {worker_id}: Not enough samples to process, this should not happen")

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
        

def shard_parquet_files_across_sp_groups_and_workers(
        path: str,
        num_sp_groups: int,
        num_workers: int,
        seed: int = 42,
    ) -> Tuple[List[List[str]], List[int], List[List[int]]]:
    """
    Shard parquet files across SP groups and workers in a balanced way.
    
    Args:
        path: Directory containing parquet files
        num_sp_groups: Number of SP groups to shard across
        num_workers: Number of workers per SP group
        drop_last: Whether to drop the last incomplete shard
        seed: Random seed for shuffling
        
    Returns:
        Tuple containing:
        - List of lists of parquet files for each shard
        - List of total samples per shard
        - List of lists of file lengths per shard
    """
    # Check if sharding plan already exists
    sharding_info_dir = os.path.join(path, f"sharding_info_{num_sp_groups}_sp_groups_{num_workers}_workers")
    if os.path.exists(sharding_info_dir):
        logger.info(f"Sharding plan already exists")
        logger.info(f"Loading sharding plan from {sharding_info_dir}")
        try:
            with open(os.path.join(sharding_info_dir, "shard_parquet_files.pkl"), "rb") as f:
                shard_parquet_files = pickle.load(f)
            with open(os.path.join(sharding_info_dir, "shard_total_samples.pkl"), "rb") as f:
                shard_total_samples = pickle.load(f)
            with open(os.path.join(sharding_info_dir, "shard_parquet_lengths.pkl"), "rb") as f:
                shard_parquet_lengths = pickle.load(f)
            return shard_parquet_files, shard_total_samples, shard_parquet_lengths
        except Exception as e:
            logger.error(f"Error loading sharding plan: {str(e)}")
            logger.info("Falling back to creating new sharding plan")
    
    logger.info(f"Scanning for parquet files in {path}")
    
    # Find all parquet files
    parquet_files = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

        
    if not parquet_files:
        raise ValueError(f"No parquet files found in {path}")
    
    # Calculate file lengths efficiently using a single pass
    logger.info("Calculating file lengths...")
    lengths = []
    for file in tqdm.tqdm(parquet_files, desc="Reading parquet files"):
        lengths.append(pq.ParquetFile(file).metadata.num_rows)

            
    total_samples = sum(lengths)
    logger.info(f"Found {len(parquet_files)} files with {total_samples} total samples")
    
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
    for file, length in zip(reversed(sorted_files), reversed(sorted_lengths)):
        # Find shard with minimum current length
        target_shard = np.argmin(shard_total_samples)
        shard_parquet_files[target_shard].append(file)
        shard_total_samples[target_shard] += length
        shard_parquet_lengths[target_shard][file] = length
    #randomize each shard
    for shard in shard_parquet_files:
        random.seed(seed)
        random.shuffle(shard)
    
    # Save sharding info if we're rank 0
    if get_world_rank() == 0:
        save_dir = os.path.join(path, f"sharding_info_{num_sp_groups}_sp_groups_{num_workers}_workers")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "shard_parquet_files.pkl"), "wb") as f:
            pickle.dump(shard_parquet_files, f)
        with open(os.path.join(save_dir, "shard_total_samples.pkl"), "wb") as f:
            pickle.dump(shard_total_samples, f)
        with open(os.path.join(save_dir, "shard_parquet_lengths.pkl"), "wb") as f:
            pickle.dump(shard_parquet_lengths, f)
        logger.info(f"Saved sharding info to {save_dir}")

            
    return shard_parquet_files, shard_total_samples, shard_parquet_lengths

def build_parquet_iterable_style_dataloader(
        path: str,
        batch_size: int,
        num_data_workers: int,
        cfg_rate: float = 0.0,
        frame_len: int = 8,
        drop_last: bool = True,
        text_padding_length: int = 512,
        seed: int = 42) -> Tuple[LatentsParquetIterStyleDataset, StatefulDataLoader]:
    """Build a dataloader for the LatentsParquetIterStyleDataset."""
    dataset = LatentsParquetIterStyleDataset(
        path=path,
        batch_size=batch_size,
        cfg_rate=cfg_rate,
        frame_len=frame_len,
        num_workers=num_data_workers,
        drop_last=drop_last,
        text_padding_length=text_padding_length,
        seed=seed
    )

    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader