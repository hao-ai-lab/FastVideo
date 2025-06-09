import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
from einops import rearrange
from torch import distributed as dist
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.v1.distributed import (get_sp_group, get_sp_parallel_rank,
                                      get_sp_world_size, get_world_rank,
                                      get_world_size)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class ParquetVideoTextDataset(Dataset):
    """Efficient loader for video-text data from a directory of Parquet files."""

    def __init__(self,
                 path: str,
                 batch_size,
                 cfg_rate: float = 0.0,
                 num_latent_t: int = 2,
                 seed: int = 0,
                 validation: bool = False):
        super().__init__()
        self.path = str(path)
        self.batch_size = batch_size
        self.global_rank = get_world_rank()
        self.rank_in_sp_group = get_sp_parallel_rank()
        self.sp_group = get_sp_group()
        self.sp_world_size = get_sp_world_size()
        self.world_size = get_world_size()
        self.cfg_rate = cfg_rate
        self.num_latent_t = num_latent_t
        self.local_indices = None
        self.validation = validation

        # Negative prompt caching
        self.neg_metadata = None
        self.cached_neg_prompt: Dict[str, Any] | None = None

        self.plan_output_dir = os.path.join(
            self.path,
            f"data_plan_world_size_{self.world_size}_sp_size_{self.sp_world_size}.json"
        )

        # group_ranks: a list of lists
        # len(group_ranks) = self.world_size
        # len(group_ranks[i]) = self.sp_world_size
        # group_ranks[i] represents the ranks of the SP group for the i-th GPU
        # For example, if self.world_size = 4, self.sp_world_size = 2, then
        # group_ranks = [[0, 1], [0, 1], [2, 3], [2, 3]]
        sp_group_ranks = get_sp_group().ranks
        group_ranks: List[List] = [[] for _ in range(self.world_size)]
        dist.all_gather_object(group_ranks, sp_group_ranks)

        if self.global_rank == 0:
            # If a plan already exists, then skip creating a new plan
            # This will be useful when resume training
            if os.path.exists(self.plan_output_dir):
                logger.info("Using existing plan from %s", self.plan_output_dir)
            else:
                logger.info("Creating new plan for %s", self.plan_output_dir)
                metadatas = []
                for root, _, files in os.walk(self.path):
                    for file in sorted(files):
                        if file.endswith('.parquet'):
                            file_path = os.path.join(root, file)
                            num_rows = pq.ParquetFile(
                                file_path).metadata.num_rows
                            for row_idx in range(num_rows):
                                metadatas.append((file_path, row_idx))

                # the negative prompt is always the first row in the first
                # parquet file
                if validation:
                    self.neg_metadata = metadatas[0]
                    metadatas = metadatas[1:]

                # Generate the plan that distribute rows among workers
                random.seed(seed)
                random.shuffle(metadatas)

                # Get all sp groups
                # e.g. if num_gpus = 4, sp_size = 2
                # group_ranks = [(0, 1), (0, 1), (2, 3), (2, 3)]
                # We will assign the same batches of data to ranks in the same sp group, and we'll assign different batches to ranks in different sp groups
                # e.g. plan = {0: [row 1, row 4], 1: [row 1, row 4], 2: [row 2, row 3], 3: [row 2, row 3]}
                group_ranks_list: List[Any] = list(
                    set(tuple(r) for r in group_ranks))
                num_sp_groups = len(group_ranks_list)
                plan = defaultdict(list)
                for idx, metadata in enumerate(metadatas):
                    sp_group_idx = idx % num_sp_groups
                    for global_rank in group_ranks_list[sp_group_idx]:
                        plan[global_rank].append(metadata)

                if validation:
                    assert self.neg_metadata is not None
                    plan["negative_prompt"] = [self.neg_metadata]
                with open(self.plan_output_dir, "w") as f:
                    json.dump(plan, f)
        else:
            pass
        dist.barrier()
        if validation:
            with open(self.plan_output_dir) as f:
                plan = json.load(f)
            self.neg_metadata = plan["negative_prompt"][0]

    def _load_and_cache_negative_prompt(self) -> None:
        """Load and cache the negative prompt. Only rank 0 in each SP group should call this."""
        if not self.validation or self.neg_metadata is None:
            return

        if self.cached_neg_prompt is not None:
            return

        # Only rank 0 in each SP group should read the negative prompt
        try:
            file_path, row_idx = self.neg_metadata
            parquet_file = pq.ParquetFile(file_path)

            # Since negative prompt is always the first row (row_idx = 0),
            # it's always in the first row group
            row_group_index = 0
            local_index = row_idx  # This will be 0 for the negative prompt

            row_group = parquet_file.read_row_group(row_group_index).to_pydict()
            row_dict = {k: v[local_index] for k, v in row_group.items()}
            del row_group

            # Process the negative prompt row
            self.cached_neg_prompt = self._process_row(row_dict)

        except Exception as e:
            logger.error("Failed to load negative prompt: %s", e)
            self.cached_neg_prompt = None

    def get_validation_negative_prompt(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get the negative prompt for validation. 
        This method ensures the negative prompt is loaded and cached properly.
        Returns the processed negative prompt data (latents, embeddings, masks, info).
        """
        if not self.validation:
            raise ValueError(
                "get_validation_negative_prompt() can only be called in validation mode"
            )

        # Load and cache if needed (only rank 0 in SP group will actually load)
        if self.cached_neg_prompt is None:
            self._load_and_cache_negative_prompt()

        if self.cached_neg_prompt is None:
            raise RuntimeError(
                f"Rank {self.global_rank} (SP rank {self.rank_in_sp_group}): Could not retrieve negative prompt data"
            )

        # Extract the components
        lat, emb, mask, info = (self.cached_neg_prompt["latents"],
                                self.cached_neg_prompt["embeddings"],
                                self.cached_neg_prompt["masks"],
                                self.cached_neg_prompt["info"])

        # Apply the same processing as in __getitem__
        if lat.numel() == 0:  # Validation parquet
            return lat, emb, mask, info
        else:
            lat = lat[:, -self.num_latent_t:]
            if self.sp_world_size > 1:
                lat = rearrange(lat,
                                "t (n s) h w -> t n s h w",
                                n=self.sp_world_size).contiguous()
                lat = lat[:, self.rank_in_sp_group, :, :, :]
            return lat, emb, mask, info

    def __len__(self):
        if self.local_indices is None:
            try:
                with open(self.plan_output_dir) as f:
                    plan = json.load(f)
                self.local_indices = plan[str(self.global_rank)]
            except Exception as err:
                raise Exception(
                    "The data plan hasn't been created yet") from err
        assert self.local_indices is not None
        return len(self.local_indices)

    def __getitem__(self, idx):
        if self.local_indices is None:
            try:
                with open(self.plan_output_dir) as f:
                    plan = json.load(f)
                self.local_indices = plan[self.global_rank]
            except Exception as err:
                raise Exception(
                    "The data plan hasn't been created yet") from err
        assert self.local_indices is not None
        file_path, row_idx = self.local_indices[idx]
        parquet_file = pq.ParquetFile(file_path)

        # Calculate the row group to read into memory and the local idx
        # This way we can avoid reading in the entire parquet file
        cumulative = 0
        for i in range(parquet_file.num_row_groups):
            num_rows = parquet_file.metadata.row_group(i).num_rows
            if cumulative + num_rows > row_idx:
                row_group_index = i
                local_index = row_idx - cumulative
                break
            cumulative += num_rows

        row_group = parquet_file.read_row_group(row_group_index).to_pydict()
        row_dict = {k: v[local_index] for k, v in row_group.items()}
        del row_group

        processed = self._process_row(row_dict)
        lat, emb, mask, info = processed["latents"], processed[
            "embeddings"], processed["masks"], processed["info"]
        if lat.numel() == 0:  # Validation parquet
            return lat, emb, mask, info
        else:
            lat = lat[:, -self.num_latent_t:]
            if self.sp_world_size > 1:
                lat = rearrange(lat,
                                "t (n s) h w -> t n s h w",
                                n=self.sp_world_size).contiguous()
                lat = lat[:, self.rank_in_sp_group, :, :, :]
            return lat, emb, mask, info

    def _process_row(self, row) -> Dict[str, Any]:
        """Process a PyArrow batch into tensors."""

        vae_latent_bytes = row["vae_latent_bytes"]
        vae_latent_shape = row["vae_latent_shape"]
        text_embedding_bytes = row["text_embedding_bytes"]
        text_embedding_shape = row["text_embedding_shape"]
        text_attention_mask_bytes = row["text_attention_mask_bytes"]
        text_attention_mask_shape = row["text_attention_mask_shape"]

        # Process latent
        if not vae_latent_shape:  # No VAE latent is stored. Split is validation
            lat = np.array([])
        else:
            lat = np.frombuffer(vae_latent_bytes,
                                dtype=np.float32).reshape(vae_latent_shape)
            # Make array writable
            lat = np.copy(lat)

        if random.random() < self.cfg_rate:
            emb = np.zeros((512, 4096), dtype=np.float32)
        else:
            emb = np.frombuffer(text_embedding_bytes,
                                dtype=np.float32).reshape(text_embedding_shape)
            # Make array writable
            emb = np.copy(emb)
        if emb.shape[0] < 512:
            padded_emb = np.zeros((512, emb.shape[1]), dtype=np.float32)
            padded_emb[:emb.shape[0], :] = emb
            emb = padded_emb
        elif emb.shape[0] > 512:
            emb = emb[:512, :]

        # Process mask
        if len(text_attention_mask_bytes) > 0 and len(
                text_attention_mask_shape) > 0:
            msk = np.frombuffer(text_attention_mask_bytes,
                                dtype=np.uint8).astype(np.bool_)
            msk = msk.reshape(1, -1)
            # Make array writable
            msk = np.copy(msk)
            if msk.shape[1] < 512:
                padded_msk = np.zeros((1, 512), dtype=np.bool_)
                padded_msk[:, :msk.shape[1]] = msk
                msk = padded_msk
            elif msk.shape[1] > 512:
                msk = msk[:, :512]
        else:
            msk = np.ones((1, 512), dtype=np.bool_)

        # Collect metadata
        info = {
            "width": row["width"],
            "height": row["height"],
            "num_frames": row["num_frames"],
            "duration_sec": row["duration_sec"],
            "fps": row["fps"],
            "file_name": row["file_name"],
            "caption": row["caption"],
        }

        return {
            "latents": torch.from_numpy(lat),
            "embeddings": torch.from_numpy(emb),
            "masks": torch.from_numpy(msk),
            "info": info
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark Parquet dataset loading speed')
    parser.add_argument('--path',
                        type=str,
                        default="your/dataset/path",
                        help='Path to Parquet dataset')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Batch size for DataLoader')
    parser.add_argument('--num_batches',
                        type=int,
                        default=100,
                        help='Number of batches to benchmark')
    parser.add_argument('--vae_debug', action="store_true")
    args = parser.parse_args()

    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Initialize CUDA device first
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize distributed training
    if world_size > 1:
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                world_size=world_size,
                                rank=rank)
        print(
            f"Initialized process: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}"
        )

    # Create dataset
    dataset = ParquetVideoTextDataset(
        args.path,
        batch_size=args.batch_size,
    )

    # Create DataLoader with proper settings
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,  # Reduce number of workers to avoid memory issues
        prefetch_factor=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True)

    # Example of how to load dataloader state
    # if os.path.exists("/workspace/FastVideo/dataloader_state.pt"):
    #     dataloader_state = torch.load("/workspace/FastVideo/dataloader_state.pt")
    #     dataloader.load_state_dict(dataloader_state[rank])

    # Warm-up with synchronization
    if rank == 0:
        print("Warming up...")
    for i, (latents, embeddings, masks, infos) in enumerate(dataloader):
        # Example of how to save dataloader state
        # if i == 30:
        #     dist.barrier()
        #     local_data = {rank: dataloader.state_dict()}
        #     gathered_data = [None] * world_size
        #     dist.all_gather_object(gathered_data, local_data)
        #     if rank == 0:
        #         global_state_dict = {}
        #         for d in gathered_data:
        #             global_state_dict.update(d)
        #         torch.save(global_state_dict, "dataloader_state.pt")
        assert torch.sum(masks[0]).item() == torch.count_nonzero(
            embeddings[0]).item() // 4096
        if args.vae_debug:
            from diffusers.utils import export_to_video
            from diffusers.video_processor import VideoProcessor

            from fastvideo.v1.configs.models.vaes import WanVAEConfig
            from fastvideo.v1.fastvideo_args import FastVideoArgs
            from fastvideo.v1.models.loader.component_loader import VAELoader
            VAE_PATH = "/workspace/data/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae"
            fastvideo_args = FastVideoArgs(
                model_path=VAE_PATH,
                vae_config=WanVAEConfig(load_encoder=False),
                vae_precision="fp32")
            vae_loader = VAELoader()
            vae = vae_loader.load(model_path=VAE_PATH,
                                  architecture="",
                                  fastvideo_args=fastvideo_args)

            videoprocessor = VideoProcessor(vae_scale_factor=8)

            with torch.inference_mode():
                video = vae.decode(latents[0].unsqueeze(0).to(device))
                video = videoprocessor.postprocess_video(video)
                video_path = os.path.join("/workspace/FastVideo/debug_videos",
                                          infos["caption"][0][:50] + ".mp4")
                export_to_video(video[0], video_path, fps=16)

        # Move data to device
        # latents = latents.to(device)
        # embeddings = embeddings.to(device)

    if world_size > 1:
        dist.barrier()

    # Benchmark
    if rank == 0:
        print(f"Benchmarking with batch_size={args.batch_size}")
    start_time = time.time()
    total_samples = 0
    for i, (latents, embeddings, masks,
            infos) in enumerate(tqdm.tqdm(dataloader, total=args.num_batches)):
        if i >= args.num_batches:
            break

        # Move data to device
        latents = latents.to(device)
        embeddings = embeddings.to(device)

        # Calculate actual batch size
        batch_size = latents.size(0)
        total_samples += batch_size

        # Print progress only from rank 0
        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed
            print(
                f"Batch {i+1}/{args.num_batches}, Speed: {samples_per_sec:.2f} samples/sec"
            )

    # Final statistics
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        elapsed = time.time() - start_time
        samples_per_sec = total_samples / elapsed

        print("\nBenchmark Results:")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Total samples: {total_samples}")
        print(f"Average speed: {samples_per_sec:.2f} samples/sec")
        print(f"Time per batch: {elapsed/args.num_batches*1000:.2f} ms")

    if world_size > 1:
        dist.destroy_process_group()
