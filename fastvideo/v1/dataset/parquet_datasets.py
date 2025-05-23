import time
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import itertools
import numpy as np
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader
import tqdm
import argparse
import random
import os
import glob
import pyarrow.parquet as pq
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from torch import distributed as dist

# Path to your dataset
dataset_path = "/mnt/sharefs/users/hao.zhang/Vchitect-2M/Vchitect-2M-laten-93x512x512/train/"


class ParquetVideoTextDataset(IterableDataset):
    """Efficient loader for video-text data from a directory of Parquet files."""
    
    def __init__(self, path: str, batch_size: int = 1024, rank: int = 0, world_size: int = 1, cfg_rate: float = 0.0, num_workers: int = 1, row_per_parquet: int = 32, split: str = "train"):
        super().__init__()
        self.path = str(path)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.cfg_rate = cfg_rate
        
        assert split in ["train", "validation"]
        # Find all parquet files recursively
        print(f"Scanning for parquet files in {self.path}")
        self.parquet_files = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith('.parquet'):
                    self.parquet_files.append(os.path.join(root, file))
        # Sort files for consistent ordering
        self.parquet_files.sort()

        # Distribute files among workers
        # drop last unenven files
        print(f"Total files: {len(self.parquet_files)}")
        total_files = len(self.parquet_files)
        base_count = total_files // world_size
        extra_files = total_files % world_size

        if rank < extra_files:
            start_idx = rank * (base_count + 1)
            end_idx = start_idx + base_count + 1
        else:
            start_idx = rank * base_count + extra_files
            end_idx = start_idx + base_count

        self.parquet_files = self.parquet_files[start_idx:end_idx]

        print(f"Files assigned to rank {rank}: {len(self.parquet_files)}")        
        if len(self.parquet_files) > 0:
            print(f"First file: {self.parquet_files[0]}")
            print(f"Last file: {self.parquet_files[-1]}")
        
        # Initialize current file index
        self.current_file_idx = 0
        self.current_reader = None
        self.current_batches = None
        self.total_samples = 0

    def _open_next_file(self):
        """Open the next parquet file for reading."""
        num_workers = get_worker_info().num_workers
        worker_id = get_worker_info().id
        total_files = len(self.parquet_files)
        base_count = total_files // num_workers
        extra_files = total_files % num_workers

        if worker_id < extra_files:
            start_idx = worker_id * (base_count + 1)
            end_idx = start_idx + base_count + 1
        else:
            start_idx = worker_id * base_count + extra_files
            end_idx = start_idx + base_count

        worker_parquet_files = self.parquet_files[start_idx:end_idx]
        if self.current_file_idx >= len(worker_parquet_files):
            print(f"Rank {self.rank}, Worker {worker_id}: No more files to open (current_idx={self.current_file_idx}, total_files={len(worker_parquet_files)})")
            return False
            
        if self.current_reader is not None:
            self.current_reader.close()
            
        file_path = worker_parquet_files[self.current_file_idx]
        print(f"Rank {self.rank}, Worker {worker_id}: Opening file {self.current_file_idx + 1}/{len(worker_parquet_files)}: {file_path}")
        
        try:
            self.current_reader = pq.ParquetFile(file_path)
            self.current_batches = self.current_reader.iter_batches(batch_size=self.batch_size)
            self.current_file_idx += 1
            return True
        except Exception as e:
            print(f"Error opening file {file_path}: {str(e)}")
            return False


    def __iter__(self):
        """Iterate over the dataset in a streaming fashion."""
        print(f"Rank {self.rank}: Starting iteration")
        
        # First try to open a file
        if not self._open_next_file():
            print(f"Rank {self.rank}: Failed to open first file")
            return
            
        while True:
            try:
                # Get next batch from current file
                batch = next(self.current_batches)
                batch_dict = batch.to_pydict()
                processed = self._process_batch(batch_dict)
                
                # Update sample count
                batch_size = len(processed["latents"])
                self.total_samples += batch_size
                
                # Print progress
                if self.total_samples % 1000 == 0:
                    print(f"Rank {self.rank}: Processed {self.total_samples} samples")
                
                # Yield each item in the batch
                for lat, emb, mask, info in zip(processed["latents"], processed["embeddings"], processed["masks"], processed["info"]):
                    yield lat, emb, mask, info
                    
            except StopIteration:
                # Current file is exhausted, try next file
                print(f"Rank {self.rank}: Current file exhausted, trying next file")
                self.current_batches = None
                if not self._open_next_file():
                    print(f"Rank {self.rank}: No more files to process. Total samples: {self.total_samples}")
                    break
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                self.current_batches = None
                if not self._open_next_file():
                    print(f"Rank {self.rank}: Failed to open next file after error")
                    break
                
        # Clean up
        if self.current_reader is not None:
            self.current_reader.close()

    def _process_batch(self, batch):
        """Process a PyArrow batch into tensors."""
        out = {"lat": [], "emb": [], "msk": [], "info": []}
        
        for i in range(len(batch["vae_latent_bytes"])):
            vae_latent_bytes = batch["vae_latent_bytes"][i]
            vae_latent_shape = batch["vae_latent_shape"][i]
            text_embedding_bytes = batch["text_embedding_bytes"][i]
            text_embedding_shape = batch["text_embedding_shape"][i]
            text_attention_mask_bytes = batch["text_attention_mask_bytes"][i]
            text_attention_mask_shape = batch["text_attention_mask_shape"][i]
            
            # Process latent
            if not vae_latent_shape: # No VAE latent is stored. Split is validation
                lat = np.array([])
            else:
                lat = np.frombuffer(vae_latent_bytes, dtype=np.float32).reshape(vae_latent_shape)
                # Make array writable
                lat = np.copy(lat)

            if random.random() < self.cfg_rate:
                emb = np.zeros((512, 4096), dtype=np.float32)
            else:
                emb = np.frombuffer(text_embedding_bytes, dtype=np.float32).reshape(text_embedding_shape)
                # Make array writable
                emb = np.copy(emb)
            if emb.shape[0] < 512:
                padded_emb = np.zeros((512, emb.shape[1]), dtype=np.float32)
                padded_emb[:emb.shape[0], :] = emb
                emb = padded_emb
            elif emb.shape[0] > 512:
                emb = emb[:512, :]
            
            # Process mask
            if len(text_attention_mask_bytes) > 0 and len(text_attention_mask_shape) > 0:
                msk = np.frombuffer(text_attention_mask_bytes, dtype=np.uint8).astype(np.bool_)
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
            # to string
            file_name = str(batch["file_name"][i])
            # Collect metadata
            info = {
                "width": batch["width"][i],
                "height": batch["height"][i],
                "num_frames": batch["num_frames"][i],
                "duration_sec": batch["duration_sec"][i],
                "fps": batch["fps"][i],
                "file_name": batch["file_name"][i],
                "caption": batch["caption"][i],
            }
            
            out["lat"].append(torch.from_numpy(lat))
            out["emb"].append(torch.from_numpy(emb))
            out["msk"].append(torch.from_numpy(msk))
            out["info"].append(info)
            
        return {
            "latents": torch.stack(out["lat"]) if out["lat"] else None,
            "embeddings": torch.stack(out["emb"]) if out["emb"] else None,
            "masks": torch.stack(out["msk"]) if out["msk"] else None,
            "info": out["info"]
        }

def bind_cpu_cores(local_rank, cpu_per_process=16):
    """根据local_rank绑定固定cpu核。"""
    start = local_rank * cpu_per_process
    end = start + cpu_per_process
    cores = list(range(start, end))
    print(f"[Rank {local_rank}] Binding to CPU cores: {cores}")
    os.sched_setaffinity(0, cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Parquet dataset loading speed')
    parser.add_argument('--path', type=str, default=dataset_path, 
                        help='Path to Parquet dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to benchmark')
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
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        print(f"Initialized process: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}")
    
    # Bind CPU cores after distributed initialization
    # bind_cpu_cores(local_rank, cpu_per_process=16)
    
    # Create dataset
    dataset = ParquetVideoTextDataset(
        args.path, 
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=1,
        split="train",
    )
    
    # Create DataLoader with proper settings
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,  # Reduce number of workers to avoid memory issues
        prefetch_factor=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

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
        assert torch.sum(masks[0]).item() == torch.count_nonzero(embeddings[0]).item() // 4096
        if args.vae_debug:
            from fastvideo.v1.fastvideo_args import FastVideoArgs
            from fastvideo.v1.configs.models.vaes import WanVAEConfig
            from fastvideo.v1.models.loader.component_loader import VAELoader
            from diffusers.utils import export_to_video
            from diffusers.video_processor import VideoProcessor
            VAE_PATH = "/workspace/data/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae"
            fastvideo_args = FastVideoArgs(
                model_path=VAE_PATH,
                vae_config=WanVAEConfig(load_encoder=False),
                vae_precision="fp32"
            )
            fastvideo_args.device = device
            vae_loader = VAELoader()
            vae = vae_loader.load(model_path=VAE_PATH, architecture="", fastvideo_args=fastvideo_args)

            videoprocessor = VideoProcessor(vae_scale_factor=8)

            with torch.inference_mode():
                video = vae.decode(latents[0].unsqueeze(0).to(device))
                video = videoprocessor.postprocess_video(video)
                video_path = os.path.join("/workspace/FastVideo/debug_videos", infos["caption"][0][:50] + ".mp4")
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
    for i, (latents, embeddings, masks, infos) in enumerate(tqdm.tqdm(dataloader, total=args.num_batches)):
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
            print(f"Batch {i+1}/{args.num_batches}, Speed: {samples_per_sec:.2f} samples/sec")
    
    # Final statistics
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        elapsed = time.time() - start_time
        samples_per_sec = total_samples / elapsed
        
        print(f"\nBenchmark Results:")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Total samples: {total_samples}")
        print(f"Average speed: {samples_per_sec:.2f} samples/sec")
        print(f"Time per batch: {elapsed/args.num_batches*1000:.2f} ms")
    
    if world_size > 1:
        dist.destroy_process_group()