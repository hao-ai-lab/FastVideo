# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import pathlib
import time

import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp

from fastvideo.v1.dataset.parquet_dataset_iterable_style import (
    build_parquet_iterable_style_dataloader)
from fastvideo.v1.distributed import get_world_rank
from fastvideo.v1.distributed.parallel_state import (
    cleanup_dist_env_and_memory, get_torch_device,
    maybe_init_distributed_environment_and_model_parallel)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parquet iterable style dataset loading speed")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to parquet dataset",
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="Batch size for DataLoader")
    parser.add_argument("--num_data_workers",
                        type=int,
                        help="Number of DataLoader workers")
    parser.add_argument("--num_epoch",
                        type=int,
                        default=2,
                        help="Number of epoches to benchmark")
    parser.add_argument("--verify_resume",
                        action="store_true",
                        help="Verify resume")
    parser.add_argument(
        "--num_batches_per_epoch",
        type=int,
        default=1000,
        help="Number of batches to benchmark",
    )
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='dataloader_checkpoint',
                        help='Path to save/load checkpoint')
    '''
    example launch command:
    torchrun --nproc_per_node=1 --master_port=12358 fastvideo/v1/dataset/benchmarks/benchmark_parquet_dataset_iterable_style.py --path data/crush-smol/latents/combined_parquet_dataset --batch_size 2 --num_data_workers 2 --num_epoch 2 --num_batches_per_epoch 2 --verify_resume
    torchrun --nproc_per_node=8 --master_port=12358 fastvideo/v1/dataset/benchmarks/benchmark_parquet_dataset_iterable_style.py --path data/crush-smol/latents/combined_parquet_dataset --batch_size 2 --num_data_workers 1 --num_epoch 2 --num_batches_per_epoch 5 --verify_resume
    torchrun --nproc_per_node=8 --master_port=12358 fastvideo/v1/dataset/benchmarks/benchmark_parquet_dataset_iterable_style.py --path /mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents/ --batch_size 2 --num_data_workers 4 --num_epoch 2 --num_batches_per_epoch 100 
    '''
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=(world_size + 1) // 2, sp_size=(world_size + 1) // 2)
    logger.info("Initialized distributed environment with world_size=%d",
                world_size)

    # Create DataLoader with proper settings
    dataset, dataloader = build_parquet_iterable_style_dataloader(
        args.path, args.batch_size, args.num_data_workers)
    logger.info("Initialized dataloader")

    if args.verify_resume:
        # First pass - record latent sums
        first_pass_sums = []
        for i, (latents, embeddings, masks,
                caption_text) in enumerate(dataloader):
            latent_sum = latents.sum().item()
            first_pass_sums.append(latent_sum)
            logger.info("Batch %d latent sum: %f", i, latent_sum)
            if i >= args.num_batches_per_epoch - 1:
                break

        # Save dataloader state using distributed checkpoint
        checkpoint_dir = pathlib.Path(args.checkpoint_path)
        logger.info("Rank %d: Saving dataloader state to %s", get_world_rank(),
                    checkpoint_dir)
        states = {"dataloader": dataloader}

        begin_time = time.monotonic()
        dist_cp.save(states, checkpoint_id=checkpoint_dir.as_posix())
        end_time = time.monotonic()

        logger.info("Rank %d: Saved checkpoint in %.2f seconds",
                    get_world_rank(), end_time - begin_time)

        # Make sure all processes wait for checkpoint to be saved
        if world_size > 1:
            dist.barrier()

        # Recreate dataloader and load state
        dataset, dataloader = build_parquet_iterable_style_dataloader(
            args.path, args.batch_size, args.num_data_workers)
        load_states = {"dataloader": dataloader}
        dist_cp.load(load_states, checkpoint_id=checkpoint_dir.as_posix())
        logger.info("Rank %d: Loaded dataloader state from %s",
                    get_world_rank(), checkpoint_dir)

        # Second pass - verify latent sums match
        for i, (latents, embeddings, masks) in enumerate(dataloader):
            latent_sum = latents.sum().item()
            first_pass_sums.append(latent_sum)
            logger.info("Batch %d latent sum: %f",
                        i + args.num_batches_per_epoch, latent_sum)
            if i >= args.num_batches_per_epoch - 1:
                break

        dataset, dataloader = build_parquet_iterable_style_dataloader(
            args.path, args.batch_size, args.num_data_workers)
        # Second pass - verify latent sums match
        second_pass_sums = []
        for i, (latents, embeddings, masks,
                caption_text) in enumerate(dataloader):
            latent_sum = latents.sum().item()
            second_pass_sums.append(latent_sum)
            logger.info("Batch %d latent sum: %f (should match first pass: %f)",
                        i, latent_sum, first_pass_sums[i])
            if i >= args.num_batches_per_epoch * 2 - 1:
                break

        # Verify all sums match
        if all(
                abs(a - b) < 1e-6
                for a, b in zip(first_pass_sums, second_pass_sums)):
            logger.info(
                "All latent sums match between passes - resume verification successful!"
            )
        else:
            raise ValueError(
                "Latent sums do not match between passes - resume verification failed!"
            )

    start_time = time.time()
    total_samples = 0
    total_batches = 0
    for _ in range(args.num_epoch):
        for i, (latents, embeddings, masks,
                caption_text) in enumerate(dataloader):
            if i >= args.num_batches_per_epoch:
                break

            # Move data to device
            latents = latents.to(get_torch_device())
            embeddings = embeddings.to(get_torch_device())

            # Calculate actual batch size
            batch_size = latents.size(0)
            total_samples += batch_size
            total_batches += 1

            # Print progress only from rank 0
            if get_world_rank() == 0 and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed
                logger.info("Batch %d/%d, Speed: %.2f samples/sec", i + 1,
                            args.num_batches_per_epoch, samples_per_sec)

    # Final statistics
    if world_size > 1:
        dist.barrier()

    if get_world_rank() == 0:
        elapsed = time.time() - start_time
        samples_per_sec = total_samples / elapsed

        logger.info("\nBenchmark Results:")
        logger.info("Total time: %.2f seconds", elapsed)
        logger.info("Total samples: %d", total_samples)
        logger.info("Average speed: %.2f samples/sec", samples_per_sec)
        logger.info("Time per batch: %.2f ms", elapsed / total_batches * 1000)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_dist_env_and_memory()
