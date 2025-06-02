# SPDX-License-Identifier: Apache-2.0
"""
T2V Data Preprocessing pipeline implementation.

This module contains an implementation of the T2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
import gc
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fastvideo.v1.dataset import getdataset
from fastvideo.v1.dataset.dataloader.schema import pyarrow_schema
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages import TextEncodingStage

# TODO(will): move PRECISION_TO_TYPE to better place

logger = init_logger(__name__)


class PreprocessPipeline(ComposedPipelineBase):

    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        args,
    ):
        # Initialize class variables for data sharing
        self.video_data: Dict[str, Any] = {}  # Store video metadata and paths
        self.latent_data: Dict[str, Any] = {}  # Store latent tensors
        self.preprocess_validation_text(fastvideo_args, args)
        self.preprocess_video_and_text(fastvideo_args, args)

    def preprocess_video_and_text(self, fastvideo_args: FastVideoArgs, args):
        os.makedirs(args.output_dir, exist_ok=True)
        # Create directory for combined data
        combined_parquet_dir = os.path.join(args.output_dir,
                                            "combined_parquet_dataset")
        os.makedirs(combined_parquet_dir, exist_ok=True)
        local_rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))

        # Get how many samples have already been processed
        start_idx = 0
        for root, _, files in os.walk(combined_parquet_dir):
            for file in files:
                if file.endswith('.parquet'):
                    table = pq.read_table(os.path.join(root, file))
                    start_idx += table.num_rows

        # Loading dataset
        train_dataset = getdataset(args, start_idx=start_idx)
        sampler = DistributedSampler(train_dataset,
                                     rank=local_rank,
                                     num_replicas=world_size,
                                     shuffle=False)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        num_processed_samples = 0
        # Add progress bar for video preprocessing
        pbar = tqdm(train_dataloader,
                    desc="Processing videos",
                    unit="batch",
                    disable=local_rank != 0)
        for batch_idx, data in enumerate(pbar):
            if data is None:
                continue

            with torch.inference_mode():
                # Filter out invalid samples (those with all zeros)
                valid_indices = []
                for i, pixel_values in enumerate(data["pixel_values"]):
                    if not torch.all(
                            pixel_values == 0):  # Check if all values are zero
                        valid_indices.append(i)
                num_processed_samples += len(valid_indices)

                if not valid_indices:
                    continue

                # Create new batch with only valid samples
                valid_data = {
                    "pixel_values":
                    torch.stack(
                        [data["pixel_values"][i] for i in valid_indices]),
                    "text": [data["text"][i] for i in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                    "fps": [data["fps"][i] for i in valid_indices],
                    "duration": [data["duration"][i] for i in valid_indices],
                }

                # VAE
                with torch.autocast("cuda", dtype=torch.float32):
                    latents = self.get_module("vae").encode(
                        valid_data["pixel_values"].to(
                            fastvideo_args.device)).mean

                batch_captions = valid_data["text"]

                batch = ForwardBatch(
                    data_type="video",
                    prompt=batch_captions,
                    prompt_embeds=[],
                    prompt_attention_mask=[],
                )
                assert hasattr(self, "prompt_encoding_stage")
                result_batch = self.prompt_encoding_stage(batch, fastvideo_args)
                prompt_embeds, prompt_attention_mask = result_batch.prompt_embeds[
                    0], result_batch.prompt_attention_mask[0]
                assert prompt_embeds.shape[0] == prompt_attention_mask.shape[0]

                # Get sequence lengths from attention masks (number of 1s)
                seq_lens = prompt_attention_mask.sum(dim=1)

                non_padded_embeds = []
                non_padded_masks = []

                # Process each item in the batch
                for i in range(prompt_embeds.size(0)):
                    seq_len = seq_lens[i].item()
                    # Slice the embeddings and masks to keep only non-padding parts
                    non_padded_embeds.append(prompt_embeds[i, :seq_len])
                    non_padded_masks.append(prompt_attention_mask[i, :seq_len])

                # Update the tensors with non-padded versions
                prompt_embeds = non_padded_embeds
                prompt_attention_mask = non_padded_masks

            # Prepare batch data for Parquet dataset
            batch_data = []

            # Add progress bar for saving outputs
            save_pbar = tqdm(enumerate(valid_data["path"]),
                             desc="Saving outputs",
                             unit="item",
                             leave=False)
            for idx, video_path in save_pbar:
                # Get the corresponding latent and info using video name
                latent = latents[idx].cpu()
                video_name = os.path.basename(video_path).split(".")[0]
                height, width = valid_data["pixel_values"][idx].shape[-2:]

                # Convert tensors to numpy arrays
                vae_latent = latent.cpu().numpy()
                text_embedding = prompt_embeds[idx].cpu().numpy()
                text_attention_mask = prompt_attention_mask[idx].cpu().numpy(
                ).astype(np.uint8)

                # Create record for Parquet dataset
                record = {
                    "id": video_name,
                    "vae_latent_bytes": vae_latent.tobytes(),
                    "vae_latent_shape": list(vae_latent.shape),
                    "vae_latent_dtype": str(vae_latent.dtype),
                    "text_embedding_bytes": text_embedding.tobytes(),
                    "text_embedding_shape": list(text_embedding.shape),
                    "text_embedding_dtype": str(text_embedding.dtype),
                    "text_attention_mask_bytes": text_attention_mask.tobytes(),
                    "text_attention_mask_shape":
                    list(text_attention_mask.shape),
                    "text_attention_mask_dtype": str(text_attention_mask.dtype),
                    "file_name": video_name,
                    "caption": valid_data["text"][idx],
                    "media_type": "video",
                    "width": width,
                    "height": height,
                    "num_frames": latents[idx].shape[1],
                    "duration_sec": float(valid_data["duration"][idx]),
                    "fps": float(valid_data["fps"][idx]),
                }
                batch_data.append(record)

            if batch_data:
                # Add progress bar for writing to Parquet dataset
                write_pbar = tqdm(total=1,
                                  desc="Writing to Parquet dataset",
                                  unit="batch")
                # Convert batch data to PyArrow arrays
                arrays = [
                    pa.array([record["id"] for record in batch_data]),
                    pa.array(
                        [record["vae_latent_bytes"] for record in batch_data],
                        type=pa.binary()),
                    pa.array(
                        [record["vae_latent_shape"] for record in batch_data],
                        type=pa.list_(pa.int32())),
                    pa.array(
                        [record["vae_latent_dtype"] for record in batch_data]),
                    pa.array([
                        record["text_embedding_bytes"] for record in batch_data
                    ],
                             type=pa.binary()),
                    pa.array([
                        record["text_embedding_shape"] for record in batch_data
                    ],
                             type=pa.list_(pa.int32())),
                    pa.array([
                        record["text_embedding_dtype"] for record in batch_data
                    ]),
                    pa.array([
                        record["text_attention_mask_bytes"]
                        for record in batch_data
                    ],
                             type=pa.binary()),
                    pa.array([
                        record["text_attention_mask_shape"]
                        for record in batch_data
                    ],
                             type=pa.list_(pa.int32())),
                    pa.array([
                        record["text_attention_mask_dtype"]
                        for record in batch_data
                    ]),
                    pa.array([record["file_name"] for record in batch_data]),
                    pa.array([record["caption"] for record in batch_data]),
                    pa.array([record["media_type"] for record in batch_data]),
                    pa.array([record["width"] for record in batch_data],
                             type=pa.int32()),
                    pa.array([record["height"] for record in batch_data],
                             type=pa.int32()),
                    pa.array([record["num_frames"] for record in batch_data],
                             type=pa.int32()),
                    pa.array([record["duration_sec"] for record in batch_data],
                             type=pa.float32()),
                    pa.array([record["fps"] for record in batch_data],
                             type=pa.float32()),
                ]
                table = pa.Table.from_arrays(
                    arrays, names=[f.name for f in pyarrow_schema])
                write_pbar.update(1)
                write_pbar.close()

                # Store the table in a list for later processing
                if not hasattr(self, 'all_tables'):
                    self.all_tables = []
                self.all_tables.append(table)

                logger.info("Collected batch with %s samples", len(table))

            if num_processed_samples >= args.flush_frequency:
                assert hasattr(self, 'all_tables') and self.all_tables
                print(f"Combining {len(self.all_tables)} batches...")
                combined_table = pa.concat_tables(self.all_tables)
                assert len(combined_table) == num_processed_samples
                print(f"Total samples collected: {len(combined_table)}")

                # Calculate total number of chunks needed, discarding remainder
                total_chunks = max(
                    num_processed_samples // args.samples_per_file, 1)

                print(
                    f"Fixed samples per parquet file: {args.samples_per_file}")
                print(f"Total number of parquet files: {total_chunks}")
                print(
                    f"Total samples to be processed: {total_chunks * args.samples_per_file} (discarding {num_processed_samples % args.samples_per_file} samples)"
                )

                # Split work among processes
                num_workers = int(min(multiprocessing.cpu_count(),
                                      total_chunks))
                chunks_per_worker = (total_chunks + num_workers -
                                     1) // num_workers

                print(
                    f"Using {num_workers} workers to process {total_chunks} chunks"
                )
                logger.info("Chunks per worker: %s", chunks_per_worker)

                # Prepare work ranges
                work_ranges = []
                for i in range(num_workers):
                    start_idx = i * chunks_per_worker
                    end_idx = min((i + 1) * chunks_per_worker, total_chunks)
                    if start_idx < total_chunks:
                        work_ranges.append(
                            (start_idx, end_idx, combined_table, i,
                             combined_parquet_dir, args.samples_per_file))

                total_written = 0
                failed_ranges = []
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(self.process_chunk_range, work_range):
                        work_range
                        for work_range in work_ranges
                    }
                    for future in tqdm(futures, desc="Processing chunks"):
                        try:
                            written = future.result()
                            total_written += written
                            logger.info("Processed chunk with %s samples",
                                        written)
                        except Exception as e:
                            work_range = futures[future]
                            failed_ranges.append(work_range)
                            logger.error("Failed to process range %s-%s: %s",
                                         work_range[0], work_range[1], str(e))

                # Retry failed ranges sequentially
                if failed_ranges:
                    logger.warning("Retrying %s failed ranges sequentially",
                                   len(failed_ranges))
                    for work_range in failed_ranges:
                        try:
                            total_written += self.process_chunk_range(
                                work_range)
                        except Exception as e:
                            logger.error(
                                "Failed to process range %s-%s after retry: %s",
                                work_range[0], work_range[1], str(e))

                logger.info("Total samples written: %s", total_written)

                num_processed_samples = 0
                self.all_tables = []

    def preprocess_validation_text(self, fastvideo_args: FastVideoArgs, args):
        # Create Parquet dataset directory for validation
        validation_parquet_dir = os.path.join(args.output_dir,
                                              "validation_parquet_dataset")
        os.makedirs(validation_parquet_dir, exist_ok=True)

        with open(args.validation_prompt_txt, encoding="utf-8") as file:
            lines = file.readlines()
        prompts = [line.strip() for line in lines]

        # Prepare batch data for Parquet dataset
        batch_data = []

        # Add progress bar for validation text preprocessing
        pbar = tqdm(enumerate(prompts),
                    desc="Processing validation prompts",
                    unit="prompt")
        for prompt_idx, prompt in pbar:
            with torch.inference_mode():
                # Text Encoder
                batch = ForwardBatch(
                    data_type="video",
                    prompt=prompt,
                    prompt_embeds=[],
                    prompt_attention_mask=[],
                )
                assert hasattr(self, "prompt_encoding_stage")
                result_batch = self.prompt_encoding_stage(batch, fastvideo_args)
            prompt_embeds = result_batch.prompt_embeds[0]
            prompt_attention_mask = result_batch.prompt_attention_mask[0]

            file_name = prompt.split(".")[0]

            # Get the sequence length from attention mask (number of 1s)
            seq_len = prompt_attention_mask.sum().item()

            text_embedding = prompt_embeds[0, :seq_len].cpu().numpy()
            text_attention_mask = prompt_attention_mask[
                0, :seq_len].cpu().numpy().astype(np.uint8)

            # Log the shapes after removing padding
            logger.info(
                "Shape after removing padding - Embeddings: %s, Mask: %s",
                text_embedding.shape, text_attention_mask.shape)

            # Create record for Parquet dataset
            record = {
                "id": file_name,
                "vae_latent_bytes": b"",  # Not available for validation
                "vae_latent_shape": [],
                "vae_latent_dtype": "",
                "text_embedding_bytes": text_embedding.tobytes(),
                "text_embedding_shape": list(text_embedding.shape),
                "text_embedding_dtype": str(text_embedding.dtype),
                "text_attention_mask_bytes": text_attention_mask.tobytes(),
                "text_attention_mask_shape": list(text_attention_mask.shape),
                "text_attention_mask_dtype": str(text_attention_mask.dtype),
                "file_name": file_name,
                "caption": prompt,
                "media_type": "video",
                "width": 0,  # Not available for validation
                "height": 0,  # Not available for validation
                "num_frames": 0,  # Not available for validation
                "duration_sec": 0.0,  # Not available for validation
                "fps": 0.0,  # Not available for validation
            }
            batch_data.append(record)

            logger.info("Saved validation sample: %s", file_name)

        if batch_data:
            # Add progress bar for writing to Parquet dataset
            write_pbar = tqdm(total=1,
                              desc="Writing to Parquet dataset",
                              unit="batch")
            # Convert batch data to PyArrow arrays
            arrays = [
                pa.array([record["id"] for record in batch_data]),
                pa.array([record["vae_latent_bytes"] for record in batch_data],
                         type=pa.binary()),
                pa.array([record["vae_latent_shape"] for record in batch_data],
                         type=pa.list_(pa.int32())),
                pa.array([record["vae_latent_dtype"] for record in batch_data]),
                pa.array(
                    [record["text_embedding_bytes"] for record in batch_data],
                    type=pa.binary()),
                pa.array(
                    [record["text_embedding_shape"] for record in batch_data],
                    type=pa.list_(pa.int32())),
                pa.array(
                    [record["text_embedding_dtype"] for record in batch_data]),
                pa.array([
                    record["text_attention_mask_bytes"] for record in batch_data
                ],
                         type=pa.binary()),
                pa.array([
                    record["text_attention_mask_shape"] for record in batch_data
                ],
                         type=pa.list_(pa.int32())),
                pa.array([
                    record["text_attention_mask_dtype"] for record in batch_data
                ]),
                pa.array([record["file_name"] for record in batch_data]),
                pa.array([record["caption"] for record in batch_data]),
                pa.array([record["media_type"] for record in batch_data]),
                pa.array([record["width"] for record in batch_data],
                         type=pa.int32()),
                pa.array([record["height"] for record in batch_data],
                         type=pa.int32()),
                pa.array([record["num_frames"] for record in batch_data],
                         type=pa.int32()),
                pa.array([record["duration_sec"] for record in batch_data],
                         type=pa.float32()),
                pa.array([record["fps"] for record in batch_data],
                         type=pa.float32()),
            ]
            table = pa.Table.from_arrays(arrays,
                                         names=[f.name for f in pyarrow_schema])
            write_pbar.update(1)
            write_pbar.close()

            logger.info("Total validation samples: %s", len(table))

            work_range = (0, 1, table, 0, validation_parquet_dir, len(table))

            total_written = 0
            failed_ranges = []
            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = {
                    executor.submit(self.process_chunk_range, work_range):
                    work_range
                }
                for future in tqdm(futures, desc="Processing chunks"):
                    try:
                        total_written += future.result()
                    except Exception as e:
                        work_range = futures[future]
                        failed_ranges.append(work_range)
                        logger.error("Failed to process range %s-%s: %s",
                                     work_range[0], work_range[1], str(e))

            if failed_ranges:
                logger.warning("Retrying %s failed ranges sequentially",
                               len(failed_ranges))
                for work_range in failed_ranges:
                    try:
                        total_written += self.process_chunk_range(work_range)
                    except Exception as e:
                        logger.error(
                            "Failed to process range %s-%s after retry: %s",
                            work_range[0], work_range[1], str(e))

            logger.info("Total validation samples written: %s", total_written)

            # Clear memory
            del table
            gc.collect()  # Force garbage collection

    @staticmethod
    def process_chunk_range(args: Any) -> int:
        start_idx, end_idx, table, worker_id, output_dir, samples_per_file = args
        try:
            total_written = 0
            num_samples = len(table)

            # Create worker-specific subdirectory
            worker_dir = os.path.join(output_dir, f"worker_{worker_id}")
            os.makedirs(worker_dir, exist_ok=True)

            # Check how many files there are already in the dir, and update i accordingly
            num_parquets = 0
            for root, _, files in os.walk(worker_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        num_parquets += 1

            for i in range(start_idx, end_idx):
                start_sample = i * samples_per_file
                end_sample = min((i + 1) * samples_per_file, num_samples)
                chunk = table.slice(start_sample, end_sample - start_sample)

                # Create chunk file in worker's directory
                chunk_path = os.path.join(
                    worker_dir, f"data_chunk_{i + num_parquets}.parquet")
                temp_path = chunk_path + '.tmp'

                try:
                    # Write to temporary file
                    pq.write_table(chunk, temp_path, compression='zstd')

                    # Rename temporary file to final file
                    if os.path.exists(chunk_path):
                        os.remove(
                            chunk_path)  # Remove existing file if it exists
                    os.rename(temp_path, chunk_path)

                    total_written += len(chunk)
                except Exception as e:
                    # Clean up temporary file if it exists
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise e

            return total_written
        except Exception as e:
            logger.error("Error processing chunks %s-%s for worker %s: %s",
                         start_idx, end_idx, worker_id, str(e))
            raise


EntryClass = PreprocessPipeline