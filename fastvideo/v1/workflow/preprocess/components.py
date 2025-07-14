from typing import Any, Callable, Optional
import os
import random

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch, PreprocessBatch

logger = init_logger(__name__)

class PreprocessingDataValidator:
    def __init__(self,
                 max_height: int = 1024,
                 max_width: int = 1024,
                 max_h_div_w_ratio: float = 17 / 16,
                 min_h_div_w_ratio: float = 8 / 16,
                 num_frames: int = 16,
                 train_fps: int = 24,
                 speed_factor: int = 1,
                 video_length_tolerance_range: float = 5.0,
                 drop_short_ratio: float = 0.0,
                 hw_aspect_threshold: float = 1.5):
        self.max_height = max_height
        self.max_width = max_width
        self.max_h_div_w_ratio = max_h_div_w_ratio
        self.min_h_div_w_ratio = min_h_div_w_ratio
        self.num_frames = num_frames
        self.train_fps = train_fps
        self.speed_factor = speed_factor
        self.video_length_tolerance_range = video_length_tolerance_range
        self.drop_short_ratio = drop_short_ratio
        self.hw_aspect_threshold = hw_aspect_threshold
        self.validators = {}
        self.filter_counts = {}

        self.num_items_before_filtering = 0
        self.num_items_after_filtering = 0

        self.register_validators()
    
    def register_validators(self):
        self.add_validator("data_type_validator", self._validate_data_type)
        self.add_validator("resolution_validator", self._validate_resolution)
        self.add_validator("frame_sampling_validator", self._validate_frame_sampling)

    def add_validator(self, name: str, validator: Callable[[ForwardBatch], bool]):
        self.validators[name] = validator
        self.filter_counts[name] = 0

    def __call__(self, batch: ForwardBatch) -> bool:
        """
        Validate whether the preprocessing data batch is valid.
        """
        self.num_items_before_filtering += 1

        for name, validator in self.validators.items():
            if not validator(batch):
                self.filter_counts[name] += 1
                return False

        self.num_items_after_filtering += 1
        return True

    def _validate_data_type(self, batch: ForwardBatch) -> bool:
        """Validate basic validity of data items"""
        if batch.prompt is None or batch.prompt == "":
            return False
        
        if self._is_video_batch(batch):
            if batch.fps is None or batch.fps <= 0:
                return False
            if batch.num_frames is None or batch.num_frames <= 0:
                return False
        
        return True
    
    def _validate_resolution(self, batch: ForwardBatch) -> bool:
        """Validate resolution constraints"""
        if not self._is_video_batch(batch):
            return True
        
        if batch.height is None or batch.width is None:
            return False
        
        aspect = self.max_height / self.max_width
        
        return self._filter_resolution(
            batch.height,
            batch.width,
            max_h_div_w_ratio=self.hw_aspect_threshold * aspect,
            min_h_div_w_ratio=1 / self.hw_aspect_threshold * aspect,
        )
    
    def _filter_resolution(self, h: int, w: int, max_h_div_w_ratio: float,
                          min_h_div_w_ratio: float) -> bool:
        """Filter based on aspect ratio"""
        return h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio and h / w <= self.max_h_div_w_ratio and h / w >= self.min_h_div_w_ratio

    
    def _validate_frame_sampling(self, batch: ForwardBatch) -> bool:
        """Validate frame sampling constraints"""
        if not self._is_video_batch(batch):
            return True
        
        if batch.fps is None or batch.num_frames is None:
            return False
        
        if (batch.num_frames / batch.fps > self.video_length_tolerance_range *
            (self.num_frames / self.train_fps * self.speed_factor)):
            return False
        
        frame_interval = batch.fps / self.train_fps
        start_frame_idx = 0
        frame_indices = np.arange(start_frame_idx, batch.num_frames,
                                  frame_interval).astype(int)
        
        return not (len(frame_indices) < self.num_frames
                    and random.random() < self.drop_short_ratio)
    
    def _is_video_batch(self, batch: ForwardBatch) -> bool:
        return batch.data_type == "video"

    def log_validation_stats(self):
        info = ""
        for name, count in self.filter_counts.items():
            info += f"failed in {name}: {count}, "
        info += f"number of items before filtering: {self.num_items_before_filtering}, "
        info += f"number of items after filtering: {self.num_items_after_filtering}"

        logger.info(info)


class VideoForwardBatchBuilder:
    def __call__(self, batch: list) -> PreprocessBatch:
        forward_batch = PreprocessBatch()
        if len(batch) == 1:
            item = batch[0]
            forward_batch.video_loader.append(item["video"])
            forward_batch.name.append(item["name"])
            forward_batch.height = item["resolution"]["height"]
            forward_batch.width = item["resolution"]["width"]
            forward_batch.fps = item["fps"]
            forward_batch.num_frames = item["num_frames"]
            forward_batch.prompt = item["caption"]
            forward_batch.data_type = "video"
        else:
            raise ValueError("Batch size must = 1")
            # TODO(will): implement batching
            if False:
                for item in batch:
                    forward_batch.video_loader.append(item["video"])
                    forward_batch.height = item["resolution"]["height"]
                    forward_batch.width = item["resolution"]["width"]
                    forward_batch.fps = item["fps"]
                    forward_batch.num_frames = item["num_frames"]
                    forward_batch.prompt = item["caption"]
                    forward_batch.data_type = "video"

        return forward_batch


class ParquetDatasetSaver:
    """Component for saving and writing Parquet datasets"""
    
    def __init__(self, 
                 schema_fields_provider: Callable[[], list[str]],
                 record_creator: Callable[..., dict[str, Any]],
                 chunk_processor: Optional[Callable] = None):
        """
        Initialize ParquetDatasetSaver
        
        Args:
            schema_fields_provider: Function that returns schema fields list
            record_creator: Function for creating records
            chunk_processor: Function for processing chunks, uses default implementation if None
        """
        self.get_schema_fields = schema_fields_provider
        self.create_record = record_creator
        self.process_chunk_range = chunk_processor or self._default_process_chunk_range
        self.all_tables = []
        
    def save_and_write_parquet_batch(self,
                                   batch: PreprocessBatch,
                                   latents: torch.Tensor,
                                   prompt_embeds: list,
                                   prompt_attention_mask: Optional[torch.Tensor],
                                   extra_features: Optional[dict[str, Any]] = None,
                                   num_processed_samples: int = 0,
                                   args = None,
                                   combined_parquet_dir: str = None) -> int:
        """
        Save and write Parquet dataset batch
        
        Args:
            batch: PreprocessBatch containing video and metadata information
            latents: VAE latent vectors
            prompt_embeds: Text embeddings
            prompt_attention_mask: Attention mask
            extra_features: Extra features
            num_processed_samples: Number of processed samples
            args: Arguments object
            combined_parquet_dir: Combined parquet directory
            
        Returns:
            Number of processed samples
        """
        # Process non-padded embeddings (if needed)
        if prompt_attention_mask is not None:
            prompt_embeds, prompt_attention_mask = self._process_non_padded_embeddings(
                prompt_embeds, prompt_attention_mask
            )

        # Prepare batch data for Parquet dataset
        batch_data = []

        # Add progress bar for saving outputs
        save_pbar = tqdm(enumerate(batch.name),
                         desc="Saving outputs",
                         unit="item",
                         leave=False)
        
        for idx, video_name in save_pbar:
            # Get the corresponding latent and info using video name
            latent = latents[idx].cpu()

            # Convert tensors to numpy arrays
            vae_latent = latent.cpu().numpy()
            text_embedding = prompt_embeds[idx].cpu().numpy()

            # Get extra features for this sample if needed
            sample_extra_features = {}
            if extra_features:
                for key, value in extra_features.items():
                    if isinstance(value, torch.Tensor):
                        sample_extra_features[key] = value[idx].cpu().numpy()
                    else:
                        sample_extra_features[key] = value[idx]

            # Create record for Parquet dataset
            record = self.create_record_from_batch(
                video_name=video_name,
                vae_latent=vae_latent,
                text_embedding=text_embedding,
                batch=batch,
                idx=idx,
                extra_features=sample_extra_features)
            batch_data.append(record)

        if batch_data:
            # Add progress bar for writing to Parquet dataset
            write_pbar = tqdm(total=1,
                              desc="Writing to Parquet dataset",
                              unit="batch")
            
            # Convert batch data to PyArrow arrays
            table = self._convert_batch_to_pyarrow_table(batch_data)
            write_pbar.update(1)
            write_pbar.close()

            # Store the table in a list for later processing
            self.all_tables.append(table)
            logger.info("Collected batch with %s samples", len(table))

        # If flush is needed
        if args and num_processed_samples >= args.flush_frequency:
            self.flush_tables(num_processed_samples, args, combined_parquet_dir)
            return 0  # Reset count
            
        return len(batch_data) if batch_data else 0

    def _process_non_padded_embeddings(self, prompt_embeds, prompt_attention_mask):
        """Process non-padded embeddings"""
        if isinstance(prompt_embeds, torch.Tensor) and isinstance(prompt_attention_mask, torch.Tensor):
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

            return non_padded_embeds, non_padded_masks
        
        return prompt_embeds, prompt_attention_mask

    def _convert_batch_to_pyarrow_table(self, batch_data: list[dict]) -> pa.Table:
        """Convert batch data to PyArrow table"""
        arrays = []
        
        for field in self.get_schema_fields():
            if field.endswith('_bytes'):
                arrays.append(
                    pa.array([record[field] for record in batch_data],
                             type=pa.binary()))
            elif field.endswith('_shape'):
                arrays.append(
                    pa.array([record[field] for record in batch_data],
                             type=pa.list_(pa.int32())))
            elif field in ['width', 'height', 'num_frames']:
                arrays.append(
                    pa.array([record[field] for record in batch_data],
                             type=pa.int32()))
            elif field in ['duration_sec', 'fps']:
                arrays.append(
                    pa.array([record[field] for record in batch_data],
                             type=pa.float32()))
            else:
                arrays.append(
                    pa.array([record[field] for record in batch_data]))

        return pa.Table.from_arrays(arrays, names=self.get_schema_fields())

    def flush_tables(self, num_processed_samples: int, args, combined_parquet_dir: str):
        """Flush collected tables to disk"""
        if not hasattr(self, 'all_tables') or not self.all_tables:
            return

        print(f"Combining {len(self.all_tables)} batches...")
        combined_table = pa.concat_tables(self.all_tables)
        assert len(combined_table) == num_processed_samples
        print(f"Total samples collected: {len(combined_table)}")

        # Calculate total number of chunks needed, discarding remainder
        total_chunks = max(num_processed_samples // args.samples_per_file, 1)

        print(f"Fixed samples per parquet file: {args.samples_per_file}")
        print(f"Total number of parquet files: {total_chunks}")
        print(
            f"Total samples to be processed: {total_chunks * args.samples_per_file} (discarding {num_processed_samples % args.samples_per_file} samples)"
        )

        # Split work among processes
        num_workers = int(min(multiprocessing.cpu_count(), total_chunks))
        chunks_per_worker = (total_chunks + num_workers - 1) // num_workers

        print(f"Using {num_workers} workers to process {total_chunks} chunks")
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
                    logger.info("Processed chunk with %s samples", written)
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
                    total_written += self.process_chunk_range(work_range)
                except Exception as e:
                    logger.error(
                        "Failed to process range %s-%s after retry: %s",
                        work_range[0], work_range[1], str(e))

        logger.info("Total samples written: %s", total_written)
        
        # Clear tables list
        self.all_tables = []

    def _default_process_chunk_range(self, args_tuple):
        """Default chunk processing implementation"""
        start_idx, end_idx, combined_table, worker_id, output_dir, samples_per_file = args_tuple
        
        written_count = 0
        for chunk_idx in range(start_idx, end_idx):
            start_row = chunk_idx * samples_per_file
            end_row = min(start_row + samples_per_file, len(combined_table))
            
            if start_row >= len(combined_table):
                break
                
            chunk_table = combined_table.slice(start_row, end_row - start_row)
            
            # Write to file
            output_file = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.parquet")
            pq.write_table(chunk_table, output_file)
            written_count += len(chunk_table)
            
        return written_count

    def create_record_from_batch(self,
                                video_name: str,
                                vae_latent: np.ndarray,
                                text_embedding: np.ndarray,
                                batch: PreprocessBatch,
                                idx: int,
                                extra_features: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a record for the Parquet dataset from PreprocessBatch."""
        # For batch processing, we need to handle the case where some fields might be single values
        # or lists depending on the batch size
        
        # Get caption - handle both single string and list cases
        caption = ""
        if batch.prompt:
            if isinstance(batch.prompt, list):
                caption = batch.prompt[idx] if idx < len(batch.prompt) else batch.prompt[0]
            else:
                caption = batch.prompt
        
        # Get dimensions - these are typically single values in PreprocessBatch
        width = batch.width if batch.width is not None else 0
        height = batch.height if batch.height is not None else 0
        
        # Get FPS - single value in PreprocessBatch
        fps_val = float(batch.fps) if batch.fps is not None else 0.0
        
        # For duration, we need to calculate it or use a default since it's not in PreprocessBatch
        # duration = num_frames / fps if available
        duration_val = 0.0
        if batch.num_frames and batch.fps and batch.fps > 0:
            duration_val = float(batch.num_frames) / float(batch.fps)
        
        record = {
            "id": video_name,
            "vae_latent_bytes": vae_latent.tobytes(),
            "vae_latent_shape": list(vae_latent.shape),
            "vae_latent_dtype": str(vae_latent.dtype),
            "text_embedding_bytes": text_embedding.tobytes(),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype),
            "file_name": video_name,
            "caption": caption,
            "media_type": "video",
            "width": int(width),
            "height": int(height),
            "num_frames": vae_latent.shape[1] if len(vae_latent.shape) > 1 else 0,
            "duration_sec": duration_val,
            "fps": fps_val,
        }
        
        if extra_features:
            record.update(extra_features)
        return record

    def clear_tables(self):
        """Clear all tables"""
        self.all_tables = []