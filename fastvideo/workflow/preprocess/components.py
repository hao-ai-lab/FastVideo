import dataclasses
import gc
import multiprocessing
import os
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, Video, load_dataset

from fastvideo.configs.configs import DatasetType, PreprocessConfig
from fastvideo.distributed.parallel_state import get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch

logger = init_logger(__name__)


class PreprocessingDataValidator:

    def __init__(self,
                 max_height: int = 1024,
                 max_width: int = 1024,
                 max_h_div_w_ratio: float = 17 / 16,
                 min_h_div_w_ratio: float = 8 / 16,
                 num_frames: int = 16,
                 train_fps: int = 24,
                 speed_factor: float = 1.0,
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
        self.validators: dict[str, Callable[[dict[str, Any]], bool]] = {}
        self.filter_counts: dict[str, int] = {}

        self.num_items_before_filtering = 0
        self.num_items_after_filtering = 0

        self.register_validators()

    def register_validators(self) -> None:
        self.add_validator("data_type_validator", self._validate_data_type)
        self.add_validator("resolution_validator", self._validate_resolution)
        self.add_validator("frame_sampling_validator",
                           self._validate_frame_sampling)

    def add_validator(self, name: str, validator: Callable[[dict[str, Any]],
                                                           bool]) -> None:
        self.validators[name] = validator
        self.filter_counts[name] = 0

    def __call__(self, batch: dict[str, Any]) -> bool:
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

    def _validate_data_type(self, batch: dict[str, Any]) -> bool:
        """Validate basic validity of data items"""
        return not (batch["caption"] is None or batch["caption"] == ""
                    or batch["fps"] is None or batch["fps"] <= 0
                    or batch["num_frames"] is None or batch["num_frames"] <= 0)

    def _validate_resolution(self, batch: dict[str, Any]) -> bool:
        """Validate resolution constraints"""

        aspect = self.max_height / self.max_width
        if batch["resolution"] is not None:
            height = batch["resolution"].get("height", None)
            width = batch["resolution"].get("width", None)

        if height is None or width is None:
            return False

        return self._filter_resolution(
            height,
            width,
            max_h_div_w_ratio=self.hw_aspect_threshold * aspect,
            min_h_div_w_ratio=1 / self.hw_aspect_threshold * aspect,
        )

    def _filter_resolution(self, h: int, w: int, max_h_div_w_ratio: float,
                           min_h_div_w_ratio: float) -> bool:
        """Filter based on aspect ratio"""
        return (min_h_div_w_ratio <= h / w <= max_h_div_w_ratio) and (
            self.min_h_div_w_ratio <= h / w <= self.max_h_div_w_ratio)

    def _validate_frame_sampling(self, batch: dict[str, Any]) -> bool:
        """Validate frame sampling constraints"""

        if (batch["num_frames"] / batch["fps"]
                > self.video_length_tolerance_range *
            (self.num_frames / self.train_fps * self.speed_factor)):
            return False

        frame_interval = batch["fps"] / self.train_fps
        start_frame_idx = 0
        frame_indices = np.arange(start_frame_idx, batch["num_frames"],
                                  frame_interval).astype(int)
        return not (len(frame_indices) < self.num_frames
                    and random.random() < self.drop_short_ratio)

    def log_validation_stats(self):
        info = ""
        for name, count in self.filter_counts.items():
            info += f"failed in {name}: {count}, "
        info += f"number of items before filtering: {self.num_items_before_filtering}, "
        info += f"number of items after filtering: {self.num_items_after_filtering}"

        logger.info(info)


class VideoForwardBatchBuilder:

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, batch: list) -> PreprocessBatch:
        forward_batch = PreprocessBatch(
            video_loader=[item["video"] for item in batch],
            video_file_name=[item["name"] for item in batch],
            height=[item["resolution"]["height"] for item in batch],
            width=[item["resolution"]["width"] for item in batch],
            fps=[item["fps"] for item in batch],
            num_frames=[item["num_frames"] for item in batch],
            prompt=[item["caption"] for item in batch],
            prompt_attention_mask=[],
            data_type="video",
            generator=torch.Generator("cpu").manual_seed(self.seed),
        )
        return forward_batch


class ParquetDatasetSaver:
    """Component for saving and writing Parquet datasets"""

    def __init__(self,
                 flush_frequency: int,
                 samples_per_file: int,
                 schema_fields: list[str],
                 record_creator: Callable[..., list[dict[str, Any]]],
                 file_writer_fn: Callable | None = None):
        """
        Initialize ParquetDatasetSaver
        
        Args:
            schema_fields: schema fields list
            record_creator: Function for creating records
            file_writer_fn: Function for writing records to files, uses default implementation if None
        """
        self.flush_frequency = flush_frequency
        self.samples_per_file = samples_per_file
        self.schema_fields = schema_fields
        self.create_records_from_batch = record_creator
        self.file_writer_fn: Callable[
            [tuple], int] = file_writer_fn or self._default_file_writer_fn
        self.all_tables: list[pa.Table] = []
        self.num_processed_samples: int = 0
        self.num_saved_files: int = 0

    def save_and_write_parquet_batch(
            self,
            batch: PreprocessBatch,
            output_dir: str,
            extra_features: dict[str, Any] | None = None) -> None:
        """
        Save and write Parquet dataset batch
        
        Args:
            batch: PreprocessBatch containing video and metadata information
            output_dir: Output directory
            extra_features: Extra features
            
        Returns:
            Number of processed samples
        """
        assert isinstance(batch.latents, torch.Tensor)
        assert isinstance(batch.prompt_embeds, list)
        assert isinstance(batch.prompt_attention_mask, list)

        # Process non-padded embeddings (if needed)
        if batch.prompt_attention_mask is not None:
            batch.prompt_embeds = self._process_non_padded_embeddings(
                batch.prompt_embeds[0], batch.prompt_attention_mask[0])
        else:
            raise ValueError("prompt_attention_mask is None")

        # Prepare batch data for Parquet dataset
        batch_data: list[dict[str, Any]] = []

        for key in dataclasses.fields(batch):
            value = getattr(batch, key.name)
            if isinstance(value, list):
                for idx in range(len(value)):
                    if isinstance(value[idx], torch.Tensor):
                        value[idx] = value[idx].cpu().numpy()
            elif isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
                setattr(batch, key.name, value)

        # Create record for Parquet dataset
        records = self.create_records_from_batch(batch)
        batch_data.extend(records)

        if batch_data:
            self.num_processed_samples += len(batch_data)
            # Convert batch data to PyArrow arrays
            table = self._convert_batch_to_pyarrow_table(batch_data)

            # Store the table in a list for later processing
            self.all_tables.append(table)
            logger.debug("Collected batch with %s samples", len(table))

        # If flush is needed
        if self.num_processed_samples >= self.flush_frequency:
            self.flush_tables(output_dir)

    def _process_non_padded_embeddings(
            self, prompt_embeds: torch.Tensor,
            prompt_attention_mask: torch.Tensor) -> list[torch.Tensor]:
        """Process non-padded embeddings"""
        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(prompt_attention_mask, torch.Tensor)
        assert prompt_embeds.shape[0] == prompt_attention_mask.shape[0]

        # Get sequence lengths from attention masks (number of 1s)
        seq_lens = prompt_attention_mask.sum(dim=1)

        non_padded_embeds = []

        # Process each item in the batch
        for i in range(prompt_embeds.size(0)):
            seq_len = seq_lens[i].item()
            # Slice the embeddings and masks to keep only non-padding parts
            non_padded_embeds.append(prompt_embeds[i, :seq_len])

        return non_padded_embeds

    def _convert_batch_to_pyarrow_table(self,
                                        batch_data: list[dict]) -> pa.Table:
        """Convert batch data to PyArrow table"""
        arrays = []

        for field in self.schema_fields:
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
                arrays.append(pa.array([record[field]
                                        for record in batch_data]))

        return pa.Table.from_arrays(arrays, names=self.schema_fields)

    def flush_tables(self, output_dir: str):
        """Flush collected tables to disk"""
        if not hasattr(self, 'all_tables') or not self.all_tables:
            return

        logger.debug("Combining %d batches...", len(self.all_tables))
        combined_table = pa.concat_tables(self.all_tables)
        assert len(combined_table) == self.num_processed_samples
        logger.debug("Total samples collected: %d", len(combined_table))

        # Calculate total number of chunks needed, putting remainder into self.all_tables
        total_files = max(self.num_processed_samples // self.samples_per_file,
                          1)

        logger.debug("Fixed samples per parquet file: %d",
                     self.samples_per_file)
        logger.debug("Total number of parquet files: %d", total_files)
        logger.debug(
            "Total samples to be processed: %d (putting %d samples into self.all_tables)",
            total_files * self.samples_per_file,
            self.num_processed_samples % self.samples_per_file)

        # Split work among processes
        num_workers = int(min(multiprocessing.cpu_count(), total_files))
        files_per_worker = (total_files + num_workers - 1) // num_workers

        logger.debug("Using %d workers to process %d files", num_workers,
                     total_files)
        logger.debug("Files per worker: %s", files_per_worker)

        # Prepare work ranges
        work_ranges = []
        for i in range(num_workers):
            start_idx = i * files_per_worker
            end_idx = min((i + 1) * files_per_worker, total_files)
            if start_idx < total_files:
                work_ranges.append((start_idx, end_idx, combined_table, i,
                                    output_dir, self.samples_per_file))

        total_written = 0
        failed_ranges = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.file_writer_fn, work_range): work_range
                for work_range in work_ranges
            }
            for future in futures:
                try:
                    written = future.result()
                    total_written += written
                    logger.info("Processed file with %s samples", written)
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
                    total_written += self.file_writer_fn(work_range)
                except Exception as e:
                    logger.error(
                        "Failed to process range %s-%s after retry: %s",
                        work_range[0], work_range[1], str(e))

        self.num_saved_files += total_files

        # Clear tables list
        self.all_tables = []
        if self.num_processed_samples > self.samples_per_file:
            saved_samples = total_files * self.samples_per_file
            self.all_tables.append(combined_table.slice(saved_samples))
            self.num_processed_samples -= saved_samples
        else:
            self.num_processed_samples = 0

        del combined_table
        gc.collect()

    def clean_up(self) -> None:
        """Clean up all tables"""
        self.all_tables = []
        self.num_processed_samples = 0
        self.num_saved_files = 0
        gc.collect()

    def _default_file_writer_fn(self, args_tuple: tuple) -> int:
        """Default chunk processing implementation"""
        start_idx, end_idx, combined_table, worker_id, output_dir, samples_per_file = args_tuple

        written_count = 0
        for file_idx in range(start_idx, end_idx):
            start_row = file_idx * samples_per_file
            end_row = min(start_row + samples_per_file, len(combined_table))

            if start_row >= len(combined_table):
                break

            chunk_table = combined_table.slice(start_row, end_row - start_row)

            # Write to file
            output_file = os.path.join(
                output_dir,
                f"chunk_{file_idx + self.num_saved_files:06d}.parquet")
            pq.write_table(chunk_table, output_file)
            written_count += len(chunk_table)

        return written_count


def build_dataset(preprocess_config: PreprocessConfig, split: str,
                  validator: Callable[[dict[str, Any]], bool]) -> Dataset:
    if preprocess_config.dataset_type == DatasetType.HF:
        dataset = load_dataset(preprocess_config.dataset_path, split=split)
        dataset = dataset.filter(validator)
        dataset = dataset.shard(num_shards=get_world_size(),
                                index=get_world_rank())
    elif preprocess_config.dataset_type == DatasetType.MERGED:
        metadata_json_path = os.path.join(preprocess_config.dataset_path,
                                          "videos2caption.json")
        video_folder = os.path.join(preprocess_config.dataset_path, "videos")
        dataset = load_dataset("json",
                               data_files=metadata_json_path,
                               split=split)
        column_names = dataset.column_names
        # rename columns to match the schema
        if "cap" in column_names:
            dataset = dataset.rename_column("cap", "caption")
        if "path" in column_names:
            dataset = dataset.rename_column("path", "name")

        dataset = dataset.filter(validator)
        dataset = dataset.shard(num_shards=get_world_size(),
                                index=get_world_rank())

        # add video column
        def add_video_column(item: dict[str, Any]) -> dict[str, Any]:
            item["video"] = os.path.join(video_folder, item["name"])
            return item

        dataset = dataset.map(add_video_column)
        dataset = dataset.cast_column("video", Video())
    else:
        raise ValueError(
            f"Invalid dataset type: {preprocess_config.dataset_type}")

    return dataset
