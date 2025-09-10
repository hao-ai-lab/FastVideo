# SPDX-License-Identifier: Apache-2.0
"""
Text-only Data Preprocessing pipeline implementation.

This module contains an implementation of the Text-only Data Preprocessing pipeline
using the modular pipeline architecture, based on the ODE Trajectory preprocessing.
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from fastvideo.dataset import gettextdataset
from fastvideo.dataset.dataloader.schema import pyarrow_schema_text_only
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.pipelines.stages import (TextEncodingStage)

logger = init_logger(__name__)


class PreprocessPipeline_Text(BasePreprocessPipeline):
    """Text-only preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer"
    ]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]

    def get_schema_fields(self):
        """Get the schema fields for text-only pipeline."""
        return [f.name for f in pyarrow_schema_text_only]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

    def preprocess_text_only(self,
                            fastvideo_args: FastVideoArgs,
                            args):
        """Preprocess text-only data."""
        
        for batch_idx, data in enumerate(self.pbar):
            if data is None:
                continue

            with torch.inference_mode():
                # For text-only processing, we only need text data
                # Filter out samples without text
                valid_indices = []
                for i, text in enumerate(data["text"]):
                    if text and text.strip():  # Check if text is not empty
                        valid_indices.append(i)
                self.num_processed_samples += len(valid_indices)

                if not valid_indices:
                    continue

                # Create new batch with only valid samples (text-only)
                valid_data = {
                    "text": [data["text"][i] for i in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                }

                batch_captions = valid_data["text"]
                # Encode text using the standalone TextEncodingStage API
                prompt_embeds_list, prompt_masks_list = self.prompt_encoding_stage.encode_text(
                    batch_captions,
                    fastvideo_args,
                    encoder_index=[0],
                    return_attention_mask=True,
                )
                prompt_embeds = prompt_embeds_list[0]
                prompt_attention_masks = prompt_masks_list[0]
                assert prompt_embeds.shape[0] == prompt_attention_masks.shape[0]

                logger.info(f"===== prompt_embeds: {prompt_embeds.shape}")
                logger.info(f"===== prompt_attention_masks: {prompt_attention_masks.shape}")

                # Prepare batch data for Parquet dataset
                batch_data = []

                # Add progress bar for saving outputs
                save_pbar = tqdm(enumerate(valid_data["path"]),
                                 desc="Saving outputs",
                                 unit="item",
                                 leave=False)
                
                for idx, text_path in save_pbar:
                    text_name = os.path.basename(text_path).split(".")[0]

                    # Convert tensors to numpy arrays
                    text_embedding = prompt_embeds[idx].cpu().numpy()

                    # Create record for Parquet dataset (text-only)
                    record = self.create_text_only_record(
                        text_name=text_name,
                        text_embedding=text_embedding,
                        valid_data=valid_data,
                        idx=idx,
                        caption=valid_data["text"][idx])
                    batch_data.append(record)

                if batch_data:
                    # Add progress bar for writing to Parquet dataset
                    write_pbar = tqdm(total=1,
                                      desc="Writing to Parquet dataset",
                                      unit="batch")
                    # Convert batch data to PyArrow arrays
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
                        else:
                            arrays.append(
                                pa.array([record[field] for record in batch_data]))

                    table = pa.Table.from_arrays(arrays,
                                                 names=self.get_schema_fields())
                    write_pbar.update(1)
                    write_pbar.close()

                    # Store the table in a list for later processing
                    if not hasattr(self, 'all_tables'):
                        self.all_tables = []
                    self.all_tables.append(table)

                    logger.info("Collected batch with %s samples", len(table))

                if self.num_processed_samples >= args.flush_frequency:
                    self._flush_tables(self.num_processed_samples, args,
                                       self.combined_parquet_dir)
                    self.num_processed_samples = 0
                    self.all_tables = []
        
        # Final flush for any remaining samples
        if hasattr(self, 'all_tables') and self.all_tables and self.num_processed_samples > 0:
            logger.info(f"Final flush with {self.num_processed_samples} remaining samples")
            self._flush_tables(self.num_processed_samples, args, self.combined_parquet_dir)
            self.num_processed_samples = 0
            self.all_tables = []

    def create_text_only_record(
            self,
            text_name: str,
            text_embedding: np.ndarray,
            valid_data: dict[str, Any],
            idx: int,
            caption: str) -> dict[str, Any]:
        """Create a record for text-only preprocessing using text-only schema."""
        
        # Create base record using only fields from text-only schema
        record = {
            "id": f"text_{text_name}_{idx}",
            "text_embedding_bytes": text_embedding.tobytes(),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype),
            "caption": caption,
        }

        return record

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs, args):
        if not self.post_init_called:
            self.post_init()

        self.local_rank = int(os.getenv("RANK", 0))
        os.makedirs(args.output_dir, exist_ok=True)
        # Create directory for combined data
        self.combined_parquet_dir = os.path.join(args.output_dir,
                                                 "combined_parquet_dataset")
        os.makedirs(self.combined_parquet_dir, exist_ok=True)

        # Loading text dataset
        train_dataset = gettextdataset(args)

        self.preprocess_dataloader = DataLoader(
            train_dataset,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        self.preprocess_loader_iter = iter(self.preprocess_dataloader)

        self.num_processed_samples = 0
        # Add progress bar for text preprocessing
        self.pbar = tqdm(self.preprocess_loader_iter,
                         desc="Processing text",
                         unit="batch",
                         disable=self.local_rank != 0)

        # Initialize class variables for data sharing
        self.text_data: dict[str, Any] = {}  # Store text metadata and paths
        
        self.preprocess_text_only(fastvideo_args, args)


EntryClass = PreprocessPipeline_Text
