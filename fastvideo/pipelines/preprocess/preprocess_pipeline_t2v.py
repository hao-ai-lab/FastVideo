# SPDX-License-Identifier: Apache-2.0
"""
T2V Data Preprocessing pipeline implementation.

This module contains an implementation of the T2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)


class PreprocessPipeline_T2V(BasePreprocessPipeline):
    """T2V preprocessing pipeline implementation."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def get_schema_fields(self):
        """Get the schema fields for T2V pipeline."""
        return [f.name for f in pyarrow_schema_t2v]


EntryClass = PreprocessPipeline_T2V
