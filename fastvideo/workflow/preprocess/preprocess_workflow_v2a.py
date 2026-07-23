# SPDX-License-Identifier: Apache-2.0
"""Shared raw-media workflow for video-to-audio feature preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fastvideo.configs.configs import DatasetType
from fastvideo.dataset.v2a_feature_cache import V2AFeatureShardWriter
from fastvideo.dataset.vggsound import VGGSoundDataset
from fastvideo.distributed.parallel_state import get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow

if TYPE_CHECKING:
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

logger = init_logger(__name__)


def build_v2a_dataset(
    dataset_type: DatasetType,
    dataset_path: str,
    *,
    dataset_metadata_path: str = "",
    dataset_split: str = "train",
):
    """Build a raw-media dataset for a V2A preprocessing workflow."""
    if isinstance(dataset_type, str):
        dataset_type = DatasetType.from_string(dataset_type)
    if dataset_type is DatasetType.VGGSOUND:
        metadata_path = dataset_metadata_path or None
        return VGGSoundDataset(
            dataset_path,
            split=dataset_split,
            metadata_path=metadata_path,
        )
    raise ValueError(f"V2A preprocessing currently supports dataset_type=vggsound; got {dataset_type.value!r}")


class V2AForwardBatchBuilder:
    """Translate standardized raw-media rows into a FastVideo batch."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, rows: list[dict[str, Any]]) -> PreprocessBatch:
        return PreprocessBatch(
            data_type="video",
            video_loader=[str(row["video_path"]) for row in rows],
            video_file_name=[str(row["id"]) for row in rows],
            prompt=[str(row["caption"]) for row in rows],
            batch_size=len(rows),
            generator=torch.Generator("cpu").manual_seed(self.seed),
        )


class PreprocessWorkflowV2A(PreprocessWorkflow):
    """Run model-specific V2A encoders with shared data/cache orchestration."""

    training_dataloader: DataLoader
    preprocess_pipeline: ComposedPipelineBase
    forward_batch_builder: V2AForwardBatchBuilder
    feature_writer: V2AFeatureShardWriter

    def register_components(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        config = self.fastvideo_args.preprocess_config
        dataset = build_v2a_dataset(
            config.dataset_type,
            config.dataset_path,
            dataset_metadata_path=config.dataset_metadata_path,
            dataset_split=config.dataset_split,
        )

        rank = get_world_rank()
        world_size = get_world_size()
        rank_indices = list(range(rank, len(dataset), world_size))
        rank_dataset = Subset(dataset, rank_indices)
        dataloader = DataLoader(
            rank_dataset,
            batch_size=config.preprocess_video_batch_size,
            num_workers=config.dataloader_num_workers,
            collate_fn=lambda rows: rows,
            shuffle=False,
        )
        self.add_component("training_dataloader", dataloader)
        self.add_component("forward_batch_builder", V2AForwardBatchBuilder(config.seed))
        self.add_component(
            "feature_writer",
            V2AFeatureShardWriter(
                config.dataset_output_dir,
                rank=rank,
                samples_per_shard=config.samples_per_file,
            ),
        )

    def prepare_system_environment(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        output_root = Path(self.fastvideo_args.preprocess_config.dataset_output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self.failure_log = output_root / f"failures_rank_{get_world_rank():05d}.jsonl"
        logger.info("V2A feature cache output: %s", output_root)

    def _write_failures(self, failures: list[dict[str, str]]) -> None:
        if not failures:
            return
        with self.failure_log.open("a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    def run(self) -> None:
        processed = 0
        skipped = 0
        failed = 0
        try:
            for rows in tqdm(
                    self.training_dataloader,
                    desc="Preprocessing V2A training data",
                    unit="batch",
            ):
                pending_rows = [row for row in rows if not self.feature_writer.contains(str(row["id"]))]
                skipped += len(rows) - len(pending_rows)
                if not pending_rows:
                    continue
                batch = self.forward_batch_builder(pending_rows)
                batch = self.preprocess_pipeline.forward(batch, self.fastvideo_args)
                failures = batch.extra.get("failed_samples", [])
                failed += len(failures)
                self._write_failures(failures)
                features = batch.extra.get("precomputed_features", {})
                metadata = batch.extra.get("precomputed_metadata", [])
                self.feature_writer.append(features, metadata)
                processed += len(metadata)
        finally:
            self.feature_writer.close()
        logger.info(
            "Finished V2A preprocessing on rank %s: processed=%s skipped=%s failed=%s output=%s",
            get_world_rank(),
            processed,
            skipped,
            failed,
            self.output_root,
        )


__all__ = [
    "PreprocessWorkflowV2A",
    "V2AForwardBatchBuilder",
    "build_v2a_dataset",
]
