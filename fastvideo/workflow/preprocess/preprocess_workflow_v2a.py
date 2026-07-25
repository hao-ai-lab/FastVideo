# SPDX-License-Identifier: Apache-2.0
"""Shared raw-media workflow for video-to-audio feature preprocessing."""

from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fastvideo.configs.configs import DatasetType, VideoLoaderType
from fastvideo.dataset.v2a_feature_cache import V2AFeatureShardWriter
from fastvideo.dataset.vggsound import VGGSoundDataset
from fastvideo.distributed.parallel_state import get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.pipelines.preprocess.mmaudio.torio_media_reader import (MMAudioTorioRowPreprocessor,
                                                                       PREPROCESSED_MEDIA_KEY, PREPROCESS_ERROR_KEY)
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow

if TYPE_CHECKING:
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

logger = init_logger(__name__)


def _identity_collate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


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
        extra: dict[str, Any] = {}
        preprocessed_media = [row.get(PREPROCESSED_MEDIA_KEY) for row in rows]
        preprocess_errors = [row.get(PREPROCESS_ERROR_KEY) for row in rows]
        if any(media is not None for media in preprocessed_media):
            extra[PREPROCESSED_MEDIA_KEY] = preprocessed_media
        if any(error is not None for error in preprocess_errors):
            extra[PREPROCESS_ERROR_KEY] = preprocess_errors
        return PreprocessBatch(
            data_type="video",
            video_loader=[str(row["video_path"]) for row in rows],
            video_file_name=[str(row["id"]) for row in rows],
            prompt=[str(row["caption"]) for row in rows],
            batch_size=len(rows),
            generator=torch.Generator("cpu").manual_seed(self.seed),
            extra=extra,
        )


class PreprocessWorkflowV2A(PreprocessWorkflow):
    """Run model-specific V2A encoders with shared data/cache orchestration."""

    training_dataloader: DataLoader
    preprocess_pipeline: ComposedPipelineBase
    forward_batch_builder: V2AForwardBatchBuilder
    feature_writer: V2AFeatureShardWriter

    def register_components(self) -> None:
        self.torio_row_preprocessor: MMAudioTorioRowPreprocessor | None = None
        self.torio_decode_workers = 0
        self.initial_skipped = 0
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
        feature_writer = V2AFeatureShardWriter(
            config.dataset_output_dir,
            rank=rank,
            samples_per_shard=config.samples_per_file,
        )
        rank_indices = list(range(rank, len(dataset), world_size))
        pending_indices = [index for index in rank_indices if not feature_writer.contains(str(dataset[index]["id"]))]
        self.initial_skipped = len(rank_indices) - len(pending_indices)
        rank_indices = pending_indices
        rank_dataset = Subset(dataset, rank_indices)

        dataloader_workers = config.dataloader_num_workers
        if config.video_loader_type == VideoLoaderType.TORIO:
            pc = self.fastvideo_args.pipeline_config
            duration_s = float(pc.duration_s)
            sample_rate = int(pc.sampling_rate)
            alignment = int(pc.spectrogram_frame_rate * pc.latent_downsample_rate)
            target_samples = math.ceil(duration_s * sample_rate / alignment) * alignment
            self.torio_row_preprocessor = MMAudioTorioRowPreprocessor(
                duration_s=duration_s,
                target_sample_rate=sample_rate,
                target_samples=target_samples,
                normalize_audio=config.dataset_split.lower() == "train",
                clip_fps=pc.clip_frame_rate,
                sync_fps=pc.sync_frame_rate,
                clip_size=pc.clip_image_size,
                sync_size=pc.sync_image_size,
            )
            self.torio_decode_workers = max(0, dataloader_workers)
            # Large video tensors cannot pass safely through a small /dev/shm.
            # Threads also avoid tensor serialization while overlapping the
            # next decode batch with GPU feature extraction.
            dataloader_workers = 0
            logger.info(
                "Torio preprocessing uses %d background decode threads on rank %d",
                self.torio_decode_workers,
                rank,
            )

        dataloader = DataLoader(
            rank_dataset,
            batch_size=config.preprocess_video_batch_size,
            num_workers=dataloader_workers,
            collate_fn=_identity_collate,
            shuffle=False,
            persistent_workers=dataloader_workers > 0,
        )
        self.add_component("training_dataloader", dataloader)
        self.add_component("forward_batch_builder", V2AForwardBatchBuilder(config.seed))
        self.add_component("feature_writer", feature_writer)

    def _iter_preprocessed_rows(self) -> Iterator[list[dict[str, Any]]]:
        preprocessor = self.torio_row_preprocessor
        if preprocessor is None:
            yield from self.training_dataloader
            return

        if self.torio_decode_workers <= 0:
            for rows in self.training_dataloader:
                yield [preprocessor(row) for row in rows]
            return

        raw_batches = iter(self.training_dataloader)
        queued_batches: deque[list[Future[dict[str, Any]]]] = deque()
        with ThreadPoolExecutor(
                max_workers=self.torio_decode_workers,
                thread_name_prefix="mmaudio-torio",
        ) as executor:

            def submit_next() -> bool:
                try:
                    rows = next(raw_batches)
                except StopIteration:
                    return False
                queued_batches.append([executor.submit(preprocessor, row) for row in rows])
                return True

            submit_next()
            submit_next()
            while queued_batches:
                futures = queued_batches.popleft()
                rows = [future.result() for future in futures]
                submit_next()
                yield rows

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
        skipped = self.initial_skipped
        failed = 0
        try:
            for rows in tqdm(
                    self._iter_preprocessed_rows(),
                    total=len(self.training_dataloader),
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
