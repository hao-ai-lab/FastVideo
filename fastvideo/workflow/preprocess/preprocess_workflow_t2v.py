from typing import TYPE_CHECKING

from tqdm import tqdm

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.workflow.preprocess.components import ParquetDatasetSaver
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.workflow.preprocess.components import (
        VideoForwardBatchBuilder)


class PreprocessWorkflowT2V(PreprocessWorkflow):
    training_dataloader: "DataLoader"
    validation_dataloader: "DataLoader"
    preprocess_pipeline: "ComposedPipelineBase"
    processed_dataset_saver: "ParquetDatasetSaver"
    video_forward_batch_builder: "VideoForwardBatchBuilder"

    def register_components(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        super().register_components()
        self.add_component(
            "processed_dataset_saver",
            ParquetDatasetSaver(
                flush_frequency=self.fastvideo_args.preprocess_config.
                flush_frequency,
                samples_per_file=self.fastvideo_args.preprocess_config.
                samples_per_file,
                schema_fields=[f.name for f in pyarrow_schema_t2v],
            ))

    def run(self) -> None:
        # Training dataset preprocessing
        for batch in tqdm(self.training_dataloader,
                          desc="Preprocessing training dataset",
                          unit="batch"):
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(
                batch)

            forward_batch = self.preprocess_pipeline.forward(
                forward_batch, self.fastvideo_args)

            self.processed_dataset_saver.save_and_write_parquet_batch(
                forward_batch, self.training_dataset_output_dir)

        self.processed_dataset_saver.flush_tables(
            self.training_dataset_output_dir)
        self.processed_dataset_saver.clean_up()

        # Validation dataset preprocessing
        for batch in tqdm(self.validation_dataloader,
                          desc="Preprocessing validation dataset",
                          unit="batch"):
            forward_batch = self.video_forward_batch_builder(batch)

            forward_batch = self.preprocess_pipeline.forward(
                forward_batch, self.fastvideo_args)

            self.processed_dataset_saver.save_and_write_parquet_batch(
                forward_batch, self.validation_dataset_output_dir)
        self.processed_dataset_saver.flush_tables(
            self.validation_dataset_output_dir)
        self.processed_dataset_saver.clean_up()
