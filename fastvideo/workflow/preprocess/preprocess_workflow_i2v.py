from typing import TYPE_CHECKING

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.workflow.preprocess.components import ParquetDatasetSaver
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.workflow.preprocess.components import (
        VideoForwardBatchBuilder)


class PreprocessWorkflowI2V(PreprocessWorkflow):
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
                schema_fields=[f.name for f in pyarrow_schema_i2v],
            ))

    def register_pipelines(self) -> None:
        pass
