import os
from typing import cast

from torch.utils.data import DataLoader

from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.schema import WorkloadType
from fastvideo.dataset.dataloader.record_schema import (basic_t2v_record_creator, i2v_record_creator)
from fastvideo.dataset.dataloader.schema import (pyarrow_schema_i2v, pyarrow_schema_t2v)
from fastvideo.distributed.parallel_state import get_world_rank
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.workflow.preprocess.components import (ParquetDatasetSaver, PreprocessingDataValidator,
                                                      VideoForwardBatchBuilder, build_dataset)
from fastvideo.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


class PreprocessWorkflow(WorkflowBase):

    def register_pipelines(self) -> None:
        self.add_pipeline_config("preprocess_pipeline", (PipelineType.PREPROCESS, self.resolved_config))

    def register_components(self) -> None:
        """Register the data validator, dataloaders, batch builder, and parquet saver that ``preprocess``
        configures."""
        preprocess_config = self.resolved_config.preprocess

        # raw data validator
        raw_data_validator = PreprocessingDataValidator(
            max_height=preprocess_config.max_height,
            max_width=preprocess_config.max_width,
            num_frames=preprocess_config.num_frames,
            train_fps=preprocess_config.train_fps,
            speed_factor=preprocess_config.speed_factor,
            video_length_tolerance_range=preprocess_config.video_length_tolerance_range,
            drop_short_ratio=preprocess_config.drop_short_ratio,
        )
        self.add_component("raw_data_validator", raw_data_validator)

        # training dataset
        try:
            training_dataset = build_dataset(preprocess_config, split="train", validator=raw_data_validator)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Training dataset not found, please use the matching download script under examples/datasets/ to download the dataset first. Error: {e}"
            ) from e

        # we do not use collate_fn here because we use iterable-style Dataset
        # and want to keep the original type of the dataset
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=preprocess_config.preprocess_video_batch_size,
            num_workers=preprocess_config.dataloader_num_workers,
            collate_fn=lambda x: x,
        )
        self.add_component("training_dataloader", training_dataloader)

        # try to load validation dataset if it exists
        try:
            validation_dataset = build_dataset(preprocess_config, split="validation", validator=raw_data_validator)
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=preprocess_config.preprocess_video_batch_size,
                num_workers=preprocess_config.dataloader_num_workers,
                collate_fn=lambda x: x,
            )
        except ValueError:
            logger.warning("Validation dataset not found, skipping validation dataset preprocessing.")
            validation_dataloader = None

        self.add_component("validation_dataloader", validation_dataloader)

        # forward batch builder
        video_forward_batch_builder = VideoForwardBatchBuilder(seed=preprocess_config.seed)
        self.add_component("video_forward_batch_builder", video_forward_batch_builder)

        # record creator
        if self.resolved_config.pipeline.workload_type == WorkloadType.I2V:
            record_creator = i2v_record_creator
            schema = pyarrow_schema_i2v
        else:
            record_creator = basic_t2v_record_creator
            schema = pyarrow_schema_t2v
        processed_dataset_saver = ParquetDatasetSaver(
            flush_frequency=preprocess_config.flush_frequency,
            samples_per_file=preprocess_config.samples_per_file,
            schema=schema,
            record_creator=record_creator,
        )
        self.add_component("processed_dataset_saver", processed_dataset_saver)

    def prepare_system_environment(self) -> None:
        dataset_output_dir = self.resolved_config.preprocess.dataset_output_dir
        os.makedirs(dataset_output_dir, exist_ok=True)

        validation_dataset_output_dir = os.path.join(dataset_output_dir, "validation_dataset",
                                                     f"worker_{get_world_rank()}")
        os.makedirs(validation_dataset_output_dir, exist_ok=True)
        self.validation_dataset_output_dir = validation_dataset_output_dir

        training_dataset_output_dir = os.path.join(dataset_output_dir, "training_dataset", f"worker_{get_world_rank()}")
        os.makedirs(training_dataset_output_dir, exist_ok=True)
        self.training_dataset_output_dir = training_dataset_output_dir

    @classmethod
    def get_workflow_cls(cls, resolved_config: ResolvedGeneratorConfig) -> "PreprocessWorkflow":
        """The workflow class of ``pipeline.workload_type``; LTX-2 text-to-video has its own."""
        workload_type = resolved_config.pipeline.workload_type
        is_ltx2_t2v = (workload_type == WorkloadType.T2V
                       and resolved_config.pipeline_config.__class__.__name__ == "LTX2T2VConfig")
        if is_ltx2_t2v:
            from fastvideo.workflow.preprocess.preprocess_workflow_ltx2_t2v import (PreprocessWorkflowLTX2T2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowLTX2T2V)
        if workload_type == WorkloadType.T2V:
            from fastvideo.workflow.preprocess.preprocess_workflow_t2v import (PreprocessWorkflowT2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowT2V)
        elif workload_type == WorkloadType.I2V:
            from fastvideo.workflow.preprocess.preprocess_workflow_i2v import (PreprocessWorkflowI2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowI2V)
        else:
            raise ValueError(f"Workload type: {workload_type} is not supported in preprocessing workflow.")
