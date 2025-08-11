import os
from typing import cast

from datasets import load_dataset
from torch.utils.data import DataLoader

from fastvideo.configs.configs import PreprocessConfig
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.workflow.preprocess.components import (
    PreprocessingDataValidator, VideoForwardBatchBuilder)
from fastvideo.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


class PreprocessWorkflow(WorkflowBase):

    def register_pipelines(self) -> None:
        self.add_pipeline_config("preprocess_pipeline",
                                 (PipelineType.PREPROCESS, self.fastvideo_args))

    def register_components(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        preprocess_config: PreprocessConfig = self.fastvideo_args.preprocess_config

        # raw data validator
        raw_data_validator = PreprocessingDataValidator(
            max_height=preprocess_config.max_height,
            max_width=preprocess_config.max_width,
            num_frames=preprocess_config.num_frames,
            train_fps=preprocess_config.train_fps,
            speed_factor=preprocess_config.speed_factor,
            video_length_tolerance_range=preprocess_config.
            video_length_tolerance_range,
            drop_short_ratio=preprocess_config.drop_short_ratio,
        )
        self.add_component("raw_data_validator", raw_data_validator)

        # training dataset
        training_dataset = load_dataset(preprocess_config.dataset_path,
                                        split="train")
        training_dataset = training_dataset.filter(raw_data_validator)
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
            validation_dataset = load_dataset(preprocess_config.dataset_path,
                                              split="validation")
            validation_dataset = validation_dataset.filter(raw_data_validator)
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=preprocess_config.preprocess_video_batch_size,
                num_workers=preprocess_config.dataloader_num_workers,
                collate_fn=lambda x: x,
            )
        except ValueError:
            logger.warning(
                "Validation dataset not found, skipping validation dataset preprocessing."
            )
            validation_dataloader = None

        self.add_component("validation_dataloader", validation_dataloader)

        # forward batch builder
        video_forward_batch_builder = VideoForwardBatchBuilder()
        self.add_component("video_forward_batch_builder",
                           video_forward_batch_builder)

    def prepare_system_environment(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        dataset_output_dir = self.fastvideo_args.preprocess_config.dataset_output_dir
        os.makedirs(dataset_output_dir, exist_ok=True)

        validation_dataset_output_dir = os.path.join(dataset_output_dir,
                                                     "validation_dataset")
        os.makedirs(validation_dataset_output_dir, exist_ok=True)
        self.validation_dataset_output_dir = validation_dataset_output_dir

        training_dataset_output_dir = os.path.join(dataset_output_dir,
                                                   "training_dataset")
        os.makedirs(training_dataset_output_dir, exist_ok=True)
        self.training_dataset_output_dir = training_dataset_output_dir

    @classmethod
    def get_workflow_cls(cls,
                         fastvideo_args: FastVideoArgs) -> "PreprocessWorkflow":
        if fastvideo_args.workload_type == WorkloadType.T2V:
            from fastvideo.workflow.preprocess.preprocess_workflow_t2v import (
                PreprocessWorkflowT2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowT2V)
        elif fastvideo_args.workload_type == WorkloadType.I2V:
            from fastvideo.workflow.preprocess.preprocess_workflow_i2v import (
                PreprocessWorkflowI2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowI2V)
        else:
            raise ValueError(
                f"Workload type: {fastvideo_args.workload_type} is not supported in preprocessing workflow."
            )
