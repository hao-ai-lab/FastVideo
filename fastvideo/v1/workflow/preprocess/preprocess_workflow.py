import os

from datasets import load_dataset
from torch.utils.data import DataLoader

from fastvideo.v1.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.v1.workflow.workflow_base import WorkflowBase

class PreprocessWorkflow(WorkflowBase):
    def register_pipelines(self) -> None:
        self.add_pipeline_config(
            "preprocess_pipeline",
            ("preprocessing", self.fastvideo_args)
        )

    def register_components(self) -> None:
        training_dataset = load_dataset(self.fastvideo_args.preprocess_config.training_dataset_path, split="training")
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=self.fastvideo_args.preprocess_config.preprocess_video_batch_size,
            num_workers=self.fastvideo_args.preprocess_config.dataloader_num_workers,
        )

        validation_dataset = load_dataset(self.fastvideo_args.preprocess_config.validation_dataset_path, split="validation")
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.fastvideo_args.preprocess_config.batch_size,
            num_workers=self.fastvideo_args.preprocess_config.num_workers,
        )

        self.add_component(
            "training_dataloader",
            training_dataloader
        )
        self.add_component(
            "validation_dataloader",
            validation_dataloader
        )
    
    def prepare_system_environment(self) -> None:
        dataset_output_dir = self.fastvideo_args.preprocess_config.dataset_output_dir
        os.makedirs(dataset_output_dir, exist_ok=True)

        validation_dataset_output_dir = os.path.join(dataset_output_dir, "validation_dataset")
        os.makedirs(validation_dataset_output_dir, exist_ok=True)

        training_dataset_output_dir = os.path.join(dataset_output_dir, "training_dataset")
        os.makedirs(training_dataset_output_dir, exist_ok=True)


    @classmethod
    def get_workflow_cls(cls, fastvideo_args: FastVideoArgs) -> "PreprocessWorkflow":
        if fastvideo_args.workload_type == WorkloadType.T2V:
            from fastvideo.v1.workflow.preprocess.preprocess_workflow_t2v import PreprocessWorkflowT2V
            return PreprocessWorkflowT2V
        elif fastvideo_args.workload_type == WorkloadType.I2V:
            from fastvideo.v1.workflow.preprocess.preprocess_workflow_i2v import PreprocessWorkflowI2V
            return PreprocessWorkflowI2V
        else:
            raise ValueError(f"Workload type: {fastvideo_args.workload_type} is not supported in preprocessing workflow.")