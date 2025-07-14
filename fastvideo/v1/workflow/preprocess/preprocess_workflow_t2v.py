from tqdm import tqdm

from fastvideo.v1.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.v1.workflow.preprocess.components import ParquetDatasetSaver
from fastvideo.v1.workflow.preprocess.preprocess_workflow import PreprocessWorkflow
from fastvideo.v1.dataset.dataloader.schema import pyarrow_schema_t2v

class PreprocessWorkflowT2V(PreprocessWorkflow):
    def register_components(self) -> None:
        super().register_components()
        self.add_component(
            "processed_dataset_saver",
            ParquetDatasetSaver(
                flush_frequency=self.fastvideo_args.preprocess_config.flush_frequency,
                samples_per_file=self.fastvideo_args.preprocess_config.samples_per_file,
                schema_fields=[f.name for f in pyarrow_schema_t2v],
            )
        )

    def run(self) -> None:
        for batch in tqdm(self.training_dataloader,
                          desc="Preprocessing training dataset",
                          unit="batch"):
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(batch)

            forward_batch = self.preprocess_pipeline.forward(forward_batch, self.fastvideo_args)

            self.processed_dataset_saver.save_and_write_parquet_batch(
                forward_batch,
                self.training_dataset_output_dir
            )

        for batch in tqdm(self.validation_dataloader,
                          desc="Preprocessing validation dataset",
                          unit="batch"):
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(batch)

            forward_batch = self.preprocess_pipeline.forward(forward_batch, self.fastvideo_args)

            self.processed_dataset_saver.save_and_write_parquet_batch(
                forward_batch,
                self.validation_dataset_output_dir
            )
