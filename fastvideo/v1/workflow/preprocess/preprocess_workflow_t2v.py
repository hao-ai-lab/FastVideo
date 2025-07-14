from tqdm import tqdm

from fastvideo.v1.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.v1.workflow.preprocess.preprocess_workflow import PreprocessWorkflow

class PreprocessWorkflowT2V(PreprocessWorkflow):
    def run(self):
        for batch in tqdm(self.training_dataloader,
                          desc="Preprocessing training dataset",
                          unit="batch"):
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(batch)

            if not self.raw_data_validator(forward_batch):
                continue

            forward_batch = self.preprocess_pipeline(forward_batch)



        
