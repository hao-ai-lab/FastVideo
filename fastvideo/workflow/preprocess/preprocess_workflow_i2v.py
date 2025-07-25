from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow


class PreprocessWorkflowI2V(PreprocessWorkflow):

    def __init__(self, fastvideo_args: FastVideoArgs):
        super().__init__(fastvideo_args)

    def register_pipelines(self) -> None:
        pass

    def register_components(self) -> None:
        pass
