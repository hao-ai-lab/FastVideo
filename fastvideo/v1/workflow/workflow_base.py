from abc import ABC, abstractmethod

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines import ComposedPipelineBase, build_pipeline


class WorkflowBase(ABC):
    pipeline_configs: dict[str, FastVideoArgs] = {}
    pipelines: dict[str, ComposedPipelineBase] = {}

    def __init__(self, fastvideo_args: FastVideoArgs):
        self.fastvideo_args = fastvideo_args

    def register_pipelines(self, pipeline_configs: dict[str, FastVideoArgs]):
        self.pipeline_configs.update(pipeline_configs)

    def load_pipelines(self):
        for pipeline_name, pipeline_config in self.pipeline_configs.items():
            pipeline = build_pipeline(pipeline_config)
            self.pipelines[pipeline_name] = pipeline

    @abstractmethod
    def get_components(self):
        pass

    @abstractmethod
    def run(self):
        pass
