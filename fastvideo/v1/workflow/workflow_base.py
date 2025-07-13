from abc import ABC, abstractmethod
from typing import Any, Callable

from fastvideo.v1.configs.pipelines.base import PipelineConfig
from fastvideo.v1.fastvideo_args import ExecutionMode, FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase, build_pipeline

logger = init_logger(__name__)

class WorkflowBase(ABC):

    def __init__(self, fastvideo_args: FastVideoArgs):
        self.fastvideo_args = fastvideo_args

        # TODO: pipeline_config should be: dict[str, PipelineConfig]
        # pipeline_type should be included in the PipelineConfig
        # pipeline_config[pipeline_name] = (pipeline_type, fastvideo_args)
        self._pipeline_configs: dict[str, tuple[str, FastVideoArgs]] = {}
        self._pipelines: dict[str, ComposedPipelineBase] = {}
        self._components: dict[str, Any] = {}
        self.register_pipelines()
        self.register_components()

        self.prepare_system_environment()
        self.load_pipelines()

    def load_pipelines(self) -> None:
        for pipeline_name, pipeline_config in self._pipeline_configs.items():
            pipeline_type, fastvideo_args = pipeline_config
            pipeline = build_pipeline(fastvideo_args, pipeline_type)
            self._pipelines[pipeline_name] = pipeline

    def add_component(self, component_name: str, component: Any) -> None:
        self._components[component_name] = component
        setattr(self, component_name, component)

    def add_pipeline_config(self, pipeline_name: str, pipeline: ComposedPipelineBase) -> None:
        self._pipelines[pipeline_name] = pipeline
        setattr(self, pipeline_name, pipeline)

    @abstractmethod
    def register_components(self) -> None:
        pass

    @abstractmethod
    def register_pipelines(self) -> None:
        pass

    @abstractmethod
    def prepare_system_environment(self) -> None:
        pass

    @abstractmethod
    def get_components(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def run(self):
        pass

    @classmethod
    def get_workflow_cls(cls, fastvideo_args: FastVideoArgs) -> "WorkflowBase":
        if fastvideo_args.mode == ExecutionMode.PREPROCESS:
            from fastvideo.v1.workflow.preprocess.preprocess_workflow import PreprocessWorkflow
            return PreprocessWorkflow.get_workflow_cls(fastvideo_args)
            
