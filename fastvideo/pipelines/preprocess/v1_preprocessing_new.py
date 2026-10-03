from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.training_schema import PreprocessRunConfig, load_resolved_run_config
from fastvideo.distributed import (maybe_init_distributed_environment_and_model_parallel)
from fastvideo.logger import init_logger
from fastvideo.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


def main(resolved_config: ResolvedGeneratorConfig) -> None:
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    preprocess_workflow_cls = WorkflowBase.get_workflow_cls(resolved_config)
    preprocess_workflow = preprocess_workflow_cls(resolved_config)
    preprocess_workflow.run()


if __name__ == "__main__":
    main(load_resolved_run_config(PreprocessRunConfig))
