from fastvideo.distributed import (maybe_init_distributed_environment_and_model_parallel)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


def main(resolved_config: FastVideoArgs) -> None:
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    preprocess_workflow_cls = WorkflowBase.get_workflow_cls(resolved_config)
    preprocess_workflow = preprocess_workflow_cls(resolved_config)
    preprocess_workflow.run()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    resolved_config = FastVideoArgs.from_cli_args(args)
    main(resolved_config)
