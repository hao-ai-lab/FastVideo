import os
import warnings

from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.training_schema import PreprocessRunConfig, load_resolved_run_config
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.logger import init_logger
from fastvideo.pipelines.preprocess.preprocess_pipeline_i2v import (PreprocessPipeline_I2V)
from fastvideo.pipelines.preprocess.preprocess_pipeline_ode_trajectory import (PreprocessPipeline_ODE_Trajectory)
from fastvideo.pipelines.preprocess.preprocess_pipeline_t2v import (PreprocessPipeline_T2V)
from fastvideo.pipelines.preprocess.preprocess_pipeline_text import (PreprocessPipeline_Text)
from fastvideo.pipelines.preprocess.matrixgame2.matrixgame2_preprocess_pipeline import (PreprocessPipeline_MatrixGame2)
from fastvideo.pipelines.preprocess.matrixgame2.matrixgame2_preprocess_pipeline_ode_trajectory import (
    PreprocessPipeline_MatrixGame2_ODE_Trajectory)
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


def main(resolved_config: ResolvedGeneratorConfig) -> None:
    """Download the model and run the pipeline of ``preprocess.preprocess_task`` on one GPU.

    The local checkpoint directory becomes ``model_path`` and ``preprocess.model_path``, which the pipelines and the
    dataset tokenizer read.
    """
    model_path = maybe_download_model(resolved_config.model_path)
    resolved_config = resolved_config.with_override("v1_preprocess.main", {
        "model_path": model_path,
        "preprocess.model_path": model_path,
    })
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    num_gpus = int(os.environ["WORLD_SIZE"])
    assert num_gpus == 1, "Only support 1 GPU"

    preprocess_task = resolved_config.preprocess.preprocess_task
    if preprocess_task == "t2v":
        PreprocessPipeline = PreprocessPipeline_T2V
    elif preprocess_task == "i2v":
        PreprocessPipeline = PreprocessPipeline_I2V
    elif preprocess_task == "text_only":
        PreprocessPipeline = PreprocessPipeline_Text
    elif preprocess_task == "ode_trajectory":
        assert resolved_config.is_explicit("pipeline.flow_shift"), "pipeline.flow_shift is required for ode_trajectory"
        PreprocessPipeline = PreprocessPipeline_ODE_Trajectory
    elif preprocess_task in ("matrixgame2", "matrixgame"):
        if preprocess_task == "matrixgame":
            warnings.warn("preprocess.preprocess_task=matrixgame is deprecated; use matrixgame2",
                          DeprecationWarning,
                          stacklevel=2)
        PreprocessPipeline = PreprocessPipeline_MatrixGame2
    elif preprocess_task in ("matrixgame2_ode_trajectory", "matrixgame_ode_trajectory"):
        if preprocess_task == "matrixgame_ode_trajectory":
            warnings.warn(
                "preprocess.preprocess_task=matrixgame_ode_trajectory is deprecated; use matrixgame2_ode_trajectory",
                DeprecationWarning,
                stacklevel=2)
        PreprocessPipeline = PreprocessPipeline_MatrixGame2_ODE_Trajectory
    else:
        raise ValueError(f"Invalid preprocess task: {preprocess_task}. "
                         f"Valid options: t2v, i2v, ode_trajectory, text_only, matrixgame2, matrixgame2_ode_trajectory")

    logger.info("Preprocess task: %s using %s", preprocess_task, PreprocessPipeline.__name__)

    pipeline = PreprocessPipeline(model_path, resolved_config)
    pipeline.forward(batch=None, resolved_config=pipeline.resolved_config)


if __name__ == "__main__":
    main(load_resolved_run_config(PreprocessRunConfig))
