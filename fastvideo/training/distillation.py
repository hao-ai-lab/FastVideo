# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any, Callable

from fastvideo.distillation import DistillTrainer
from fastvideo.distillation.builder import build_wan_dmd2_method
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)

_PipelineFactory = Callable[[Any], Any]
_MethodBuilder = Callable[[Any], Any]


def _build_pipeline_factories() -> dict[str, _PipelineFactory]:
    return {
        "wan":
        lambda args: WanDistillationPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            args=args,
        ),
    }


def _build_method_builders() -> dict[tuple[str, str], _MethodBuilder]:
    return {
        ("wan", "dmd2"): build_wan_dmd2_method,
    }


def run_distillation(
    args: Any,
    *,
    distill_model: str,
    distill_method: str,
) -> None:
    pipeline_factories = _build_pipeline_factories()
    method_builders = _build_method_builders()

    if distill_model not in pipeline_factories:
        raise ValueError(
            f"Unknown distill_model={distill_model!r}. Supported: {sorted(pipeline_factories)}"
        )

    builder_key = (distill_model, distill_method)
    if builder_key not in method_builders:
        supported = sorted({m for (model, m) in method_builders if model == distill_model})
        raise ValueError(
            f"Unknown distill_method={distill_method!r} for distill_model={distill_model!r}. "
            f"Supported methods for {distill_model}: {supported}"
        )

    pipeline = pipeline_factories[distill_model](args)
    training_args = pipeline.training_args

    method = method_builders[builder_key](pipeline)

    trainer = DistillTrainer(training_args, tracker=pipeline.tracker)
    trainer.run(
        method,
        dataloader=pipeline.train_dataloader,
        max_steps=training_args.max_train_steps,
        start_step=pipeline.init_steps,
    )


def main(args: Any) -> None:
    distill_model = str(getattr(args, "distill_model"))
    distill_method = str(getattr(args, "distill_method"))
    logger.info(
        "Starting distillation: distill_model=%s, distill_method=%s",
        distill_model,
        distill_method,
    )
    run_distillation(args, distill_model=distill_model, distill_method=distill_method)
    logger.info("Distillation completed")


if __name__ == "__main__":
    argv = sys.argv
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--distill-model",
        type=str,
        default="wan",
        help="Distillation model family (Phase 1 supports: wan).",
    )
    parser.add_argument(
        "--distill-method",
        type=str,
        default="dmd2",
        help="Distillation method (Phase 1 supports: dmd2).",
    )
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args(argv[1:])
    main(args)
