# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)

_PipelineFactory = Callable[[Any], Any]
_MethodBuilder = Callable[[Any], Any]


def _build_pipeline_factories() -> dict[str, _PipelineFactory]:
    from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline

    return {
        "wan":
        lambda args: WanDistillationPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            args=args,
        ),
    }


def _build_method_builders() -> dict[tuple[str, str], _MethodBuilder]:
    from fastvideo.distillation.builder import build_wan_dmd2_method

    return {
        ("wan", "dmd2"): build_wan_dmd2_method,
    }


def run_distillation(
    args: Any,
    *,
    distill_model: str,
    distill_method: str,
) -> None:
    """Legacy Phase 1 entrypoint (CLI-driven, uses legacy pipelines)."""

    from fastvideo.distillation import DistillTrainer

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


def run_distillation_from_config(config_path: str, *, dry_run: bool = False) -> None:
    """Phase 2 entrypoint (YAML-only, standalone runtime builder)."""

    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distillation import DistillTrainer
    from fastvideo.distillation.builder import build_wan_dmd2_runtime_from_config
    from fastvideo.distillation.yaml_config import load_distill_run_config

    cfg = load_distill_run_config(config_path)
    training_args = cfg.training_args

    maybe_init_distributed_environment_and_model_parallel(
        training_args.tp_size,
        training_args.sp_size,
    )

    if cfg.distill.model == "wan" and cfg.distill.method == "dmd2":
        runtime = build_wan_dmd2_runtime_from_config(cfg)
    else:
        raise ValueError(
            f"Unsupported distillation config: model={cfg.distill.model!r}, "
            f"method={cfg.distill.method!r}"
        )

    if dry_run:
        logger.info("Dry-run: config parsed and runtime built successfully.")
        return

    trainer = DistillTrainer(training_args, tracker=runtime.tracker)
    trainer.run(
        runtime.method,
        dataloader=runtime.dataloader,
        max_steps=training_args.max_train_steps,
        start_step=runtime.start_step,
    )


def main(args: Any) -> None:
    config_path = str(args.config)
    dry_run = bool(args.dry_run)
    logger.info("Starting Phase 2 distillation from config=%s", config_path)
    run_distillation_from_config(config_path, dry_run=dry_run)
    logger.info("Distillation completed")


if __name__ == "__main__":
    argv = sys.argv
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to distillation YAML config (Phase 2 entrypoint).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and build runtime, but do not start training.",
    )
    args = parser.parse_args(argv[1:])
    main(args)
