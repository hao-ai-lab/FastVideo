# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)

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
    # NOTE: do not use `FlexibleArgumentParser` here.
    # It treats `--config` specially (loads and inlines CLI args from YAML),
    # which conflicts with Phase 2 distillation where `--config` points to the
    # distillation run YAML itself.
    parser = argparse.ArgumentParser()
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
