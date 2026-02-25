# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def run_distillation_from_config(
    config_path: str,
    *,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    override_output_dir: str | None = None,
) -> None:
    """YAML-only distillation entrypoint (schema v2)."""

    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distillation import DistillTrainer
    from fastvideo.distillation.utils.checkpoint import (
        DistillCheckpointConfig,
        DistillCheckpointManager,
    )
    from fastvideo.distillation.dispatch import build_runtime_from_config
    from fastvideo.distillation.utils.config import load_distill_run_config

    cfg = load_distill_run_config(config_path)
    training_args = cfg.training_args

    if resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = str(resume_from_checkpoint)
    if override_output_dir is not None:
        training_args.output_dir = str(override_output_dir)

    maybe_init_distributed_environment_and_model_parallel(
        training_args.tp_size,
        training_args.sp_size,
    )

    runtime = build_runtime_from_config(cfg)

    if dry_run:
        logger.info("Dry-run: config parsed and runtime built successfully.")
        return

    # Attach the exact YAML used for this run to the tracker (e.g., W&B Files).
    # This helps reproducibility and makes runs easy to inspect later.
    runtime.tracker.log_file(os.path.abspath(os.path.expanduser(config_path)), name="run.yaml")

    ckpt_config = DistillCheckpointConfig(
        save_steps=int(getattr(training_args, "training_state_checkpointing_steps", 0) or 0),
        keep_last=int(getattr(training_args, "checkpoints_total_limit", 0) or 0),
    )

    get_rng_generators = getattr(runtime.method, "get_rng_generators", None)
    if not callable(get_rng_generators):
        adapter = getattr(runtime.method, "adapter", None)
        get_rng_generators = getattr(adapter, "get_rng_generators", None)
        if not callable(get_rng_generators):
            get_rng_generators = None

    checkpoint_manager = DistillCheckpointManager(
        bundle=runtime.method.bundle,
        dataloader=runtime.dataloader,
        output_dir=training_args.output_dir,
        config=ckpt_config,
        get_rng_generators=get_rng_generators,
    )

    trainer = DistillTrainer(training_args, tracker=runtime.tracker)
    trainer.run(
        runtime.method,
        dataloader=runtime.dataloader,
        max_steps=training_args.max_train_steps,
        start_step=runtime.start_step,
        checkpoint_manager=checkpoint_manager,
    )


def main(args: Any) -> None:
    config_path = str(args.config)
    dry_run = bool(args.dry_run)
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    override_output_dir = getattr(args, "override_output_dir", None)
    logger.info("Starting distillation from config=%s", config_path)
    run_distillation_from_config(
        config_path,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint,
        override_output_dir=override_output_dir,
    )
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
        help="Path to distillation YAML config (schema v2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and build runtime, but do not start training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory (checkpoint-<step>), its 'dcp/' subdir, "
            "or an output_dir containing checkpoints (auto-picks latest)."
        ),
    )
    parser.add_argument(
        "--override-output-dir",
        type=str,
        default=None,
        help="Override training.output_dir from YAML (useful for repeated runs).",
    )
    args = parser.parse_args(argv[1:])
    main(args)
