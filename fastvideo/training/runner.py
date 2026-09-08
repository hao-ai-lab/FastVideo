# SPDX-License-Identifier: Apache-2.0
import importlib

from fastvideo.logger import init_logger
from fastvideo.utils import build_parser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info("Starting training pipeline %s...", args.pipeline_class)

    module = importlib.import_module(args.pipeline_module)
    pipeline_class = getattr(module, args.pipeline_class)

    pipeline = pipeline_class.from_pretrained(args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    parser = build_parser()

    args = parser.parse_args()
    args.dit_cpu_offload = False

    main(args)
