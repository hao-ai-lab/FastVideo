# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline)
from fastvideo.training.self_forcing_distillation_pipeline import (
    SelfForcingDistillationPipeline)
from fastvideo.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanSelfForcingDistillationPipeline(SelfForcingDistillationPipeline):
    """
    A self-forcing distillation pipeline for Wan that uses the self-forcing methodology
    with DMD for video generation.
    """
    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
    ]

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        validation_pipeline = WanCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "transformer_2": self.get_module("transformer_2")
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    logger.info("Starting Wan self-forcing distillation pipeline...")

    pipeline = WanSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    pipeline.train()
    logger.info("Wan self-forcing distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
