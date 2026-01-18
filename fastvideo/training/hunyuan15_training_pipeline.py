# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.training.training_pipeline import TrainingPipeline

from fastvideo.pipelines.basic.hunyuan15.hunyuan15_pipeline import HunyuanVideo15Pipeline

logger = init_logger(__name__)


class Hunyuan15TrainingPipeline(TrainingPipeline):
    """
    A training pipeline for HunyuanVideo-1.5.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        if training_args.enable_gradient_checkpointing_type is None:
            training_args.enable_gradient_checkpointing_type = "full"
        super().initialize_training_pipeline(training_args)

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        # reserved for future refactors
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline (Hunyuan15)...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True

        validation_pipeline = HunyuanVideo15Pipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                # reuse the training transformer weights for validation sampling
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True,
            # dit_layerwise_offload=True,
            use_fsdp_inference=True,
        )

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    logger.info("Starting Hunyuan15 training pipeline...")

    pipeline = Hunyuan15TrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        args=args,
    )
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()

    args.dit_cpu_offload = False
    main(args)
