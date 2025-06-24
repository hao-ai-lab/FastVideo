# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

from fastvideo.v1.fastvideo_args import FastVideoArgs, DistillationArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.v1.pipelines.wan.wan_pipeline import WanValidationPipeline
from fastvideo.v1.training.distillation_pipeline import DistillationPipeline
from fastvideo.v1.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanDistillationPipeline(DistillationPipeline):
    """
    A distillation pipeline for Wan that uses a single transformer model.
    The main transformer serves as the student model, and copies are made for teacher and critic.
    """
    _required_config_modules = ["scheduler", "transformer"]  # Only need scheduler and transformer

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize Wan-specific scheduler."""
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: DistillationArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: DistillationArgs):
        """Initialize Wan validation pipeline."""
        logger.info("Initializing Wan validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.pipeline_config.vae_config.load_encoder = False
        validation_pipeline = WanValidationPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus)

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    logger.info("Starting Wan distillation pipeline...")

    # Create pipeline with original args
    pipeline = WanDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    
    # Convert args to DistillationArgs for training
    distillation_args = DistillationArgs.from_cli_args(args)
    
    # Initialize the distillation pipeline
    pipeline.initialize_training_pipeline(distillation_args)
    pipeline.initialize_validation_pipeline(distillation_args)
    
    # Start training
    pipeline.train()
    logger.info("Wan distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.v1.fastvideo_args import DistillationArgs, TrainingArgs
    from fastvideo.v1.utils import FlexibleArgumentParser
    from fastvideo.v1.fastvideo_args import FastVideoArgs
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = DistillationArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    args.use_cpu_offload = False
    main(args) 