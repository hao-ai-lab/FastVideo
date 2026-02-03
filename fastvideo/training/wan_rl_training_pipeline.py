# SPDX-License-Identifier: Apache-2.0
import sys

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler)
from fastvideo.training.rl.rl_pipeline import RLPipeline
from fastvideo.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanRLTrainingPipeline(RLPipeline):
    """
    A training pipeline for Wan with RL/GRPO support.
    
    This pipeline extends RLPipeline with Wan-specific initialization.
    """
    _required_config_modules = [
        "scheduler", "transformer", "vae", "text_encoder", "tokenizer"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # self.modules["scheduler"] = UniPCMultistepScheduler.from_pretrained(
        #     fastvideo_args.model_path, subfolder="scheduler"
        # )
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        self.validation_pipeline = self._create_inference_pipeline(
            training_args, dit_cpu_offload=True)

def main(args) -> None:
    logger.info("Starting RL training pipeline...")

    pipeline = WanRLTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("RL training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    # Enable RL mode
    args.rl_mode = True
    main(args)
