# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.basic.ovis_image.ovis_image_pipeline import (OvisImagePipeline)
from fastvideo.training.training_pipeline import TrainingPipeline

logger = init_logger(__name__)


class OvisImageTrainingPipeline(TrainingPipeline):
    """Ovis-Image training: the shared TrainingPipeline loop plus an Ovis
    FlowMatchEuler scheduler and OvisImagePipeline validation."""

    _required_config_modules = [
        "scheduler",
        "transformer",
        "text_encoder",
        "tokenizer",
        "vae",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up the FlowMatchEuler scheduler with Ovis-Image defaults."""
        pipeline_cfg = fastvideo_args.pipeline_config
        flow_shift = getattr(pipeline_cfg, "flow_shift", 3.0)

        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            use_dynamic_shifting=False,
        )
        logger.info("OvisImageTrainingPipeline: scheduler initialised "
                    "(flow_shift=%s)", flow_shift)

    def initialize_validation_pipeline(self, training_args: TrainingArgs) -> None:
        """Build a validation OvisImagePipeline sharing the trained transformer."""
        logger.info("Initialising validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True

        self.validation_pipeline = OvisImagePipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True,
        )


def main(args: TrainingArgs) -> None:
    logger.info("Starting Ovis-Image training pipeline...")
    pipeline = OvisImageTrainingPipeline.from_pretrained(args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Ovis-Image training pipeline finished.")


if __name__ == "__main__":
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
