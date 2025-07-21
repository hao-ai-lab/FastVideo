# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

import torch
from fastvideo.v1.distributed import get_local_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.v1.pipelines.wan.wan_dmd_pipeline import WanDmdPipeline
from fastvideo.v1.training.distillation_pipeline import DistillationPipeline
from fastvideo.v1.pipelines.pipeline_batch_info import (ForwardBatch,
                                                        TrainingBatch)

from fastvideo.v1.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanDistillationPipeline(DistillationPipeline):
    """
    A distillation pipeline for Wan that uses a single transformer model.
    The main transformer serves as the student model, and copies are made for teacher and critic.
    """
    _required_config_modules = ["scheduler", "transformer", "vae", "teacher_transformer", "critic_transformer"]
    
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize Wan-specific scheduler."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.use_cpu_offload = True
        args_copy.pipeline_config.vae_config.load_encoder = False
        validation_pipeline = WanDmdPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus)

        self.validation_pipeline = validation_pipeline

    def _build_input_kwargs(self, noise_input: torch.Tensor, timestep: torch.Tensor, text_dict: dict[str, torch.Tensor],
                            training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.input_kwargs = {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep[0][:1],
            "return_dict":
            False,
        }
        training_batch.noise_latents = noise_input
        return training_batch

def main(args) -> None:
    logger.info("Starting Wan distillation pipeline...")

    # Create pipeline with original args
    pipeline = WanDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    
    args = pipeline.training_args
    # Start training
    pipeline.train()
    logger.info("Wan distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.v1.fastvideo_args import TrainingArgs
    from fastvideo.v1.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)   
    args = parser.parse_args()
    main(args) 