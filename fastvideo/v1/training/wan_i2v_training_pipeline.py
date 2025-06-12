import sys

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.training.wan_training_pipeline import WanTrainingPipeline
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.training.training_utils import shard_latents_across_sp

logger = init_logger(__name__)

# Manual gradient checking flag - set to True to enable gradient verification
ENABLE_GRADIENT_CHECK = False


class WanI2VTrainingPipeline(WanTrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def _prepare_extra_train_inputs(self, batch, input_kwargs):
        _, _, _, _, extra_latents = batch

        extra_kwargs = {}
        encoder_hidden_states_image = extra_latents["encoder_hidden_states_image"]
        image_latents = extra_latents["image_latents"]
        # Image Embeds
        assert torch.isnan(encoder_hidden_states_image).sum() == 0
        encoder_hidden_states_image = encoder_hidden_states_image.to(get_torch_device(),
                                        dtype=torch.bfloat16)
        extra_kwargs["encoder_hidden_states_image"] = encoder_hidden_states_image

        # Image Latents
        assert torch.isnan(image_latents).sum() == 0
        image_latents = image_latents.to(get_torch_device(),
                                          dtype=torch.bfloat16)
        image_latents = shard_latents_across_sp(image_latents, self.training_args.num_latent_t)

        noisy_model_input = torch.cat(
            [input_kwargs["hidden_states"], image_latents],
            dim=1)
        extra_kwargs["hidden_states"] = noisy_model_input

        input_kwargs.update(extra_kwargs)
        return input_kwargs
    
    def _prepare_extra_validation_inputs(self, batch, input_kwargs):
        _, _, _, info, extra_latents = batch

        logger.info("info: %s", info)
        logger.info("extra_latents: %s", extra_latents)
        encoder_hidden_states_image = extra_latents["encoder_hidden_states_image"]
        image_latents = extra_latents["image_latents"]
        logger.info("image_latents: %s", image_latents.shape)

        image_latents = image_latents.to(get_torch_device())
        image_latents = shard_latents_across_sp(image_latents, self.training_args.num_latent_t)

        extra_kwargs = {
            "image_embeds": [encoder_hidden_states_image.to(get_torch_device())],
            "image_latent": image_latents,
        }

        input_kwargs.update(extra_kwargs)
        return input_kwargs

def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanI2VTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.forward(None, args)
    logger.info("Training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.v1.fastvideo_args import TrainingArgs
    from fastvideo.v1.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.use_cpu_offload = False
    main(args)