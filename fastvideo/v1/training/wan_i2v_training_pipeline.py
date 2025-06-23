# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

import torch

from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.v1.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.v1.pipelines.wan.wan_i2v_pipeline import (
    WanImageToVideoValidationPipeline)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import shard_latents_across_sp
from fastvideo.v1.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanI2VTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
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
        args_copy.pipeline_config.vae_config.load_encoder = False
        validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus)

        self.validation_pipeline = validation_pipeline

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None
        assert training_batch.extra_latents is not None

        extra_latents = training_batch.extra_latents
        if extra_latents:
            image_embeds, image_latents = extra_latents[
                "clip_feature"], extra_latents["first_frame_latent"]
            # Image Embeds
            assert torch.isnan(image_embeds).sum() == 0
            image_embeds = image_embeds.to(get_torch_device(),
                                           dtype=torch.bfloat16)
            encoder_hidden_states_image = image_embeds

            # Image Latents
            assert torch.isnan(image_latents).sum() == 0
            image_latents = image_latents.to(get_torch_device(),
                                             dtype=torch.bfloat16)
            image_latents = shard_latents_across_sp(
                image_latents, num_latent_t=self.training_args.num_latent_t)

            training_batch.noisy_model_input = torch.cat(
                [training_batch.noisy_model_input, image_latents], dim=1)

        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "timestep":
            training_batch.timesteps.to(get_torch_device(),
                                        dtype=torch.bfloat16),
            "encoder_attention_mask":
            training_batch.encoder_attention_mask,
            "encoder_hidden_states_image":
            encoder_hidden_states_image,
            "return_dict":
            False,
        }
        return training_batch

    def _prepare_validation_inputs(
            self, sampling_param: SamplingParam, training_args: TrainingArgs,
            validation_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    List[str], Dict[str, Any],
                                    List[Dict[str,
                                              Any]]], num_inference_steps: int,
            negative_prompt_embeds: torch.Tensor | None,
            negative_prompt_attention_mask: torch.Tensor | None
    ) -> ForwardBatch:
        latents, embeddings, masks, caption_text, extra_latents, infos = validation_batch

        prompt_embeds = embeddings.to(get_torch_device())
        prompt_attention_mask = masks.to(get_torch_device())
        clip_features = extra_latents.get("clip_feature")
        first_frame_latent = extra_latents.get("first_frame_latent")
        pil_image = extra_latents.get("pil_image")

        if clip_features is not None and clip_features.numel() > 0:
            clip_features = clip_features.to(get_torch_device())
        if first_frame_latent is not None and first_frame_latent.numel() > 0:
            first_frame_latent = first_frame_latent.to(get_torch_device())
        if pil_image is not None and pil_image[0] is not None and pil_image[
                0].numel() > 0:
            pil_image = pil_image[0].to(get_torch_device())
        else:
            clip_features = None
            first_frame_latent = None
            pil_image = None

        # Calculate sizes
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1

        # Prepare batch for validation
        batch = ForwardBatch(
            data_type="video",
            latents=None,
            seed=self.seed,  # Use deterministic seed
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            prompt_embeds=[prompt_embeds],
            prompt_attention_mask=[prompt_attention_mask],
            negative_prompt_embeds=[negative_prompt_embeds],
            negative_attention_mask=[negative_prompt_attention_mask],
            image_embeds=[clip_features],
            preprocessed_image=pil_image,
            height=training_args.num_height,
            width=training_args.num_width,
            num_frames=num_frames,
            num_inference_steps=
            num_inference_steps,  # Use the current validation step
            guidance_scale=sampling_param.guidance_scale,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )
        return batch


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanI2VTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
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
