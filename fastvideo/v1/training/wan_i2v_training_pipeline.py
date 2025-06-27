# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any, Dict

import torch

from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset.dataloader.schema import (
    pyarrow_schema_i2v, pyarrow_schema_i2v_validation)
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.v1.pipelines.pipeline_batch_info import (ForwardBatch,
                                                        TrainingBatch)
from fastvideo.v1.pipelines.wan.wan_i2v_pipeline import (
    WanImageToVideoValidationPipeline)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
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

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_i2v
        self.validation_dataset_schema = pyarrow_schema_i2v_validation

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

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        assert self.train_dataloader is not None

        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        clip_features = batch['clip_feature']
        image_latents = batch['first_frame_latent']
        image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = batch['pil_image']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            get_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            get_torch_device(), dtype=torch.bfloat16)
        training_batch.preprocessed_image = pil_image.to(get_torch_device())
        training_batch.image_embeds = clip_features.to(get_torch_device())
        training_batch.image_latents = image_latents.to(get_torch_device())
        training_batch.infos = infos

        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""
        assert self.training_args is not None
        assert training_batch.latents is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert self.noise_random_generator is not None
        assert training_batch.image_latents is not None

        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(get_torch_device(),
                                                        dtype=torch.bfloat16)

        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, image_latents], dim=1)

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None
        assert training_batch.image_embeds is not None

        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_torch_device(), dtype=torch.bfloat16)
        encoder_hidden_states_image = image_embeds

        # NOTE: noisy_model_input already contains concatenated image_latents from _prepare_dit_inputs
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
            validation_batch: Dict[str, Any], num_inference_steps: int,
            negative_prompt_embeds: torch.Tensor | None,
            negative_prompt_attention_mask: torch.Tensor | None
    ) -> ForwardBatch:
        embeddings = validation_batch['text_embedding']
        masks = validation_batch['text_attention_mask']
        clip_features = validation_batch['clip_feature']
        pil_image = validation_batch['pil_image']
        infos = validation_batch['info_list']
        prompt = infos[0]['prompt']

        prompt_embeds = embeddings.to(get_torch_device())
        prompt_attention_mask = masks.to(get_torch_device())
        clip_features = clip_features.to(get_torch_device())

        # Calculate sizes
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1

        # Prepare batch for validation
        batch = ForwardBatch(
            prompt=prompt,
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
