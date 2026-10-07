# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any

import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.basic.wan.wan_i2v_dmd_pipeline import (WanImageToVideoDmdPipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.utils import is_vsa_available, shallow_asdict

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class WanI2VDistillationPipeline(DistillationPipeline):
    """
    A distillation pipeline for Wan that uses a single transformer model.
    The main transformer serves as the student model, and copies are made for teacher and critic.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize Wan-specific scheduler."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_i2v

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.dit_cpu_offload = False
        # args_copy.pipeline_config.vae_config.load_encoder = False
        # validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
        validation_pipeline = WanImageToVideoDmdPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
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

        training_batch.latents = latents.to(get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(get_local_torch_device(),
                                                                          dtype=torch.bfloat16)
        training_batch.preprocessed_image = pil_image.to(get_local_torch_device())
        training_batch.image_embeds = clip_features.to(get_local_torch_device())
        training_batch.image_latents = image_latents.to(get_local_torch_device())
        training_batch.infos = infos

        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam, training_args: TrainingArgs,
                                  validation_batch: dict[str, Any], num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch['video_path']
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1, sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t - 1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch

    def _prepare_dit_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""
        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(get_local_torch_device(), dtype=torch.bfloat16)
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        training_batch.image_latents = self._build_image_conditioning(
            image_latents,
            num_latent_t=self.training_args.num_latent_t,
            temporal_compression_ratio=vae_config.temporal_compression_ratio,
        )

        return training_batch

    @staticmethod
    def _build_image_conditioning(
        image_latents: torch.Tensor,
        *,
        num_latent_t: int,
        temporal_compression_ratio: int = 4,
    ) -> torch.Tensor:
        """Build full-length Wan I2V mask and latent conditioning.

        WanTransformer3DModel applies sequence-parallel sharding after patch
        embedding. Pre-sharding only the image conditioning here would make its
        temporal length differ from the full noise input before concatenation.
        """
        batch_size, _, latent_t, latent_height, latent_width = image_latents.shape
        if latent_t != num_latent_t:
            raise ValueError("Wan I2V image conditioning temporal length must match num_latent_t: "
                             f"got image_latents.shape[2]={latent_t}, num_latent_t={num_latent_t}")

        # The Wan I2V conditioning mask is one on the first latent frame and zero
        # elsewhere across the temporal-compression channels; the repeat/view/
        # transpose form used by the inference path reduces to this.
        mask_lat_size = torch.zeros(
            batch_size,
            temporal_compression_ratio,
            num_latent_t,
            latent_height,
            latent_width,
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        mask_lat_size[:, :, 0] = 1

        return torch.cat([mask_lat_size, image_latents], dim=1)

    def _build_distill_input_kwargs(self, noise_input: torch.Tensor, timestep: torch.Tensor,
                                    text_dict: dict[str, torch.Tensor], training_batch: TrainingBatch) -> TrainingBatch:
        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(), dtype=torch.bfloat16)

        image_conditioning = training_batch.image_latents  # bs, c, t, h, w
        noise_latents = noise_input.permute(0, 2, 1, 3, 4)  # bs, c, t, h, w
        matching_batch_and_time = (noise_latents.shape[0] == image_conditioning.shape[0]
                                   and noise_latents.shape[2] == image_conditioning.shape[2])
        matching_spatial_shape = noise_latents.shape[3:] == image_conditioning.shape[3:]
        if not matching_batch_and_time or not matching_spatial_shape:
            raise ValueError("Wan I2V noise and image conditioning must have matching batch, temporal, and spatial "
                             f"dimensions, got noise={tuple(noise_latents.shape)} and "
                             f"conditioning={tuple(image_conditioning.shape)}")
        # Concatenate in the transformer's (bs, c, t, h, w) layout so the patch
        # embedding receives a contiguous tensor.
        hidden_states = torch.cat([noise_latents, image_conditioning], dim=1)

        training_batch.input_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep,
            "encoder_hidden_states_image": image_embeds,
            "return_dict": False,
        }
        training_batch.noise_latents = noise_input

        return training_batch


def main(args) -> None:
    logger.info("Starting Wan distillation pipeline...")

    # Create pipeline with original args
    pipeline = WanI2VDistillationPipeline.from_pretrained(args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args

    # Start training
    pipeline.train()
    logger.info("Wan distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
