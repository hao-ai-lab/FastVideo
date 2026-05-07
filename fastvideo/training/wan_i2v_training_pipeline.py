# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any

import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.distributed import get_local_torch_device, get_sp_group
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.wan.wan_i2v_pipeline import (WanImageToVideoPipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.utils import is_vsa_available, shallow_asdict

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class WanI2VTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        self._validate_condition_latent_args(training_args)
        super().initialize_training_pipeline(training_args)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_i2v

    @staticmethod
    def _get_non_empty_batch_tensor(batch: dict[str, Any], key: str) -> torch.Tensor | None:
        # The I2V schema fills missing optional image-conditioning columns with
        # empty tensors. Treat them like absent fields so we can issue an
        # explicit error for required CLIP features and still ignore optional
        # legacy first_frame_latent / pil_image columns.
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return value
        return None

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.dit_cpu_offload = True
        # args_copy.pipeline_config.vae_config.load_encoder = False
        # validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
        self.validation_pipeline = WanImageToVideoPipeline.from_pretrained(training_args.model_path,
                                                                           args=None,
                                                                           inference_mode=True,
                                                                           loaded_modules={
                                                                               "transformer":
                                                                               self.get_module("transformer"),
                                                                           },
                                                                           tp_size=training_args.tp_size,
                                                                           sp_size=training_args.sp_size,
                                                                           num_gpus=training_args.num_gpus,
                                                                           dit_cpu_offload=True)

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
        # clip_feature is required for real Wan-Fun I2V training. It may be the
        # legacy first-frame tensor [B, 257, 1280] or endpoint features
        # [B, K, 257, 1280] for variable-context training.
        clip_features = self._get_non_empty_batch_tensor(batch, 'clip_feature')
        if clip_features is None:
            raise ValueError("Wan-Fun I2V training requires clip_feature in the parquet cache.")

        # first_frame_latent is optional only for opt-in latent-prefix
        # continuation because that path reconstructs Wan-Fun's 16-channel
        # clean context branch from vae_latent.
        image_latents = self._get_non_empty_batch_tensor(batch, 'first_frame_latent')
        if image_latents is None and not self._random_context_enabled(self.training_args):
            raise ValueError("Wan-Fun legacy I2V training requires first_frame_latent in the parquet cache.")
        if image_latents is not None:
            image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = self._get_non_empty_batch_tensor(batch, 'pil_image')
        infos = batch['info_list']

        training_batch.latents = latents.to(get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(get_local_torch_device(),
                                                                          dtype=torch.bfloat16)
        training_batch.preprocessed_image = (pil_image.to(get_local_torch_device()) if pil_image is not None else None)
        training_batch.image_embeds = (clip_features.to(get_local_torch_device())
                                       if clip_features is not None else None)
        training_batch.image_latents = (image_latents.to(get_local_torch_device())
                                        if image_latents is not None else None)
        training_batch.infos = infos

        return training_batch

    def _prepare_dit_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""

        # First, let the base pipeline sample noise/timesteps for the full
        # video latent tensor. The selected I2V conditioning path then prepares
        # Wan-Fun's mask and 16-channel clean-context branch before they are
        # appended below.
        training_batch = super()._prepare_dit_inputs(training_batch)
        if self._random_context_enabled(self.training_args):
            training_batch = self._apply_random_context_conditioning(training_batch)
        else:
            training_batch = self._apply_legacy_i2v_conditioning(training_batch)

        assert training_batch.mask_lat_size is not None
        assert training_batch.image_latents is not None
        assert training_batch.noisy_model_input is not None
        # Wan-Fun InP declares 36 input channels: 16 noisy video channels, then
        # 4 mask channels, then 16 clean-context channels.
        training_batch.noisy_model_input = torch.cat([
            training_batch.noisy_model_input,
            training_batch.mask_lat_size,
            training_batch.image_latents,
        ],
                                                     dim=1)

        return training_batch

    @staticmethod
    def _random_context_enabled(training_args: TrainingArgs) -> bool:
        return training_args.max_condition_latents >= 1

    @staticmethod
    def _validate_condition_latent_args(training_args: TrainingArgs) -> None:
        min_condition_latents = training_args.min_condition_latents
        max_condition_latents = training_args.max_condition_latents
        num_latent_t = training_args.num_latent_t

        if min_condition_latents == 0 and max_condition_latents == 0:
            return
        if min_condition_latents < 1:
            raise ValueError("min_condition_latents must be >= 1, or 0 with max_condition_latents=0")
        if max_condition_latents < min_condition_latents:
            raise ValueError("max_condition_latents must be >= min_condition_latents")
        # Keep at least one latent timestep unconditioned so every sample has a
        # generated suffix to supervise after masking the clean prefix.
        if max_condition_latents >= num_latent_t:
            raise ValueError("max_condition_latents must be < num_latent_t")

    def _sample_condition_latent_counts(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample one clean-prefix length per training sample."""
        min_condition_latents = self.training_args.min_condition_latents
        max_condition_latents = self.training_args.max_condition_latents
        return torch.randint(
            low=min_condition_latents,
            high=max_condition_latents + 1,
            size=(batch_size, ),
            generator=getattr(self, "noise_gen_cuda", None),
            device=device,
        )

    def _apply_legacy_i2v_conditioning(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare the pre-existing Wan-Fun first-frame conditioning tensors."""
        assert training_batch.image_latents is not None

        image_latents = training_batch.image_latents.to(get_local_torch_device(), dtype=torch.bfloat16)
        temporal_compression_ratio = 4
        num_frames = (self.training_args.num_latent_t - 1) * temporal_compression_ratio + 1
        batch_size, _, _, latent_height, latent_width = image_latents.shape

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=temporal_compression_ratio)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, temporal_compression_ratio, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(image_latents.device, dtype=torch.bfloat16)

        training_batch.image_latents = image_latents
        training_batch.mask_lat_size = mask_lat_size
        training_batch.condition_latent_counts = None
        training_batch.generation_loss_mask = None
        return training_batch

    def _apply_random_context_conditioning(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Convert first-frame I2V conditioning into clean-prefix continuation.

        This opt-in path intentionally overwrites the dataset's
        first_frame_latent, which is VAE([frame_0, zero, zero, ...]). The
        replacement is the normalized vae_latent prefix from the cached target
        video, zero-padded to the full latent length. Those tensors are not
        equivalent: Wan's temporal VAE sees different neighboring frames for
        first-frame I2V versus full-video prefix continuation. Use this path
        only when training and inference share the same clean-prefix latent
        contract.
        """
        assert training_batch.latents is not None
        assert training_batch.noisy_model_input is not None

        latents = training_batch.latents
        batch_size, _, latent_t, latent_height, latent_width = latents.shape
        counts = self._sample_condition_latent_counts(
            batch_size=batch_size,
            device=latents.device,
        )

        if self.training_args.sp_size > 1:
            # Sequence-parallel ranks hold different temporal shards but must
            # agree on the sampled prefix length for each batch element.
            get_sp_group().broadcast(counts, src=0)

        time_ids = torch.arange(latent_t, device=latents.device)
        # prefix_mask selects clean context latents; generation_mask is its
        # complement and is later used to train only the generated suffix.
        prefix_mask = time_ids.view(1, 1, latent_t, 1, 1) < counts.view(batch_size, 1, 1, 1, 1)
        generation_mask = ~prefix_mask

        # The model sees clean latents in the prefix and noisy latents in the
        # suffix, matching the autoregressive continuation task.
        training_batch.noisy_model_input = torch.where(
            prefix_mask.expand_as(training_batch.noisy_model_input),
            latents,
            training_batch.noisy_model_input,
        )

        # Rebuild the Wan-Fun image/context branch directly from normalized
        # vae_latent. This is separate from CLIP image embeddings, which feed
        # cross-attention rather than the DiT input projection.
        condition_latents = torch.where(
            prefix_mask,
            latents,
            torch.zeros_like(latents),
        ).to(dtype=torch.bfloat16)
        mask_lat_size = prefix_mask.expand(
            batch_size,
            4,
            latent_t,
            latent_height,
            latent_width,
        ).to(dtype=torch.bfloat16)

        training_batch.image_latents = condition_latents
        training_batch.mask_lat_size = mask_lat_size
        training_batch.condition_latent_counts = counts
        training_batch.generation_loss_mask = generation_mask
        return training_batch

    @staticmethod
    def _select_clip_features_for_context(
        image_embeds: torch.Tensor | None,
        condition_latent_counts: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_embeds is None or image_embeds.numel() == 0:
            raise ValueError("Wan-Fun random-context training requires non-empty CLIP image features.")

        if image_embeds.dim() == 3:
            # Legacy I2V cache: one first-frame CLIP sequence per sample.
            return image_embeds

        if image_embeds.dim() != 4:
            raise ValueError("clip_feature must have shape [B, 257, 1280] or [B, K, 257, 1280], "
                             f"got {tuple(image_embeds.shape)}.")

        batch_size, num_context_endpoints = image_embeds.shape[:2]
        if condition_latent_counts is None:
            # Offline pre-encoded parquet caches may store endpoint CLIP as
            # [B, K, 257, 1280] even when the trainer is run in explicit legacy
            # mode (min/max_condition_latents == 0/0). Legacy I2V has no
            # sampled context length, but semantically it conditions on the
            # first frame / first latent endpoint. Consume endpoint 0 here so
            # the same offline cache can serve both default random-context
            # training and optional legacy first-frame I2V compatibility.
            if num_context_endpoints < 1:
                raise ValueError("Endpoint CLIP cache must contain at least one context endpoint.")
            return image_embeds[:, 0]

        if condition_latent_counts.shape != (batch_size, ):
            raise ValueError("condition_latent_counts must have shape [B] when selecting endpoint CLIP features, "
                             f"got {tuple(condition_latent_counts.shape)} for batch size {batch_size}.")

        context_indices = condition_latent_counts.to(device=image_embeds.device, dtype=torch.long) - 1
        context_indices_cpu = context_indices.detach().cpu()
        min_context_index = int(context_indices_cpu.min().item())
        max_context_index = int(context_indices_cpu.max().item())
        if min_context_index < 0 or max_context_index >= num_context_endpoints:
            raise ValueError(
                "Sampled context length is outside available endpoint CLIP features: "
                f"counts={(context_indices_cpu + 1).tolist()}, num_context_endpoints={num_context_endpoints}.")

        batch_indices = torch.arange(batch_size, device=image_embeds.device)
        return image_embeds[batch_indices, context_indices]

    def _build_input_kwargs(self, training_batch: TrainingBatch) -> TrainingBatch:

        # The latent-prefix branch carries low-level temporal context. CLIP
        # carries the image cross-attention context. For endpoint caches
        # [B, K, 257, 1280], choose the CLIP sequence that matches the sampled
        # context endpoint; legacy [B, 257, 1280] caches keep first-frame CLIP.
        image_embeds = self._select_clip_features_for_context(
            training_batch.image_embeds,
            training_batch.condition_latent_counts,
        )
        assert torch.isnan(image_embeds).sum() == 0
        encoder_hidden_states_image = image_embeds.to(get_local_torch_device(), dtype=torch.bfloat16)

        # NOTE: noisy_model_input already contains concatenated image_latents from _prepare_dit_inputs
        training_batch.input_kwargs = {
            "hidden_states": training_batch.noisy_model_input,
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "timestep": training_batch.timesteps.to(get_local_torch_device(), dtype=torch.bfloat16),
            "encoder_attention_mask": training_batch.encoder_attention_mask,
            "encoder_hidden_states_image": encoder_hidden_states_image,
            "return_dict": False,
        }
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
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanI2VTrainingPipeline.from_pretrained(args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


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
