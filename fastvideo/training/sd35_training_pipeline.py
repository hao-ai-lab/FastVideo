# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler

from fastvideo.dataset.dataloader.schema import pyarrow_schema_sd35
from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.sd35.sd35_pipeline import SD35Pipeline
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    get_sigmas, normalize_dit_input)

logger = init_logger(__name__)

# Combined sequence length: 77 CLIP tokens + up to 154 T5 tokens + slack
SD35_TEXT_SEQ_LEN = 256

class SD35TrainingPipeline(TrainingPipeline):
    """Training pipeline for Stable Diffusion 3.5 text-to-image."""

    _required_config_modules = ["scheduler", "transformer", "vae"]

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_sd35

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        flow_shift = getattr(fastvideo_args.pipeline_config, "flow_shift",
                             None) or 1.0
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=flow_shift)

    def initialize_training_pipeline(
            self, training_args: TrainingArgs) -> None:
        original_text_len = (training_args.pipeline_config.text_encoder_configs[
            0].arch_config.text_len)
        training_args.pipeline_config.text_encoder_configs[
            0].arch_config.text_len = SD35_TEXT_SEQ_LEN
        super().initialize_training_pipeline(training_args)
        training_args.pipeline_config.text_encoder_configs[
            0].arch_config.text_len = original_text_len

    def initialize_validation_pipeline(
            self, training_args: TrainingArgs) -> None:
        logger.info("Initializing SD3.5 validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        validation_pipeline = SD35Pipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True,
        )
        self.validation_pipeline = validation_pipeline

    def _log_validation(self, transformer, training_args,
                        global_step) -> None:
        mp = getattr(training_args, "mixed_precision", None)
        original_precision = training_args.pipeline_config.dit_precision
        if mp and mp != "no":
            training_args.pipeline_config.dit_precision = mp
        try:
            super()._log_validation(transformer, training_args, global_step)
        finally:
            training_args.pipeline_config.dit_precision = original_precision

    def _get_next_batch(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        with self.tracker.timed("timing/get_next_batch"):
            batch = next(self.train_loader_iter, None)  # type: ignore
            if batch is None:
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                self.train_loader_iter = iter(self.train_dataloader)
                batch = next(self.train_loader_iter)

            device = get_local_torch_device()

            # SD3.5 latents are stored as (C, T, H, W) with T=1
            latents = batch["vae_latent"].to(device, dtype=torch.bfloat16)

            # Combined CLIP+T5 sequence embeddings
            encoder_hidden_states = batch["text_embedding"].to(
                device, dtype=torch.bfloat16)

            # Concatenated CLIP pooled projections
            pooled_projections = batch["pooled_projection"].to(
                device, dtype=torch.bfloat16)

            training_batch.latents = latents
            training_batch.encoder_hidden_states = encoder_hidden_states
            training_batch.pooled_projections = pooled_projections
            training_batch.encoder_attention_mask = None
            training_batch.infos = batch["info_list"]

        return training_batch

    def _normalize_dit_input(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        with self.tracker.timed("timing/normalize_input"):
            training_batch.latents = normalize_dit_input(
                "sd3",
                training_batch.latents,
                self.get_module("vae"),
            )
        return training_batch

    def _prepare_dit_inputs(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        with self.tracker.timed("timing/prepare_dit_inputs"):
            latents = training_batch.latents
            assert latents is not None

            # latents are (B, C, T, H, W) with T=1 — squeeze to 4D for the
            # SD3 transformer which expects (B, C, H, W).
            if latents.ndim == 5:
                latents = latents.squeeze(2)
                training_batch.latents = latents

            batch_size = latents.shape[0]
            noise = torch.randn(
                latents.shape,
                generator=self.noise_gen_cuda,
                device=latents.device,
                dtype=latents.dtype,
            )
            timesteps = self._sample_timesteps(batch_size, latents.device)
            sigmas = get_sigmas(
                self.noise_scheduler,
                latents.device,
                timesteps,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

            training_batch.noisy_model_input = noisy_model_input
            training_batch.timesteps = timesteps
            training_batch.sigmas = sigmas
            training_batch.noise = noise
            training_batch.raw_latent_shape = latents.shape

        return training_batch

    def _build_input_kwargs(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.timesteps is not None
        assert training_batch.pooled_projections is not None

        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "pooled_projections":
            training_batch.pooled_projections,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "return_dict":
            False,
        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        assert training_batch.input_kwargs is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.latents is not None
        assert training_batch.noise is not None
        assert training_batch.sigmas is not None

        with self.tracker.timed("timing/forward_backward"), \
                set_forward_context(current_timestep=training_batch.current_timestep,
                                    attn_metadata=None):
            model_pred = self.transformer(**training_batch.input_kwargs)[0]

            if self.training_args.precondition_outputs:
                model_pred = (training_batch.noisy_model_input -
                              model_pred * training_batch.sigmas)
                target = training_batch.latents
            else:
                target = training_batch.noise - training_batch.latents

            assert model_pred.shape == target.shape, (
                f"model_pred.shape={model_pred.shape}, "
                f"target.shape={target.shape}")

            loss = (torch.mean((model_pred.float() - target.float())**2) /
                    self.training_args.gradient_accumulation_steps)
            loss.backward()
            avg_loss = loss.detach().clone()

        with self.tracker.timed("timing/reduce_loss"):
            world_group = get_world_group()
            avg_loss = world_group.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        training_batch.total_loss += avg_loss.item()
        return training_batch

    def train_one_step(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)
            training_batch = self._normalize_dit_input(training_batch)
            training_batch = self._prepare_dit_inputs(training_batch)
            training_batch = self._build_attention_metadata(training_batch)
            training_batch = self._build_input_kwargs(training_batch)
            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        training_batch = self._clip_grad_norm(training_batch)

        with self.tracker.timed("timing/optimizer_step"):
            self.optimizer.step()
            self.lr_scheduler.step()

        return training_batch


def main(args) -> None:
    logger.info("Starting SD3.5 training pipeline...")
    pipeline = SD35TrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("SD3.5 training pipeline done")


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
