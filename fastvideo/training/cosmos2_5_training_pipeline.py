# SPDX-License-Identifier: Apache-2.0
"""Cosmos 2.5 training pipeline (text-to-world, full fine-tuning + LoRA).

Follows the same structure as wan_training_pipeline.py. Key Cosmos 2.5 specifics:
- Text embeddings are (B, seq, 100352) from Reason1 full-concat (28 layers × 3584 dim)
- forward() takes `padding_mask` (B, 1, H, W) and optional `fps` int
- Normalisation is handled by Cosmos25WanVAEWrapper (handles_latent_norm=True),
  so _normalize_dit_input is a no-op and stored latents are already normalised.
- Timesteps are sigma values in [0, 1] (flow matching); (B,) is auto-expanded
  inside the model to (B, 1).
"""
import torch

from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.schema import ExecutionMode
from fastvideo.api.training_schema import TrainingRunConfig, load_resolved_run_config
from fastvideo.distributed import get_local_torch_device
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.cosmos.cosmos2_5_pipeline import Cosmos2_5Pipeline
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline, resolve_validation_config

logger = init_logger(__name__)


class Cosmos25TrainingPipeline(TrainingPipeline):
    """Training pipeline for Cosmos 2.5 (text-to-world).

    Supports:
    - Full fine-tuning (all transformer parameters)
    - LoRA via the inherited LoRAPipeline mechanism
      (lora_param_names_mapping is set on Cosmos25Transformer3DModel)
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, resolved_config: ResolvedGeneratorConfig):
        """Create the flow-matching scheduler with Cosmos 2.5's shift=5.0."""
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=resolved_config.pipeline.flow_shift)

    def initialize_validation_pipeline(self, resolved_config: ResolvedGeneratorConfig):
        """Build a full Cosmos2_5Pipeline that reuses the training transformer."""
        logger.info("Initializing Cosmos 2.5 validation pipeline...")
        validation_config = resolve_validation_config(
            resolved_config,
            offload={
                "pin_cpu_memory": resolved_config.engine.offload.pin_cpu_memory,
                "dit": True
            },
        )

        validation_pipeline = Cosmos2_5Pipeline.from_pretrained(
            resolved_config.model_path,
            resolved_config=validation_config,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
        )
        self.validation_pipeline = validation_pipeline

    # ------------------------------------------------------------------
    # Cosmos 2.5-specific overrides
    # ------------------------------------------------------------------

    def _normalize_dit_input(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Skip: Cosmos25WanVAEWrapper normalises latents during encoding.

        Pre-computed latents stored in the dataset are already normalised
        (per-channel mean/std applied inside the VAE wrapper), so applying a
        second normalisation here would corrupt the inputs.
        """
        return training_batch

    def _build_input_kwargs(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Build the keyword arguments for Cosmos25Transformer3DModel.forward().

        Cosmos 2.5 forward() signature (T2W training subset):
            hidden_states : (B, 16, T, H, W)
            timestep      : (B,)  sigma values in [0, 1] — model expands to (B, 1)
            encoder_hidden_states : (B, seq, 100352)
            padding_mask  : (B, 1, H, W)  — all-ones for T2W (no spatial padding)
            fps           : int            — used only when rope_enable_fps_modulation=True
        """
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.sigmas is not None

        noisy = training_batch.noisy_model_input  # (B, 16, T, H, W)
        batch_size, _c, _t, height, width = noisy.shape
        device = get_local_torch_device()

        # sigmas stored as (B, 1, 1, 1, 1) after get_sigmas; flatten to (B,)
        sigma_vals = training_batch.sigmas.view(batch_size).to(dtype=torch.bfloat16, device=device)

        # T2W: no spatial padding, all-ones padding mask (16 latent + 1 padding).
        padding_mask = torch.ones(
            batch_size,
            1,
            height,
            width,
            device=device,
            dtype=torch.bfloat16,
        )

        # T2W: no inpainting, all-zeros condition mask (adds 1 channel → total 18).
        condition_mask = torch.zeros(
            batch_size,
            1,
            _t,
            height,
            width,
            device=device,
            dtype=torch.bfloat16,
        )

        training_batch.input_kwargs = {
            "hidden_states": noisy,
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            # (B,) sigma in [0, 1]; model auto-expands to (B, 1) at line 901
            "timestep": sigma_vals,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            # rope_enable_fps_modulation=False for the 2B checkpoint, so fps
            # has no effect on the embeddings, but the argument is still forwarded.
            "fps": 24,
        }
        return training_batch


# ---------------------------------------------------------------------------
# Entry point (mirrors wan_training_pipeline.py)
# ---------------------------------------------------------------------------


def main(resolved_config: ResolvedGeneratorConfig) -> None:
    logger.info("Starting Cosmos 2.5 training pipeline...")
    pipeline = Cosmos25TrainingPipeline.from_pretrained(resolved_config.model_path, resolved_config=resolved_config)
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    main(load_resolved_run_config(TrainingRunConfig, mode=ExecutionMode.FINETUNING))
