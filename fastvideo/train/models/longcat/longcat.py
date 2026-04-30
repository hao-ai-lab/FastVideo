# SPDX-License-Identifier: Apache-2.0
"""LongCat model plugin (per-role instance)."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.wan.wan import WanModel

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


class LongCatModel(WanModel):
    """LongCat per-role model for training and distillation."""

    _transformer_cls_name: str = "LongCatTransformer3DModel"

    @staticmethod
    def _validate_flow_shift(flow_shift: float | None) -> float:
        if flow_shift is None:
            return 12.0

        validated = float(flow_shift)
        if validated == 0.0:
            raise ValueError(
                "LongCat training does not support flow_shift=0.0 because "
                "it collapses FlowMatch training timesteps. Use 12.0 to "
                "match the released LongCat scheduler config."
            )
        return validated

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 12.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            flow_shift=self._validate_flow_shift(flow_shift),
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
        )

    def _init_timestep_mechanics(self) -> None:
        assert self.training_config is not None
        tc = self.training_config
        flow_shift = getattr(tc.pipeline_config, "flow_shift", None)  # type: ignore[union-attr]
        self.timestep_shift = self._validate_flow_shift(flow_shift)
        self.num_train_timestep = int(self.noise_scheduler.num_train_timesteps)
        self.min_timestep = 0
        self.max_timestep = self.num_train_timestep

    def _build_attention_metadata(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.attn_metadata = None
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for LongCat distillation")

        batch_size = int(noise_input.shape[0])
        if timestep.ndim == 0:
            timestep = timestep.view(1).expand(batch_size)
        elif timestep.ndim == 1 and int(timestep.shape[0]) == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        return {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep,
        }

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        """Adapt LongCat's sign convention to FineTuneMethod's target.

        ``LongCatTransformer3DModel`` is pretrained to output the
        ``clean - noise`` direction; ``LongCatDenoisingStage`` (the
        bidirectional inference pipeline) explicitly negates the
        transformer output before handing it to
        ``FlowMatchEulerDiscreteScheduler.step``. Training methods on
        the other hand (``FineTuneMethod``,
        ``DiffusionForcingSFTMethod``) target ``noise - clean``
        directly (the standard rectified-flow velocity Wan uses).

        Without the negation here, the loss MSE pushes the transformer
        toward ``noise - clean``, flipping its native output sign over
        training. Inference then applies its own negation on top, so
        the scheduler receives the wrong direction and produces noise
        even while the training loss is dropping. Verified empirically
        on a 100-step LongCat overfit run: step 0 generated meaningful
        video, step 100 was pure noise despite low loss.

        Negating in ``predict_noise`` keeps the transformer's
        pretrained sign convention intact while presenting the
        training methods with a Wan-compatible
        ``pred ≈ noise - clean`` for MSE.
        """
        pred = super().predict_noise(
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        return -pred
