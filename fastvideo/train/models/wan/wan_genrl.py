# SPDX-License-Identifier: Apache-2.0
"""Wan model extended for GenRL (RL training with text prompts).

Overrides ``init_preprocessors`` to load a T5 text encoder
and tokenizer instead of the standard parquet video
dataloader.  Provides a trivial dummy dataloader so the
trainer's outer loop has something to iterate over — the
real prompt dataloaders are created by the GenRLMethod.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.logger import init_logger
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.moduleloader import (
    load_module_from_path,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig,
    )

logger = init_logger(__name__)


class _InfiniteDummyLoader:
    """Trivial iterable that yields empty dicts forever."""

    def __iter__(self):
        while True:
            yield {}


class GenRLWanModel(WanModel):
    """Wan model with text encoder for RL training.

    Compared to the base :class:`WanModel`, this variant:

    * Loads the T5 text encoder and tokenizer from the
      pretrained model path.
    * Sets a dummy dataloader so the trainer can iterate
      without blocking.
    * Does **not** build the standard parquet video
      dataloader.
    """

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 3.0,
        enable_gradient_checkpointing_type: str
        | None = None,
        transformer_override_safetensor: str
        | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=(
                disable_custom_init_weights
            ),
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=(
                enable_gradient_checkpointing_type
            ),
            transformer_override_safetensor=(
                transformer_override_safetensor
            ),
        )
        self.text_encoder: Any = None
        self.tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(
        self,
        training_config: TrainingConfig,
    ) -> None:  # type: ignore[override]
        """Load VAE, text encoder, and tokenizer."""
        # Load VAE.
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()

        # Load text encoder and tokenizer.
        model_path = str(training_config.model_path)
        self._load_text_encoder(
            model_path, training_config
        )

        # Dummy dataloader for the trainer's outer loop.
        self.dataloader = _InfiniteDummyLoader()
        self.start_step = 0

    def _load_text_encoder(
        self,
        model_path: str,
        training_config: TrainingConfig,
    ) -> None:
        from transformers import AutoTokenizer

        logger.info(
            "Loading tokenizer from %s", model_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )

        logger.info(
            "Loading text encoder from %s", model_path
        )
        self.text_encoder = load_module_from_path(
            model_path=model_path,
            module_type="text_encoder",
            training_config=training_config,
        )
        self.text_encoder.requires_grad_(False)

    def on_train_start(self) -> None:
        """Skip negative conditioning (handled by method)."""
