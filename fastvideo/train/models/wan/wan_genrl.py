# SPDX-License-Identifier: Apache-2.0
"""Wan model extended for GenRL (RL training with text prompts).

Overrides ``init_preprocessors`` to load a T5 text encoder
and tokenizer instead of the standard parquet video
dataloader.  Provides a trivial dummy dataloader so the
trainer's outer loop has something to iterate over — the
real prompt dataloaders are created by the GenRLMethod.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.logger import init_logger
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.moduleloader import (
    load_module_from_path, )

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )

logger = init_logger(__name__)


def _is_lora_target(
    module_name: str,
    target_modules: list[str],
) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def _apply_fastvideo_lora(
    transformer: Any,
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules: list[str],
    init_weights: str,
) -> int:
    from fastvideo.layers.lora.linear import (
        get_lora_layer,
        replace_submodule,
    )

    transformer.requires_grad_(False)
    converted_count = 0
    for name, layer in list(transformer.named_modules()):
        if not _is_lora_target(name, target_modules):
            continue
        lora_layer = get_lora_layer(
            layer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_mode=True,
        )
        if lora_layer is None:
            continue
        _init_lora_weights(lora_layer, init_weights, lora_rank)
        replace_submodule(transformer, name, lora_layer)
        converted_count += 1
    return converted_count


def _init_lora_weights(
    lora_layer: Any,
    init_weights: str,
    lora_rank: int,
) -> None:
    """Match PEFT's useful LoRA initialization modes."""
    init = init_weights.lower()
    if init == "default":
        return

    lora_A = getattr(lora_layer, "lora_A", None)
    lora_B = getattr(lora_layer, "lora_B", None)
    if lora_A is None or lora_B is None:
        return

    if init == "gaussian":
        torch.nn.init.normal_(lora_A, std=1 / max(1, lora_rank))
        torch.nn.init.zeros_(lora_B)
        return

    raise ValueError("Unsupported GenRLWanModel LoRA init_weights="
                     f"{init_weights!r}. Use 'gaussian' or 'default'.")


@contextmanager
def _disable_lora_adapters(transformer: Any):
    """Temporarily run a LoRA-wrapped transformer as its frozen base model."""
    lora_layers = [module for module in transformer.modules() if hasattr(module, "disable_lora")]
    previous = [bool(module.disable_lora) for module in lora_layers]
    try:
        for module in lora_layers:
            module.disable_lora = True
        yield
    finally:
        for module, was_disabled in zip(lora_layers, previous, strict=True):
            module.disable_lora = was_disabled


def _attach_disable_adapter(transformer: Any) -> None:
    """Expose a PEFT-compatible disable_adapter context manager."""

    def disable_adapter(self):
        return _disable_lora_adapters(self)

    transformer.disable_adapter = MethodType(  # type: ignore[attr-defined]
        disable_adapter,
        transformer,
    )


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
        use_lora: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_target_modules: list[str] | None = None,
        lora_path: str | None = None,
        lora_init_weights: str = "gaussian",
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
            disable_custom_init_weights=(disable_custom_init_weights),
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=(enable_gradient_checkpointing_type),
            transformer_override_safetensor=(transformer_override_safetensor),
        )
        if use_lora:
            if lora_target_modules is None:
                raise ValueError("GenRLWanModel use_lora=True requires "
                                 "lora_target_modules.")
            if lora_path:
                raise ValueError("GenRLWanModel lora_path is not supported for "
                                 "FastVideo LoRA training yet.")
            converted_count = _apply_fastvideo_lora(
                self.transformer,
                lora_rank=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=lora_target_modules,
                init_weights=lora_init_weights,
            )
            if converted_count == 0:
                raise ValueError("GenRLWanModel use_lora=True did not match any "
                                 f"FastVideo linear layers: {lora_target_modules}")
            logger.info(
                "Converted %d GenRL Wan transformer layers to LoRA",
                converted_count,
            )
            _attach_disable_adapter(self.transformer)
        self.text_encoder: Any = None
        self.tokenizer: Any = None

    def disable_adapter(self):
        """PEFT-compatible context manager for reference KL with LoRA."""
        return _disable_lora_adapters(self.transformer)

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
        self._load_text_encoder(model_path, training_config)

        # Dummy dataloader for the trainer's outer loop.
        self.dataloader = _InfiniteDummyLoader()
        self.start_step = 0

    def _load_text_encoder(
        self,
        model_path: str,
        training_config: TrainingConfig,
    ) -> None:
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        logger.info("Loading text encoder from %s", model_path)
        self.text_encoder = load_module_from_path(
            model_path=model_path,
            module_type="text_encoder",
            training_config=training_config,
        )
        self.text_encoder.requires_grad_(False)

    def on_train_start(self) -> None:
        """Skip negative conditioning (handled by method)."""
