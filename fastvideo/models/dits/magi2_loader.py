# SPDX-License-Identifier: Apache-2.0
"""Production checkpoint loaders for the MAGI-2 preview and refiner DiTs."""

from __future__ import annotations

import gc

import torch

from fastvideo.configs.models.dits.magi2 import Magi2PreviewVideoConfig, Magi2RefinerVideoConfig
from fastvideo.models.dits.magi2 import Magi2PreviewDiT
from fastvideo.models.dits.magi2_checkpointing import load_magi2_model_state_dict, load_safetensors_dir
from fastvideo.models.dits.magi2_refiner import Magi2RefinerDiT


def load_magi2_preview_model(
    checkpoint_dir: str,
    config: Magi2PreviewVideoConfig,
    device: torch.device | str,
) -> Magi2PreviewDiT:
    """Build the EP-local preview model and strictly load its official state."""
    model = Magi2PreviewDiT(config=config)
    state_dict = load_magi2_model_state_dict(model, checkpoint_dir)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict MAGI-2 preview loading reported incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    gc.collect()
    model = model.to(device=device).eval()
    torch.cuda.empty_cache()
    return model


def load_magi2_refiner_model(
    checkpoint_dir: str,
    config: Magi2RefinerVideoConfig,
    device: torch.device | str,
) -> Magi2RefinerDiT:
    """Build a replicated refiner and strictly load every official tensor."""
    model = Magi2RefinerDiT(config=config)
    state_dict = load_safetensors_dir(checkpoint_dir, desc="Loading MAGI-2 refiner shards")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict MAGI-2 refiner loading reported incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    gc.collect()
    model = model.to(device=device).eval()
    torch.cuda.empty_cache()
    return model
