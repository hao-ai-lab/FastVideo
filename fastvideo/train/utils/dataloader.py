# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        DataConfig, )


def build_parquet_t2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int | None,
    parquet_schema: Any,
    cfg_rate: float | None = None,
    max_presentation_tokens: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> Any:
    """Build a parquet dataloader for T2V-style datasets.

    ``cfg_rate`` overrides ``data_config.training_cfg_rate`` for models
    that perform CFG dropout themselves (e.g. LTX2Model swaps in the
    unconditional embedding in ``prepare_batch``); pass 0.0 to keep the
    collate's zeroing drop off so the two drops do not stack.
    """

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    packing = getattr(data_config, "cross_sample_sequence_packing", None)
    packing_enabled = bool(getattr(packing, "enabled", False))
    if packing_enabled and gradient_accumulation_steps is None:
        raise ValueError("Packed dataloading requires gradient_accumulation_steps for document-mean loss weighting")

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=(float(data_config.training_cfg_rate or 0.0) if cfg_rate is None else float(cfg_rate)),
        drop_last=True,
        text_padding_length=None if text_len is None else int(text_len),
        max_presentation_tokens=max_presentation_tokens,
        native_shape_bucketing=bool(getattr(data_config, "native_shape_bucketing", False)),
        require_existing_parquet_cache=getattr(data_config, "require_existing_parquet_cache", False),
        fixed_sequence_capacities=getattr(data_config, "fixed_sequence_capacities", None),
        sequence_packing_enabled=packing_enabled,
        sequence_packing_max_joint_tokens=getattr(packing, "max_joint_tokens", None),
        sequence_packing_max_samples_per_pack=getattr(packing, "max_samples_per_pack", None),
        gradient_accumulation_steps=1 if gradient_accumulation_steps is None else gradient_accumulation_steps,
        seed=int(data_config.seed or 0),
    ))
    return dataloader


def build_parquet_matrixgame2_train_dataloader(
    data_config: DataConfig,
    *,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for Matrix-Game 2.0 datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=float(data_config.training_cfg_rate or 0.0),
        drop_last=True,
        text_padding_length=512,
        require_existing_parquet_cache=getattr(data_config, "require_existing_parquet_cache", False),
        seed=int(data_config.seed or 0),
    ))
    return dataloader
