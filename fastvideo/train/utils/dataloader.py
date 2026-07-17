# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        DataConfig,
    )


def build_parquet_t2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for T2V-style datasets."""

    dataloader_type = str(getattr(data_config, "dataloader_type", "map")).strip().lower()
    if dataloader_type == "streaming":
        from fastvideo.dataset.parquet_dataset_streaming_style import (
            build_parquet_streaming_style_dataloader,
        )

        _dataset, dataloader = build_parquet_streaming_style_dataloader(
            data_config.data_path,
            data_config.train_batch_size,
            num_data_workers=data_config.dataloader_num_workers,
            parquet_schema=parquet_schema,
            manifest_path=str(data_config.streaming_manifest_path),
            cfg_rate=data_config.training_cfg_rate,
            drop_last=True,
            text_padding_length=int(text_len),
            seed=int(data_config.seed or 0),
            read_batch_size=int(data_config.streaming_read_batch_size),
            shuffle_row_groups=bool(data_config.streaming_shuffle_row_groups),
        )
        return dataloader
    if dataloader_type != "map":
        raise ValueError(f"Unsupported T2V dataloader_type: {dataloader_type!r}; expected 'map' or 'streaming'")

    from fastvideo.dataset import build_parquet_map_style_dataloader

    _dataset, dataloader = build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=data_config.training_cfg_rate,
        drop_last=True,
        text_padding_length=int(text_len),
        seed=int(data_config.seed or 0),
    )
    return dataloader


def build_parquet_matrixgame2_train_dataloader(
    data_config: DataConfig,
    *,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for Matrix-Game 2.0 datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader,
    )

    _dataset, dataloader = build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=float(data_config.training_cfg_rate or 0.0),
        drop_last=True,
        text_padding_length=512,
        seed=int(data_config.seed or 0),
    )
    return dataloader
