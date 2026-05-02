# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        DataConfig, )


def _build_parquet_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for preprocessed parquet datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=data_config.training_cfg_rate,
        drop_last=True,
        text_padding_length=int(text_len),
        seed=int(data_config.seed or 0),
    ))
    return dataloader


def build_parquet_t2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for T2V-style datasets."""

    return _build_parquet_train_dataloader(
        data_config,
        text_len=text_len,
        parquet_schema=parquet_schema,
    )


def build_parquet_i2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for I2V/TI2V-style datasets."""

    return _build_parquet_train_dataloader(
        data_config,
        text_len=text_len,
        parquet_schema=parquet_schema,
    )


def build_parquet_matrixgame_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for MatrixGame datasets."""

    return _build_parquet_train_dataloader(
        data_config,
        text_len=text_len,
        parquet_schema=parquet_schema,
    )
