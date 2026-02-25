# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def build_parquet_t2v_train_dataloader(
    training_args: Any,
    *,
    parquet_schema: Any,
) -> Any:
    """Build a parquet map-style dataloader for T2V-style latent datasets."""

    from fastvideo.dataset import build_parquet_map_style_dataloader

    text_len = training_args.pipeline_config.text_encoder_configs[0].arch_config.text_len  # type: ignore[attr-defined]
    _dataset, dataloader = build_parquet_map_style_dataloader(
        training_args.data_path,
        training_args.train_batch_size,
        num_data_workers=training_args.dataloader_num_workers,
        parquet_schema=parquet_schema,
        cfg_rate=training_args.training_cfg_rate,
        drop_last=True,
        text_padding_length=int(text_len),
        seed=int(training_args.seed or 0),
    )
    return dataloader

