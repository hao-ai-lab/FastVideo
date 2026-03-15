# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from fastvideo.logger import init_logger

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        DataConfig, )

logger = init_logger(__name__)


def _maybe_apply_bot_died_filter(
    *,
    data_config: "DataConfig",
    dataset: Any,
) -> None:
    if not bool(getattr(data_config, "apply_bot_died_filter", False)):
        return

    from fastvideo.dataset.parquet_dataset_map_style import (
        build_bot_died_excluded_indices,
    )
    from fastvideo.distributed import (
        get_world_group,
        get_world_rank,
    )

    world_group = get_world_group()
    world_rank = int(get_world_rank())
    total = int(sum(dataset.lengths))
    bot_died_excluded: set[int] | None = None
    if world_rank == 0:
        bot_died_excluded = build_bot_died_excluded_indices(
            data_path=str(data_config.data_path),
            parquet_files=list(dataset.parquet_files),
            lengths=list(dataset.lengths),
        )

    bot_died_excluded = world_group.broadcast_object(bot_died_excluded, src=0)

    excluded_count = int(len(bot_died_excluded or set()))
    remaining_count = int(total - excluded_count)

    if bot_died_excluded:
        valid_indices = [i for i in range(total) if i not in bot_died_excluded]
        dataset.sampler.set_candidate_indices(valid_indices, epoch=0)

    local_samples = int(len(getattr(dataset.sampler, "sp_group_local_indices", [])))
    local_batches = int(len(dataset.sampler))

    if world_rank == 0:
        logger.info(
            "BOT_DIED_FILTER_SUMMARY enabled=true total=%d excluded=%d remaining=%d",
            total,
            excluded_count,
            remaining_count,
        )
    logger.info(
        "BOT_DIED_FILTER_LOCAL rank=%d samples=%d batches=%d",
        world_rank,
        local_samples,
        local_batches,
    )


def build_parquet_t2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for T2V-style datasets."""

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
    _maybe_apply_bot_died_filter(
        data_config=data_config,
        dataset=_dataset,
    )
    return dataloader


def build_parquet_wangame_train_dataloader(
    data_config: DataConfig,
    *,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for WanGame datasets."""

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
        seed=int(data_config.seed or 0),
    ))
    _maybe_apply_bot_died_filter(
        data_config=data_config,
        dataset=_dataset,
    )
    return dataloader
