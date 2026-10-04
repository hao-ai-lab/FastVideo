# SPDX-License-Identifier: Apache-2.0
from torchvision import transforms
from torchvision.transforms import Lambda

from fastvideo.dataset.parquet_dataset_map_style import (build_parquet_map_style_dataloader)
from fastvideo.dataset.ltx2_precomputed_dataset import (build_ltx2_precomputed_dataloader, LTX2PrecomputedDataset)
from fastvideo.dataset.preprocessing_datasets import VideoCaptionMergedDataset, TextDataset
from fastvideo.dataset.transform import (CenterCropResizeVideo, Normalize255, TemporalRandomCrop)
from fastvideo.dataset.validation_dataset import ValidationDataset


def getdataset(preprocess_config, model_path: str) -> VideoCaptionMergedDataset:
    """Build the video-caption dataset of ``preprocess_config``, the ``preprocess`` section of a resolved
    ``PreprocessRunConfig``. ``model_path`` is the local checkpoint directory that holds the tokenizer."""
    temporal_sample = (TemporalRandomCrop(preprocess_config.num_frames)
                       if preprocess_config.do_temporal_sample else None)  # 16 x
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    resize_topcrop = [
        CenterCropResizeVideo((preprocess_config.max_height, preprocess_config.max_width), top_crop=True),
    ]
    resize = [
        CenterCropResizeVideo((preprocess_config.max_height, preprocess_config.max_width)),
    ]
    transform = transforms.Compose([
        # Normalize255(),
        *resize,
    ])
    transform_topcrop = transforms.Compose([
        Normalize255(),
        *resize_topcrop,
        norm_fun,
    ])
    return VideoCaptionMergedDataset(data_merge_path=preprocess_config.data_merge_path,
                                     preprocess_config=preprocess_config,
                                     model_path=model_path,
                                     transform=transform,
                                     temporal_sample=temporal_sample,
                                     transform_topcrop=transform_topcrop,
                                     seed=preprocess_config.seed)


def gettextdataset(preprocess_config, model_path: str) -> TextDataset:
    """Build the text dataset of ``preprocess_config``; ``model_path`` is the local checkpoint directory that holds
    the tokenizer."""
    return TextDataset(data_merge_path=preprocess_config.data_merge_path,
                       preprocess_config=preprocess_config,
                       model_path=model_path,
                       seed=preprocess_config.seed)


__all__ = [
    "build_parquet_map_style_dataloader",
    "build_ltx2_precomputed_dataloader",
    "LTX2PrecomputedDataset",
    "ValidationDataset",
    "VideoCaptionMergedDataset",
    "TextDataset",
]
