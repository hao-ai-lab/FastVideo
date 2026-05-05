# SPDX-License-Identifier: Apache-2.0
from torchvision import transforms
from torchvision.transforms import Lambda

from fastvideo.dataset.parquet_dataset_map_style import (
    build_parquet_map_style_dataloader)
from fastvideo.dataset.ltx2_precomputed_dataset import (
    build_ltx2_precomputed_dataloader, LTX2PrecomputedDataset)
from fastvideo.dataset.preprocessing_datasets import VideoCaptionMergedDataset, TextDataset
from fastvideo.dataset.transform import (CenterCropResizeVideo,
                                         LetterboxResizeVideo, Normalize255,
                                         TemporalRandomCrop)
from fastvideo.dataset.validation_dataset import ValidationDataset


def getdataset(args) -> VideoCaptionMergedDataset:
    temporal_sample = TemporalRandomCrop(args.num_frames) if args.do_temporal_sample else None  # 16 x
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    resize_mode = getattr(args, "resize_mode", "center_crop")
    if resize_mode == "letterbox":
        # Pad with -1 because the downstream `(video / 127.5) - 1.0` step
        # in VideoTransformStage maps uint8(0) -> -1.0; we feed pre-uint8
        # tensors here, so 0 yields the same post-normalize value (-1).
        resize_topcrop = [
            LetterboxResizeVideo((args.max_height, args.max_width), fill=0),
        ]
        resize = [
            LetterboxResizeVideo((args.max_height, args.max_width), fill=0),
        ]
    elif resize_mode == "center_crop":
        resize_topcrop = [
            CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True),
        ]
        resize = [
            CenterCropResizeVideo((args.max_height, args.max_width)),
        ]
    else:
        raise ValueError(
            f"Unknown resize_mode={resize_mode!r}; "
            f"valid options: center_crop, letterbox")
    transform = transforms.Compose([
        # Normalize255(),
        *resize,
    ])
    transform_topcrop = transforms.Compose([
        Normalize255(),
        *resize_topcrop,
        norm_fun,
    ])
    return VideoCaptionMergedDataset(data_merge_path=args.data_merge_path,
                                     args=args,
                                     transform=transform,
                                     temporal_sample=temporal_sample,
                                     transform_topcrop=transform_topcrop,
                                     seed=args.seed)
                                    

def gettextdataset(args) -> TextDataset:
    return TextDataset(data_merge_path=args.data_merge_path,
                       args=args,
                       seed=args.seed)


__all__ = [
    "build_parquet_map_style_dataloader",
    "build_ltx2_precomputed_dataloader",
    "LTX2PrecomputedDataset",
    "ValidationDataset",
    "VideoCaptionMergedDataset",
    "TextDataset",
]
