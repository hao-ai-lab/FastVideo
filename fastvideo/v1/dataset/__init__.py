import os

from torchvision import transforms
from torchvision.transforms import Lambda
from transformers import AutoTokenizer

from fastvideo.v1.dataset.t2v_datasets import T2V_dataset
from fastvideo.v1.dataset.transform import (CenterCropResizeVideo, Normalize255,
                                            TemporalRandomCrop)

from .parquet_dataset_map_style import build_parquet_map_style_dataloader

__all__ = ["build_parquet_map_style_dataloader"]


def getdataset(args, start_idx=0) -> T2V_dataset:
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    resize_topcrop = [
        CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True),
    ]
    resize = [
        CenterCropResizeVideo((args.max_height, args.max_width)),
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
    tokenizer_path = os.path.join(args.model_path, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              cache_dir=args.cache_dir)
    if args.dataset == "t2v":
        return T2V_dataset(args,
                           transform=transform,
                           temporal_sample=temporal_sample,
                           tokenizer=tokenizer,
                           transform_topcrop=transform_topcrop,
                           start_idx=start_idx)

    raise NotImplementedError(args.dataset)
