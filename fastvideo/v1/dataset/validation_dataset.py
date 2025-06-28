# SPDX-License-Identifier: Apache-2.0
# adapted from: https://github.com/a-r-r-o-w/finetrainers/blob/main/finetrainers/data/dataset.py
import os
import pathlib

import datasets
import torch

from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.vision_utils import load_image, load_video

logger = init_logger(__name__)


class ValidationDataset(torch.utils.data.IterableDataset):

    def __init__(self, filename: str):
        super().__init__()

        self.filename = pathlib.Path(filename)
        # get directory of filename
        self.dir = os.path.abspath(self.filename.parent)

        if not self.filename.exists():
            raise FileNotFoundError(
                f"File {self.filename.as_posix()} does not exist")

        if self.filename.suffix == ".csv":
            data = datasets.load_dataset("csv",
                                         data_files=self.filename.as_posix(),
                                         split="train")
        elif self.filename.suffix == ".json":
            data = datasets.load_dataset("json",
                                         data_files=self.filename.as_posix(),
                                         split="train",
                                         field="data")
        elif self.filename.suffix == ".parquet":
            data = datasets.load_dataset("parquet",
                                         data_files=self.filename.as_posix(),
                                         split="train")
        elif self.filename.suffix == ".arrow":
            data = datasets.load_dataset("arrow",
                                         data_files=self.filename.as_posix(),
                                         split="train")
        else:
            _SUPPORTED_FILE_FORMATS = [".csv", ".json", ".parquet", ".arrow"]
            raise ValueError(
                f"Unsupported file format {self.filename.suffix} for validation dataset. Supported formats are: {_SUPPORTED_FILE_FORMATS}"
            )

        self._data = data.to_iterable_dataset()

    def __iter__(self):
        for sample in self._data:
            # For consistency reasons, we mandate that "caption" is always present in the validation dataset.
            # However, since the model specifications use "prompt", we create an alias here.
            sample["prompt"] = sample["caption"]

            # Load image or video if the path is provided
            # TODO(aryan): need to handle custom columns here for control conditions
            sample["image"] = None
            sample["video"] = None

            if sample.get("image_path", None) is not None:
                image_path = sample["image_path"]
                image_path = os.path.join(self.dir, image_path)
                if not pathlib.Path(image_path).is_file(
                ) and not image_path.startswith("http"):
                    logger.warning("Image file %s does not exist.", image_path)
                else:
                    sample["image"] = load_image(image_path)

            if sample.get("video_path", None) is not None:
                video_path = sample["video_path"]
                video_path = os.path.join(self.dir, video_path)
                if not pathlib.Path(video_path).is_file(
                ) and not video_path.startswith("http"):
                    logger.warning("Video file %s does not exist.", video_path)
                else:
                    sample["video"] = load_video(video_path)

            if sample.get("control_image_path", None) is not None:
                control_image_path = sample["control_image_path"]
                control_image_path = os.path.join(self.dir, control_image_path)
                if not pathlib.Path(control_image_path).is_file(
                ) and not control_image_path.startswith("http"):
                    logger.warning("Control Image file %s does not exist.",
                                   control_image_path)
                else:
                    sample["control_image"] = load_image(control_image_path)

            if sample.get("control_video_path", None) is not None:
                control_video_path = sample["control_video_path"]
                control_video_path = os.path.join(self.dir, control_video_path)
                if not pathlib.Path(control_video_path).is_file(
                ) and not control_video_path.startswith("http"):
                    logger.warning("Control Video file %s does not exist.",
                                   control_video_path)
                else:
                    sample["control_video"] = load_video(control_video_path)

            sample = {k: v for k, v in sample.items() if v is not None}
            yield sample
