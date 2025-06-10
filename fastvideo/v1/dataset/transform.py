# SPDX-License-Identifier: Apache-2.0
import random

import torch


def _is_tensor_video_clip(clip) -> bool:
    if not torch.is_tensor(clip):
        raise TypeError(f"clip should be Tensor. Got {type(clip)}")

    if not clip.ndimension() == 4:
        raise ValueError(f"clip should be 4D. Got {clip.dim()}D")

    return True


def crop(clip, i, j, h, w) -> torch.Tensor:
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i:i + h, j:j + w]


def resize(clip, target_size, interpolation_mode) -> torch.Tensor:
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip,
        size=target_size,
        mode=interpolation_mode,
        align_corners=True,
        antialias=True,
    )


def center_crop_th_tw(clip, th, tw, top_crop) -> torch.Tensor:
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")

    # import ipdb;ipdb.set_trace()
    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)


def normalize_video(clip) -> torch.Tensor:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            f"clip tensor should have data type uint8. Got {clip.dtype}")
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
        self,
        size,
        top_crop=False,
        interpolation_mode="bilinear",
    ) -> None:
        if len(size) != 2:
            raise ValueError(
                f"size should be tuple (height, width), instead got {size}")
        self.size = size
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip) -> torch.Tensor:
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_th_tw(clip,
                                             self.size[0],
                                             self.size[1],
                                             top_crop=self.top_crop)
        clip_center_crop_resize = resize(
            clip_center_crop,
            target_size=self.size,
            interpolation_mode=self.interpolation_mode,
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class Normalize255:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    """

    def __init__(self) -> None:
        pass

    def __call__(self, clip) -> torch.Tensor:
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return normalize_video(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class TemporalRandomCrop:
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, total_frames) -> tuple[int, int]:
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index
