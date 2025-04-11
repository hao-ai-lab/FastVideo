# SPDX-License-Identifier: Apache-2.0

import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
import torch
from packaging import version

if version.parse(version.parse(
        PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def pil_to_numpy(
        images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    r"""
    Convert a PIL image or a list of PIL images to NumPy arrays.

    Args:
        images (`PIL.Image.Image` or `List[PIL.Image.Image]`):
            The PIL image or list of images to convert to NumPy format.

    Returns:
        `np.ndarray`:
            A NumPy array representation of the images.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images_arr: np.ndarray = np.stack(images, axis=0)

    return images_arr


def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    r"""
    Convert a NumPy image to a PyTorch tensor.

    Args:
        images (`np.ndarray`):
            The NumPy image array to convert to PyTorch format.

    Returns:
        `torch.Tensor`:
            A PyTorch tensor representation of the images.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def normalize(
        images: Union[np.ndarray,
                      torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Normalize an image array to [-1,1].

    Args:
        images (`np.ndarray` or `torch.Tensor`):
            The image array to normalize.

    Returns:
        `np.ndarray` or `torch.Tensor`:
            The normalized image array.
    """
    return 2.0 * images - 1.0


def load_image(
    image: Union[str, PIL.Image.Image],
    convert_method: Optional[Callable[[PIL.Image.Image],
                                      PIL.Image.Image]] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], *optional*):
            A conversion method to apply to the image after loading it. When set to `None` the image will be converted
            "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


def get_default_height_width(
    image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
    vae_scale_factor: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[int, int]:
    r"""
    Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
            should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
            tensor, it should have shape `[batch, channels, height, width]`.
        height (`Optional[int]`, *optional*, defaults to `None`):
            The height of the preprocessed image. If `None`, the height of the `image` input will be used.
        width (`Optional[int]`, *optional*, defaults to `None`):
            The width of the preprocessed image. If `None`, the width of the `image` input will be used.

    Returns:
        `Tuple[int, int]`:
            A tuple containing the height and width, both resized to the nearest integer multiple of
            `vae_scale_factor`.
    """

    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[2]
        else:
            height = image.shape[1]

    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[3]
        else:
            width = image.shape[2]

    width, height = (x - x % vae_scale_factor for x in (width, height)
                     )  # resize to integer multiple of vae_scale_factor

    return height, width


def resize(
    image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
    height: int,
    width: int,
    resize_mode: str = "default",  # "default", "fill", "crop"
    resample: str = "lanczos",
) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
    """
    Resize image.

    Args:
        image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
            The image input, can be a PIL image, numpy array or pytorch tensor.
        height (`int`):
            The height to resize to.
        width (`int`):
            The width to resize to.
        resize_mode (`str`, *optional*, defaults to `default`):
            The resize mode to use, can be one of `default` or `fill`. If `default`, will resize the image to fit
            within the specified width and height, and it may not maintaining the original aspect ratio. If `fill`,
            will resize the image to fit within the specified width and height, maintaining the aspect ratio, and
            then center the image within the dimensions, filling empty with data from image. If `crop`, will resize
            the image to fit within the specified width and height, maintaining the aspect ratio, and then center
            the image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
            supported for PIL image input.

    Returns:
        `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
            The resized image.
    """
    if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
        raise ValueError(
            f"Only PIL image input is supported for resize_mode {resize_mode}")
    assert isinstance(image, PIL.Image.Image)
    if resize_mode == "default":
        image = image.resize((width, height),
                             resample=PIL_INTERPOLATION[resample])
    else:
        raise ValueError(f"resize_mode {resize_mode} is not supported")
    return image
