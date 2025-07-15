import hashlib
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from comfy.cli_args import args
from PIL import ImageFile, UnidentifiedImageError

T = TypeVar('T')


def conditioning_set_values(conditioning: list[Any],
                            values: dict[str, Any] | None = None) -> list[Any]:
    if values is None:
        values = {}
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c


def pillow(fn: Callable[[Any], T], arg: Any) -> T:
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError
            ):  #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher() -> Callable[[], Any]:
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    return hashfuncs[args.default_hashing_function]


def string_to_torch_dtype(string: str) -> torch.dtype | None:
    if string == "fp32":
        return torch.float32
    if string == "fp16":
        return torch.float16
    if string == "bf16":
        return torch.bfloat16
    return None


def image_alpha_fix(destination: torch.Tensor,
                    source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., :destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        destination = torch.nn.functional.pad(destination, (0, 1))
        destination[..., -1] = 1.0
    return destination, source
