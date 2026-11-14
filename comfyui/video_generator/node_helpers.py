import hashlib
from collections.abc import Callable
from typing import Any, TypeVar

from comfy.cli_args import args
from PIL import ImageFile, UnidentifiedImageError

T = TypeVar('T')


def pillow(fn: Callable[[Any], T], arg: Any) -> T:
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):  #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher() -> Callable[[], Any]:
    hashfuncs = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}
    return hashfuncs[args.default_hashing_function]
