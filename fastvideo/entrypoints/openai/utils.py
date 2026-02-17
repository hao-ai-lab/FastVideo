# Adapted from SGLang (https://github.com/sgl-project/sglang)

import base64
import os
import re
from typing import Any, List, Optional, Union

import httpx
from fastapi import UploadFile

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def parse_size(size: str) -> tuple[int, int] | tuple[None, None]:
    """Parse a 'WIDTHxHEIGHT' string into (width, height)"""
    try:
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        return None, None


def choose_image_ext(output_format: Optional[str], background: Optional[str]) -> str:
    """Pick a file extension for image outputs"""
    fmt = (output_format or "").lower()
    if fmt in {"png", "webp", "jpeg", "jpg"}:
        return "jpg" if fmt == "jpeg" else fmt
    if (background or "auto").lower() == "transparent":
        return "png"
    return "jpg"


async def save_image_to_path(image: Union[UploadFile, str], target_path: str) -> str:
    """Save an uploaded file or download from URL to *target_path*"""
    input_path = await _maybe_url_image(image, target_path)
    if input_path is None:
        input_path = await _save_upload_to_path(image, target_path)
    return input_path


async def _save_upload_to_path(upload: UploadFile, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    content = await upload.read()
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path


async def _maybe_url_image(img_url: str, target_path: str) -> str | None:
    if not isinstance(img_url, str):
        return None
    if img_url.lower().startswith(("http://", "https://")):
        return await _save_url_image_to_path(img_url, target_path)
    elif img_url.startswith("data:image"):
        return await _save_base64_image_to_path(img_url, target_path)
    else:
        raise ValueError("Unsupported image url format")


async def _save_url_image_to_path(image_url: str, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()

            if not os.path.splitext(target_path)[1]:
                content_type = response.headers.get("content-type", "").lower()
                url_path = image_url.split("?")[0]
                _, url_ext = os.path.splitext(url_path)
                url_ext = url_ext.lower()
                if url_ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                    ext = ".jpg" if url_ext == ".jpeg" else url_ext
                elif "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"
                else:
                    ext = ".jpg"
                target_path = f"{target_path}{ext}"

            with open(target_path, "wb") as f:
                f.write(response.content)
            return target_path
    except Exception as e:
        raise RuntimeError(f"Failed to download image from URL: {e}")


async def _save_base64_image_to_path(base64_data: str, target_path: str) -> str:
    pattern = r"data:(.*?)(;base64)?,(.*)"
    match = re.match(pattern, base64_data)
    if not match or not match.group(2):
        raise ValueError("Invalid base64 image data URL")
    media_type = match.group(1)
    data = match.group(3)
    if not data:
        raise ValueError("Empty base64 image data")
    ext = media_type.split("/")[-1].lower() if media_type.startswith("image/") else "jpg"
    if ext == "jpeg":
        ext = "jpg"
    target_path = f"{target_path}.{ext}"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        image_data = base64.b64decode(data)
        with open(target_path, "wb") as f:
            f.write(image_data)
        return target_path
    except Exception as e:
        raise RuntimeError(f"Failed to decode base64 image: {e}")


def merge_image_input_list(*inputs: Union[List, Any, None]) -> List:
    """Merge multiple image input sources into a single flat list"""
    result = []
    for input_item in inputs:
        if input_item is not None:
            if isinstance(input_item, list):
                result.extend(input_item)
            else:
                result.append(input_item)
    return result
