from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_video(source: str | torch.Tensor | list, **kwargs) -> torch.Tensor:
    """Load a video as a ``(T, C, H, W)`` float32 tensor in ``[0, 1]``.

    Supported *source* types:

    * **str / Path** – path to ``.mp4`` / ``.avi`` / ``.gif`` file, or a
      directory of frame images (sorted alphabetically).
    * **torch.Tensor** – returned as-is after shape validation.
    * **list[PIL.Image]** – stacked into a tensor.
    """
    if isinstance(source, torch.Tensor):
        if source.ndim != 4:
            raise ValueError(
                f"Expected video tensor with 4 dims (T,C,H,W), got {source.ndim}"
            )
        return source.float()

    if isinstance(source, list):
        frames = [_pil_to_tensor(img) for img in source]
        return torch.stack(frames)

    path = Path(source)
    if path.is_dir():
        return _load_frame_dir(path)
    return _load_video_file(str(path))


def extract_frames(video: torch.Tensor, n_frames: int | None = None) -> torch.Tensor:
    """Uniformly sample *n_frames* from a ``(T, C, H, W)`` video tensor."""
    if n_frames is None or n_frames >= video.shape[0]:
        return video
    indices = torch.linspace(0, video.shape[0] - 1, n_frames).long()
    return video[indices]


# --- Internal helpers ---

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))  # (H, W, 3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _load_frame_dir(path: Path) -> torch.Tensor:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No image files found in {path}")
    frames = [_pil_to_tensor(Image.open(f)) for f in files]
    return torch.stack(frames)


def _load_video_file(path: str) -> torch.Tensor:
    # Try decord first (faster), fall back to torchvision
    try:
        return _load_with_decord(path)
    except ImportError:
        pass
    return _load_with_torchvision(path)


def _load_with_decord(path: str) -> torch.Tensor:
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    # (T, H, W, C) uint8
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    # → (T, C, H, W) float32 [0, 1]
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def _load_with_torchvision(path: str) -> torch.Tensor:
    import torchvision.io

    video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    # torchvision returns (T, H, W, C) uint8
    return video.permute(0, 3, 1, 2).float() / 255.0
