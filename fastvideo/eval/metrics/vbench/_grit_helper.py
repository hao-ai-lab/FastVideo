"""Shared GRiT model loading and detection utilities for VBench metrics.

All 4 GRiT-based metrics (object_class, multiple_objects, color,
spatial_relationship) use the same model and detection API.
"""

from __future__ import annotations

import numpy as np
import torch


def _patch_detectron2_registries():
    """Make detectron2's fvcore registries skip duplicate names instead of raising.

    Both pip-installed ``vbench`` and wm-eval vendor the same GRiT / CenterNet2
    source with 20+ ``@REGISTRY.register()`` calls. When both are imported in the
    same process (e.g. the parity test), detectron2's global registries see two
    different Python classes with the same name and raise ``AssertionError``.

    This one-time patch makes ``_do_register`` silently skip if the name is
    already registered, which is safe because the classes are identical.
    """
    # fvcore Registry (META_ARCH, ROI_HEADS, BACKBONE, PROPOSAL_GENERATOR, ...)
    try:
        from fvcore.common.registry import Registry
        orig = Registry._do_register
        if not getattr(orig, "_patched_idempotent", False):
            def _safe_do_register(self, name, obj):
                if name in self._obj_map:
                    return
                orig(self, name, obj)
            _safe_do_register._patched_idempotent = True
            Registry._do_register = _safe_do_register
    except ImportError:
        pass

    # detectron2 DatasetCatalog (object365_train, vg_train, etc.)
    try:
        from detectron2.data import DatasetCatalog
        orig_ds = DatasetCatalog.register
        if not getattr(orig_ds, "_patched_idempotent", False):
            def _safe_ds_register(name, func):
                if name in DatasetCatalog:
                    return
                orig_ds(name, func)
            _safe_ds_register._patched_idempotent = True
            DatasetCatalog.register = _safe_ds_register
    except (ImportError, AttributeError):
        pass


_patch_detectron2_registries()


def load_grit_model(device: str | torch.device, task: str = "DenseCap"):
    """Load the GRiT DenseCaptioning model.

    Parameters
    ----------
    task : "DenseCap" | "ObjectDet"
        VBench uses "ObjectDet" for object_class / multiple_objects /
        spatial_relationship and "DenseCap" for color (which needs the
        actual caption text). The two heads return predictions in
        different formats:
            - DenseCap → ``[(caption, bbox, [class_label]), ...]``
            - ObjectDet → ``[(class_label, bbox, [class_label]), ...]``
        spatial_relationship matches on ``pred[0]`` (the first field),
        so it must run in ObjectDet mode to compare against class names.
    """
    from vbench.third_party.grit_model import DenseCaptioning
    from fastvideo.eval.models import ensure_checkpoint

    ckpt = ensure_checkpoint(
        "grit_b_densecap_objectdet.pth",
        source="OpenGVLab/VBench_Used_Models",
        filename="grit_b_densecap_objectdet.pth",
    )
    # GRiT internals call .type on device, so coerce to torch.device
    if isinstance(device, str):
        device = torch.device(device)
    model = DenseCaptioning(device)
    if task == "ObjectDet":
        model.initialize_model_det(ckpt)
    else:
        model.initialize_model(ckpt)
    return model


def detect_frames(model, frames_np: list[np.ndarray]) -> list:
    """Run GRiT detection on a list of (H, W, C) uint8 numpy frames.

    Returns per-frame predictions in the format used by VBench metrics.
    Each frame's predictions is a list of (description, bbox, object_types).
    """
    predictions = []
    with torch.no_grad():
        for frame in frames_np:
            ret = model.run_caption_tensor(frame)
            predictions.append(ret[0] if len(ret[0]) > 0 else [])
    return predictions


def _vbench_middle_indices(vlen: int, num_frames: int) -> list[int]:
    """Replicate VBench's get_frame_indices(sample="middle"): split [0, vlen)
    into num_frames equal intervals and pick the midpoint of each.

    Without this, wm-eval's torch.linspace-based sampler picks different
    indices than VBench, producing different GRiT predictions and scores
    on long videos. See vbench/utils.py:get_frame_indices.
    """
    acc = min(num_frames, vlen)
    intervals = np.linspace(0, vlen, acc + 1).astype(int)
    indices = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(acc)]
    if len(indices) < num_frames:
        indices = indices + [indices[-1]] * (num_frames - len(indices))
    return indices


def prepare_frames(video_tensor: torch.Tensor, n_frames: int = 16,
                   max_short_side: int = 768) -> list[np.ndarray]:
    """Convert (T, C, H, W) float [0,1] tensor to list of (H, W, C) numpy frames
    in VBench's exact format: float32 [0, 255] HWC.

    VBench's load_video casts the decord uint8 buffer with ``torch.Tensor(...)``,
    yielding **float32 with values in [0, 255]**, then runs torchvision
    ``Resize`` (which preserves float dtype and produces fractional bilinear
    outputs), then ``.permute(0,2,3,1).numpy()`` for GRiT. We must replicate
    this exactly — round-tripping through uint8 truncates the fractional
    bilinear outputs and shifts GRiT detection counts (e.g. 1 vs 6 persons
    per frame), which then breaks spatial_relationship/multiple_objects.

    Sampling matches VBench's ``get_frame_indices(sample="middle")``.
    """
    from torchvision import transforms

    T = video_tensor.shape[0]
    indices = _vbench_middle_indices(T, n_frames)
    # Recover the original uint8 values from the float [0,1] loader
    # (round, not truncate), then re-cast to float32 [0,255] like VBench's
    # ``torch.Tensor(decord_uint8_array)`` path.
    frames_uint8 = (video_tensor[indices] * 255).round().clamp(0, 255).to(torch.uint8)
    frames_f = frames_uint8.float()

    h, w = frames_f.shape[-2], frames_f.shape[-1]
    if min(h, w) > max_short_side:
        scale = 720.0 / min(h, w)
        new_h, new_w = int(scale * h), int(scale * w)
        # VBench (object_class.py:55) uses transforms.Resize without
        # antialias kwarg → torchvision default (BILINEAR, no antialias).
        # Float input ⇒ fractional bilinear outputs are preserved.
        frames_f = transforms.Resize(size=(new_h, new_w))(frames_f)

    frames_np = frames_f.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    return list(frames_np)
