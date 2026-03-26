"""Test i2v_record_creator handles missing CLIP embeddings (Wan2.2 I2V).

Wan2.2 I2V conditions on the input image through VAE only (no CLIP).
The preprocessing pipeline skips ImageEncodingStage, so image_embeds
stays empty. The record creator must handle this gracefully.

Usage:
    python tests/local_tests/test_i2v_record_no_clip.py
"""

import importlib
import importlib.util
import os
import sys
import types

# Remove tests/local_tests from sys.path to avoid shadowing
# the real 'transformers' package.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np  # noqa: E402
import torch  # noqa: E402

# We need exactly two leaf modules:
#   fastvideo.dataset.dataloader.record_schema
#   fastvideo.pipelines.pipeline_batch_info
# But their intermediate __init__.py files pull in heavy deps.
# Stub out all intermediate packages so only the leaf files load.
_repo = os.path.dirname(os.path.dirname(_here))
_fv = os.path.join(_repo, "fastvideo")


def _stub_pkg(name, path):
    """Register a stub package in sys.modules."""
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


_stub_pkg("fastvideo", _fv)
_stub_pkg("fastvideo.dataset", os.path.join(_fv, "dataset"))
_stub_pkg("fastvideo.dataset.dataloader",
          os.path.join(_fv, "dataset", "dataloader"))
_stub_pkg("fastvideo.pipelines", os.path.join(_fv, "pipelines"))


def _load_module(fqn, filepath):
    """Load a single .py file as a module without triggering __init__."""
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


# Load leaf modules directly from file.
batch_info = _load_module(
    "fastvideo.pipelines.pipeline_batch_info",
    os.path.join(_fv, "pipelines", "pipeline_batch_info.py"),
)
record_schema = _load_module(
    "fastvideo.dataset.dataloader.record_schema",
    os.path.join(_fv, "dataset", "dataloader", "record_schema.py"),
)

i2v_record_creator = record_schema.i2v_record_creator
PreprocessBatch = batch_info.PreprocessBatch


def _make_batch(image_embeds: list) -> PreprocessBatch:
    """Create a minimal PreprocessBatch for i2v record creation."""
    return PreprocessBatch(
        data_type="video",
        prompt=["a cat walking"],
        width=[512],
        height=[512],
        fps=[16],
        num_frames=[77],
        video_file_name=["test.mp4"],
        latents=[
            np.random.randn(16, 20, 64, 64).astype(np.float32)
        ],
        prompt_embeds=[
            np.random.randn(512, 4096).astype(np.float32)
        ],
        image_embeds=image_embeds,
        image_latent=np.random.randn(1, 16, 1, 64, 64).astype(
            np.float32),
        pil_image=np.random.randn(1, 3, 512, 512).astype(
            np.float32),
    )


def test_i2v_record_with_clip():
    """Normal case: CLIP embeddings present (Wan2.1 I2V)."""
    clip_embed = np.random.randn(1, 768).astype(np.float32)
    batch = _make_batch(image_embeds=[clip_embed])
    records = i2v_record_creator(batch)
    assert len(records) == 1
    assert len(records[0]["clip_feature_bytes"]) > 0


def test_i2v_record_no_clip():
    """Wan2.2 I2V: no CLIP encoder, image_embeds is empty."""
    batch = _make_batch(image_embeds=[])
    records = i2v_record_creator(batch)
    assert len(records) == 1
    assert records[0]["clip_feature_bytes"] == b""
    assert records[0]["clip_feature_shape"] == []
    # image_latent and pil_image should still be present
    assert len(records[0]["first_frame_latent_bytes"]) > 0
    assert len(records[0]["pil_image_bytes"]) > 0


if __name__ == "__main__":
    print("test_i2v_record_with_clip ... ", end="", flush=True)
    test_i2v_record_with_clip()
    print("PASS")

    print("test_i2v_record_no_clip ... ", end="", flush=True)
    try:
        test_i2v_record_no_clip()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL (AssertionError: {e})")
        sys.exit(1)
