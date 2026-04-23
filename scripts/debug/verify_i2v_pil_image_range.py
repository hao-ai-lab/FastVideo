"""One-shot reproducer for the I2V pil_image normalization bug.

Run against pre-fix and post-fix commits to generate before/after evidence
for the PR description. Removed after use.

Usage (from FastVideo/):
    PYTHONPATH=. path/to/python scripts/debug/verify_i2v_pil_image_range.py
"""
import torch

from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage

# Bypass __init__ so we don't need a real VAE. preprocess() doesn't use self.
stage = ImageVAEEncodingStage.__new__(ImageVAEEncodingStage)

# Deterministic input spanning full uint8 range.
uint8_in = torch.tensor(
    [[[[0, 64, 128, 192, 255]],
      [[0, 64, 128, 192, 255]],
      [[0, 64, 128, 192, 255]]]],
    dtype=torch.uint8,
)
float_in = uint8_in.float() / 255.0

try:
    out_uint8 = stage.preprocess(uint8_in, vae_scale_factor=8, height=1, width=5)
    print(
        f"uint8 input  (range [0, 255]) -> out range "
        f"[{out_uint8.min().item():.3f}, {out_uint8.max().item():.3f}], "
        f"dtype={out_uint8.dtype}"
    )
except (ValueError, TypeError) as e:
    print(f"uint8 input  (range [0, 255]) -> rejected: {type(e).__name__}: {e}")
    out_uint8 = None

out_float = stage.preprocess(float_in, vae_scale_factor=8, height=1, width=5)
print(
    f"float input  (range [0,   1]) -> out range "
    f"[{out_float.min().item():.3f}, {out_float.max().item():.3f}], "
    f"dtype={out_float.dtype}"
)

if out_uint8 is not None:
    print(f"numerically equal: {torch.equal(out_uint8.float(), out_float)}")
