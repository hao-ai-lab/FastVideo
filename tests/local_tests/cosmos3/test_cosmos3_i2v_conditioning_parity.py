# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo I2V conditioning pixel video vs the framework.

The Cosmos3 I2V path conditions on a *static repeat* of the input image. The
framework (``cosmos_framework.inference.vision``):

  * ``load_conditioning_image``: aspect-preserving resize + center crop + uint8
    quantization, then ``/127.5 - 1`` -> ``[3, 1, h, w]`` in [-1, 1];
  * ``build_conditioned_video_batch``: frame 0 = the image, and every remaining
    frame **repeats the last conditioning frame** (a static video) -> the clip
    is then VAE-encoded and only the latent condition frame(s) are kept clean.

Because the VAE is temporal, zero-filling the non-condition frames (the earlier
FastVideo behavior) changes the encoded condition latent, so the repeat-fill is
correctness-critical. This pins FastVideo's
``Cosmos3DenoisingStage._image_to_video_tensor`` against the framework's
image preprocessing + repeat-fill.

CPU / float32. The framework is the parity ORACLE.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_i2v_conditioning_parity.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

# The official framework provides the parity oracle.
vision = pytest.importorskip(
    "cosmos_framework.inference.vision",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

from fastvideo.pipelines.stages.cosmos3_stages import (  # noqa: E402
    Cosmos3DenoisingStage,
)

pytestmark = [pytest.mark.local]


def _make_image(path, h_in: int, w_in: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h_in, w_in, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


# (input H, input W, target H, target W, num_frames)
_CASES = [
    pytest.param(120, 200, 256, 256, 9, id="square_from_landscape"),
    pytest.param(200, 120, 704, 1280, 13, id="wide_from_portrait"),
    pytest.param(256, 256, 256, 256, 5, id="same_size"),
]


class TestCosmos3I2VConditioningParity:

    @pytest.mark.parametrize(("h_in", "w_in", "h", "w", "num_frames"), _CASES)
    def test_conditioning_video_matches_framework(self, tmp_path, h_in, w_in, h, w, num_frames):
        img_path = tmp_path / "cond.png"
        _make_image(img_path, h_in, w_in)

        # ---- Framework oracle ----
        # load_conditioning_image -> [3, 1, h, w] in [-1, 1].
        cond = vision.load_conditioning_image(img_path, target_h=h, target_w=w).float()
        # Mirror build_conditioned_video_batch (vision.py lines 117-123) in fp32/CPU:
        # frame 0 = image; remaining frames repeat the last conditioning frame.
        t_cond = cond.shape[1]
        expected = torch.zeros(1, 3, num_frames, h, w, dtype=torch.float32)
        t_fill = min(t_cond, num_frames)
        expected[0, :, :t_fill] = cond[:, :t_fill]
        if t_fill < num_frames:
            expected[0, :, t_fill:] = expected[0, :, t_fill - 1:t_fill].expand(-1, num_frames - t_fill, -1, -1)

        # ---- FastVideo: same PIL image through the stage helper ----
        pil = Image.open(img_path).convert("RGB")
        got = Cosmos3DenoisingStage._image_to_video_tensor(
            pil, num_frames, h, w, torch.device("cpu"), torch.float32)

        assert got.shape == expected.shape, f"shape: got={got.shape} expected={expected.shape}"
        max_abs = (got - expected).abs().max().item()
        print(f"\n[i2v_cond {h}x{w} nf={num_frames}] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(got, expected)

        # Static repeat (not zero-fill): every frame equals frame 0, and the
        # frames past frame 0 are non-zero.
        assert torch.equal(got[0, :, 0], got[0, :, -1]), "non-condition frames must repeat the image"
        assert got[0, :, 1:].abs().sum() > 0, "non-condition frames must not be zero-filled"
