# SPDX-License-Identifier: Apache-2.0
"""MMAudio Synchformer parity against the official reference.

Coverage scope: implementation_subcomponent. Production-loader coverage is
added after the converted component directory exists.
"""

from __future__ import annotations

import os
import gc
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_WEIGHTS = Path(
    os.environ.get(
        "MMAUDIO_SYNCHFORMER_WEIGHTS",
        REPO_ROOT.parent / "MMAudio/ext_weights/synchformer_state_dict.pth",
    )
)


def _build_models(device: torch.device, dtype: torch.dtype):
    from mmaudio.ext.synchformer import Synchformer

    from fastvideo.configs.models.encoders.mmaudio_synchformer import (
        MMAudioSynchformerConfig,
    )
    from fastvideo.models.encoders.mmaudio_synchformer import (
        MMAudioSynchformerVisualEncoder,
    )
    from fastvideo.models.loader.utils import set_default_torch_dtype

    with torch.device(device), set_default_torch_dtype(dtype):
        official = Synchformer()
        fastvideo = MMAudioSynchformerVisualEncoder(MMAudioSynchformerConfig())
    return official, fastvideo


def test_mmaudio_synchformer_state_structure() -> None:
    # MotionFormer calls ``Tensor.item`` while building its stochastic-depth
    # schedule, so it cannot be constructed on the meta device. Instantiate
    # the two large models sequentially to keep peak host memory bounded.
    from mmaudio.ext.synchformer import Synchformer

    official = Synchformer()
    official_state = official.state_dict()
    official_shapes = {name: tensor.shape for name, tensor in official_state.items()}
    del official_state, official
    gc.collect()

    from fastvideo.configs.models.encoders.mmaudio_synchformer import (
        MMAudioSynchformerConfig,
    )
    from fastvideo.models.encoders.mmaudio_synchformer import (
        MMAudioSynchformerVisualEncoder,
    )

    fastvideo = MMAudioSynchformerVisualEncoder(MMAudioSynchformerConfig())
    fastvideo_shapes = {name: tensor.shape for name, tensor in fastvideo.state_dict().items()}

    assert official_shapes == fastvideo_shapes


def test_mmaudio_synchformer_numerical_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio Synchformer parity requires CUDA")
    if not OFFICIAL_WEIGHTS.is_file():
        pytest.skip(
            "Official MMAudio Synchformer weights are absent. Set "
            "MMAUDIO_SYNCHFORMER_WEIGHTS or download the released checkpoint."
        )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    official, fastvideo = _build_models(device, dtype)
    state = torch.load(OFFICIAL_WEIGHTS, map_location="cpu", weights_only=True)
    official.load_state_dict(state, strict=True)
    fastvideo.load_state_dict(state, strict=True)
    official.eval()
    fastvideo.eval()

    generator = torch.Generator(device=device).manual_seed(1234)
    segments = torch.randn((1, 1, 16, 3, 224, 224), generator=generator, device=device, dtype=dtype)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        expected = official(segments)
        actual = fastvideo.forward_segmented(segments)

    difference = (actual.float() - expected.float()).abs()
    print("max_abs", difference.max().item())
    print("mean_abs", difference.mean().item())
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
