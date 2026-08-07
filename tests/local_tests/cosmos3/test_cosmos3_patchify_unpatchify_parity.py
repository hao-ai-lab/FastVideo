# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 ``[B,C,T,H,W] <-> [B, T*hp*wp, p*p*C]`` patchify roundtrip (Tier A).

Reference: ``vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py``
lines 1009-1036 (``Cosmos3VFMTransformer.patchify`` /
``Cosmos3VFMTransformer.unpatchify``) and the reference assertion at
``tests/diffusion/models/cosmos3/test_cosmos3_transformer.py:98-101``.

Invariant: ``unpatchify(patchify(x)) == x`` for any ``x`` with shape
``(B, C, t, h, w)`` where ``h, w`` are divisible by ``latent_patch_size``.
Also exercises a non-trivial channel count (3) to ensure the
``permute([0, 2, 3, 5, 4, 6, 1])`` reordering is correct.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.local]


def test_patchify_unpatchify_roundtrip() -> None:
    """Asserts that the FastVideo Cosmos3 transformer's patchify/unpatchify
    pair are exact inverses for ``latent_patch_size=2``, ``latent_channel=3``.

    Once FastVideo's ``fastvideo.models.dits.cosmos3.Cosmos3VFMTransformer``
    lands, replace the skip with the upstream-equivalent assertion path.
    """
    try:
        from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer  # type: ignore
    except ImportError:
        pytest.skip("FastVideo Cosmos3 not yet implemented (Phase 2b)")

    from torch import nn

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.latent_patch_size = 2
    model.latent_channel_size = 3

    latents = torch.arange(1 * 3 * 1 * 3 * 5, dtype=torch.float32).reshape(1, 3, 1, 3, 5)
    roundtrip = model.unpatchify(model.patchify(latents, t=1, h=3, w=5), t=1, h=3, w=5)
    torch.testing.assert_close(roundtrip, latents)


def test_patchify_default_patch_size() -> None:
    """Asserts shape contract for the default ``latent_patch_size=[1,2,2]``
    (i.e. spatial-only patching) with a representative video latent.

    With ``[B,C,T,H,W] = [1, 16, 2, 8, 8]`` and patch=2 on H/W, expected
    flattened tokens = ``T * (H/2) * (W/2) = 2 * 4 * 4 = 32`` and each token
    carries ``2*2*C = 4*16 = 64`` channels.
    """
    try:
        from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer  # type: ignore
    except ImportError:
        pytest.skip("FastVideo Cosmos3 not yet implemented (Phase 2b)")

    from torch import nn

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.latent_patch_size = 2
    model.latent_channel_size = 16

    latents = torch.zeros(1, 16, 2, 8, 8)
    tokens = model.patchify(latents, t=2, h=8, w=8)
    assert tuple(tokens.shape) == (1, 32, 64)
    restored = model.unpatchify(tokens, t=2, h=8, w=8)
    assert tuple(restored.shape) == (1, 16, 2, 8, 8)
