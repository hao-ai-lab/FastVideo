# SPDX-License-Identifier: Apache-2.0
"""MMAudio transformer parity against the official reference.

Coverage scope: implementation_subcomponent. Production-loader coverage is
added after the converted Diffusers component directory exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_WEIGHTS = Path(
    os.environ.get(
        "MMAUDIO_TRANSFORMER_WEIGHTS",
        REPO_ROOT.parent / "MMAudio/weights/mmaudio_large_44k_v2.pth",
    )
)
SMALL_44K_OFFICIAL_WEIGHTS = Path(
    os.environ.get(
        "MMAUDIO_SMALL_44K_TRANSFORMER_WEIGHTS",
        REPO_ROOT.parent / "MMAudio/weights/mmaudio_small_44k.pth",
    )
)


def _require_cuda_and_weights() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio full-transformer parity requires CUDA")
    if not OFFICIAL_WEIGHTS.is_file():
        pytest.skip(
            "Official MMAudio transformer weights are absent. Set "
            "MMAUDIO_TRANSFORMER_WEIGHTS or download large_44k_v2 assets."
        )


@pytest.mark.parametrize("v2", [False, True])
def test_mmaudio_transformer_implementation_parity(v2: bool) -> None:
    from mmaudio.model.networks import MMAudio

    from fastvideo.configs.models.dits.mmaudio import (
        MMAudioArchConfig,
        MMAudioTransformerConfig,
    )
    from fastvideo.models.dits.mmaudio import MMAudioTransformer

    kwargs = {
        "latent_dim": 8,
        "clip_dim": 16,
        "sync_dim": 12,
        "text_dim": 16,
        "hidden_dim": 64,
        "depth": 3,
        "fused_depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "latent_seq_len": 5,
        "clip_seq_len": 3,
        "sync_seq_len": 8,
        "text_seq_len": 4,
        "v2": v2,
    }
    official = MMAudio(**kwargs)
    fastvideo = MMAudioTransformer(MMAudioTransformerConfig(arch_config=MMAudioArchConfig(**kwargs)), hf_config={})
    assert {name: tensor.shape for name, tensor in official.state_dict().items()} == {
        name: tensor.shape for name, tensor in fastvideo.state_dict().items()
    }
    fastvideo.load_state_dict(official.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 5, 8), generator=generator)
    clip = torch.randn((1, 3, 16), generator=generator)
    sync = torch.randn((1, 8, 12), generator=generator)
    text = torch.randn((1, 4, 16), generator=generator)
    timestep = torch.tensor([0.375])
    official.eval()
    fastvideo.eval()
    with torch.inference_mode():
        expected = official(latent, clip, sync, text, timestep)
        actual = fastvideo(latent, (clip, sync, text), timestep)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mmaudio_small_44k_v1_transformer_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio full-transformer parity requires CUDA")
    if not SMALL_44K_OFFICIAL_WEIGHTS.is_file():
        pytest.skip(
            "Official MMAudio small_44k weights are absent. Set "
            "MMAUDIO_SMALL_44K_TRANSFORMER_WEIGHTS."
        )

    from mmaudio.model.networks import small_44k

    from fastvideo.configs.models.dits.mmaudio import (
        MMAudioArchConfig,
        MMAudioTransformerConfig,
    )
    from fastvideo.models.dits.mmaudio import MMAudioTransformer
    from fastvideo.models.loader.utils import set_default_torch_dtype

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    state = torch.load(
        SMALL_44K_OFFICIAL_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )
    arch = MMAudioArchConfig(
        hidden_dim=448,
        depth=12,
        fused_depth=8,
        num_heads=7,
        v2=False,
    )
    with torch.device(device), set_default_torch_dtype(dtype):
        official = small_44k()
        fastvideo = MMAudioTransformer(
            MMAudioTransformerConfig(arch_config=arch),
            hf_config={},
        )
    official.load_weights(dict(state))
    state.pop("latent_rot", None)
    state.pop("clip_rot", None)
    missing, unexpected = fastvideo.load_state_dict(state, strict=True)
    assert missing == []
    assert unexpected == []
    official.eval()
    fastvideo.eval()

    generator = torch.Generator(device=device).manual_seed(5678)
    latent = torch.randn(
        (1, 345, 40), generator=generator, device=device, dtype=dtype
    )
    clip = torch.randn(
        (1, 64, 1024), generator=generator, device=device, dtype=dtype
    )
    sync = torch.randn(
        (1, 192, 768), generator=generator, device=device, dtype=dtype
    )
    text = torch.randn(
        (1, 77, 1024), generator=generator, device=device, dtype=dtype
    )
    timestep = torch.tensor([0.375], device=device, dtype=dtype)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        expected = official(latent, clip, sync, text, timestep)
        actual = fastvideo(latent, (clip, sync, text), timestep)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mmaudio_large_44k_v2_transformer_parity() -> None:
    _require_cuda_and_weights()
    from mmaudio.model.networks import large_44k_v2

    from fastvideo.configs.models.dits.mmaudio import MMAudioTransformerConfig
    from fastvideo.models.dits.mmaudio import MMAudioTransformer
    from fastvideo.models.loader.utils import set_default_torch_dtype

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    state = torch.load(OFFICIAL_WEIGHTS, map_location="cpu", weights_only=True)

    with torch.device(device), set_default_torch_dtype(dtype):
        official = large_44k_v2()
        fastvideo = MMAudioTransformer(MMAudioTransformerConfig(), hf_config={})
    official.load_weights(dict(state))
    # The released checkpoint contains a stale derived buffer which the
    # official ``load_weights`` explicitly discards. Validate it before
    # applying the same canonicalization used by the converter.
    checkpoint_freqs = state.pop("t_embed.freqs")
    torch.testing.assert_close(checkpoint_freqs,
                               fastvideo.t_embed.freqs.cpu(),
                               atol=1e-6,
                               rtol=1e-6)
    missing, unexpected = fastvideo.load_state_dict(state, strict=True)
    assert missing == []
    assert unexpected == []
    del state
    official.eval()
    fastvideo.eval()

    generator = torch.Generator(device=device).manual_seed(1234)
    latent = torch.randn((1, 345, 40), generator=generator, device=device, dtype=dtype)
    clip = torch.randn((1, 64, 1024), generator=generator, device=device, dtype=dtype)
    sync = torch.randn((1, 192, 768), generator=generator, device=device, dtype=dtype)
    text = torch.randn((1, 77, 1024), generator=generator, device=device, dtype=dtype)
    timestep = torch.tensor([0.375], device=device, dtype=dtype)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        expected = official(latent, clip, sync, text, timestep)
        actual = fastvideo(latent, (clip, sync, text), timestep)

    difference = (actual.float() - expected.float()).abs()
    print("max_abs", difference.max().item())
    print("mean_abs", difference.mean().item())
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
