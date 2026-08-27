# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 video VAE parity: FastVideo-native PyTorch reference vs MLX.

Tiny random-weight models exercise every primitive (causal conv3d, per-frame
GroupNorm, ViT attention with QK-RMSNorm + partial RoPE, SwiGLU, chunked
clip decoding, spatial tiling); a real-weight bounded-tile case validates the
released FP32 checkpoint end to end.

Runs in any environment with torch + mlx (CPU is fine):

    pytest tests/local_tests/minimax_h3/test_mlx_video_vae_parity.py -v

The real-weight case activates automatically when the released snapshot is
present; point MINIMAX_H3_MODEL_ROOT at another checkout otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch reference needed for video VAE parity")
mx = pytest.importorskip("mlx.core", reason="MLX needed for video VAE parity")

from fastvideo.configs.models.vaes.minimax_h3_video import (  # noqa: E402
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_video_vae import (  # noqa: E402
    MLXMiniMaxH3VideoVAE,
    MiniMaxH3VideoVAEConfigView,
    _rotary_cos_sin,
)

SEED = 4112

TINY_ARCH = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    block_out_channels=(8, 16),
    layers_per_block=1,
    spatial_downsample_factors=(2, 2),
    temporal_downsample_factors=(1, 2),
    norm_num_groups=4,
    norm_eps=1e-6,
    decoder_num_layers=2,
    decoder_num_attention_heads=4,
    decoder_attention_head_dim=16,
    decoder_num_register_tokens=2,
    decoder_ffn_mult=2,
    decoder_rope_theta=100.0,
    decoder_rope_dim_ratio=0.75,
    decoder_norm_eps=1e-5,
    clip_length=5,
    token_drop=1,
    latents_mean=(0.0, 0.1, -0.1, 0.2),
    latents_std=(1.0, 0.9, 1.1, 0.8),
)


def build_torch_model() -> AutoencoderKLMiniMaxH3:
    arch = MiniMaxH3VideoVAEArchConfig(**TINY_ARCH)
    config = MiniMaxH3VideoVAEConfig(arch_config=arch)
    config.use_tiling = False
    torch.manual_seed(SEED)
    model = AutoencoderKLMiniMaxH3(config)
    return model.float().eval()


def transfer_weights(model: AutoencoderKLMiniMaxH3) -> dict:
    """state_dict -> MLX weight dict (conv3d kernels transpose to channels-last)."""
    weights = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().float().numpy()
        if array.ndim == 5:  # conv3d (O, I, kT, kH, kW) -> (O, kT, kH, kW, I)
            array = np.ascontiguousarray(array.transpose(0, 2, 3, 4, 1))
        weights[name] = mx.array(np.ascontiguousarray(array))
        # The torch encoder keeps conv_shortcut absent when channels match.
    return weights


def build_mlx_model(model=None) -> MLXMiniMaxH3VideoVAE:
    model = model or build_torch_model()
    view = MiniMaxH3VideoVAEConfigView(
        in_channels=TINY_ARCH["in_channels"],
        out_channels=TINY_ARCH["out_channels"],
        latent_channels=TINY_ARCH["latent_channels"],
        block_out_channels=TINY_ARCH["block_out_channels"],
        layers_per_block=TINY_ARCH["layers_per_block"],
        spatial_downsample_factors=TINY_ARCH["spatial_downsample_factors"],
        temporal_downsample_factors=TINY_ARCH["temporal_downsample_factors"],
        norm_num_groups=TINY_ARCH["norm_num_groups"],
        norm_eps=TINY_ARCH["norm_eps"],
        decoder_num_layers=TINY_ARCH["decoder_num_layers"],
        decoder_num_attention_heads=TINY_ARCH["decoder_num_attention_heads"],
        decoder_attention_head_dim=TINY_ARCH["decoder_attention_head_dim"],
        decoder_num_register_tokens=TINY_ARCH["decoder_num_register_tokens"],
        decoder_ffn_mult=TINY_ARCH["decoder_ffn_mult"],
        decoder_rope_theta=TINY_ARCH["decoder_rope_theta"],
        decoder_rope_dim_ratio=TINY_ARCH["decoder_rope_dim_ratio"],
        decoder_norm_eps=TINY_ARCH["decoder_norm_eps"],
        clip_length=TINY_ARCH["clip_length"],
        token_drop=TINY_ARCH["token_drop"],
        latents_mean=TINY_ARCH["latents_mean"],
        latents_std=TINY_ARCH["latents_std"],
    )
    return MLXMiniMaxH3VideoVAE(transfer_weights(model), view)


def _stats(name: str, actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> None:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denom = np.linalg.norm(expected.astype(np.float64)) + 1e-12
    rel_l2 = float(np.linalg.norm(diff) / denom)
    print(f"[{name}] max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} rel_l2={rel_l2:.3e} "
          f"shape={actual.shape} dtype={actual.dtype}")
    assert diff.max() <= atol + rtol * np.abs(expected).max(), f"{name} drifted beyond tolerance"


def _to_np(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def test_rotary_embedding_matches_torch() -> None:
    from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoRotaryPosEmbed

    dim = int(TINY_ARCH["decoder_attention_head_dim"] * TINY_ARCH["decoder_rope_dim_ratio"])
    rope = MiniMaxH3VideoRotaryPosEmbed(dim, theta=TINY_ARCH["decoder_rope_theta"])
    rng = np.random.default_rng(SEED)
    positions = rng.standard_normal((1, 7, 3)).astype(np.float32) * 0.5
    cos_t, sin_t = rope(torch.from_numpy(positions))
    cos_t = cos_t.numpy()[0].reshape(7, 1, 1, -1)
    sin_t = sin_t.numpy()[0].reshape(7, 1, 1, -1)
    cos_m, sin_m = _rotary_cos_sin(mx.array(positions[0]), rotary_dim=dim,
                                   theta=TINY_ARCH["decoder_rope_theta"])
    _stats("rope.cos", _to_np(cos_m), cos_t, atol=2e-5, rtol=2e-5)
    _stats("rope.sin", _to_np(sin_m), sin_t, atol=2e-5, rtol=2e-5)


def test_reflect_padding_matches_numpy() -> None:
    from fastvideo.mlx_runtime.minimax_h3_video_vae import _reflect_pad_axis

    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4, 1)
    actual = np.asarray(_reflect_pad_axis(mx.array(values), axis=2, left=2, right=1))
    expected = np.pad(values, ((0, 0), (0, 0), (2, 1), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(actual, expected)


def test_encoder_moments_match_torch() -> None:
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 1)
    frames = rng.standard_normal((1, 3, 5, 16, 16)).astype(np.float32)  # one clip
    pixels = mx.array(frames)
    mean_m, logvar_m = vae.encode(pixels)
    with torch.no_grad():
        moments = model._encode(torch.from_numpy(frames))
    mean_t, logvar_t = torch.chunk(moments, 2, dim=1)
    logvar_t = logvar_t.clamp(-30.0, 20.0)
    _stats("encoder.mean", _to_np(mean_m), mean_t.numpy(), atol=2e-4, rtol=2e-4)
    _stats("encoder.logvar", _to_np(logvar_m), logvar_t.numpy(), atol=2e-4, rtol=2e-4)


def test_encode_keyframe_matches_torch() -> None:
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 2)
    frame = rng.standard_normal((1, 3, 1, 16, 16)).astype(np.float32)
    mean_m, logvar_m = vae.encode_keyframe(mx.array(frame))
    with torch.no_grad():
        moments = model.encode_keyframe(torch.from_numpy(frame)).latent_dist.parameters
    mean_t, logvar_t = torch.chunk(moments, 2, dim=1)
    logvar_t = logvar_t.clamp(-30.0, 20.0)
    _stats("keyframe.mean", _to_np(mean_m), mean_t.numpy(), atol=2e-4, rtol=2e-4)
    _stats("keyframe.logvar", _to_np(logvar_m), logvar_t.numpy(), atol=2e-4, rtol=2e-4)


def test_decoder_clip_matches_torch() -> None:
    """One clip through post_quant_conv + ViT decoder, unchunked both sides."""
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 3)
    z_np = rng.standard_normal((1, TINY_ARCH["latent_channels"], 3, 4, 4)).astype(np.float32)
    with torch.no_grad():
        ref = model.decoder(model.post_quant_conv(torch.from_numpy(z_np))).numpy()
    out = vae._decode_clip(mx.array(z_np))
    assert out.shape == ref.shape
    _stats("decoder.clip", _to_np(out), ref, atol=5e-4, rtol=5e-4)


def test_chunked_decode_matches_torch() -> None:
    """The full clip-chunking / overlap-blending decode path."""
    model = build_torch_model()
    vae = build_mlx_model(model)
    cfg = vae.config
    # decode() adds token_drop before chunking, so this hits the padding branch.
    num_tokens = 2 * cfg.tokens_chunk_size
    rng = np.random.default_rng(SEED + 4)
    z_np = rng.standard_normal((1, TINY_ARCH["latent_channels"], num_tokens, 4, 4)).astype(np.float32)
    with torch.no_grad():
        ref = model.decode(torch.from_numpy(z_np)).sample.numpy()
    out = vae.decode(mx.array(z_np))
    assert out.shape == ref.shape
    _stats("decoder.chunked", _to_np(out), ref, atol=5e-4, rtol=5e-4)


def test_tiled_decode_matches_untiled() -> None:
    vae = build_mlx_model()
    rng = np.random.default_rng(SEED + 5)
    z = mx.array(rng.standard_normal((1, TINY_ARCH["latent_channels"], 3, 4, 8)).astype(np.float32))
    untiled = vae._decode_clip(z)
    tiled = vae.decode_clip_tiled(
        z,
        tile_sample_min_height=untiled.shape[-2] // 2,
        tile_sample_min_width=untiled.shape[-1] // 2,
        tile_sample_min_overlap_height=vae.spatial_compression_ratio,
        tile_sample_min_overlap_width=vae.spatial_compression_ratio,
    )
    assert tiled.shape == untiled.shape
    _stats("tiled-vs-untiled", _to_np(tiled), _to_np(untiled), atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# Real released weights (bounded tile)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ROOT = Path(os.environ.get("MINIMAX_H3_MODEL_ROOT", Path.home() / "models/FastH3-Preview-v0.2"))
REAL_WEIGHTS_PRESENT = (DEFAULT_MODEL_ROOT / "vae").is_dir() and any(
    (DEFAULT_MODEL_ROOT / "vae").glob("*.safetensors"))


@pytest.mark.skipif(not REAL_WEIGHTS_PRESENT, reason="released H3 VAE snapshot not found")
def test_real_weights_bounded_tile_decode() -> None:
    from fastvideo.mlx_runtime.minimax_h3_video_vae import mlx_h3_video_vae_from_dir

    torch_vae = AutoencoderKLMiniMaxH3(MiniMaxH3VideoVAEConfig(arch_config=MiniMaxH3VideoVAEArchConfig()))
    index = json_load_index(DEFAULT_MODEL_ROOT / "vae")
    load_released_weights_into_torch(torch_vae, DEFAULT_MODEL_ROOT / "vae", index)
    torch_vae = torch_vae.float().eval()

    vae_dir = DEFAULT_MODEL_ROOT / "vae"
    mlx_vae = mlx_h3_video_vae_from_dir(vae_dir, include_encoder=False, storage_dtype="fp32")

    rng = np.random.default_rng(SEED + 6)
    z_np = rng.standard_normal((1, 24, 7, 4, 6)).astype(np.float32)
    with torch.no_grad():
        ref = torch_vae.decoder(torch_vae.post_quant_conv(torch.from_numpy(z_np))).numpy()
    out = mlx_vae._decode_clip(mx.array(z_np))
    _stats("real.decoder.tile(fp32)", _to_np(out), ref, atol=2e-4, rtol=2e-4)


def json_load_index(vae_dir: Path) -> dict:
    import json
    return json.loads((vae_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())


def load_released_weights_into_torch(model: AutoencoderKLMiniMaxH3, vae_dir: Path, index: dict) -> None:
    from safetensors.torch import safe_open

    weight_map = index["weight_map"]
    shards = {}
    state = {}
    for key, shard in weight_map.items():
        if not key.startswith(("encoder.", "quant_conv.", "post_quant_conv.", "decoder.")):
            continue
        if shard not in shards:
            handle = safe_open(str(Path(vae_dir) / shard), framework="pt", device="cpu")
            shards[shard] = handle
        state[key] = shards[shard].get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected]
    assert not unexpected, f"unexpected released keys: {unexpected[:5]}"
