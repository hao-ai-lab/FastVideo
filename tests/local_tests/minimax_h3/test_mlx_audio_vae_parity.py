# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 audio VAE parity: FastVideo-native PyTorch reference vs MLX.

Tiny random-weight models exercise every primitive (Snake/SnakeBeta,
Kaiser-window up/down sampling, DAC encoder blocks, causal attention
projection, GeGLU MLP, BigVGAN AMP blocks); a real-weight bounded-segment
case validates the released FP32 checkpoint end to end.

    pytest tests/local_tests/minimax_h3/test_mlx_audio_vae_parity.py -v

The real-weight case activates automatically when the released snapshot is
present; point MINIMAX_H3_MODEL_ROOT at another checkout otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch reference needed for audio VAE parity")
mx = pytest.importorskip("mlx.core", reason="MLX needed for audio VAE parity")

from fastvideo.configs.models.vaes.minimax_h3_audio import (  # noqa: E402
    MiniMaxH3AudioVAEArchConfig,
    MiniMaxH3AudioVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_audio_vae import (  # noqa: E402
    MLXMiniMaxH3AudioVAE,
    MiniMaxH3AudioVAEConfigView,
    kaiser_sinc_filter1d,
)

SEED = 9112

# decoder_rates product must equal the encoder hop length (config invariant).
TINY_ARCH = dict(
    encoder_dim=8,
    encoder_rates=(2, 2),
    latent_dim=32,
    latent_channels=4,
    num_attention_heads=2,
    decoder_dim=16,
    decoder_rates=(2, 2),
    decoder_kernel_sizes=(5, 4),
    resblock_kernel_sizes=(3,),
    resblock_dilation_sizes=((1, 2),),
    sampling_rate=32000,
    latents_mean=[0.05 * i for i in range(4)],
    latents_std=[1.0 + 0.1 * i for i in range(4)],
)


def initialize_model_parameters(model: torch.nn.Module) -> None:
    """ReplicatedLinear parameters are allocated with torch.empty and need
    explicit initialization to avoid undefined (often zero) values."""
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    param.fill_(1.0)
                else:
                    param.normal_(mean=0.0, std=0.02)
                continue
            if "weight_v" in name or param.ndim >= 2:
                torch.nn.init.xavier_uniform_(param)


def build_torch_model() -> MiniMaxH3AudioVAE:
    arch = MiniMaxH3AudioVAEArchConfig(**TINY_ARCH)
    config = MiniMaxH3AudioVAEConfig(arch_config=arch)
    config.use_tiling = False
    torch.manual_seed(SEED)
    model = MiniMaxH3AudioVAE(config)
    initialize_model_parameters(model)
    return model.float().eval()


def build_mlx_model(model=None) -> MLXMiniMaxH3AudioVAE:
    model = model or build_torch_model()
    weights = {}
    for name, tensor in model.state_dict().items():
        weights[name] = mx.array(np.ascontiguousarray(tensor.detach().float().numpy()))
    view = MiniMaxH3AudioVAEConfigView(
        encoder_dim=TINY_ARCH["encoder_dim"],
        encoder_rates=TINY_ARCH["encoder_rates"],
        latent_dim=TINY_ARCH["latent_dim"],
        latent_channels=TINY_ARCH["latent_channels"],
        num_attention_heads=TINY_ARCH["num_attention_heads"],
        decoder_dim=TINY_ARCH["decoder_dim"],
        decoder_rates=TINY_ARCH["decoder_rates"],
        decoder_kernel_sizes=TINY_ARCH["decoder_kernel_sizes"],
        resblock_kernel_sizes=TINY_ARCH["resblock_kernel_sizes"],
        resblock_dilation_sizes=tuple(tuple(d) for d in TINY_ARCH["resblock_dilation_sizes"]),
        sampling_rate=TINY_ARCH["sampling_rate"],
        latents_mean=tuple(TINY_ARCH["latents_mean"]),
        latents_std=tuple(TINY_ARCH["latents_std"]),
    )
    return MLXMiniMaxH3AudioVAE(weights, view)


def _stats(name: str, actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> None:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denom = np.linalg.norm(expected.astype(np.float64)) + 1e-12
    print(f"[{name}] max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} "
          f"rel_l2={float(np.linalg.norm(diff)/denom):.3e} shape={actual.shape}")
    assert diff.max() <= atol + rtol * np.abs(expected).max(), f"{name} drifted beyond tolerance"


def test_kaiser_filter_matches_torch() -> None:
    from fastvideo.models.vaes.minimax_h3_audio import kaiser_sinc_filter1d as torch_filter

    for cutoff, half_width, size in ((0.25, 0.6, 12), (0.125, 0.15, 12), (0.5, 0.6, 13)):
        ref = torch_filter(cutoff, half_width, size).detach().numpy().reshape(-1)
        out = kaiser_sinc_filter1d(cutoff, half_width, size)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)


def test_snake_activations_match_torch() -> None:
    import mlx.core as mx_
    from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioSnakeBeta
    from fastvideo.mlx_runtime.minimax_h3_audio_vae import _snake_beta

    rng = np.random.default_rng(SEED)
    x = (rng.standard_normal((1, 6, 20)) * 0.5).astype(np.float32)

    mod = MiniMaxH3AudioSnakeBeta(6)
    with torch.no_grad():
        mod.alpha.fill_(0.1)
        mod.beta.fill_(-0.2)
        ref = mod(torch.from_numpy(x)).numpy()

    alpha = np.full((6, ), 0.1, dtype=np.float32)
    beta = np.full((6, ), -0.2, dtype=np.float32)
    got = np.asarray(_snake_beta(mx_.array(x), mx_.array(alpha), mx_.array(beta)))
    _stats("snake_beta", got, ref, atol=1e-5, rtol=1e-5)


def test_activation1d_matches_torch() -> None:
    """Alias-free SnakeBeta block with released-style filters."""
    from fastvideo.models.vaes.minimax_h3_audio import (
        MiniMaxH3AudioActivation1d,
        MiniMaxH3AudioDownSample1d,
        MiniMaxH3AudioSnakeBeta,
        MiniMaxH3AudioUpSample1d,
    )

    torch.manual_seed(SEED + 1)
    act = MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(4))
    with torch.no_grad():
        act.act.alpha.normal_(0.0, 0.1)
        act.act.beta.normal_(0.0, 0.1)
    rng = np.random.default_rng(SEED + 2)
    x = rng.standard_normal((1, 4, 50)).astype(np.float32)
    with torch.no_grad():
        ref = act(torch.from_numpy(x)).numpy()

    up_f = act.upsample.filter.detach().numpy().reshape(-1)
    down_f = act.downsample.lowpass.filter.detach().numpy().reshape(-1)
    alpha = act.act.alpha.detach().numpy().reshape(-1)
    beta = act.act.beta.detach().numpy().reshape(-1)

    from fastvideo.mlx_runtime.minimax_h3_audio_vae import _activation1d

    got = np.asarray(_activation1d(mx.array(x), mx.array(alpha), mx.array(beta),
                                   mx.array(up_f), mx.array(down_f)))
    _stats("activation1d", got, ref, atol=1e-4, rtol=1e-4)


def test_encode_moments_match_torch() -> None:
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 3)
    samples = 400  # exact multiple of hop 4
    wave = rng.standard_normal((1, 1, samples)).astype(np.float32) * 0.1
    mean_m, logvar_m = vae.encode(mx.array(wave))
    with torch.no_grad():
        post = model.encode(torch.from_numpy(wave)).latent_dist
    _stats("encode.mean", np.asarray(mean_m), post.mean.numpy(), atol=2e-4, rtol=2e-4)
    _stats("encode.logvar", np.asarray(logvar_m), post.logs.numpy(), atol=2e-4, rtol=2e-4)


def test_encode_right_pads_like_reference() -> None:
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 4)
    samples = 401  # forces right padding to 404
    wave = rng.standard_normal((1, 1, samples)).astype(np.float32) * 0.1
    mean_m, logvar_m = vae.encode(mx.array(wave))
    padded = torch.from_numpy(np.concatenate([wave, np.zeros((1, 1, 3), np.float32)], -1))
    with torch.no_grad():
        post = model.encode(padded).latent_dist
    _stats("encode.pad.mean", np.asarray(mean_m), post.mean.numpy(), atol=2e-4, rtol=2e-4)
    _stats("encode.pad.logvar", np.asarray(logvar_m), post.logs.numpy(), atol=2e-4, rtol=2e-4)


def test_decode_waveform_matches_torch() -> None:
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 5)
    z_np = rng.standard_normal((1, TINY_ARCH["latent_channels"], 12)).astype(np.float32) * 0.5
    with torch.no_grad():
        ref = model.decode(torch.from_numpy(z_np)).sample.numpy()
    out = vae.decode(mx.array(z_np))
    assert out.shape[-1] == ref.shape[-1]
    _stats("decode.wave", np.asarray(out), ref, atol=5e-4, rtol=5e-4)


def test_stereo_decode_is_independent_and_ordered() -> None:
    vae = build_mlx_model()
    rng = np.random.default_rng(SEED + 6)
    left = rng.standard_normal((1, TINY_ARCH["latent_channels"], 10)).astype(np.float32) * 0.5
    right = rng.standard_normal((1, TINY_ARCH["latent_channels"], 10)).astype(np.float32) * 0.5
    stereo = np.concatenate([left, right], 0)
    both = np.asarray(vae.decode(mx.array(stereo)))
    single_l = np.asarray(vae.decode(mx.array(left)))
    single_r = np.asarray(vae.decode(mx.array(right)))
    np.testing.assert_allclose(both[0], single_l[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(both[1], single_r[0], rtol=1e-5, atol=1e-6)


def test_latent_normalization_roundtrip() -> None:
    model = build_torch_model()
    vae = build_mlx_model(model)
    rng = np.random.default_rng(SEED + 7)
    latents = rng.standard_normal((1, TINY_ARCH["latent_channels"], 5)).astype(np.float32)
    normalized = np.asarray(vae.normalize_latents(mx.array(latents)))
    ref = ((torch.from_numpy(latents) - model.latents_mean) / model.latents_std).numpy()
    _stats("normalize", normalized, ref, atol=1e-5, rtol=1e-5)
    denorm = np.asarray(vae.denormalize_latents(mx.array(normalized)))
    _stats("denormalize", denorm, latents, atol=1e-5, rtol=1e-5)


def test_directory_loader_rejects_non_fp32_storage(tmp_path) -> None:
    from fastvideo.mlx_runtime.minimax_h3_audio_vae import mlx_h3_audio_vae_from_dir

    with pytest.raises(ValueError, match="storage_dtype='fp32'"):
        mlx_h3_audio_vae_from_dir(tmp_path, storage_dtype="bf16")


def test_audio_loader_validates_decoder_weights_before_construction(tmp_path) -> None:
    from fastvideo.mlx_runtime.minimax_h3_audio_vae import mlx_h3_audio_vae_from_file

    weights_path = tmp_path / "audio.safetensors"
    mx.save_safetensors(str(weights_path), {"unrelated": mx.zeros((1, ))})
    with pytest.raises(KeyError, match="dec_in_proj.weight"):
        mlx_h3_audio_vae_from_file(weights_path, include_encoder=False, config=build_mlx_model().config)


# ---------------------------------------------------------------------------
# Real released weights (bounded segment)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ROOT = Path(os.environ.get("MINIMAX_H3_MODEL_ROOT", Path.home() / "models/FastH3-Preview-v0.2"))
REAL_WEIGHTS_PRESENT = (DEFAULT_MODEL_ROOT / "audio_vae").is_dir() and any(
    (DEFAULT_MODEL_ROOT / "audio_vae").glob("*.safetensors"))


@pytest.mark.skipif(not REAL_WEIGHTS_PRESENT, reason="released H3 audio VAE snapshot not found")
def test_real_weights_bounded_segment_decode() -> None:
    from fastvideo.mlx_runtime.minimax_h3_audio_vae import mlx_h3_audio_vae_from_dir

    torch_vae = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=MiniMaxH3AudioVAEArchConfig())).float().eval()
    state = load_released_state(DEFAULT_MODEL_ROOT / "audio_vae")
    missing, unexpected = torch_vae.load_state_dict(state, strict=False)
    assert not [k for k in unexpected]

    mlx_vae = mlx_h3_audio_vae_from_dir(DEFAULT_MODEL_ROOT / "audio_vae")

    rng = np.random.default_rng(SEED + 8)
    z_np = rng.standard_normal((1, 32, 10)).astype(np.float32) * 0.5
    with torch.no_grad():
        ref = torch_vae.decode(torch.from_numpy(z_np)).sample.numpy()
    out = np.asarray(mlx_vae.decode(mx.array(z_np)))
    _stats("real.audio.segment(fp32)", out, ref, atol=2e-4, rtol=2e-4)


def load_released_state(component_dir: Path) -> dict:
    from safetensors.torch import safe_open

    state = {}
    index_path = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = {}
        for key, shard in weight_map.items():
            if shard not in shards:
                shards[shard] = safe_open(str(component_dir / shard), framework="pt", device="cpu")
            state[key] = shards[shard].get_tensor(key)
    else:
        with safe_open(str(component_dir / "diffusion_pytorch_model.safetensors"), framework="pt",
                       device="cpu") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
    return state
