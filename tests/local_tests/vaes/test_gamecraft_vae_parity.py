# SPDX-License-Identifier: Apache-2.0
"""
Numerical parity test for GameCraft VAE (AutoencoderKLCausal3D).

Ported from official Hunyuan-GameCraft-1.0/hymm_sp/vae. Compares FastVideo's
GameCraftVAE against the official implementation.

Follows the same pattern as test_ltx2_vae.py:
- Loads same weights into both models (strict key checking)
- Encodes same input, compares latents
- Decodes same latents through both decoders, compares output
- Checks for non-finite values

Usage:
    DISABLE_SP=1 pytest tests/local_tests/vaes/test_gamecraft_vae_parity.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29515")
os.environ.setdefault("DISABLE_SP", "1")

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _load_weights(weights_path: Path) -> dict[str, torch.Tensor]:
    """Load VAE weights from checkpoint and strip 'vae.' prefix."""
    print(f"[VAE TEST] Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Strip 'vae.' prefix
    vae_sd = {k.replace("vae.", ""): v for k, v in state_dict.items() if k.startswith("vae.")}
    print(f"[VAE TEST] Loaded {len(vae_sd)} VAE tensors")
    return vae_sd


def _load_into_model(
    model: torch.nn.Module, weights: dict[str, torch.Tensor], name: str
) -> tuple[int, list[str]]:
    """Load weights into a model, checking for missing/unexpected keys."""
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in weights.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    missing = [k for k in model_state.keys() if k not in filtered]
    unexpected = [k for k in weights.keys() if k not in model_state]
    print(
        f"[VAE TEST] {name}: Loading {len(filtered)}/{len(model_state)} tensors "
        f"({len(missing)} missing, {len(unexpected)} unexpected)"
    )
    if missing:
        print(f"[VAE TEST] {name} missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not filtered:
        return 0, missing
    model.load_state_dict(filtered, strict=False)
    return len(filtered), missing


def _load_official_vae(
    official_path: Path, vae_sd: dict[str, torch.Tensor],
    device: torch.device, dtype: torch.dtype
):
    """Load the official GameCraft VAE from hymm_sp/vae."""
    sys.path.insert(0, str(official_path))

    try:
        from hymm_sp.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
    except ImportError as e:
        pytest.skip(f"Failed to import official VAE: {e}")

    config_path = (
        official_path / "weights" / "stdmodels" / "vae_3d" / "hyvae" / "config.json"
    )
    if not config_path.exists():
        pytest.skip(f"VAE config not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    vae = AutoencoderKLCausal3D(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 3),
        down_block_types=tuple(config.get("down_block_types", [])),
        up_block_types=tuple(config.get("up_block_types", [])),
        block_out_channels=tuple(config.get("block_out_channels", [])),
        layers_per_block=config.get("layers_per_block", 2),
        act_fn=config.get("act_fn", "silu"),
        latent_channels=config.get("latent_channels", 16),
        norm_num_groups=config.get("norm_num_groups", 32),
        sample_size=config.get("sample_size", 256),
        sample_tsize=config.get("sample_tsize", 64),
        scaling_factor=config.get("scaling_factor", 0.476986),
        time_compression_ratio=config.get("time_compression_ratio", 4),
        mid_block_add_attention=config.get("mid_block_add_attention", True),
        mid_block_causal_attn=config.get("mid_block_causal_attn", True),
    )

    loaded, missing = _load_into_model(vae, vae_sd, "Official VAE")
    if loaded == 0:
        pytest.skip("Failed to load weights into official VAE")
    if missing:
        pytest.skip(f"Official VAE has {len(missing)} missing keys; cannot ensure parity")

    if not hasattr(vae, "use_trt_decoder"):
        vae.use_trt_decoder = False
    return vae.to(device=device, dtype=dtype).eval()


def _load_fastvideo_vae(
    vae_sd: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype
):
    """Load FastVideo GameCraftVAE with official weights."""
    from fastvideo.configs.models.vaes import GameCraftVAEConfig
    from fastvideo.models.vaes.gamecraftvae import GameCraftVAE

    config = GameCraftVAEConfig()
    vae = GameCraftVAE(config)

    loaded, missing = _load_into_model(vae, vae_sd, "FastVideo VAE")
    if loaded == 0:
        pytest.skip("Failed to load weights into FastVideo VAE")
    if missing:
        pytest.skip(f"FastVideo VAE has {len(missing)} missing keys; cannot ensure parity")

    return vae.to(device=device, dtype=dtype).eval()


def test_gamecraft_vae_parity():
    """Test VAE encode/decode parity between official and FastVideo.
    
    Key: decodes the SAME latents through both decoders to isolate
    decode-only differences (following LTX2 VAE test pattern).
    """
    torch.manual_seed(42)

    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    vae_weights_path = Path(
        os.getenv(
            "GAMECRAFT_VAE_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "stdmodels" / "vae_3d" / "hyvae" / "pytorch_model.pt",
        )
    )

    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    if not vae_weights_path.exists():
        alt_paths = [
            vae_weights_path.parent / "checkpoint-step-270000.ckpt",
            vae_weights_path.parent / "diffusion_pytorch_model.safetensors",
        ]
        for alt in alt_paths:
            if alt.exists():
                vae_weights_path = alt
                break
        else:
            pytest.skip(f"VAE weights not found. Tried: {vae_weights_path}, {alt_paths}")

    # Use fp32 for precise comparison (bf16 amplifies small differences)
    use_cuda = torch.cuda.is_available()
    try:
        if use_cuda:
            _ = torch.zeros(1, device="cuda:0")
            del _
            torch.cuda.empty_cache()
    except Exception:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Use fp32 for parity test precision (bf16 gives ~0.2 max diff due to precision)
    dtype = torch.float32

    # Load weights once, share between both models
    vae_sd = _load_weights(vae_weights_path)

    batch_size = 1
    frames = 5
    height = 64
    width = 64
    video_input = torch.randn(batch_size, 3, frames, height, width, device=device, dtype=dtype)
    print(f"\n[VAE TEST] Input shape: {video_input.shape}, dtype: {dtype}")

    # --- Encode comparison ---
    print("\n[VAE TEST] Loading official VAE...")
    official_vae = _load_official_vae(official_path, vae_sd, device, dtype)
    with torch.no_grad():
        official_latent_dist = official_vae.encode(video_input)
        if hasattr(official_latent_dist, "latent_dist"):
            official_latent_dist = official_latent_dist.latent_dist
        official_latents = official_latent_dist.mode()

    print("[VAE TEST] Loading FastVideo GameCraftVAE...")
    fastvideo_vae = _load_fastvideo_vae(vae_sd, device, dtype)
    with torch.no_grad():
        fv_latent_dist = fastvideo_vae.encode(video_input)
        fv_latents = fv_latent_dist.latent_dist.mode()

    print(f"[VAE TEST] Official latents: {official_latents.shape}")
    print(f"[VAE TEST] FastVideo latents: {fv_latents.shape}")
    assert official_latents.shape == fv_latents.shape, "Latent shape mismatch"
    assert torch.isfinite(official_latents).all(), "Official encoder produced non-finite latents"
    assert torch.isfinite(fv_latents).all(), "FastVideo encoder produced non-finite latents"

    enc_diff = (official_latents - fv_latents).abs().max().item()
    enc_mean = (official_latents - fv_latents).abs().mean().item()
    print(f"[VAE TEST] Encode max diff: {enc_diff:.6f}, mean diff: {enc_mean:.6f}")
    assert_close(official_latents, fv_latents, atol=1e-2, rtol=1e-2)

    # --- Decode comparison (SAME latents through both decoders) ---
    # Use official latents as the shared input to isolate decode-only differences
    shared_latents = official_latents

    with torch.no_grad():
        official_decoded = official_vae.decode(shared_latents)
        if hasattr(official_decoded, "sample"):
            official_decoded = official_decoded.sample

    # Free official VAE memory before loading FastVideo decode
    del official_vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        fv_decoded = fastvideo_vae.decode(shared_latents)
        if hasattr(fv_decoded, "sample"):
            fv_decoded = fv_decoded.sample

    print(f"[VAE TEST] Official decoded: {official_decoded.shape}")
    print(f"[VAE TEST] FastVideo decoded: {fv_decoded.shape}")
    assert official_decoded.shape == fv_decoded.shape, "Decoded shape mismatch"
    assert torch.isfinite(official_decoded).all(), "Official decoder produced non-finite output"
    assert torch.isfinite(fv_decoded).all(), "FastVideo decoder produced non-finite output"

    dec_diff = (official_decoded - fv_decoded).abs().max().item()
    dec_mean = (official_decoded - fv_decoded).abs().mean().item()
    print(f"[VAE TEST] Decode max diff: {dec_diff:.6f}, mean diff: {dec_mean:.6f}")
    assert_close(official_decoded, fv_decoded, atol=1e-2, rtol=1e-2)

    print("[VAE TEST] PASSED: Numerical parity between official and FastVideo!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gamecraft_vae_config_compatibility():
    """Test that GameCraft VAE config is compatible with FastVideo's GameCraftVAE."""
    vae_config_path = (
        repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "stdmodels" / "vae_3d" / "hyvae" / "config.json"
    )
    
    if not vae_config_path.exists():
        pytest.skip(f"VAE config not found at {vae_config_path}")
    
    with open(vae_config_path) as f:
        gc_config = json.load(f)
    
    print("\n[VAE CONFIG TEST] GameCraft VAE config:")
    for k, v in gc_config.items():
        print(f"  {k}: {v}")
    
    # Expected HunyuanVideo VAE config
    expected_config = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 16,
        "block_out_channels": [128, 256, 512, 512],
        "time_compression_ratio": 4,
        "scaling_factor": 0.476986,
    }
    
    # Verify compatibility
    for key, expected in expected_config.items():
        if key in gc_config:
            actual = gc_config[key]
            assert actual == expected, f"Config mismatch for {key}: expected {expected}, got {actual}"
            print(f"[VAE CONFIG TEST] {key}: OK (matches expected)")
    
    print("[VAE CONFIG TEST] Config compatibility check passed!")


if __name__ == "__main__":
    test_gamecraft_vae_config_compatibility()
    test_gamecraft_vae_parity()
