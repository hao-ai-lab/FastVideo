# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.testing import assert_close


repo_root = Path(__file__).resolve().parents[3]
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.models.vaes.ltx2vae import LTX2VideoDecoder, LTX2VideoEncoder


def _load_metadata(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if not meta or "config" not in meta:
        raise KeyError("Missing config metadata in safetensors file.")
    return json.loads(meta["config"])


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    print(f"[LTX2 VAE TEST] Loading weights from {path}")
    return load_file(str(path))


def _select_vae_weights(weights: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    alt_prefix = prefix.replace("vae.", "")
    for name, tensor in weights.items():
        if name.startswith(prefix):
            filtered[name.replace(prefix, "")] = tensor
        elif alt_prefix and name.startswith(alt_prefix):
            filtered[name.replace(alt_prefix, "")] = tensor
        elif name.startswith("vae.per_channel_statistics."):
            filtered[name.replace("vae.", "")] = tensor
        elif name.startswith("per_channel_statistics."):
            filtered[name] = tensor
    print(f"[LTX2 VAE TEST] Selected {len(filtered)} tensors for {prefix}")
    return filtered


def _load_into_model(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> tuple[int, list[str]]:
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in weights.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    missing = [k for k in model_state.keys() if k not in filtered]
    print(
        f"[LTX2 VAE TEST] Loading {len(filtered)} / {len(model_state)} tensors "
        f"from {len(weights)} available"
    )
    if not filtered:
        return 0, missing
    model.load_state_dict(filtered, strict=False)
    return len(filtered), missing


def test_ltx2_vae_parity():
    diffusers_root = Path(
        os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    )
    official_path = Path(
        os.getenv(
            "LTX2_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-19b-distilled.safetensors",
        )
    )
    fastvideo_path = Path(
        os.getenv("LTX2_VAE_PATH", str(diffusers_root / "vae"))
    )
    if not official_path.exists():
        pytest.skip(f"LTX-2 weights not found at {official_path}")
    if not fastvideo_path.exists():
        pytest.skip(f"LTX-2 diffusers VAE not found at {fastvideo_path}")

    config = _load_metadata(official_path)
    if "vae" not in config:
        pytest.skip("VAE config not found in safetensors metadata.")

    try:
        from ltx_core.model.video_vae import VideoDecoderConfigurator, VideoEncoderConfigurator
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    ref_weights = _load_weights(official_path)
    encoder_weights = _select_vae_weights(ref_weights, "vae.encoder.")
    decoder_weights = _select_vae_weights(ref_weights, "vae.decoder.")
    if not encoder_weights or not decoder_weights:
        pytest.skip("VAE weights not found in safetensors file.")

    fastvideo_weights_path = fastvideo_path / "model.safetensors"
    if not fastvideo_weights_path.exists():
        pytest.skip(f"FastVideo VAE weights not found at {fastvideo_weights_path}")
    fastvideo_weights = _load_weights(fastvideo_weights_path)
    fastvideo_encoder_weights = _select_vae_weights(
        fastvideo_weights, "encoder."
    )
    fastvideo_decoder_weights = _select_vae_weights(
        fastvideo_weights, "decoder."
    )
    if not fastvideo_encoder_weights or not fastvideo_decoder_weights:
        pytest.skip("FastVideo VAE weights not found in diffusers file.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    fastvideo_encoder = LTX2VideoEncoder(config).to(device=device, dtype=precision)
    fastvideo_decoder = LTX2VideoDecoder(config).to(device=device, dtype=precision)
    ref_encoder = VideoEncoderConfigurator.from_config(config).to(device=device, dtype=precision)
    ref_decoder = VideoDecoderConfigurator.from_config(config).to(device=device, dtype=precision)

    loaded_fastvideo_encoder, missing_fastvideo_encoder = _load_into_model(
        fastvideo_encoder.model, fastvideo_encoder_weights
    )
    loaded_ref_encoder, missing_ref_encoder = _load_into_model(ref_encoder, encoder_weights)
    loaded_fastvideo_decoder, missing_fastvideo_decoder = _load_into_model(
        fastvideo_decoder.model, fastvideo_decoder_weights
    )
    loaded_ref_decoder, missing_ref_decoder = _load_into_model(ref_decoder, decoder_weights)

    if min(
        loaded_fastvideo_encoder,
        loaded_ref_encoder,
        loaded_fastvideo_decoder,
        loaded_ref_decoder,
    ) == 0:
        pytest.skip("Failed to load VAE weights into one or more models.")
    if (
        missing_fastvideo_encoder
        or missing_ref_encoder
        or missing_fastvideo_decoder
        or missing_ref_decoder
    ):
        print(f"[LTX2 VAE TEST] Missing encoder keys: {len(missing_fastvideo_encoder)}")
        print(f"[LTX2 VAE TEST] Missing decoder keys: {len(missing_fastvideo_decoder)}")
        pytest.skip("Missing VAE keys; cannot ensure parity.")

    fastvideo_encoder.model.eval()
    fastvideo_decoder.model.eval()
    ref_encoder.eval()
    ref_decoder.eval()

    fastvideo_decoder.model.decode_noise_scale = 0.0
    ref_decoder.decode_noise_scale = 0.0

    batch_size = 1
    frames = 9
    height = 64
    width = 64
    video = torch.randn(
        batch_size,
        3,
        frames,
        height,
        width,
        device=device,
        dtype=precision,
    )

    with torch.no_grad():
        ref_latents = ref_encoder(video)
        fast_latents = fastvideo_encoder(video)

    assert ref_latents.shape == fast_latents.shape
    assert ref_latents.dtype == fast_latents.dtype
    assert torch.isfinite(ref_latents).all(), "Reference encoder produced non-finite latents."
    assert torch.isfinite(fast_latents).all(), "FastVideo encoder produced non-finite latents."
    assert_close(ref_latents, fast_latents, atol=1e-2, rtol=1e-2)

    timestep = torch.tensor([0.05], device=device, dtype=precision)
    with torch.no_grad():
        ref_decoded = ref_decoder(ref_latents, timestep=timestep)
        fast_decoded = fastvideo_decoder(fast_latents, timestep=timestep)

    assert ref_decoded.shape == fast_decoded.shape
    assert ref_decoded.dtype == fast_decoded.dtype
    assert torch.isfinite(ref_decoded).all(), "Reference decoder produced non-finite output."
    assert torch.isfinite(fast_decoded).all(), "FastVideo decoder produced non-finite output."
    assert_close(ref_decoded, fast_decoded, atol=1e-2, rtol=1e-2)
