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

from fastvideo.models.audio.ltx2_audio_vae import (
    LTX2AudioDecoder,
    LTX2AudioEncoder,
    LTX2Vocoder,
)


def _load_metadata(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if not meta or "config" not in meta:
        raise KeyError("Missing config metadata in safetensors file.")
    return json.loads(meta["config"])


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    print(f"[LTX2 AUDIO VAE TEST] Loading weights from {path}")
    return load_file(str(path))


def _select_audio_vae_weights(
    weights: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    alt_prefix = prefix.replace("audio_vae.", "")
    for name, tensor in weights.items():
        if name.startswith(prefix):
            filtered[name.replace(prefix, "")] = tensor
        elif alt_prefix and name.startswith(alt_prefix):
            filtered[name.replace(alt_prefix, "")] = tensor
        elif name.startswith("audio_vae.per_channel_statistics."):
            filtered[name.replace("audio_vae.", "")] = tensor
        elif name.startswith("per_channel_statistics."):
            filtered[name] = tensor
    print(f"[LTX2 AUDIO VAE TEST] Selected {len(filtered)} tensors for {prefix}")
    return filtered


def _select_vocoder_weights(
    weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if any(name.startswith("vocoder.") for name in weights):
        filtered = {
            name.replace("vocoder.", ""): tensor
            for name, tensor in weights.items()
            if name.startswith("vocoder.")
        }
    else:
        filtered = dict(weights)
    print(f"[LTX2 AUDIO VAE TEST] Selected {len(filtered)} tensors for vocoder.")
    return filtered


def _load_into_model(
    model: torch.nn.Module, weights: dict[str, torch.Tensor]
) -> tuple[int, list[str]]:
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in weights.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    missing = [k for k in model_state.keys() if k not in filtered]
    print(
        f"[LTX2 AUDIO VAE TEST] Loading {len(filtered)} / {len(model_state)} tensors "
        f"from {len(weights)} available"
    )
    if not filtered:
        return 0, missing
    model.load_state_dict(filtered, strict=False)
    return len(filtered), missing


def test_ltx2_audio_vae_vocoder_parity():
    diffusers_root = Path(
        os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    )
    official_path = Path(
        os.getenv(
            "LTX2_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-19b-distilled.safetensors",
        )
    )
    audio_vae_path = Path(
        os.getenv("LTX2_AUDIO_VAE_PATH", str(diffusers_root / "audio_vae"))
    )
    vocoder_path = Path(
        os.getenv("LTX2_VOCODER_PATH", str(diffusers_root / "vocoder"))
    )
    if not official_path.exists():
        pytest.skip(f"LTX-2 weights not found at {official_path}")
    if not audio_vae_path.exists():
        pytest.skip(f"LTX-2 audio VAE weights not found at {audio_vae_path}")
    if not vocoder_path.exists():
        pytest.skip(f"LTX-2 vocoder weights not found at {vocoder_path}")

    config = _load_metadata(official_path)
    if "audio_vae" not in config or "vocoder" not in config:
        pytest.skip("Audio VAE or vocoder config not found in safetensors metadata.")

    try:
        from ltx_core.model.audio_vae import (
            AudioDecoderConfigurator,
            AudioEncoderConfigurator,
            VocoderConfigurator,
        )
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    ref_weights = _load_weights(official_path)
    encoder_weights = _select_audio_vae_weights(ref_weights, "audio_vae.encoder.")
    decoder_weights = _select_audio_vae_weights(ref_weights, "audio_vae.decoder.")
    vocoder_weights = _select_vocoder_weights(ref_weights)
    if not encoder_weights or not decoder_weights or not vocoder_weights:
        pytest.skip("Audio VAE or vocoder weights not found in safetensors file.")

    fastvideo_audio_weights_path = audio_vae_path / "model.safetensors"
    fastvideo_vocoder_weights_path = vocoder_path / "model.safetensors"
    if not fastvideo_audio_weights_path.exists():
        pytest.skip(
            f"FastVideo audio VAE weights not found at {fastvideo_audio_weights_path}"
        )
    if not fastvideo_vocoder_weights_path.exists():
        pytest.skip(
            f"FastVideo vocoder weights not found at {fastvideo_vocoder_weights_path}"
        )
    fastvideo_audio_weights = _load_weights(fastvideo_audio_weights_path)
    fastvideo_encoder_weights = _select_audio_vae_weights(
        fastvideo_audio_weights, "encoder."
    )
    fastvideo_decoder_weights = _select_audio_vae_weights(
        fastvideo_audio_weights, "decoder."
    )
    fastvideo_vocoder_weights = _select_vocoder_weights(
        _load_weights(fastvideo_vocoder_weights_path)
    )
    if (not fastvideo_encoder_weights or not fastvideo_decoder_weights
            or not fastvideo_vocoder_weights):
        pytest.skip("FastVideo audio VAE/vocoder weights not found in diffusers files.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    fastvideo_encoder = LTX2AudioEncoder(config).to(device=device, dtype=precision)
    fastvideo_decoder = LTX2AudioDecoder(config).to(device=device, dtype=precision)
    fastvideo_vocoder = LTX2Vocoder(config).to(device=device, dtype=precision)

    ref_encoder = AudioEncoderConfigurator.from_config(config).to(
        device=device, dtype=precision
    )
    ref_decoder = AudioDecoderConfigurator.from_config(config).to(
        device=device, dtype=precision
    )
    ref_vocoder = VocoderConfigurator.from_config(config).to(
        device=device, dtype=precision
    )

    loaded_fastvideo_encoder, missing_fastvideo_encoder = _load_into_model(
        fastvideo_encoder.model, fastvideo_encoder_weights
    )
    loaded_ref_encoder, missing_ref_encoder = _load_into_model(
        ref_encoder, encoder_weights
    )
    loaded_fastvideo_decoder, missing_fastvideo_decoder = _load_into_model(
        fastvideo_decoder.model, fastvideo_decoder_weights
    )
    loaded_ref_decoder, missing_ref_decoder = _load_into_model(
        ref_decoder, decoder_weights
    )
    loaded_fastvideo_vocoder, missing_fastvideo_vocoder = _load_into_model(
        fastvideo_vocoder.model, fastvideo_vocoder_weights
    )
    loaded_ref_vocoder, missing_ref_vocoder = _load_into_model(
        ref_vocoder, vocoder_weights
    )

    if min(
        loaded_fastvideo_encoder,
        loaded_ref_encoder,
        loaded_fastvideo_decoder,
        loaded_ref_decoder,
        loaded_fastvideo_vocoder,
        loaded_ref_vocoder,
    ) == 0:
        pytest.skip("Failed to load audio VAE or vocoder weights.")
    if (
        missing_fastvideo_encoder
        or missing_ref_encoder
        or missing_fastvideo_decoder
        or missing_ref_decoder
        or missing_fastvideo_vocoder
        or missing_ref_vocoder
    ):
        print(
            f"[LTX2 AUDIO VAE TEST] Missing encoder keys: {len(missing_fastvideo_encoder)}"
        )
        print(
            f"[LTX2 AUDIO VAE TEST] Missing decoder keys: {len(missing_fastvideo_decoder)}"
        )
        print(
            f"[LTX2 AUDIO VAE TEST] Missing vocoder keys: {len(missing_fastvideo_vocoder)}"
        )
        pytest.skip("Missing audio VAE/vocoder keys; cannot ensure parity.")

    fastvideo_encoder.model.eval()
    fastvideo_decoder.model.eval()
    fastvideo_vocoder.model.eval()
    ref_encoder.eval()
    ref_decoder.eval()
    ref_vocoder.eval()

    ddconfig = config["audio_vae"]["model"]["params"]["ddconfig"]
    in_channels = ddconfig.get("in_channels", 2)
    resolution = ddconfig.get("resolution", 256)
    mel_bins = ddconfig.get("mel_bins", 64)
    batch_size = 1
    spectrogram = torch.randn(
        batch_size,
        in_channels,
        resolution,
        mel_bins,
        device=device,
        dtype=precision,
    )

    with torch.no_grad():
        ref_latents = ref_encoder(spectrogram)
        fast_latents = fastvideo_encoder(spectrogram)

    assert ref_latents.shape == fast_latents.shape
    assert ref_latents.dtype == fast_latents.dtype
    assert torch.isfinite(ref_latents).all(), "Reference encoder produced non-finite latents."
    assert torch.isfinite(fast_latents).all(), "FastVideo encoder produced non-finite latents."
    assert_close(ref_latents, fast_latents, atol=1e-2, rtol=1e-2)

    with torch.no_grad():
        ref_decoded = ref_decoder(ref_latents)
        fast_decoded = fastvideo_decoder(ref_latents)

    assert ref_decoded.shape == fast_decoded.shape
    assert ref_decoded.dtype == fast_decoded.dtype
    assert torch.isfinite(ref_decoded).all(), "Reference decoder produced non-finite output."
    assert torch.isfinite(fast_decoded).all(), "FastVideo decoder produced non-finite output."
    assert_close(ref_decoded, fast_decoded, atol=1e-2, rtol=1e-2)

    with torch.no_grad():
        ref_audio = ref_vocoder(ref_decoded)
        fast_audio = fastvideo_vocoder(ref_decoded)

    assert ref_audio.shape == fast_audio.shape
    assert ref_audio.dtype == fast_audio.dtype
    assert torch.isfinite(ref_audio).all(), "Reference vocoder produced non-finite audio."
    assert torch.isfinite(fast_audio).all(), "FastVideo vocoder produced non-finite audio."
    assert_close(ref_audio, fast_audio, atol=1e-2, rtol=1e-2)
