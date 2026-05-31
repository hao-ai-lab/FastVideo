# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

repo_root = Path(__file__).resolve().parents[3]
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.configs.models.vaes import LTX2VAEConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import VAELoader


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LTX-2 VAE parity test requires CUDA.",
)
def test_ltx2_vae_parity_official():
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

    try:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.video_vae import (
            VAE_DECODER_COMFY_KEYS_FILTER,
            VAE_ENCODER_COMFY_KEYS_FILTER,
            VideoDecoderConfigurator,
            VideoEncoderConfigurator,
        )
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=str(fastvideo_path),
        vae_cpu_offload=False,
        pipeline_config=PipelineConfig(
            vae_config=LTX2VAEConfig(),
            vae_precision=precision_str,
        ),
    )

    loader = VAELoader()
    fastvideo_vae = loader.load(str(fastvideo_path), args).to(
        device=device, dtype=precision
    )

    encoder_builder = SingleGPUModelBuilder(
        model_class_configurator=VideoEncoderConfigurator,
        model_path=str(official_path),
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    )
    decoder_builder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=str(official_path),
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    )
    ref_encoder = encoder_builder.build(
        device=device, dtype=precision
    ).to(device=device, dtype=precision)
    ref_decoder = decoder_builder.build(
        device=device, dtype=precision
    ).to(device=device, dtype=precision)

    fastvideo_vae.encoder.eval()
    fastvideo_vae.decoder.eval()
    ref_encoder.eval()
    ref_decoder.eval()

    if hasattr(fastvideo_vae.decoder, "decode_noise_scale"):
        fastvideo_vae.decoder.decode_noise_scale = 0.0
    if hasattr(ref_decoder, "decode_noise_scale"):
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
        fast_latents = fastvideo_vae.encoder(video)

    assert ref_latents.shape == fast_latents.shape
    assert ref_latents.dtype == fast_latents.dtype
    assert torch.isfinite(ref_latents).all(), "Reference encoder produced non-finite latents."
    assert torch.isfinite(fast_latents).all(), "FastVideo encoder produced non-finite latents."
    assert_close(ref_latents, fast_latents, atol=1e-2, rtol=1e-2)

    timestep = torch.tensor([0.05], device=device, dtype=precision)
    with torch.no_grad():
        ref_decoded = ref_decoder(ref_latents, timestep=timestep)
        fast_decoded = fastvideo_vae.decoder(fast_latents, timestep=timestep)

    assert ref_decoded.shape == fast_decoded.shape
    assert ref_decoded.dtype == fast_decoded.dtype
    assert torch.isfinite(ref_decoded).all(), "Reference decoder produced non-finite output."
    assert torch.isfinite(fast_decoded).all(), "FastVideo decoder produced non-finite output."
    assert_close(ref_decoded, fast_decoded, atol=1e-2, rtol=1e-2)
