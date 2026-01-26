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
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import UpsamplerLoader


def _load_metadata(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if not meta or "config" not in meta:
        raise KeyError("Missing config metadata in safetensors file.")
    return json.loads(meta["config"])


def test_ltx2_upsampler_parity():
    official_path = Path(
        os.getenv(
            "LTX2_UPSAMPLER_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        )
    )
    fastvideo_path = Path(
        os.getenv(
            "LTX2_UPSAMPLER_PATH",
            "converted/ltx2_spatial_upscaler",
        )
    )

    if not official_path.exists():
        pytest.skip(f"LTX-2 upsampler weights not found at {official_path}")
    if not fastvideo_path.exists():
        pytest.skip(f"FastVideo upsampler weights not found at {fastvideo_path}")

    try:
        from ltx_core.model.upsampler import LatentUpsamplerConfigurator
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    config = _load_metadata(official_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    precision_str = "bf16" if torch.cuda.is_available() else "fp32"

    ref_model = LatentUpsamplerConfigurator.from_config(config).to(
        device=device, dtype=precision
    )
    ref_weights = load_file(str(official_path))
    ref_model.load_state_dict(ref_weights, strict=False)

    args = FastVideoArgs(
        model_path=str(fastvideo_path),
        pipeline_config=PipelineConfig(vae_precision=precision_str),
    )
    loader = UpsamplerLoader()
    fastvideo_model = loader.load(str(fastvideo_path), args).to(
        device=device, dtype=precision
    )

    ref_model.eval()
    fastvideo_model.eval()

    batch = 1
    channels = config.get("in_channels", 128)
    frames = 3
    height = 8
    width = 8
    latent = torch.randn(
        batch,
        channels,
        frames,
        height,
        width,
        device=device,
        dtype=precision,
    )

    with torch.no_grad():
        ref_out = ref_model(latent)
        fast_out = fastvideo_model(latent)

    assert ref_out.shape == fast_out.shape
    assert ref_out.dtype == fast_out.dtype
    assert torch.isfinite(ref_out).all(), "Reference upsampler produced non-finite output."
    assert torch.isfinite(fast_out).all(), "FastVideo upsampler produced non-finite output."
    assert_close(ref_out, fast_out, atol=1e-4, rtol=1e-4)
