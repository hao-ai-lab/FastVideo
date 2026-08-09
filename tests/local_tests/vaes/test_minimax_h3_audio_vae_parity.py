# Copyright 2025 MiniMax authors and HuggingFace Team
# SPDX-License-Identifier: Apache-2.0
"""Production-loader parity for the MiniMax H3 audio VAE."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.environ.get("MINIMAX_H3_OFFICIAL_REF_DIR", REPO_ROOT / "DiffusersMiniMaxH3"))
OFFICIAL_SRC = OFFICIAL_REF_DIR / "src"
MODEL_ROOT = Path(os.environ.get("MINIMAX_H3_MODEL_ROOT", REPO_ROOT / "official_weights" / "MiniMax-H3"))
RUN_ENV = "MINIMAX_H3_RUN_AUDIO_VAE_PARITY"
PARITY_SCOPE = "production_loader"

if os.environ.get(RUN_ENV) == "1":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29663")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if OFFICIAL_SRC.is_dir():
        sys.path.insert(0, str(OFFICIAL_SRC))


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    if os.environ.get(RUN_ENV) != "1":
        yield
        return

    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _require_assets() -> tuple[torch.device, Path]:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 on an allocated CUDA node")
    if not torch.cuda.is_available():
        pytest.fail("MiniMax H3 audio VAE parity requires CUDA", pytrace=False)
    if not OFFICIAL_SRC.is_dir():
        pytest.fail(f"pinned Diffusers MiniMax H3 checkout is missing: {OFFICIAL_REF_DIR}", pytrace=False)

    component_dir = MODEL_ROOT / "audio_vae"
    required = (component_dir / "config.json", component_dir / "diffusion_pytorch_model.safetensors")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.fail(f"MiniMax H3 audio VAE assets are missing: {missing}", pytrace=False)
    return torch.device("cuda:0"), component_dir


def _load_official(component_dir: Path, device: torch.device) -> torch.nn.Module:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    from tests.local_tests.minimax_h3._reference import assert_reference_source

    assert_reference_source(
        AutoencoderKLMiniMaxH3Audio,
        "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py",
    )

    model, loading_info = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        component_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        device_map=device,
        output_loading_info=True,
    )
    load_errors = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading_info.get(name)
    }
    assert not load_errors, f"official MiniMax H3 audio VAE did not load strictly: {load_errors}"
    model = model.eval()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    return model


def _load_fastvideo(component_dir: Path) -> torch.nn.Module:
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.models.loader.component_loader import AudioDecoderLoader

    args = SimpleNamespace(
        pipeline_config=MiniMaxH3PipelineConfig(),
        vae_cpu_offload=False,
    )
    model = AudioDecoderLoader().load(str(component_dir), args)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    return model


def _make_waveform() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    # An intentionally non-divisible length exercises the production right-pad
    # path for the model's 800-sample hop length.
    return torch.randn(1, 1, 1601, generator=generator, dtype=torch.float32)


def _run(model: torch.nn.Module, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        posterior = model.encode(waveform, return_dict=False)[0]
        latents = posterior.mode()
        decoded = model.decode(latents, return_dict=False)[0]
    return {
        "mean": posterior.mean.detach().cpu(),
        "logs": posterior.logs.detach().cpu(),
        "decoded": decoded.detach().cpu(),
    }


def _normalize_with_config(model: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(model.config.latents_mean, dtype=latents.dtype).view(1, -1, 1)
    std = torch.tensor(model.config.latents_std, dtype=latents.dtype).view(1, -1, 1)
    return (latents - mean) / std


def _reclaim_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _assert_tensor_parity(name: str, actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
    assert actual.shape == expected.shape
    drift = (actual.float() - expected.float()).abs()
    reference_abs_mean = expected.float().abs().mean().item()
    relative_mean_drift = drift.mean().item() / max(reference_abs_mean, 1e-8)
    print(
        f"{name}: reference_abs_mean={reference_abs_mean:.8f} "
        f"max_abs={drift.max().item():.8f} mean_abs={drift.mean().item():.8f} "
        f"relative_mean={relative_mean_drift:.8f}",
        flush=True,
    )
    assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_minimax_h3_audio_vae_parity() -> None:
    """Match posterior, normalization, hop geometry, and deterministic decode."""
    device, component_dir = _require_assets()
    waveform = _make_waveform()

    official = _load_official(component_dir, device)
    expected = _run(official, waveform.to(device))
    expected_normalized = _normalize_with_config(official, expected["mean"])
    del official
    _reclaim_vram()

    fastvideo = _load_fastvideo(component_dir)
    actual = _run(fastvideo, waveform.to(device))
    actual_normalized = _normalize_with_config(fastvideo, actual["mean"])
    assert fastvideo.hop_length == 800
    assert fastvideo.sampling_rate == 32000
    assert fastvideo.latent_channels == 32
    del fastvideo
    _reclaim_vram()

    _assert_tensor_parity("audio_vae.mean", actual["mean"], expected["mean"], 0.0)
    _assert_tensor_parity("audio_vae.logs", actual["logs"], expected["logs"], 0.0)
    _assert_tensor_parity("audio_vae.normalized", actual_normalized, expected_normalized, 0.0)
    # The official and native FP32 BigVGAN paths differ only in the final
    # weight-norm rounding order (observed max_abs=2.4e-7 on GB200).
    _assert_tensor_parity("audio_vae.decode", actual["decoded"], expected["decoded"], 5e-7)
