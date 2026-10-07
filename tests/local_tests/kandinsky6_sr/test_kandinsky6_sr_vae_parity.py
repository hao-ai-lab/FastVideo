# SPDX-License-Identifier: Apache-2.0
"""Bitwise parity of the native ``Kandinsky6SRVAE`` against a reference checkout of the previous port, on the real
checkpoint (CUDA).

The reference is another FastVideo checkout (its package is also named ``fastvideo``), so it runs in a subprocess with
``PYTHONPATH`` pointing at it and saves its outputs to a ``.pt`` file; this process runs the native VAE and requires
``torch.equal`` on every output.  Cases follow the pipeline's use: whole-clip encode in ``encode_lq_video_to_lr_latent``
style (uint8 -> float -> ``normalize_data`` -> VAE dtype -> ``encode(x)[0]``) and ``decode(z).sample``, in fp32 and bf16
(the pipeline's ``vae_precision``) at an LQ-sized clip, plus a bf16 4x-SR-sized tile (832x1536) whose decoder
convolutions exceed the chunked-conv threshold.

Environment variables (the test skips without the weights or the reference checkout):

    KANDINSKY6_SR_VAE_DIR        ``vae/`` of an official Diffusers SR repo (config.json + safetensors)
    KANDINSKY6_SR_VAE_REFERENCE  root of the reference FastVideo checkout

GPU: needs ~80 GB free for the 832x1536 bf16 decode (run on a GB200 / H200 class device).
Run: ``pytest tests/local_tests/kandinsky6_sr/test_kandinsky6_sr_vae_parity.py -v -s``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

VAE_DIR = Path(os.environ.get("KANDINSKY6_SR_VAE_DIR", "/nonexistent"))
REFERENCE_ROOT = Path(os.environ.get("KANDINSKY6_SR_VAE_REFERENCE", "/nonexistent"))

# (name, dtype, frames, height, width): 33 frames = two encode segments (17 + 16) and two decode segments (5 + 4).
CASES = [
    ("lq_fp32", torch.float32, 33, 208, 384),
    ("lq_bf16", torch.bfloat16, 33, 208, 384),
    ("tile4x_bf16", torch.bfloat16, 33, 832, 1536),
]


def make_clip(frames: int, height: int, width: int) -> torch.Tensor:
    """Deterministic ``[T, 3, H, W]`` uint8 clip with moving structure and texture (not flat, not pure noise)."""
    t = torch.arange(frames, dtype=torch.float32)[:, None, None]
    y = torch.linspace(0, 1, height)[None, :, None]
    x = torch.linspace(0, 1, width)[None, None, :]
    channels = [
        128 + 100 * torch.sin(12 * x + 0.3 * t) * torch.cos(7 * y - 0.2 * t),
        128 + 90 * torch.cos(20 * (x + y) - 0.4 * t),
        255 * ((x - 0.5 - 0.01 * t)**2 + (y - 0.5)**2 < 0.05).float() + 20 * torch.sin(60 * x * y + t),
    ]
    noise = torch.randn((frames, 3, height, width), generator=torch.Generator().manual_seed(0)) * 8
    return (torch.stack(channels, dim=1) + noise).clamp(0, 255).to(torch.uint8)


def load_vae(vae_dir: Path, device: str):
    """Build ``Kandinsky6SRVAE`` from ``config.json`` like ``VAELoader`` and load the checkpoint strictly (fp32)."""
    from safetensors.torch import load_file

    from fastvideo.configs.models.vaes.kandinsky6_sr import Kandinsky6SRVAEConfig
    from fastvideo.models.vaes.kandinsky6_sr import Kandinsky6SRVAE

    config_dict = json.loads((vae_dir / "config.json").read_text())
    config_dict.pop("_class_name", None)
    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(config_dict)
    with torch.device(device):
        vae = Kandinsky6SRVAE(config)
    state: dict[str, torch.Tensor] = {}
    for path in sorted(vae_dir.glob("*.safetensors")):
        state.update(load_file(str(path), device=device))
    vae.load_state_dict(state, strict=True)
    return vae.eval(), sorted(state)


@torch.no_grad()
def run_cases(vae_dir: Path, device: str = "cuda") -> dict[str, object]:
    import fastvideo

    vae, checkpoint_keys = load_vae(vae_dir, device)
    outputs: dict[str, object] = {
        "fastvideo_file": fastvideo.__file__,
        "checkpoint_keys": checkpoint_keys,
        "state_keys": sorted(vae.state_dict()),
        "num_params": sum(p.numel() for p in vae.parameters()),
    }
    for name, dtype, frames, height, width in CASES:
        vae.to(dtype)
        clip = make_clip(frames, height, width)
        pixel = vae.normalize_data(clip.permute(1, 0, 2, 3).unsqueeze(0).to(device).float()).to(dtype)
        latent = vae.encode(pixel)[0]
        decoded = vae.decode(latent).sample
        outputs[f"{name}/latent"] = latent.cpu()
        outputs[f"{name}/decoded"] = decoded.cpu()
        del pixel, latent, decoded
        torch.cuda.empty_cache()
    return outputs


def _run_reference(out_path: Path) -> dict[str, object]:
    env = dict(os.environ, PYTHONPATH=str(REFERENCE_ROOT))
    subprocess.run([sys.executable, __file__, str(VAE_DIR), str(out_path)], cwd=REFERENCE_ROOT, env=env, check=True)
    return torch.load(out_path, weights_only=False)


def test_native_vae_is_bitwise_equal_to_reference_on_real_weights(tmp_path):
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if not (VAE_DIR / "config.json").is_file() or not any(VAE_DIR.glob("*.safetensors")):
        pytest.skip(f"no Kandinsky6 SR VAE weights at {VAE_DIR} (set KANDINSKY6_SR_VAE_DIR)")
    if not (REFERENCE_ROOT / "fastvideo" / "models" / "vaes" / "kandinsky6_sr_kvae").is_dir():
        pytest.skip(f"no reference checkout at {REFERENCE_ROOT} (set KANDINSKY6_SR_VAE_REFERENCE)")

    reference = _run_reference(tmp_path / "reference.pt")
    assert str(reference["fastvideo_file"]).startswith(str(REFERENCE_ROOT))
    native = run_cases(VAE_DIR)
    assert not str(native["fastvideo_file"]).startswith(str(REFERENCE_ROOT))

    # Strict load already passed in both; the native keys are the checkpoint's own, the reference's carry "model.".
    assert native["state_keys"] == native["checkpoint_keys"]
    assert native["state_keys"] == sorted(key.removeprefix("model.") for key in reference["state_keys"])
    assert native["num_params"] == reference["num_params"]

    failures = []
    for name, *_ in CASES:
        for kind in ("latent", "decoded"):
            key = f"{name}/{kind}"
            ours, theirs = native[key], reference[key]
            assert ours.shape == theirs.shape and ours.dtype == theirs.dtype, key
            equal = torch.equal(ours, theirs)
            diff = (ours.float() - theirs.float()).abs().max().item()
            print(f"{key}: shape={tuple(ours.shape)} dtype={ours.dtype} bitwise={equal} max_abs_diff={diff:.3e}")
            if not equal:
                failures.append(f"{key} max_abs_diff={diff:.3e}")
    assert not failures, failures


if __name__ == "__main__":
    # Reference worker: python test_kandinsky6_sr_vae_parity.py <vae_dir> <out.pt>
    torch.save(run_cases(Path(sys.argv[1])), sys.argv[2])
