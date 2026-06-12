# SPDX-License-Identifier: Apache-2.0
"""Component parity for the native WorldEngineVAE (Waypoint-1-Small).

Compares FastVideo's native ``WorldEngineVAE`` against the official reference
(the ``ae_model.py`` shipped in the diffusers repo, loaded here purely as a
parity reference) on matched inputs. Set ``WAYPOINT_MODEL_PATH`` to override the
model location.
"""

import glob
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

MODEL_PATH = os.environ.get(
    "WAYPOINT_MODEL_PATH",
    "models/Waypoint-1-Small-Diffusers",
)
VAE_DIR = os.path.join(MODEL_PATH, "vae")


def _weights_available() -> bool:
    return os.path.isfile(os.path.join(VAE_DIR, "config.json")) and bool(
        glob.glob(os.path.join(VAE_DIR, "*.safetensors")))


SKIP_REASON = f"WorldEngineVAE weights not found under {VAE_DIR}"


def _load_state_dict() -> dict:
    import safetensors.torch as st
    sd: dict = {}
    for f in glob.glob(os.path.join(VAE_DIR, "*.safetensors")):
        sd.update(st.load_file(f))
    return sd


def _load_official():
    pkg = "world_engine_vae_ref"
    mod = types.ModuleType(pkg)
    mod.__path__ = [VAE_DIR]
    sys.modules[pkg] = mod
    for name in ("dcae", "ae_model"):
        spec = importlib.util.spec_from_file_location(
            f"{pkg}.{name}", os.path.join(VAE_DIR, f"{name}.py"))
        sub = importlib.util.module_from_spec(spec)
        sub.__package__ = pkg
        sys.modules[f"{pkg}.{name}"] = sub
        spec.loader.exec_module(sub)
    cfg = json.load(open(os.path.join(VAE_DIR, "config.json")))
    for k in ("_class_name", "_diffusers_version", "auto_map",
              "use_middle_block"):
        cfg.pop(k, None)
    vae = sys.modules[f"{pkg}.ae_model"].WorldEngineVAE(**cfg)
    vae.load_state_dict(_load_state_dict(), strict=True)
    return vae.eval()


def _load_native():
    from fastvideo.configs.models.vaes import WorldEngineVAEConfig
    from fastvideo.models.vaes.world_engine_vae import WorldEngineVAE
    vae = WorldEngineVAE(WorldEngineVAEConfig())
    vae.load_state_dict(_load_state_dict(), strict=True)
    return vae.eval()


@pytest.mark.skipif(not _weights_available(), reason=SKIP_REASON)
def test_world_engine_vae_state_dict_strict():
    sd = set(_load_state_dict().keys())
    from fastvideo.configs.models.vaes import WorldEngineVAEConfig
    from fastvideo.models.vaes.world_engine_vae import WorldEngineVAE
    native = set(WorldEngineVAE(WorldEngineVAEConfig()).state_dict().keys())
    assert sd == native, (
        f"missing={sorted(sd - native)[:8]} unexpected={sorted(native - sd)[:8]}")


@pytest.mark.skipif(not _weights_available(), reason=SKIP_REASON)
@torch.no_grad()
def test_world_engine_vae_output_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    official = _load_official().to(device, torch.float32)
    native = _load_native().to(device, torch.float32)

    latent = torch.randn(1, 16, 32, 32, device=device, dtype=torch.float32)
    d_off = official.decode(latent).float()
    d_nat = native.decode(latent).float()
    decode_mae = (d_off - d_nat).abs().mean().item()
    print(f"\ndecode uint8 MAE: {decode_mae:.6f}/255")
    assert decode_mae < 0.5, f"decode MAE too high: {decode_mae}"

    img = torch.rand(360, 640, 3, device=device) * 255
    e_off = official.encode(img).float()
    e_nat = native.encode(img).float()
    cos = torch.nn.functional.cosine_similarity(e_off.flatten(),
                                                 e_nat.flatten(),
                                                 dim=0).item()
    encode_mae = (e_off - e_nat).abs().mean().item()
    print(f"encode cosine: {cos:.6f}  MAE: {encode_mae:.6e}")
    assert cos > 0.9999, f"encode cosine too low: {cos}"
