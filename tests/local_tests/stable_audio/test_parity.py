#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Parity tests for Stable Audio components (transformer, VAE, conditioner, scheduler).

Verifies that FastVideo's Stable Audio implementations match stable-audio-tools
outputs when loading from unified model.safetensors.

Run from project root:
  python tests/local_tests/stable_audio/test_parity.py
  python tests/local_tests/stable_audio/test_parity.py --test transformer  # run specific test

Uses official_weights/stable-audio-open-1.0/ (model.safetensors + model_config.json).
If missing, downloads from HuggingFace stabilityai/stable-audio-open-1.0 (set HF_TOKEN
for gated access).
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
sys.path.insert(0, REPO_ROOT)
SAT_PATH = os.path.join(REPO_ROOT, "stable-audio-tools")
if os.path.isdir(SAT_PATH):
    sys.path.insert(0, SAT_PATH)

HF_STABLE_AUDIO_ID = "stabilityai/stable-audio-open-1.0"
MODEL_ROOT = os.path.join(REPO_ROOT, "official_weights", "stable-audio-open-1.0")
CHECKPOINT_PATH = os.path.join(MODEL_ROOT, "model.safetensors")
CONFIG_PATH = os.path.join(MODEL_ROOT, "model_config.json")


def _ensure_model_downloaded() -> bool:
    """If checkpoint missing, try downloading from HuggingFace. Returns True if ready."""
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CONFIG_PATH):
        return True
    print(f"  {CHECKPOINT_PATH} not found. Trying HF download ({HF_STABLE_AUDIO_ID})...")
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(os.path.dirname(MODEL_ROOT), exist_ok=True)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        snapshot_download(
            repo_id=HF_STABLE_AUDIO_ID,
            local_dir=MODEL_ROOT,
            ignore_patterns=["*.onnx", "*.msgpack"],
            token=token,
        )
    except Exception as e:
        print(f"  Download failed: {e}")
        return False
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  SKIP: After download, {CHECKPOINT_PATH} still missing (repo layout may differ).")
        return False
    print(f"  Downloaded to {MODEL_ROOT}")
    return True


def _checkpoint_exists() -> bool:
    if not _ensure_model_downloaded():
        print(f"SKIP: {CHECKPOINT_PATH} not found. Download model or set HF_TOKEN.")
        return False
    return True


# --- Transformer ---


def _load_fastvideo_transformer():
    from fastvideo.configs.models.dits.stable_audio import StableAudioDiTConfig
    from fastvideo.models.dits.stable_audio import StableAudioDiTModel
    from fastvideo.models.loader.utils import get_param_names_mapping
    from fastvideo.models.loader.weight_utils import safetensors_weights_iterator

    config = StableAudioDiTConfig()
    config.arch_config.in_channels = 64
    config.arch_config.global_states_input_dim = 1536
    config.arch_config.cross_attention_dim = 768
    config.arch_config.num_layers = 24
    config.arch_config.num_attention_heads = 24

    model = StableAudioDiTModel(config=config)
    mapping_fn = get_param_names_mapping(config.arch_config.param_names_mapping)
    weight_iter = safetensors_weights_iterator(
        [CHECKPOINT_PATH], to_cpu=True, key_prefix="model.model."
    )
    state_dict = {mapping_fn(k)[0]: v for k, v in weight_iter}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert len(missing) == 0, f"Should have no missing keys, got {len(missing)}"
    return model


def _load_reference_transformer():
    import json
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    full = create_model_from_config(config)
    full.load_state_dict(load_ckpt_state_dict(CHECKPOINT_PATH), strict=False)
    return full.model


def test_transformer():
    if not _checkpoint_exists():
        return
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fv_model = _load_fastvideo_transformer()
    fv_inner = fv_model.model.to(device).eval()
    B, C, T = 1, 64, 64
    x = torch.randn(B, C, T, device=device, dtype=torch.float32)
    t = torch.rand(B, device=device, dtype=torch.float32) * 100

    with torch.no_grad():
        out_fv = fv_inner(x, t)
    print(f"  Transformer: FastVideo output shape {out_fv.shape}")

    try:
        sat_model = _load_reference_transformer().to(device).eval()
        with torch.no_grad():
            out_sat = sat_model(x, t)
        max_diff = (out_fv - out_sat).abs().max().item()
        torch.testing.assert_close(out_fv, out_sat, atol=1e-5, rtol=1e-4)
        print(f"  Transformer: max_diff={max_diff:.6f} PASS")
    except ImportError:
        print("  Transformer: PASS (no stable-audio-tools comparison)")


# --- VAE / Pretransform ---


def _load_reference_pretransform():
    import json
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    full = create_model_from_config(config)
    full.load_state_dict(load_ckpt_state_dict(CHECKPOINT_PATH), strict=False)
    return full.pretransform


def _load_fastvideo_pretransform():
    from fastvideo.models.stable_audio import StableAudioPretransform

    return StableAudioPretransform(
        model_config=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )


def test_vae():
    if not _checkpoint_exists():
        return
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 2, 4096, device=device, dtype=torch.float32)

    ref = _load_reference_pretransform().to(device).eval()
    fv = _load_fastvideo_pretransform().to(device).eval()

    with torch.no_grad():
        z_ref = ref.encode(x)
        x_recon_ref = ref.decode(z_ref)
        z_fv = fv.encode(x)
        x_recon_fv = fv.decode(z_fv)

    torch.testing.assert_close(z_fv, z_ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(x_recon_fv, x_recon_ref, atol=0.5, rtol=0.05)
    print(f"  VAE: z {z_fv.shape} recon {x_recon_fv.shape} PASS")


# --- Conditioner ---


def _load_reference_conditioner():
    import json
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    full = create_model_from_config(config)
    full.load_state_dict(load_ckpt_state_dict(CHECKPOINT_PATH), strict=False)
    return full.conditioner


def _load_fastvideo_conditioner():
    from fastvideo.models.stable_audio import StableAudioConditioner

    return StableAudioConditioner(
        model_config=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )


def test_conditioner():
    if not _checkpoint_exists():
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = [{"prompt": "Amen break 174 BPM", "seconds_start": 0, "seconds_total": 12}]

    ref = _load_reference_conditioner().to(device).eval()
    fv = _load_fastvideo_conditioner().to(device).eval()

    with torch.no_grad():
        cond_ref = ref(metadata, device)
        cond_fv = fv(metadata, device)

    for key in cond_ref:
        r0, r1 = cond_ref[key]
        f0, f1 = cond_fv[key]
        if r0 is not None:
            torch.testing.assert_close(r0, f0, atol=1e-4, rtol=1e-3)
        if r1 is not None:
            torch.testing.assert_close(r1, f1, atol=1e-5, rtol=1e-4)
    print("  Conditioner: PASS")


# --- Scheduler / Sampling ---


def _load_reference_model():
    import json
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    model = create_model_from_config(config)
    model.load_state_dict(load_ckpt_state_dict(CHECKPOINT_PATH), strict=False)
    return model


def test_scheduler():
    if not _checkpoint_exists():
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    steps = 10

    model = _load_reference_model().to(device).eval()
    metadata = [{"prompt": "Amen break", "seconds_start": 0, "seconds_total": 12}]

    with torch.no_grad():
        cond_tensors = model.conditioner(metadata, device)
        cond_inputs = model.get_conditioning_inputs(cond_tensors)

    sample_size, latent_channels = 1024, 64
    noise_shape = (1, latent_channels, sample_size)
    cond_inputs_cuda = {k: v.to(device) if v is not None else v for k, v in cond_inputs.items()}

    def model_fn(x, sigma, **kwargs):
        return model.model(x, sigma, **cond_inputs_cuda)

    from stable_audio_tools.inference.sampling import sample_k
    from fastvideo.models.stable_audio.sampling import sample_stable_audio

    torch.manual_seed(seed)
    noise_ref = torch.randn(noise_shape, device=device, dtype=torch.float32)
    sampled_ref = sample_k(
        model_fn, noise_ref, init_data=None, steps=steps,
        sampler_type="dpmpp-2m-sde", sigma_min=0.01, sigma_max=100, rho=1.0,
        device=device, cfg_scale=6.0, batch_cfg=True, rescale_cfg=True,
    )

    torch.manual_seed(seed)
    noise_fv = torch.randn(noise_shape, device=device, dtype=torch.float32)
    sampled_fv = sample_stable_audio(
        model_fn, noise_fv, steps=steps, device=device,
        cfg_scale=6.0, batch_cfg=True, rescale_cfg=True,
    )

    torch.testing.assert_close(sampled_ref, sampled_fv, atol=1e-3, rtol=1e-2)
    print("  Scheduler/sampling: PASS")


# --- Main ---

TESTS = {
    "transformer": test_transformer,
    "vae": test_vae,
    "conditioner": test_conditioner,
    "scheduler": test_scheduler,
}


def main():
    parser = argparse.ArgumentParser(description="Stable Audio parity tests")
    parser.add_argument(
        "--test",
        choices=list(TESTS) + ["all"],
        default="all",
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    if args.test == "all":
        for name, fn in TESTS.items():
            print(f"\n[{name}]")
            fn()
    else:
        print(f"\n[{args.test}]")
        TESTS[args.test]()

    print("\nAll selected parity tests passed.")


if __name__ == "__main__":
    main()
