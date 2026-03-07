#!/usr/bin/env python3
"""
Compare FastVideo vs reference (diffusers or SGLang) Flux2 VAE decode.

Uses same latent input; compares decoded image output.
Reference: diffusers AutoencoderKLFlux2 (official).
Optionally uses SGLang if PYTHONPATH includes sglang and --use-sglang is passed.

  python compare_flux2_vae_sglang.py
  python compare_flux2_vae_sglang.py --use-sglang  # requires SGLang
"""
import argparse
import os
import sys

import torch

# Add SGLang to path if available
SGLANG_PATH = os.environ.get("SGLANG_PATH", os.path.join(os.path.dirname(__file__), "..", "sglang", "python"))
if os.path.isdir(SGLANG_PATH) and SGLANG_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PATH)

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
# Flux2 VAE: 32ch latent, spatial down 8x. Use small size for speed.
LATENT_H, LATENT_W = 128, 128  # decode -> 1024x1024
SEED = 0


def _get_vae_path(model_id: str) -> str:
    """Resolve VAE path from model ID."""
    try:
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=model_id)
        path = os.path.join(root, "vae")
        if os.path.isdir(path):
            return path
    except Exception:
        pass
    if os.path.isdir(model_id):
        vae = os.path.join(model_id, "vae")
        if os.path.isdir(vae):
            return vae
    raise FileNotFoundError(f"Could not find vae for {model_id}")


def main():
    parser = argparse.ArgumentParser(description="Compare FastVideo vs reference Flux2 VAE decode.")
    parser.add_argument("--model-path", default=None, help="Path to FLUX.2-klein-4B or vae dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-sglang", action="store_true", help="Use SGLang as reference (default: diffusers)")
    parser.add_argument("--height", type=int, default=LATENT_H, help="Latent height (spatial)")
    parser.add_argument("--width", type=int, default=LATENT_W, help="Latent width (spatial)")
    args = parser.parse_args()

    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    model_path = args.model_path or _get_vae_path(MODEL_ID)
    # FastVideoArgs expects repo root (model_index.json); VAELoader expects vae component path
    root = os.path.dirname(model_path) if os.path.basename(model_path) == "vae" else model_path
    vae_path = model_path if os.path.basename(model_path) == "vae" else os.path.join(root, "vae")
    device = args.device
    dtype = torch.float32  # VAE typically fp32 for parity
    h, w = args.height, args.width

    # Same latent for both (32ch, standard VAE input)
    generator = torch.Generator(device=device).manual_seed(SEED)
    latent = torch.randn(1, 32, h, w, device=device, dtype=dtype, generator=generator)

    # 1. FastVideo VAE
    print("Loading FastVideo Flux2 VAE ...")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    from fastvideo.configs.pipelines.flux_2 import Flux2KleinPipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import VAELoader
    from fastvideo.utils import PRECISION_TO_TYPE

    fv_args = FastVideoArgs.from_kwargs(
        model_path=root,
        pipeline_config=Flux2KleinPipelineConfig(),
    )
    fv_args.pipeline_config.vae_precision = "fp32"
    fv_args.pipeline_config.vae_tiling = False
    loader = VAELoader()
    fv_vae = loader.load(vae_path, fv_args)
    fv_vae = fv_vae.to(device).eval()

    # Flux2 VAE expects normalized latent; scale by config
    arch = fv_vae.config.arch_config
    scale = getattr(arch, "scaling_factor", None) or getattr(arch, "scale_factor", 0.13025)
    shift = getattr(arch, "shift_factor", None)
    if hasattr(scale, "to"):
        scale = scale.to(device=device, dtype=dtype)
    else:
        scale = torch.tensor(scale, device=device, dtype=dtype)
    if shift is not None and hasattr(shift, "to"):
        shift = shift.to(device=device, dtype=dtype)
    else:
        shift = torch.zeros(1, device=device, dtype=dtype) if shift is None else torch.tensor(shift, device=device, dtype=dtype)
    latent_scaled = latent / scale
    if shift is not None and shift.numel() > 0:
        latent_scaled = latent_scaled + shift.view(1, -1, 1, 1)

    with torch.no_grad():
        fv_out = fv_vae.decode(latent_scaled)
    fv_out = fv_out.cpu().float()

    # 2. Reference VAE (diffusers or SGLang)
    ref_out = None
    ref_name = "diffusers"

    if args.use_sglang:
        try:
            from sglang.multimodal_gen.runtime.loader.component_loader import VAELoader as SGLangVAELoader
            from sglang.multimodal_gen.runtime.server_args import ServerArgs
            from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2KleinPipelineConfig as SGLangFlux2KleinConfig

            print("Loading SGLang Flux2 VAE ...")
            sgl_args = ServerArgs(
                model_path=root,
                pipeline_config=SGLangFlux2KleinConfig(),
            )
            sgl_loader = SGLangVAELoader()
            sgl_vae = sgl_loader.load(vae_path, sgl_args)
            sgl_vae = sgl_vae.to(device).eval()

            with torch.no_grad():
                ref_out = sgl_vae.decode(latent_scaled)
            ref_out = ref_out.cpu().float()
            ref_name = "SGLang"
        except ImportError:
            print("SGLang not available, falling back to diffusers.")
            args.use_sglang = False

    if ref_out is None:
        # Load via Flux2KleinPipeline to get correct VAE (AutoencoderKLFlux2 or equivalent)
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            from diffusers.pipelines.flux2 import Flux2KleinPipeline
        load_path = root if os.path.isdir(root) else MODEL_ID
        print("Loading diffusers Flux2 VAE (via pipeline) for reference ...")
        pipe = Flux2KleinPipeline.from_pretrained(load_path, torch_dtype=dtype)
        ref_vae = pipe.vae.to(device).eval()
        with torch.no_grad():
            out = ref_vae.decode(latent_scaled)
        ref_out = (out.sample if hasattr(out, "sample") else out).cpu().float()
        ref_name = "diffusers"

    # Compare
    print(f"\n--- VAE decode comparison (FastVideo vs {ref_name}) ---")
    print(f"  FastVideo output shape: {fv_out.shape}")
    print(f"  {ref_name:10} output shape: {ref_out.shape}")
    if fv_out.shape != ref_out.shape:
        print("  SHAPE MISMATCH")
    else:
        diff = (fv_out - ref_out).abs()
        print(f"  max abs diff:  {diff.max().item():.6f}")
        print(f"  mean abs diff: {diff.mean().item():.6f}")
        allclose = torch.allclose(fv_out, ref_out, rtol=1e-2, atol=1e-2)
        print(f"  allclose(rtol=1e-2, atol=1e-2): {allclose}")
        if allclose:
            print("  -> VAE decode outputs match.")
        else:
            print("  -> VAE decode outputs differ.")


if __name__ == "__main__":
    main()
