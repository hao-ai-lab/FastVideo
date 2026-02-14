# SPDX-License-Identifier: Apache-2.0
"""
Run official Overworld Waypoint-1-Small model (video generation).

Uses Overworld/Waypoint-1-Small transformer + FastVideo/Waypoint-1-Small-Diffusers
for VAE, text encoder, tokenizer. Outputs video (MP4), not single image.

Usage:
  python scripts/official_waypoint.py
  python scripts/official_waypoint.py --seed 123 --output official_waypoint.mp4
  python scripts/official_waypoint.py --num-frames 8 --output out.mp4 --debug-output debug.json
"""

import argparse
import glob
import importlib.util
import inspect
import json
import os
import sys
import types

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from safetensors.torch import load_file


def _save_decoded(decoded: torch.Tensor, path: str, fps: int = 8) -> None:
    """Save decoded VAE output. WorldEngine VAE outputs 0-255 (uint8/float)."""
    import numpy as np
    import imageio

    arr = decoded.detach().cpu().numpy().astype(np.float32)
    if arr.max() <= 1.1:
        if arr.min() < -0.1:
            arr = (arr * 0.5 + 0.5).clip(0, 1)
        arr = arr * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 5:
        arr = arr[0]
        if arr.shape[1] == 3:
            arr = arr.transpose(0, 2, 3, 1)
        frames = [arr[i] for i in range(len(arr))]
        imageio.mimsave(path, frames, fps=fps, format="mp4")
        return
    if arr.ndim == 4 and arr.shape[0] > 1:
        if arr.shape[1] == 3:
            arr = arr.transpose(0, 2, 3, 1)
        frames = [arr[i] for i in range(len(arr))]
        imageio.mimsave(path, frames, fps=fps, format="mp4")
        return
    arr = arr[0] if arr.ndim == 4 else arr
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".gif", ".webm"):
        imageio.mimsave(path, [arr], fps=1, format=ext[1:])
    else:
        imageio.imwrite(path, arr)


def _stats(t: torch.Tensor, include_sample: bool = False) -> dict:
    if t is None:
        return {}
    f = t.float().detach()
    s = float(f.std()) if f.numel() > 1 else 0.0
    out = {
        "shape": list(t.shape),
        "mean": float(f.mean()),
        "std": s,
        "min": float(f.min()),
        "max": float(f.max()),
        "has_nan": bool(torch.isnan(f).any()),
        "has_inf": bool(torch.isinf(f).any()),
    }
    if include_sample and f.numel() > 0:
        flat = f.flatten()
        n = min(5, flat.numel())
        out["sample"] = flat[:n].tolist()
    return out


def _stats_rgb(t: torch.Tensor) -> list[dict]:
    """Per-channel stats for RGB tensor (H,W,C) or (C,H,W)."""
    if t is None or t.numel() == 0:
        return []
    f = t.float().detach()
    if f.dim() >= 3 and f.shape[-1] == 3:
        slices = [f[..., c] for c in range(3)]
    elif f.dim() >= 3 and f.shape[0] == 3:
        slices = [f[c] for c in range(3)]
    else:
        return []
    return [
        {
            "ch": c,
            "mean": float(s.mean()),
            "std": float(s.std()) if s.numel() > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        }
        for c, s in enumerate(slices)
    ]


def load_snapshot() -> str:
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--FastVideo--Waypoint-1-Small-Diffusers"
        "/snapshots/*"
    )
    matches = glob.glob(pattern)
    if not matches:
        matches = glob.glob(
            "/root/.cache/huggingface/hub/models--FastVideo--Waypoint-1-Small-Diffusers/snapshots/*"
        )
    if not matches:
        raise FileNotFoundError(
            "FastVideo/Waypoint-1-Small-Diffusers not found. "
            "Run basic_waypoint.py once to download."
        )
    return matches[0]


def load_vae(snapshot_dir: str):
    vae_dir = os.path.join(snapshot_dir, "vae")
    with open(os.path.join(vae_dir, "config.json")) as f:
        vae_config = json.load(f)
    auto_map = vae_config.get("auto_map", {})
    target = (
        auto_map.get("AutoModel")
        or auto_map.get("AutoencoderKL")
        or auto_map.get("Autoencoder")
    )
    module_name, cls_name = target.rsplit(".", 1)
    pkg_name = "official_waypoint_vae"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [vae_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    for pyf in sorted(glob.glob(os.path.join(vae_dir, "*.py"))):
        mod_name = os.path.splitext(os.path.basename(pyf))[0]
        full_name = f"{pkg_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(
            full_name, pyf, submodule_search_locations=[]
        )
        if spec is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
    VAEClass = getattr(sys.modules[f"{pkg_name}.{module_name}"], cls_name)
    cfg = {k: v for k, v in vae_config.items() if k not in ("_class_name", "_diffusers_version", "auto_map")}
    sig = inspect.signature(VAEClass.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        filtered_cfg = cfg
    else:
        filtered_cfg = {k: v for k, v in cfg.items() if k in valid_params}
    vae = VAEClass(**filtered_cfg)
    for sf in glob.glob(os.path.join(vae_dir, "*.safetensors")):
        vae.load_state_dict(load_file(sf), strict=False)
    return vae.to("cuda").eval()


def load_official_transformer():
    from huggingface_hub import snapshot_download

    official_dir = snapshot_download(
        "Overworld/Waypoint-1-Small", allow_patterns=["transformer/*"]
    )
    t_dir = os.path.join(official_dir, "transformer")
    with open(os.path.join(t_dir, "config.json")) as f:
        t_config = json.load(f)
    pkg_name = "official_waypoint_transformer"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [t_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    for pyf in sorted(glob.glob(os.path.join(t_dir, "*.py"))):
        mod_name = os.path.splitext(os.path.basename(pyf))[0]
        full_name = f"{pkg_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(
            full_name, pyf, submodule_search_locations=[]
        )
        if spec is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
    WorldModel = getattr(sys.modules[f"{pkg_name}.model"], "WorldModel")
    model_cfg = {k: v for k, v in t_config.items() if k not in ("_class_name", "_diffusers_version", "auto_map")}
    transformer = WorldModel(**model_cfg)
    for sf in glob.glob(os.path.join(t_dir, "*.safetensors")):
        transformer.load_state_dict(load_file(sf), strict=False)
    return transformer.to("cuda", dtype=torch.bfloat16).eval(), t_config


def main():
    parser = argparse.ArgumentParser(description="Run official Overworld Waypoint (video)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A first-person gameplay video exploring a stylized world.")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of video frames")
    parser.add_argument("--output", default="official_waypoint.mp4", help="Output video path (.mp4)")
    parser.add_argument("--debug-output", default=None, help="Save extensive tensor stats to JSON")
    parser.add_argument("--fps", type=int, default=8, help="Video FPS")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    snapshot_dir = load_snapshot()
    print(f"Snapshot: {snapshot_dir}")

    print("Loading VAE...")
    vae = load_vae(snapshot_dir)

    from transformers import AutoTokenizer, T5EncoderModel
    te_dir = os.path.join(snapshot_dir, "text_encoder")
    tok_dir = os.path.join(snapshot_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    text_encoder = T5EncoderModel.from_pretrained(te_dir).to("cuda").eval()

    inputs = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    with torch.no_grad():
        enc_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_emb = enc_out.last_hidden_state * attention_mask.unsqueeze(-1).float()
        prompt_pad_mask = attention_mask.eq(0)
    prompt_emb = prompt_emb.to(torch.bfloat16)
    print(f"Prompt emb: {_stats(prompt_emb)}")

    mouse = torch.zeros(1, 1, 2, device="cuda", dtype=torch.bfloat16)
    button = torch.zeros(1, 1, 256, device="cuda", dtype=torch.bfloat16)
    button[0, 0, 17] = 1.0
    scroll = torch.zeros(1, 1, 1, device="cuda", dtype=torch.bfloat16)

    print("Loading official transformer...")
    transformer, t_config = load_official_transformer()
    sigmas = torch.tensor(
        t_config.get("scheduler_sigmas", [1.0, 0.9483, 0.8380, 0.0]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    print(f"Sigmas: {sigmas.tolist()}")

    class DummyKVCache:
        def set_frozen(self, frozen):
            pass
        def upsert(self, k, v, pos_ids, layer_idx):
            return k, v, None

    cache_mod = sys.modules.get("official_waypoint_transformer.cache")
    kv_cache = DummyKVCache()
    if cache_mod and hasattr(cache_mod, "StaticKVCache"):
        try:
            kv_cache = cache_mod.StaticKVCache(
                transformer.config, max_frames=64, device="cuda", dtype=torch.bfloat16
            )
        except Exception:
            pass

    sigma = torch.zeros(1, 1, device="cuda", dtype=torch.bfloat16)
    if kv_cache is not None and hasattr(kv_cache, "set_frozen"):
        kv_cache.set_frozen(True)

    debug_frames = []
    decoded_frames = []

    print(f"Generating {args.num_frames} frames...")
    for frame_idx in range(args.num_frames):
        g = torch.Generator(device="cuda").manual_seed(args.seed + frame_idx)
        x_init = torch.randn(1, 1, 16, 32, 32, device="cuda", dtype=torch.bfloat16, generator=g)
        frame_ts = torch.full((1, 1), frame_idx, device="cuda", dtype=torch.long)

        frame_debug = {
            "frame": frame_idx,
            "x_init": _stats(x_init, include_sample=True),
        }
        if args.debug_output:
            frame_debug["steps"] = []

        x = x_init.clone()
        for step_i, (step_sig, step_dsig) in enumerate(zip(sigmas[:-1], sigmas.diff())):
            v = transformer(
                x=x,
                sigma=sigma.fill_(step_sig),
                frame_timestamp=frame_ts,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad_mask,
                mouse=mouse,
                button=button,
                scroll=scroll,
                kv_cache=kv_cache,
            )
            if args.debug_output:
                frame_debug["steps"].append({
                    "step": step_i,
                    "sigma": float(step_sig),
                    "v_pred": _stats(v, include_sample=True),
                    "x_before": _stats(x.clone()),
                })
            x = x + step_dsig * v
            if args.debug_output:
                frame_debug["steps"][-1]["x_after"] = _stats(x, include_sample=True)

        frame_debug["denoised"] = _stats(x, include_sample=True)
        latent = x.squeeze(1).float()
        decoded = vae.decode(latent)
        d = decoded.sample if hasattr(decoded, "sample") else decoded
        frame_debug["vae_out"] = _stats(d, include_sample=True)
        rgb = _stats_rgb(d)
        if rgb:
            frame_debug["vae_out_rgb"] = rgb
        decoded_frames.append(d.detach().cpu())
        debug_frames.append(frame_debug)
        print(f"  frame {frame_idx + 1}/{args.num_frames}: denoised mean={frame_debug['denoised']['mean']:.4f}")

    # Stack frames: (T, B, C, H, W) -> (1, T, C, H, W) for _save_decoded
    stacked = torch.stack(decoded_frames, dim=0)  # (T, 1, C, H, W) or (T, 1, H, W, C)
    stacked = stacked.permute(1, 0, 2, 3, 4) if stacked.dim() == 5 else stacked  # (1, T, ...)
    _save_decoded(stacked, args.output, fps=args.fps)
    print(f"Saved video: {args.output}")

    if args.debug_output:
        debug_data = {
            "source": "official",
            "seed": args.seed,
            "prompt": args.prompt,
            "num_frames": args.num_frames,
            "sigmas": sigmas.tolist(),
            "prompt_emb": _stats(prompt_emb, include_sample=True),
            "mouse": _stats(mouse),
            "button": _stats(button),
            "scroll": _stats(scroll),
            "frames": debug_frames,
        }
        with open(args.debug_output, "w") as f:
            json.dump(debug_data, f, indent=2)
        print(f"Debug stats saved to {args.debug_output}")


if __name__ == "__main__":
    main()
