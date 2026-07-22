"""Capture golden probes from the OFFICIAL Wan2.1 implementation.

This shim runs INSIDE a scratch environment where the official repo
(https://github.com/Wan-Video/Wan2.1) is importable and the *original-layout*
checkpoint (Wan-AI/Wan2.1-T2V-1.3B) is on disk. It is executed rarely and
deliberately — at oracle bring-up, on a reference change, on an environment
bump — and never imported by anything in fastvideo2. The official repo enters
this codebase only as the commit hash recorded in the goldens manifest.

    git clone https://github.com/Wan-Video/Wan2.1 /tmp/wan-official
    python -m venv --system-site-packages /tmp/wan-env
    /tmp/wan-env/bin/pip install easydict ftfy
    PYTHONPATH=/tmp/wan-official:<this repo> /tmp/wan-env/bin/python \\
        -m fastvideo2.wan21.capture_official \\
        --wan-repo /tmp/wan-official --ckpt-dir <Wan-AI/Wan2.1-T2V-1.3B> \\
        --out <repo>/fastvideo2/evidence/goldens/wan21-official

Captured goldens (all inputs generated from pinned seeds and stored in-file):
  text_encoder.npz  official UMT5 embeddings for the probe prompt + negative
  dit.npz           WanModel velocity at fixed (latent, t, context), 2 timesteps
  vae.npz           official decode of a fixed normalized latent
  schedule.npz      FlowUniPC sigma/timestep tables for pinned (steps, shift)
  e2e_*.mp4         full official generations (probe config + official-recommended)
  manifest.json     repo, commit, weights, env, config defaults, per-capture meta
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys


def _env(wan_repo: str) -> dict:
    import torch
    env = {"python": platform.python_version(), "machine": platform.machine(),
           "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        import flash_attn  # noqa: F401 — official attention prefers flash_attn
        env["flash_attn"] = flash_attn.__version__
    except ImportError:
        env["flash_attn"] = None
    env["repo"] = "https://github.com/Wan-Video/Wan2.1"
    env["commit"] = subprocess.run(["git", "-C", wan_repo, "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wan-repo", required=True, help="clone of Wan-Video/Wan2.1")
    p.add_argument("--ckpt-dir", required=True, help="original-layout Wan-AI/Wan2.1-T2V-1.3B dir")
    p.add_argument("--out", required=True, help="goldens output dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-e2e", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, args.wan_repo)
    try:
        import wan  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"official `wan` package not importable ({e}) — this shim runs only "
                         f"inside the capture environment; see module docstring") from e

    import numpy as np
    import torch
    from wan.configs import WAN_CONFIGS

    from fastvideo2.wan21 import goldens as G

    cfg = WAN_CONFIGS["t2v-1.3B"]
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    G._merge_manifest(args.out, {"capture": {
        **_env(args.wan_repo), "ckpt_dir": os.path.abspath(args.ckpt_dir),
        "prompt": G.PROMPT, "negative": G.NEGATIVE,
        "config_defaults": {"num_train_timesteps": int(cfg.num_train_timesteps),
                            "param_dtype": str(cfg.param_dtype),
                            "sample_neg_prompt": cfg.sample_neg_prompt},
    }})

    # ---- text encoder ---------------------------------------------------- #
    from wan.modules.t5 import T5EncoderModel
    t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device,
                        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
                        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer))
    with torch.no_grad():
        prompt_emb, neg_emb = t5([G.PROMPT, G.NEGATIVE], device)   # unpadded [len, 4096]
    G.save_golden(args.out, "text_encoder", {
        "prompt_embeds": prompt_emb.to(torch.float32).cpu().numpy(),
        "prompt_len": np.int64(prompt_emb.shape[0]),
        "negative_embeds": neg_emb.to(torch.float32).cpu().numpy(),
        "negative_len": np.int64(neg_emb.shape[0]),
    }, {"note": "official wan.modules.t5 UMT5 wrapper, bf16, outputs saved fp32 unpadded"})
    print(f"captured text_encoder: prompt len {prompt_emb.shape[0]}, neg len {neg_emb.shape[0]}")

    # ---- DiT forward ------------------------------------------------------ #
    from wan.modules.model import WanModel
    dit = WanModel.from_pretrained(args.ckpt_dir).to(device).eval().requires_grad_(False)
    x_np = G.dit_probe_latent()
    context_canonical = G.pad_context(prompt_emb.to(torch.float32).cpu().numpy())
    x = torch.from_numpy(x_np[0]).to(device)                        # official API: unbatched list
    ctx = [prompt_emb.to(device)]                                   # official pads internally
    _, _, t_, h_, w_ = x_np.shape
    seq_len = t_ * (h_ // 2) * (w_ // 2)
    arrays = {"x": x_np, "context": context_canonical,
              "timesteps": np.asarray(G.DIT_TIMESTEPS, dtype=np.float32)}
    for t in G.DIT_TIMESTEPS:
        tt = torch.tensor([t], device=device, dtype=torch.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=cfg.param_dtype):
            out = dit([x], t=tt, context=ctx, seq_len=seq_len)[0]
        arrays[f"out_t{int(t)}"] = out.to(torch.float32).cpu().numpy()[None]
        print(f"captured dit t={t}: out std {arrays[f'out_t{int(t)}'].std():.4f}")
    G.save_golden(args.out, "dit", arrays,
                  {"seq_len": seq_len, "note": "official WanModel.forward under bf16 autocast; "
                   "context stored in the canonical zero-padded [512,4096] form"})
    del dit
    torch.cuda.empty_cache()

    # ---- VAE decode -------------------------------------------------------- #
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=device)
    z_np = G.vae_probe_latent()
    with torch.no_grad():
        video = vae.decode([torch.from_numpy(z_np[0]).to(device)])[0]  # [3, T, H, W] in [-1,1]
    G.save_golden(args.out, "vae", {
        "z": z_np, "video": video.to(torch.float16).cpu().numpy(),
    }, {"note": "official WanVAE wrapper decode of a NORMALIZED latent (wrapper owns denorm); "
        "ports must reproduce the full denorm+decode behavior"})
    print(f"captured vae: video {tuple(video.shape)}, range [{video.min():.3f}, {video.max():.3f}]")
    del vae
    torch.cuda.empty_cache()

    # ---- schedule tables ---------------------------------------------------- #
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sched_arrays = {}
    for steps, shift in G.SCHEDULES:
        s = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                        shift=1, use_dynamic_shifting=False)
        s.set_timesteps(steps, device="cpu", shift=shift)
        sched_arrays[f"sigmas_s{steps}_f{shift}"] = s.sigmas.numpy().astype(np.float64)
        sched_arrays[f"timesteps_s{steps}_f{shift}"] = s.timesteps.numpy().astype(np.float64)
    G.save_golden(args.out, "schedule", sched_arrays,
                  {"note": "official FlowUniPC sigma/timestep grids per (steps, shift)"})
    print("captured schedules:", ", ".join(f"s{s}_f{f}" for s, f in G.SCHEDULES))

    # ---- end-to-end (perceptual band, not trajectory parity) ---------------- #
    if not args.skip_e2e:
        import imageio.v2 as imageio
        from wan.text2video import WanT2V
        t2v = WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device.index or 0,
                     rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False)
        runs = {"e2e_probe": dict(frame_num=17, shift=3.0, sampling_steps=8, guide_scale=5.0),
                "e2e_official": dict(frame_num=81, shift=8.0, sampling_steps=50, guide_scale=6.0)}
        meta = {}
        for name, kw in runs.items():
            with torch.no_grad():
                video = t2v.generate(G.PROMPT, size=(832, 480), sample_solver="unipc",
                                     n_prompt=G.NEGATIVE, seed=1234, offload_model=False, **kw)
            frames = ((video.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()       # [T, H, W, C]
            imageio.mimsave(os.path.join(args.out, f"{name}.mp4"), list(frames), fps=16,
                            format="mp4")
            meta[name] = {**kw, "seed": 1234, "solver": "unipc",
                          "brightness": float(frames.mean()),
                          "dynamics": float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean())}
            print(f"captured {name}: {frames.shape}, brightness {meta[name]['brightness']:.1f}")
        G._merge_manifest(args.out, {"e2e": meta})

    print(f"goldens -> {args.out}")


if __name__ == "__main__":
    main()
