"""Finetune math anchor — v2.1's FinetuneStep vs fastvideo-main goldens.

Replays main's exact recorded per-step batches (latents/embeds/noise, bf16)
and timesteps/sigmas through OUR trainer on the SAME card DiT, and compares:

    loss.step{i}    per-step training loss           (step0 target: bitwise;
                    later steps inherit flash-backward atomics noise — band
                    comes from main-vs-main reruns, never hand-picked)
    grad_norm.step{i}
    params.w5       proj_out.weight[:8,:8] after 5 steps (optimizer chain)

Usage (cluster): python -m fastvideo2.train.gates.train_anchor
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def _gold_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "evidence", "goldens",
                                         "train-finetune-main"))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / (d if d else 1.0)


def run() -> int:
    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.train.finetune import FinetuneStep
    from fastvideo2.wan21.card import WAN21_T2V_1_3B
    from fastvideo2.wan21.model_fv import WanModelFV

    gold = _gold_dir()
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    device = "cuda"

    # main trains its own WanTransformer3DModel port from the Diffusers repo;
    # our bitwise-equal vendor of it loads the same weights fp32
    root = resolve_weights(WAN21_T2V_1_3B)  # Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model = WanModelFV.from_pretrained(root, torch_dtype=torch.float32,
                                       subfolder="transformer").to(device)
    trainer = FinetuneStep(model)
    w0 = trainer.master["proj_out.weight"][:8, :8].cpu().numpy().copy()

    import hashlib

    def _hash(t):
        return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()
                              ).hexdigest()[:16]

    rows: list[tuple[str, float]] = []
    for i, rec in enumerate(steps):
        z = np.load(os.path.join(gold, f"step{i}.npz"))
        latents = torch.from_numpy(z["latents"]).to(device, torch.bfloat16)
        embeds = torch.from_numpy(z["embeds"]).to(device, torch.bfloat16)
        noise = torch.from_numpy(z["noise"]).to(device, torch.bfloat16)
        timesteps = torch.tensor(rec["timesteps"], device=device)
        sigmas = torch.tensor(rec["sigmas"], device=device,
                              dtype=torch.bfloat16).view(-1, 1, 1, 1)
        if i == 0:  # input-construction triage: does OUR noisy match theirs?
            noisy = (1.0 - sigmas) * latents + sigmas * noise
            rows.append(("noisy.step0", 0.0 if _hash(noisy) == rec["noisy_hash"] else 1.0))
            print(f"  step0 shapes: latents {tuple(latents.shape)} embeds "
                  f"{tuple(embeds.shape)} t={rec['timesteps']} σ={rec['sigmas']}",
                  flush=True)
            if "pred" in z.files:
                with torch.no_grad():
                    p = trainer.model(noisy, embeds,
                                      timesteps.to(torch.bfloat16))
                rows.append(("pred.step0", rel(p.detach().to(torch.float32).cpu().numpy(),
                                               z["pred"])))
        loss, gnorm = trainer.step(latents, embeds, noise, timesteps, sigmas)
        rows.append((f"loss.step{i}", abs(loss - rec["loss"])))
        rows.append((f"gnorm.step{i}", abs(gnorm - rec["grad_norm"])))
        print(f"  step{i}: ours loss={loss:.8f} main={rec['loss']:.8f} "
              f"gnorm {gnorm:.5f}/{rec['grad_norm']:.5f}", flush=True)

    pz = np.load(os.path.join(gold, "params.npz"))
    rows.append(("params.w0", rel(w0, pz["w0"])))
    w5 = trainer.master["proj_out.weight"][:8, :8].cpu().numpy()
    rows.append(("params.w5", rel(w5, pz["w5"])))

    # Gate: exact where the computation is deterministic; a MEASURED band
    # where flash-attention backward atomics make main itself non-repeatable
    # (main-vs-main across three capture runs drifts up to 2.1e-3 in loss —
    # see manifest self_noise). gnorm/params rows are informational until a
    # dedicated gnorm self-noise capture lands.
    LOSS_BAND = 1.5 * max(manifest.get("self_noise_max", 2.15e-3), 1e-4)
    print(f"\nFinetune anchor vs {manifest['fastvideo_commit'][:9]} "
          f"(loss band {LOSS_BAND:.2e} = 1.5x measured main self-noise)")
    failed = []
    for n, v in rows:
        if n in ("noisy.step0", "pred.step0", "loss.step0", "params.w0"):
            ok, why = v == 0.0, "exact"
        elif n.startswith("loss."):
            ok, why = v <= LOSS_BAND, "band"
        else:
            ok, why = True, "info"
        if not ok:
            failed.append(n)
        print(f"  {n:14s} {v:.6e}  {'OK' if ok else 'FAIL'} ({why})")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate="anchor.train-finetune-main",
                              status="pass" if not failed else "fail",
                              model_id=WAN21_T2V_1_3B.model_id,
                              card_digest=WAN21_T2V_1_3B.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"step0": 0.0,
                                          "later": "main self-noise band (see manifest)"},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}")])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
