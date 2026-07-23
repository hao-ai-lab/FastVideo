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


def _gold_dir(mode: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "evidence", "goldens",
                                         f"train-{mode}-main"))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / (d if d else 1.0)


def run(mode: str = "finetune") -> int:
    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.train.finetune import FinetuneStep
    from fastvideo2.wan21.card import FASTWAN_T2V_1_3B, WAN21_T2V_1_3B
    from fastvideo2.wan21.model_fv import WanModelFV, WanModelFVVSA

    gold = _gold_dir(mode)
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    device = "cuda"

    # main trains its own WanTransformer3DModel port; our bitwise-equal
    # vendor loads the same weights fp32 (VSA gate: the FastWan checkpoint,
    # which carries deterministic to_gate_compress weights)
    card = FASTWAN_T2V_1_3B if mode == "vsa" else WAN21_T2V_1_3B
    cls = WanModelFVVSA if mode == "vsa" else WanModelFV
    root = resolve_weights(card)
    model = cls.from_pretrained(root, torch_dtype=torch.float32,
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
        vsa_meta = None
        if mode == "vsa":
            from fastvideo2.layers.vsa import build_vsa_meta
            _, _, tl, hl, wl = z["latents"].shape
            vsa_meta = build_vsa_meta((tl, hl // 2, wl // 2),
                                      rec["vsa_sparsity"], device)
        if i == 0:  # input-construction triage: does OUR noisy match theirs?
            noisy = (1.0 - sigmas) * latents + sigmas * noise
            rows.append(("noisy.step0", 0.0 if _hash(noisy) == rec["noisy_hash"] else 1.0))
            print(f"  step0 shapes: latents {tuple(latents.shape)} embeds "
                  f"{tuple(embeds.shape)} t={rec['timesteps']} σ={rec['sigmas']}",
                  flush=True)
            if "pred" in z.files:
                with torch.no_grad():
                    p = trainer.model(noisy, embeds,
                                      timesteps.to(torch.bfloat16), vsa=vsa_meta)
                rows.append(("pred.step0", rel(p.detach().to(torch.float32).cpu().numpy(),
                                               z["pred"])))
        loss, gnorm = trainer.step(latents, embeds, noise, timesteps, sigmas,
                                   vsa=vsa_meta)
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
    append_ledger([GateResult(gate=f"anchor.train-{mode}-main",
                              status="pass" if not failed else "fail",
                              model_id=card.model_id,
                              card_digest=card.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"step0": 0.0,
                                          "later": "main self-noise band (see manifest)"},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}")])
    return 1 if failed else 0




def run_dmd2() -> int:
    """DMD2 anchor: replay main's recorded per-phase draws through DMD2Step.
    Order per step mirrors train_one_step: student phase (rollout -> dmd loss
    -> student AdamW) only on interval steps, then critic phase (its OWN
    rollout through the just-updated student -> critic loss -> critic AdamW).
    """
    import hashlib

    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.train.dmd2 import DMD2Step
    from fastvideo2.wan21.card import WAN21_T2V_1_3B
    from fastvideo2.wan21.model_fv import WanModelFV

    gold = _gold_dir("dmd2")
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    device = "cuda"
    root = resolve_weights(WAN21_T2V_1_3B)

    def load():
        return WanModelFV.from_pretrained(root, torch_dtype=torch.float32,
                                          subfolder="transformer").to(device)

    step_ctx = DMD2Step(load(), load(), load())
    pz = np.load(os.path.join(gold, "params.npz"))
    uncond = torch.from_numpy(pz["neg_embeds"]).to(device, torch.bfloat16)

    def _hash(t):
        return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()
                              ).hexdigest()[:16]

    def tt(a):
        return torch.from_numpy(a).to(device, torch.bfloat16)

    rows: list[tuple[str, float]] = []
    for i, rec in enumerate(steps):
        z = np.load(os.path.join(gold, f"step{i}.npz"))
        cond = tt(z["embeds"])
        n_roll = len(rec["targets"])

        def roll_draws(j):
            return {"target_idx": rec["targets"][j],
                    "init_noise": tt(z[f"ro{j}_init"]),
                    "step_noises": [tt(z[f"ro{j}_n{k}"])
                                    for k in range(len(step_ctx.denoising_steps) - 1)]}

        student_step = rec["gen_loss"] != 0.0
        if student_step:
            x0 = step_ctx.student_rollout(roll_draws(0), cond)
            if i == 0 or rows == []:
                pass
            if rec.get("x0_student_hash"):
                rows.append((f"x0.step{i}",
                             0.0 if _hash(x0) == rec["x0_student_hash"] else 1.0))
            dmd = step_ctx.dmd_loss(
                x0, {"dmd_timestep": torch.tensor([rec["dmd_t"]], device=device),
                     "dmd_noise": tt(z["dmd_noise"])}, cond, uncond)
            dmd.backward()
            step_ctx.student.apply_grads_and_step()
            rows.append((f"gen.step{i}", abs(float(dmd.detach().item()) - rec["gen_loss"])))
        elif rec.get("x0_student_hash"):
            # first (only) rollout this step belongs to the critic phase
            pass

        j = n_roll - 1  # critic-phase rollout is the last one recorded
        with torch.no_grad():
            x0c = step_ctx.student_rollout(roll_draws(j), cond)
        if not student_step and rec.get("x0_student_hash"):
            rows.append((f"x0.step{i}",
                         0.0 if _hash(x0c) == rec["x0_student_hash"] else 1.0))
        closs = step_ctx.critic_loss(
            x0c, {"critic_timestep": torch.tensor([rec["critic_t"]], device=device),
                  "critic_noise": tt(z["critic_noise"])}, cond)
        closs.backward()
        step_ctx.critic.apply_grads_and_step()
        rows.append((f"fake.step{i}", abs(float(closs.detach().item()) - rec["fake_loss"])))
        print(f"  step{i}: gen ours/main "
              f"{'-' if not student_step else f'{float(dmd.detach().item()):.6f}'}/"
              f"{rec['gen_loss']:.6f}  fake {float(closs.detach().item()):.6f}/"
              f"{rec['fake_loss']:.6f}", flush=True)

    w5s = step_ctx.student.master["proj_out.weight"][:8, :8].cpu().numpy()
    w5c = step_ctx.critic.master["proj_out.weight"][:8, :8].cpu().numpy()
    rows.append(("params.w5_student", rel(w5s, pz["w5_student"])))
    rows.append(("params.w5_critic", rel(w5c, pz["w5_critic"])))

    BAND = 1.5 * max(manifest.get("self_noise_max", 2.15e-3), 1e-4)
    print(f"\nDMD2 anchor vs {manifest['fastvideo_commit'][:9]} (band {BAND:.2e})")
    failed = []
    for n, v in rows:
        if n.startswith("x0.step0"):
            ok, why = v == 0.0, "exact"
        elif n.startswith(("gen.", "fake.", "x0.")):
            ok, why = v <= BAND, "band"
        else:
            ok, why = True, "info"
        if not ok:
            failed.append(n)
        print(f"  {n:18s} {v:.6e}  {'OK' if ok else 'FAIL'} ({why})")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate="anchor.train-dmd2-main",
                              status="pass" if not failed else "fail",
                              model_id=WAN21_T2V_1_3B.model_id,
                              card_digest=WAN21_T2V_1_3B.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"x0.step0": 0.0, "later": "band"},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}")])
    return 1 if failed else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode = args[0] if args else "finetune"
    sys.exit(run_dmd2() if mode == "dmd2" else run(mode=mode))
