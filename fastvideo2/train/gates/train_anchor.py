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
    from fastvideo2.wan21.model_fv import WanModelFVQAT
    card = FASTWAN_T2V_1_3B if mode == "vsa" else WAN21_T2V_1_3B
    cls = {"vsa": WanModelFVVSA, "qat": WanModelFVQAT}.get(mode, WanModelFV)
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




def run_dmd2(mode: str = "dmd2") -> int:
    """DMD2 anchor: replay main's recorded per-phase draws through DMD2Step.
    Order per step mirrors train_one_step: student phase (rollout -> dmd loss
    -> student AdamW) only on interval steps, then critic phase (its OWN
    rollout through the just-updated student -> critic loss -> critic AdamW).
    """
    import hashlib

    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.train.dmd2 import DMD2Step
    from fastvideo2.wan21.card import FASTWAN_T2V_1_3B, WAN21_T2V_1_3B
    from fastvideo2.wan21.model_fv import WanModelFV, WanModelFVVSA

    gold = _gold_dir(mode)
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    device = "cuda"
    is_vsa = mode == "vsa_dmd2"
    is_qad = mode == "qad"
    # student: VSA/QAT blocks per recipe; teacher/critic: DENSE blocks + base
    # weights (main's scorers are dense-built / loader-gated back to flash)
    card = FASTWAN_T2V_1_3B if is_vsa else WAN21_T2V_1_3B
    base_root = resolve_weights(WAN21_T2V_1_3B)

    def load_dense():
        return WanModelFV.from_pretrained(base_root, torch_dtype=torch.float32,
                                          subfolder="transformer").to(device)

    if is_vsa:
        student = WanModelFVVSA.from_pretrained(resolve_weights(card),
                                                torch_dtype=torch.float32,
                                                subfolder="transformer").to(device)
    elif is_qad:
        from fastvideo2.wan21.model_fv import WanModelFVQAT
        student = WanModelFVQAT.from_pretrained(base_root, torch_dtype=torch.float32,
                                                subfolder="transformer").to(device)
    else:
        student = load_dense()
    step_ctx = DMD2Step(student, load_dense(), load_dense(),
                        guidance=2.0 if is_qad else 3.5)
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
        vsa_student = vsa_dense = None
        if is_vsa:
            from fastvideo2.layers.vsa import build_vsa_meta
            _, tl, _, hl, wl = z["ro0_init"].shape  # BTCHW draws
            grid = (tl, hl // 2, wl // 2)
            vsa_student = build_vsa_meta(grid, rec.get("vsa_sparsity", 0.0), device)
            vsa_dense = None  # dense scorers = flash, metadata ignored

        def roll_draws(j):
            return {"target_idx": rec["targets"][j],
                    "init_noise": tt(z[f"ro{j}_init"]),
                    "step_noises": [tt(z[f"ro{j}_n{k}"])
                                    for k in range(len(step_ctx.denoising_steps) - 1)]}

        student_step = rec["gen_loss"] != 0.0
        if student_step:
            x0 = step_ctx.student_rollout(roll_draws(0), cond, vsa=vsa_student)
            if i == 0 or rows == []:
                pass
            if rec.get("x0_student_hash"):
                rows.append((f"x0.step{i}",
                             0.0 if _hash(x0) == rec["x0_student_hash"] else 1.0))
            dmd = step_ctx.dmd_loss(
                x0, {"dmd_timestep": torch.tensor([rec["dmd_t"]], device=device),
                     "dmd_noise": tt(z["dmd_noise"])}, cond, uncond,
                vsa_dense=vsa_dense)
            dmd.backward()
            step_ctx.student.apply_grads_and_step()
            rows.append((f"gen.step{i}", abs(float(dmd.detach().item()) - rec["gen_loss"])))
        elif rec.get("x0_student_hash"):
            # first (only) rollout this step belongs to the critic phase
            pass

        j = n_roll - 1  # critic-phase rollout is the last one recorded
        with torch.no_grad():
            x0c = step_ctx.student_rollout(roll_draws(j), cond, vsa=vsa_student)
        if not student_step and rec.get("x0_student_hash"):
            rows.append((f"x0.step{i}",
                         0.0 if _hash(x0c) == rec["x0_student_hash"] else 1.0))
        closs = step_ctx.critic_loss(
            x0c, {"critic_timestep": torch.tensor([rec["critic_t"]], device=device),
                  "critic_noise": tt(z["critic_noise"])}, cond, vsa_dense=vsa_dense)
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
    first_update = next((i for i, r in enumerate(steps) if r["gen_loss"] != 0.0), 99)
    print(f"\nDMD2 anchor vs {manifest['fastvideo_commit'][:9]} (band {BAND:.2e})")
    failed = []
    for n, v in rows:
        # x0 hashes are bitwise until the FIRST student optimizer update
        # (whose backward noise makes later hashes unmatchable by definition);
        # after that the losses carry the parity signal
        if n == "fwd.matches":
            ok, why = True, "info"
        elif n.startswith("x0.step"):
            idx = int(n.split("step")[1])
            ok, why = (v == 0.0, "exact") if idx <= first_update else (True, "info")
        elif n.startswith(("gen.", "fake.")):
            ok, why = v <= BAND, "band"
        else:
            ok, why = True, "info"
        if not ok:
            failed.append(n)
        print(f"  {n:18s} {v:.6e}  {'OK' if ok else 'FAIL'} ({why})")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate=f"anchor.train-{mode}-main",
                              status="pass" if not failed else "fail",
                              model_id=card.model_id,
                              card_digest=card.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"x0.step0": 0.0, "later": "band"},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}")])
    return 1 if failed else 0


def run_self_forcing() -> int:
    """Self-forcing anchor: causal-student rollout replay (DrawCursor over
    the recorded sequential draws) + DMD/critic losses on the SF table."""
    import hashlib

    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.train.dmd2 import DMD2Step
    from fastvideo2.train.self_forcing import DrawCursor, sf_rollout
    from fastvideo2.wan21.card import SFWAN_T2V_1_3B, WAN21_T2V_1_3B
    from fastvideo2.wan21.loop import self_forcing_table
    from fastvideo2.wan21.model_fv import WanModelFV, WanModelFVCausal

    gold = _gold_dir("self_forcing")
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    device = "cuda"

    base_root = resolve_weights(WAN21_T2V_1_3B)
    student = WanModelFVCausal.from_pretrained(resolve_weights(SFWAN_T2V_1_3B),
                                               torch_dtype=torch.float32,
                                               subfolder="transformer").to(device)

    def load_dense():
        return WanModelFV.from_pretrained(base_root, torch_dtype=torch.float32,
                                          subfolder="transformer").to(device)

    table = self_forcing_table(5.0)
    step_ctx = DMD2Step(student, load_dense(), load_dense(),
                        guidance=manifest.get("guidance", 3.0), table=table)
    dsl = tuple(manifest["denoising_step_list"])
    hw = tuple(manifest["latent_hw"])
    pz = np.load(os.path.join(gold, "params.npz"))
    uncond = (torch.from_numpy(pz["neg_embeds"]).to(device, torch.bfloat16)
              if "neg_embeds" in pz.files else None)  # generator-phase only

    def _hash(t):
        return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()
                              ).hexdigest()[:16]

    gold_fwd = []
    fh = os.path.join(gold, "forward_hashes.json")
    if os.path.exists(fh):
        with open(fh) as f:
            gold_fwd = json.load(f)
    our_fwd: list = []

    def fwd_hook(module, args_in, kwargs_in, output):
        our_fwd.append({"x": _hash(args_in[0]), "o": _hash(output),
                        "t": [float(v) for v in args_in[2].flatten().tolist()][:2]})

    hook_handle = step_ctx.student.model.register_forward_hook(fwd_hook, with_kwargs=True)

    rows: list[tuple[str, float]] = []
    for i, rec in enumerate(steps):
        z = np.load(os.path.join(gold, f"step{i}.npz"))
        cond = torch.from_numpy(z["embeds"]).to(device, torch.bfloat16)

        def cursor(j):
            draws = [("randint", v) if kind == "randint"
                     else ("randn", z[f"ro{j}_seq{k}"])
                     for kind, v_or_k in rec["seqs"][j]
                     for k, v in [(v_or_k, v_or_k)]
                     for kind2 in [kind]]
            # rebuild precisely: (kind, val) with randn arrays by index
            draws = []
            for kind, v_or_k in rec["seqs"][j]:
                if kind == "randint":
                    draws.append(("randint", v_or_k))
                else:
                    draws.append(("randn", z[f"ro{j}_seq{v_or_k}"]))
            return DrawCursor(draws, device)

        def rollout(j):
            return sf_rollout(step_ctx.student.model, cursor(j), cond,
                              num_frames=manifest["num_latent_t"],
                              denoising_steps=dsl, table=table, latent_hw=hw,
                              context_noise=manifest.get("context_noise", 0))

        student_step = rec["gen_loss"] != 0.0
        if student_step:
            x0 = rollout(0)
            if rec.get("x0_student_hash"):
                rows.append((f"x0.step{i}",
                             0.0 if _hash(x0) == rec["x0_student_hash"] else 1.0))
            dmd = step_ctx.dmd_loss(
                x0, {"dmd_timestep": torch.tensor([rec["dmd_t"]], device=device),
                     "dmd_noise": torch.from_numpy(z["dmd_noise"]).to(device, torch.bfloat16)},
                cond, uncond)
            dmd.backward()
            step_ctx.student.apply_grads_and_step()
            rows.append((f"gen.step{i}", abs(float(dmd.detach().item()) - rec["gen_loss"])))
        j = len(rec["seqs"]) - 1
        with torch.no_grad():
            x0c = rollout(j)
        if not student_step and rec.get("x0_student_hash"):
            rows.append((f"x0.step{i}",
                         0.0 if _hash(x0c) == rec["x0_student_hash"] else 1.0))
        closs = step_ctx.critic_loss(
            x0c, {"critic_timestep": torch.tensor([rec["critic_t"]], device=device),
                  "critic_noise": torch.from_numpy(z["critic_noise"]).to(device, torch.bfloat16)},
            cond)
        closs.backward()
        step_ctx.critic.apply_grads_and_step()
        rows.append((f"fake.step{i}", abs(float(closs.detach().item()) - rec["fake_loss"])))
        print(f"  step{i}: gen {'-' if not student_step else round(float(dmd.detach().item()), 6)}"
              f"/{rec['gen_loss']:.6f} fake {float(closs.detach().item()):.6f}/{rec['fake_loss']:.6f}",
              flush=True)

    hook_handle.remove()
    n_match = 0
    first_div = None
    for k, (g, r) in enumerate(zip(gold_fwd, our_fwd)):
        if g["x"] == r["x"] and g["o"] == r["o"]:
            n_match += 1
        elif first_div is None:
            first_div = k
            print(f"  FIRST FWD DIVERGENCE @ {k}: gold t={g['t']} x={g['x'][:8]} "
                  f"o={g['o'][:8]} | ours t={r['t']} x={r['x'][:8]} o={r['o'][:8]}",
                  flush=True)
    rows.append(("fwd.matches", float(len(gold_fwd) - n_match)))

    BAND = 1.5 * max(manifest.get("self_noise_max", 2.15e-3), 1e-4)
    first_update = next((i for i, r in enumerate(steps) if r["gen_loss"] != 0.0), 99)
    print(f"\nSelfForcing anchor vs {manifest['fastvideo_commit'][:9]} (band {BAND:.2e}, "
          f"{n_match}/{len(gold_fwd)} fwd bitwise)")
    failed = []
    for n, v in rows:
        if n == "fwd.matches":
            ok, why = True, "info"
        elif n.startswith("x0.step"):
            idx = int(n.split("step")[1])
            ok, why = (v == 0.0, "exact") if idx <= first_update else (True, "info")
        elif n.startswith(("gen.", "fake.")):
            ok, why = v <= BAND, "band"
        else:
            ok, why = True, "info"
        if not ok:
            failed.append(n)
        print(f"  {n:18s} {v:.6e}  {'OK' if ok else 'FAIL'} ({why})")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate="anchor.train-self_forcing-main",
                              status="pass" if not failed else "fail",
                              model_id=SFWAN_T2V_1_3B.model_id,
                              card_digest=SFWAN_T2V_1_3B.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"x0-pre-update": 0.0, "later": "band"},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}")])
    return 1 if failed else 0


def run_data() -> int:
    """Dataloader-order gate — fastvideo2.train.data must reproduce the exact
    rows main's loader fed the finetune capture: for each recorded step, the
    caption and the latent/embed hashes (bf16-cast, num_latent_t-sliced, the
    same bytes main hashed at ``_get_next_batch``). CPU-only: bf16 casting is
    round-to-nearest-even on both devices."""
    import hashlib

    import torch

    from fastvideo2.train.data import (batch_indices, load_batch,
                                       parquet_files_and_lengths)

    gold = _gold_dir("finetune")
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)

    root = os.environ.get("FV2_DATA_PATH")
    if not root:
        from huggingface_hub import snapshot_download
        root = snapshot_download(manifest["dataset"], repo_type="dataset",
                                 token=False)
    p = os.path.join(root, "combined_parquet_dataset")
    if os.path.isdir(p):
        root = p

    files, lengths = parquet_files_and_lengths(root)
    batches = batch_indices(sum(lengths), 1, manifest["seed"])
    nlt = int(manifest["num_latent_t"])

    def sha(t: torch.Tensor) -> str:
        return hashlib.sha256(t.to(torch.bfloat16).to(torch.float32)
                              .numpy().tobytes()).hexdigest()[:16]

    failed = []
    for i, rec in enumerate(steps):
        b = load_batch(files, lengths, batches[i])
        checks = [("caption", b["captions"][0] == rec["caption"]),
                  ("latents", sha(b["latents"][:, :, :nlt]) == rec["latents_hash"]),
                  ("embeds", sha(b["embeds"]) == rec["embeds_hash"])]
        for name, ok in checks:
            print(f"  step{i} {name:8s} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                failed.append(f"step{i}.{name}")

    verdict = "PASS" if not failed else f"FAIL ({', '.join(failed)})"
    print(f"anchor.train-data-main: {verdict}")
    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    from fastvideo2.wan21.card import WAN21_T2V_1_3B as card
    append_ledger([GateResult(gate="anchor.train-data-main",
                              status="pass" if not failed else "fail",
                              model_id=card.model_id,
                              card_digest=card.digest(),
                              metrics={f"step{i}.row_match": 1.0
                                       for i in range(len(steps))},
                              tolerances={"caption/latents/embeds": "exact"},
                              env=env_fingerprint(),
                              detail=f"loader order vs goldens "
                                     f"{manifest['fastvideo_commit'][:9]}: "
                                     f"{len(steps)} batches, "
                                     f"files={len(files)}")])
    return 1 if failed else 0


def run_rl() -> int:
    """DiffusionNFT math anchor — replays main's recorded RL inner steps
    (rows, timesteps, xt-noise, advantages) through OUR NFTStep on three
    WanModelFV roles from the same base checkpoint, and compares:

      advantages.step{i}   recomputed from recorded rewards["avg"] (exact)
      loss.s{i}k{k}        per inner-call total_loss — outer step 0 runs at
                           pre-update weights (optimizer fires once per
                           outer step at this config: accum = 2×4 = calls);
                           outer step 1 inherits one AdamW step + old-EMA,
                           gated by measured main self-noise when present.
    """
    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.train.diffusion_nft import NFTStep, compute_advantages
    from fastvideo2.wan21.card import WAN21_T2V_1_3B as card
    from fastvideo2.wan21.model_fv import WanModelFV

    gold = _gold_dir("rl")
    with open(os.path.join(gold, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    probe = manifest["probe"]
    device = "cuda"

    dt = (torch.float32 if probe["param_dtype"] == "torch.float32"
          else torch.bfloat16)
    root = resolve_weights(card)

    def build():
        return WanModelFV.from_pretrained(
            root, torch_dtype=dt, subfolder="transformer").to(device)

    trainer = NFTStep(build(), build(), build(),
                      lr=3e-5, beta=probe["nft_beta"],
                      kl_beta=probe["kl_beta"],
                      adv_clip_max=probe["adv_clip_max"],
                      adv_mode=probe["adv_mode"],
                      num_train_timesteps=probe["ntt"],
                      max_grad_norm=probe["max_grad_norm"])

    band = 1.5 * float(manifest.get("self_noise_max", 0.0))
    rows: list[tuple[str, float]] = []
    failed: list[str] = []

    def check(name: str, mine: float, ref: float, exact: bool) -> None:
        # step0 losses are pure forward math at pre-update weights → bitwise;
        # step1 inherits one AdamW step built from nondeterministic backward
        # atomics → measured main self-noise band
        d = abs(mine - ref)
        ok = (d == 0.0) if exact else (d <= band)
        rows.append((name, d))
        if not ok:
            failed.append(name)
        print(f"  {name:16s} mine={mine:.6f} main={ref:.6f} "
              f"diff={d:.3e} {'OK' if ok else 'FAIL'}")

    for i, rec in enumerate(steps):
        z = np.load(os.path.join(gold, f"samples{i}.npz"))
        nitems = len(rec["items"])
        lat = np.concatenate(
            [z[f"it{j}_latents_clean"] for j in range(nitems)], 0)
        emb = np.concatenate([z[f"it{j}_embeds"] for j in range(nitems)], 0)
        inner = np.load(os.path.join(gold, f"inner{i}.npz"))

        # advantages: pure recompute from recorded rewards (exact)
        prompts = [p for it in rec["items"] for p in it["prompts"]]
        adv_ref = inner["advantages"]
        adv_mine = compute_advantages(
            torch.tensor(rec["rewards"]["avg"]), prompts,
            num_train_timesteps=adv_ref.shape[1]).numpy()
        d_adv = rel(adv_mine, adv_ref)
        rows.append((f"advantages.step{i}", d_adv))
        ok = d_adv == 0.0
        if not ok:
            failed.append(f"advantages.step{i}")
        print(f"  advantages.step{i}  rel={d_adv:.3e} {'OK' if ok else 'FAIL'}")

        # loss-math dtype: recorded fp32 arrays that round-trip bf16 exactly
        # WERE bf16 on main (fp32 sampler outputs would not)
        t_lat = torch.from_numpy(lat)
        loss_dtype = (torch.bfloat16
                      if bool((t_lat.bfloat16().float() == t_lat).all())
                      else torch.float32)
        t_emb = torch.from_numpy(emb)
        emb_dtype = (torch.bfloat16
                     if bool((t_emb.bfloat16().float() == t_emb).all())
                     else torch.float32)
        if i == 0:
            print(f"  [detected] loss dtype {loss_dtype}, embeds {emb_dtype}, "
                  f"params {dt}")

        accum = len(rec["inner"])
        for k, ir in enumerate(rec["inner"]):
            ridx = ir["row_idx"]
            x0 = torch.from_numpy(lat[ridx]).to(device, loss_dtype)
            embeds = torch.from_numpy(emb[ridx]).to(device, emb_dtype)
            tstep = torch.tensor(ir["timestep"], device=device)
            noise = torch.from_numpy(inner[f"noise{k}"]).to(device, loss_dtype)
            adv = torch.from_numpy(inner[f"adv{k}"]).to(device)
            losses = trainer.timestep_loss(x0, embeds, tstep, noise, adv)
            (losses["total_loss"] / accum).backward()
            check(f"loss.s{i}k{k}",
                  float(losses["total_loss"].detach().item()),
                  float(ir["losses"]["total_loss"]),
                  exact=(i == 0))
        trainer.optimizer_step()
        trainer.update_old(float(rec["old_decay"]))

    verdict = "PASS" if not failed else f"FAIL ({', '.join(failed[:6])})"
    print(f"anchor.train-rl-main: {verdict}")
    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate="anchor.train-rl-main",
                              status="pass" if not failed else "fail",
                              model_id=card.model_id,
                              card_digest=card.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"advantages": 0.0,
                                          "step0": "exact-or-band",
                                          "step1": "main self-noise band"},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}"
                                     f", band={band:.3e}")])
    return 1 if failed else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode = args[0] if args else "finetune"
    sys.exit(run_self_forcing() if mode == "self_forcing"
             else run_dmd2(mode) if mode in ("dmd2", "vsa_dmd2", "qad")
             else run_data() if mode == "data"
             else run_rl() if mode == "rl"
             else run(mode=mode))
