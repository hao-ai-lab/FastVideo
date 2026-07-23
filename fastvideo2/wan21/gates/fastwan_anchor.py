"""FastWan-QAD anchor — v2.1 vs fastvideo-main goldens, bitwise target.

Consumes evidence/goldens/fastwan-qad-main/ (see capture_fastvideo_main.py)
and runs v2.1's OWN stack — no fastvideo import here:

    text_encoder   our fp32 UMT5 path vs main's embeds        (target 0.0*)
    dit.bf16       WanModelFV probes, 3 timesteps             (target 0.0)
    dit.fp8        WanModelFVFP8 probes, 3 timesteps          (target 0.0)
    e2e.step{i}.x  per-step latent chain through OUR engine   (target 0.0)
    e2e.final      final latents                              (target 0.0)

*ASCII prompts only. main's text cleaning (ftfy) diverges from official's on
CJK width-folding; the FastWan cards currently reuse the wan21 text stage.
# ponytail: ASCII-equal today; add main's clean fn to the FastWan pipeline
# (and a CJK golden) before serving non-ASCII prompts against these cards.

Usage (cluster):
    python -m fastvideo2.wan21.gates.fastwan_anchor [--skip-e2e]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

GOLD = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "evidence", "goldens", "fastwan-qad-main"))
PROBE_TS = (1000, 757, 522)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b.astype(np.float64)))
    if denom == 0.0:
        return float(np.linalg.norm(a.astype(np.float64)))
    return float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)))) / denom


def run(skip_e2e: bool = False) -> int:
    import torch

    from fastvideo2.engine import Instance, Request
    from fastvideo2.engine import run as engine_run
    from fastvideo2.registry import resolve
    from fastvideo2.wan21.card import FASTWAN_QAD_FP8_1_3B
    from fastvideo2.wan21.model_fv import WanModelFV, WanModelFVFP8

    with open(os.path.join(GOLD, "manifest.json")) as f:
        manifest = json.load(f)
    device = "cuda"
    rows: list[tuple[str, float]] = []

    from fastvideo2.loading import resolve_weights
    root = resolve_weights(FASTWAN_QAD_FP8_1_3B)

    # ------------------------------------------------------------- probes --- #
    pi = np.load(os.path.join(GOLD, "probe_inputs.npz"))
    px = torch.from_numpy(pi["latent"])
    pc = torch.from_numpy(pi["context"])

    def probe(dit: Any, tag: str) -> None:
        for t in PROBE_TS:
            xt = px.to(device, torch.bfloat16)
            ct = pc.to(device)  # fp32
            tt = torch.tensor([t], dtype=torch.int64, device=device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = dit(xt, ct, tt)
            gold = np.load(os.path.join(GOLD, f"dit_{tag}_t{t}.npz"))["out"]
            rows.append((f"dit.{tag}.t{t}", rel_l2(o.to(torch.float32).cpu().numpy(), gold)))

    dit = WanModelFV.from_pretrained(root, torch_dtype=torch.bfloat16).to(device)
    probe(dit, "bf16")
    del dit
    torch.cuda.empty_cache()
    dit_q = WanModelFVFP8.from_pretrained(root, torch_dtype=torch.bfloat16).to(device)
    probe(dit_q, "fp8")
    del dit_q
    torch.cuda.empty_cache()

    # --------------------------------------------------------------- text --- #
    # through the served pipeline's own text path, so the gate measures what
    # generate() actually feeds the loop
    from fastvideo2.wan21.pipeline import _encode_one
    instance = Instance(FASTWAN_QAD_FP8_1_3B, device=device)
    tok = instance.component("tokenizer")
    enc = instance.component("text_encoder")
    gold_text = np.load(os.path.join(GOLD, "text_encoder.npz"))
    for name, prompt in (("e2e", manifest["e2e_prompt"]), ("probe", manifest["probe_prompt"])):
        emb = _encode_one(tok, enc, prompt)
        rows.append((f"text.{name}", rel_l2(emb[0].to(torch.float32).cpu().numpy(),
                                            gold_text[name])))

    # ---------------------------------------------------------------- e2e --- #
    if not skip_e2e:
        card, builder = resolve(FASTWAN_QAD_FP8_1_3B.model_id)
        req = Request(prompt=manifest["e2e_prompt"], seed=manifest["seed"],
                      num_frames=81, height=480, width=832,
                      capture_trajectory=True)
        out = engine_run(instance, builder(), req)
        den = out.outputs["latents"]  # the denoise slot: {latents, trajectory, steps}
        final = den["latents"].to(torch.float32).cpu().numpy()
        gold_final = np.load(os.path.join(GOLD, "e2e_final_latents.npz"))["latents"]
        rows.append(("e2e.final", rel_l2(final, gold_final)))
        # per-step input chain: golden step_i.x is what main fed the DiT at
        # step i (BCFHW — the hook sees the permuted input); our trajectory
        # stores post-advance latents in BTCHW = the input of step i+1.
        traj = [t.to(torch.float32).permute(0, 2, 1, 3, 4).numpy() for t in den["trajectory"]]
        for i in range(1, len(PROBE_TS)):
            g = np.load(os.path.join(GOLD, f"e2e_step{i}.npz"))["x"]
            rows.append((f"e2e.step{i}.x", rel_l2(traj[i - 1], g)))

    # -------------------------------------------------------------- report --- #
    print(f"\nFastWan-QAD anchor vs {manifest['fastvideo_commit'][:9]} "
          f"({manifest['attention_backend']}, fp8 {'' if skip_e2e else '+ e2e'})")
    failed = []
    for name, val in rows:
        ok = val == 0.0
        if not ok:
            failed.append(name)
        print(f"  {name:16s} rel_l2 {val:.6e}  {'OK (bitwise)' if ok else 'DIFF'}")

    from fastvideo2.verify import append_ledger, env_fingerprint
    append_ledger({"gate": "anchor.fastwan-qad-main",
                   "status": "pass" if not failed else "fail",
                   "model_id": FASTWAN_QAD_FP8_1_3B.model_id,
                   "card_digest": FASTWAN_QAD_FP8_1_3B.digest(),
                   "metrics": {n: v for n, v in rows},
                   "tolerances": {"all": 0.0},
                   "env": env_fingerprint(),
                   "detail": f"goldens {manifest['fastvideo_commit'][:9]}"})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(skip_e2e="--skip-e2e" in sys.argv))
