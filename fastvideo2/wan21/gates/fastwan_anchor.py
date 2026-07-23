"""FastWan anchors — v2.1 vs fastvideo-main goldens, bitwise target.

Consumes evidence/goldens/fastwan-{qad,vsa}-main/ (see
capture_fastvideo_main.py) and runs v2.1's OWN stack — no fastvideo import:

    text_encoder   our fp32 UMT5 path vs main's embeds        (target 0.0*)
    dit.<tag>      model_fv probes, 3 timesteps               (target 0.0)
                   qad: bf16 (plain) + fp8; vsa: sparse-kernel forward
    e2e.step{i}.x  per-step latent chain through OUR engine   (target 0.0)
    e2e.final      final latents                              (target 0.0)

*ASCII prompts only. main's text cleaning (ftfy) diverges from official's on
CJK width-folding; the FastWan cards currently reuse the wan21 text stage.
# ponytail: ASCII-equal today; add main's clean fn to the FastWan pipeline
# (and a CJK golden) before serving non-ASCII prompts against these cards.

Usage (cluster):
    python -m fastvideo2.wan21.gates.fastwan_anchor [vsa] [--skip-e2e]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

PROBE_TS = (1000, 757, 522)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b.astype(np.float64)))
    if denom == 0.0:
        return float(np.linalg.norm(a.astype(np.float64)))
    return float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)))) / denom


def _gold_dir(variant: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "evidence", "goldens",
                                         f"fastwan-{variant}-main"))


def run(variant: str = "qad", skip_e2e: bool = False) -> int:
    import torch

    from fastvideo2.engine import Instance, Request
    from fastvideo2.engine import run as engine_run
    from fastvideo2.registry import resolve
    from fastvideo2.wan21.card import FASTWAN_QAD_FP8_1_3B, FASTWAN_T2V_1_3B
    from fastvideo2.wan21.model_fv import WanModelFV, WanModelFVFP8, WanModelFVVSA

    card = {"qad": FASTWAN_QAD_FP8_1_3B, "vsa": FASTWAN_T2V_1_3B}[variant]
    gold_dir = _gold_dir(variant)
    with open(os.path.join(gold_dir, "manifest.json")) as f:
        manifest = json.load(f)
    device = "cuda"
    rows: list[tuple[str, float]] = []

    from fastvideo2.loading import resolve_weights
    root = resolve_weights(card)

    # ------------------------------------------------------------- probes --- #
    pi = np.load(os.path.join(gold_dir, "probe_inputs.npz"))
    px = torch.from_numpy(pi["latent"])
    pc = torch.from_numpy(pi["context"])

    probe_vsa = None
    if variant == "vsa":
        from fastvideo2.layers.vsa import build_vsa_meta
        _, _, f, h, w = px.shape  # BCFHW probe latent -> (1,2,2)-patch grid
        probe_vsa = build_vsa_meta((f, h // 2, w // 2), manifest["vsa_sparsity"], device)

    def probe(dit: Any, tag: str) -> None:
        for t in PROBE_TS:
            xt = px.to(device, torch.bfloat16)
            ct = pc.to(device)  # fp32
            tt = torch.tensor([t], dtype=torch.int64, device=device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = dit(xt, ct, tt, vsa=probe_vsa)
            gold = np.load(os.path.join(gold_dir, f"dit_{tag}_t{t}.npz"))["out"]
            rows.append((f"dit.{tag}.t{t}", rel_l2(o.to(torch.float32).cpu().numpy(), gold)))

    probe_models = ([(WanModelFVVSA, "vsa")] if variant == "vsa"
                    else [(WanModelFV, "bf16"), (WanModelFVFP8, "fp8")])
    for cls, tag in probe_models:
        dit = cls.from_pretrained(root, torch_dtype=torch.bfloat16).to(device)
        probe(dit, tag)
        del dit
        torch.cuda.empty_cache()

    # --------------------------------------------------------------- text --- #
    # through the served pipeline's own text path, so the gate measures what
    # generate() actually feeds the loop
    from fastvideo2.wan21.pipeline import _encode_one
    instance = Instance(card, device=device)
    tok = instance.component("tokenizer")
    enc = instance.component("text_encoder")
    gold_text = np.load(os.path.join(gold_dir, "text_encoder.npz"))
    for name, prompt in (("e2e", manifest["e2e_prompt"]), ("probe", manifest["probe_prompt"])):
        emb = _encode_one(tok, enc, prompt)
        rows.append((f"text.{name}", rel_l2(emb[0].to(torch.float32).cpu().numpy(),
                                            gold_text[name])))

    # ---------------------------------------------------------------- e2e --- #
    if not skip_e2e:
        _, builder = resolve(card.model_id)
        req = Request(prompt=manifest["e2e_prompt"], seed=manifest["seed"],
                      num_frames=81, height=480, width=832,
                      capture_trajectory=True)
        out = engine_run(instance, builder(), req)
        den = out.outputs["latents"]  # the denoise slot: {latents, trajectory, steps}
        final = den["latents"].to(torch.float32).cpu().numpy()
        gold_final = np.load(os.path.join(gold_dir, "e2e_final_latents.npz"))["latents"]
        rows.append(("e2e.final", rel_l2(final, gold_final)))
        # per-step input chain: golden step_i.x is what main fed the DiT at
        # step i (BCFHW — the hook sees the permuted input); our trajectory
        # stores post-advance latents in BTCHW = the input of step i+1.
        traj = [t.to(torch.float32).permute(0, 2, 1, 3, 4).numpy() for t in den["trajectory"]]
        for i in range(1, len(PROBE_TS)):
            g = np.load(os.path.join(gold_dir, f"e2e_step{i}.npz"))["x"]
            rows.append((f"e2e.step{i}.x", rel_l2(traj[i - 1], g)))

    # -------------------------------------------------------------- report --- #
    print(f"\nFastWan-{variant} anchor vs {manifest['fastvideo_commit'][:9]} "
          f"({manifest['attention_backend']}{'' if skip_e2e else ' + e2e'})")
    failed = []
    for name, val in rows:
        ok = val == 0.0
        if not ok:
            failed.append(name)
        print(f"  {name:16s} rel_l2 {val:.6e}  {'OK (bitwise)' if ok else 'DIFF'}")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate=f"anchor.fastwan-{variant}-main",
                              status="pass" if not failed else "fail",
                              model_id=card.model_id,
                              card_digest=card.digest(),
                              metrics={n: v for n, v in rows},
                              tolerances={"all": 0.0},
                              env=env_fingerprint(),
                              detail=f"goldens {manifest['fastvideo_commit'][:9]}")])
    return 1 if failed else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(run(variant=args[0] if args else "qad",
                 skip_e2e="--skip-e2e" in sys.argv))
