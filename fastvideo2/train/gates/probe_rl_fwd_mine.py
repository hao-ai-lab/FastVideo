"""Bisection probe (v2.1 side): rebuild the s0k0 inputs from the RL goldens
exactly like the anchor, run WanModelFV fp32 under autocast(bf16) with
per-module hooks, dump hashes, and DIFF against ``probe_main.json`` — prints
the first divergent module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

MAIN = "/mnt/fv21/probe_main.json"
OUT = "/mnt/fv21/probe_mine.json"


def _hash(t) -> str:
    import torch
    return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()
                          ).hexdigest()[:16]


def main() -> None:
    import numpy as np
    import torch

    from fastvideo2.loading import resolve_weights
    from fastvideo2.wan21.card import WAN21_T2V_1_3B as card
    from fastvideo2.wan21.model_fv import WanModelFV

    here = os.path.dirname(os.path.abspath(__file__))
    gold = os.path.normpath(os.path.join(here, "..", "..", "evidence",
                                         "goldens", "train-rl-main"))
    with open(os.path.join(gold, "steps.json")) as f:
        steps = json.load(f)
    rec0 = steps[0]
    ir = rec0["inner"][0]
    z = np.load(os.path.join(gold, "samples0.npz"))
    nitems = len(rec0["items"])
    lat = np.concatenate([z[f"it{j}_latents_clean"] for j in range(nitems)], 0)
    emb = np.concatenate([z[f"it{j}_embeds"] for j in range(nitems)], 0)
    inner = np.load(os.path.join(gold, "inner0.npz"))

    device = "cuda"
    x0 = torch.from_numpy(lat[ir["row_idx"]]).to(device, torch.bfloat16)
    embeds = torch.from_numpy(emb[ir["row_idx"]]).to(device, torch.bfloat16)
    tstep = torch.tensor(ir["timestep"], device=device)
    noise = torch.from_numpy(inner["noise0"]).to(device, torch.bfloat16)

    t = tstep.float() / 1000.0
    t_exp = t.view(-1, *([1] * (x0.ndim - 1)))
    xt = ((1 - t_exp) * x0 + t_exp * noise).to(dtype=x0.dtype)

    rec: dict = {"hooks": {}}
    rec["x0"] = _hash(x0)
    rec["x0_dtype"] = str(x0.dtype)
    rec["timestep"] = [float(v) for v in tstep.float().tolist()]
    rec["noise"] = _hash(noise)
    rec["xt"] = _hash(xt)
    rec["embeds"] = _hash(embeds)

    root = resolve_weights(card)
    model = WanModelFV.from_pretrained(root, torch_dtype=torch.float32,
                                       subfolder="transformer").to(device)
    rec["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    handles = []

    def hook(name):
        def fn(mod, args, out):
            outs = out if isinstance(out, tuple) else (out,)
            rec["hooks"][name] = [[_hash(o), str(o.dtype)]
                                  for o in outs if torch.is_tensor(o)]
        return fn

    seq: list = []

    def seq_hook(name):
        def fn(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o):
                seq.append([name, type(mod).__name__, _hash(o),
                            str(o.dtype), list(o.shape)])
        return fn

    b0 = model.blocks[0]
    rec["weights"] = {}
    for wn, wm in (("to_q", b0.attn1.to_q), ("norm_q", b0.attn1.norm_q),
                   ("to_k", b0.attn1.to_k)):
        w = wm.weight
        rec["weights"][wn] = [_hash(w), str(w.dtype), list(w.shape),
                              str(getattr(wm, "bias", None) is not None)]

    def pre(name):
        def fn(mod, args):
            if args and torch.is_tensor(args[0]):
                rec["weights"][f"{name}.in"] = [_hash(args[0]),
                                                str(args[0].dtype)]
        return fn

    handles.append(b0.attn1.to_q.register_forward_pre_hook(pre("to_q")))
    handles.append(b0.attn1.norm_q.register_forward_pre_hook(pre("norm_q")))

    rec["block0_seq"] = seq
    for name, mod in model.named_children():
        if name == "blocks":
            for i, b in enumerate(mod):
                handles.append(b.register_forward_hook(hook(f"block{i}")))
            for name2, mod2 in mod[0].named_modules():
                if name2:
                    handles.append(mod2.register_forward_hook(seq_hook(name2)))
        else:
            handles.append(mod.register_forward_hook(hook(name)))

    x = xt.permute(0, 2, 1, 3, 4)
    if x.is_floating_point():
        x = x.to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, embeds, tstep)
    pred = out.permute(0, 2, 1, 3, 4)
    rec["pred"] = _hash(pred)
    rec["pred_dtype"] = str(pred.dtype)
    for h in handles:
        h.remove()
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1)

    if os.path.exists(MAIN):
        with open(MAIN) as f:
            ref = json.load(f)
        print("== input parity ==")
        for k in ("x0", "noise", "xt", "embeds", "timestep", "pred"):
            same = rec.get(k) == ref.get(k)
            print(f"  {k:12s} {'SAME' if same else 'DIFF'}")
        print("== condition_embedder ALL outputs ==")
        print("  main:", ref["hooks"].get("condition_embedder"))
        print("  mine:", rec["hooks"].get("condition_embedder"))
        print("== block0 weights + op inputs ==")
        for k in sorted(set(ref.get("weights", {})) | set(rec["weights"])):
            a, b = ref.get("weights", {}).get(k), rec["weights"].get(k)
            print(f"  {k:10s} {'SAME' if a == b else 'DIFF'} main={a} mine={b}")
        print("== block0 internal call sequence (execution order) ==")
        ms, rs = rec["block0_seq"], ref["block0_seq"]
        n = max(len(ms), len(rs))
        shown_diff = 0
        for i in range(n):
            a = rs[i] if i < len(rs) else None
            b = ms[i] if i < len(ms) else None
            same = a is not None and b is not None and a[2] == b[2]
            if not same:
                shown_diff += 1
            print(f"  [{i:02d}] {'SAME' if same else 'DIFF'} "
                  f"main={a}  mine={b}")
            if shown_diff >= 8:
                print("  ... (stopping after 8 diffs)")
                break


if __name__ == "__main__":
    sys.exit(main())
