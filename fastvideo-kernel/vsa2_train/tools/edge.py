"""Judges a solution on one input of your choosing, beyond the five workloads: any latent extent, text length, top-K,
batch and head count, and q and k scaled to make the attention sharper. Run it in a GPU job from the repo root:

    python tools/edge.py --thw 9,10,21 --text 0 --topk 3
    python tools/edge.py --thw 13,22,39 --text 200 --topk 5 --heads 3 --batch 2 --qk-scale 8 --seed 1
    python tools/edge.py --workload s8                       # one of spec.py's workloads, as CI draws it

CI judges the five workloads with spec.py's fixed tolerances, set for their inputs. On other inputs a correct bf16
implementation can need more: where the attention is much sharper than the workloads' (large logits), D and dS cancel
and their bf16 rounding is amplified, and a small sequence can put an unlucky element just past atol. So here an output
passes when its worst error against the reference, beyond rtol * |ref|, is at most the larger of the workload
tolerance's atol and three times what a bf16 flash-style implementation of the same input needs (emulate.py). The
selection is judged as CI judges it, and the other outputs under the solution's own selection when it is valid.

Exits 0 when every output passes. --solution and --reference choose other copies, as for bench.py.
"""

import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emulate as E  # noqa: E402
import loader  # noqa: E402
import spec as S  # noqa: E402

FACTOR = 3.0  # how much worse than the bf16 flash-style implementation an output may be


def make_inputs(lay, heads, batch, topk, qk_scale, seed, device):
    """Inputs in run()'s order for a layout, drawn as spec.py draws them (a shared field over the video's positions
    and the text's order, plus noise), with q and k scaled by qk_scale."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, d = S.ROWS, S.HEAD_DIM
    p, n = lay.prefix_blocks, lay.blocks
    pos, video_valid = S._tile_positions(lay.thw, lay.grid, device)
    valid = torch.cat([torch.arange(p * rows, device=device) < lay.text, video_valid])
    x_video = pos.float() / torch.tensor(lay.thw, device=device, dtype=torch.float32)
    x_text = (torch.arange(p * rows, device=device, dtype=torch.float32) / max(lay.text, 1))[:, None]
    seq = n * rows
    q = torch.empty(batch, heads, seq, d, dtype=torch.bfloat16, device=device)
    k, v, gate, dout = (torch.empty_like(q) for _ in range(4))
    for b in range(batch):
        for h in range(heads):
            field = torch.cat([S._field(x_text, gen, device), S._field(x_video, gen, device)])
            for out in (q, k):
                noise = torch.randn(seq, d, generator=gen, device=device)
                out[b, h] = (qk_scale * (S.FIELD_WEIGHT * field + S.NOISE_WEIGHT * noise)).to(torch.bfloat16)
            v[b, h] = torch.randn(seq, d, generator=gen, device=device).to(torch.bfloat16)
            gate[b, h] = torch.tanh(torch.randn(seq, d, generator=gen, device=device)).to(torch.bfloat16)
            dout[b, h] = torch.randn(seq, d, generator=gen, device=device).to(torch.bfloat16)
    for x in (q, k, v, gate, dout):
        x[:, :, ~valid] = 0
    gate[:, :, : p * rows] = 0
    t, h_, w = lay.thw
    return [q, k, v, gate, dout, t, h_, w, lay.text, topk]


def worst(x, ref, rtol):
    """The atol every judged element needs at this rtol: max(|x - ref| - rtol |ref|) over them (inf when x is not
    finite)."""
    judged = ~torch.isnan(ref)
    x, ref = x.float()[judged], ref.float()[judged]
    if not bool(torch.isfinite(x).all()):
        return math.inf
    return float(((x - ref).abs() - rtol * ref.abs()).clamp(min=0).max()) if x.numel() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default="solution")
    ap.add_argument("--reference", default=os.path.join(os.path.dirname(HERE), "reference.py"))
    ap.add_argument("--workload", default=None, help="one of spec.py's workloads instead of the options below")
    ap.add_argument("--thw", default="9,10,21", help="the video latent's extent in tokens, T,H,W")
    ap.add_argument("--text", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--heads", type=int, default=S.HEADS)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--qk-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="build fresh instead of taking the node's cached build")
    a = ap.parse_args()

    import importlib.util

    ref_spec = importlib.util.spec_from_file_location("edge_reference", os.path.abspath(a.reference))
    R = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(R)
    dev = torch.device("cuda")
    if a.workload:
        inputs = S.make_inputs(a.workload, a.seed, dev)
        tol = S.WORKLOADS[a.workload]
    else:
        thw = tuple(int(x) for x in a.thw.split(","))
        lay0 = R.Layout(a.text, thw)
        inputs = make_inputs(lay0, a.heads, a.batch, a.topk, a.qk_scale, a.seed, dev)
        tol = next(iter(S.WORKLOADS.values()))
    q, k, v, gate, dout, t, h, w, text, topk = inputs
    lay = R.Layout(int(text), (int(t), int(h), int(w)))
    b_, h_, s, d = q.shape
    n, p = lay.blocks, lay.prefix_blocks
    scale = 1 / math.sqrt(d)
    print(f"input: batch {b_}, heads {h_}, text {text}, latent {t}x{h}x{w}, N = {p} + {lay.video_blocks}, topk {topk}, "
          f"q and k x{a.qk_scale if not a.workload else 1.0}, seed {a.seed}", flush=True)

    sol = loader.load(a.solution, cache=None if a.no_cache else loader.cache_dir())
    outs = sol(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
    torch.cuda.synchronize()
    refs = R.run(*inputs, outputs=[o.clone() for o in outs])

    # The selection the other outputs were judged under: the solution's own when valid, as the reference decides.
    valid = R.row_valid(lay, dev)
    blocks = [x.float().reshape(b_, h_, n, R.ROWS, d) for x in (q, k)]
    scores = R.router_scores(blocks[0][:, :, p:], blocks[1][:, :, p:], valid[p:], scale)
    mask = R.select_blocks(scores, topk)
    near = R.threshold_band(scores, topk)
    use = outs[1].to(dev) > 0.5 if R.valid_selection(outs[1].to(dev), mask, near, topk) else mask
    emu = E.emulate(*(x.reshape(b_, h_, n, R.ROWS, d) for x in (q, k, v, gate, dout)), lay, use, scale)

    failed = 0
    names = S.OUTPUTS
    sel_out, sel_ref = outs[1].float(), refs[1].float()
    judged = ~torch.isnan(sel_ref)
    sel_ok = bool(torch.isfinite(sel_out).all()) and bool(((sel_out - sel_ref).abs()[judged] <= 0.5).all())
    print(f"  selection  {'ok  ' if sel_ok else 'FAIL'}  ({int((~judged).sum())} pairs in the threshold band, "
          f"{'its own selection' if use is not mask else 'the reference selection'} used for the rest)")
    failed += not sel_ok
    emu_by_name = dict(zip(["out", "lse", "dq", "dk", "dv", "dgate"], emu))
    for name, x, ref in zip(names, outs, refs):
        if name == "selection":
            continue
        own = {"atol": tol["atol"], "rtol": tol["rtol"], **tol.get("outputs", {}).get(name, {})}
        ref = ref.reshape(x.shape)
        need = worst(x, ref, own["rtol"])
        need_emu = worst(emu_by_name[name].reshape(x.shape), ref, own["rtol"])
        allowed = max(own["atol"], FACTOR * need_emu)
        ok = need <= allowed
        failed += not ok
        print(f"  {name:9}  {'ok  ' if ok else 'FAIL'}  needs atol {need:.3g} at rtol {own['rtol']:g}; "
              f"allowed {allowed:.3g} (workload atol {own['atol']:g}, bf16 flash-style needs {need_emu:.3g})")
    print("passed" if not failed else f"FAILED: {failed} output(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
