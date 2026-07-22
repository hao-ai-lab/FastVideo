"""Decompose the DiT anchor delta (official vs ports) into its sources.

The bf16 anchor shows ~1.6-2.1e-2 rel L2 between the official WanModel and
both ports (diffusers, fastvideo main). This capture-env diagnostic separates
the candidate causes, each with its own experiment:

  MATH   fp32, no autocast, identical inputs: any residual is a real math /
         conversion difference (expected ~1e-6 if the ports are faithful).
         Also saves a permanent ``dit_fp32`` golden for an exact-math gate.
  PAD    official forward with the unpadded context (its native path — the
         model pads internally and may mask cross-attention to the true
         length) vs the zero-padded-512 context ports feed unmasked. If this
         reproduces the anchor delta, the divergence is the padding
         convention, and the port fix is to pass the unpadded context.
  CAST   official with fp32 x under autocast (how the golden was captured)
         vs pre-cast bf16 x (how the adapters call ports).
  ATTN   official with flash_attn (its default here) vs monkeypatched SDPA —
         the attention-backend share of the delta.

Run like capture_official.py (same env policy):
    PYTHONPATH=<wan-repo>:<extras>:<repo> python -m fastvideo2.wan21.probe_dit \\
        --wan-repo ... --ckpt-dir <official ckpt> --diffusers-root <diffusers ckpt> \\
        --goldens <repo>/fastvideo2/evidence/goldens/wan21-official
"""
from __future__ import annotations

import argparse
import re
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wan-repo", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--diffusers-root", required=True)
    p.add_argument("--goldens", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    sys.path.insert(0, args.wan_repo)

    import numpy as np
    import torch

    from fastvideo2.wan21 import goldens as G

    dev = args.device
    dit_g = G.load_golden(args.goldens, "dit")
    t5_g = G.load_golden(args.goldens, "text_encoder")
    x_np, ctx_pad = dit_g["x"], dit_g["context"]                  # [1,16,5,60,104], [512,4096]
    plen = int(t5_g["prompt_len"])
    ctx_unpad = ctx_pad[:plen]
    t750 = 750.0
    golden_bf16 = dit_g["out_t750"]
    _, _, T, H, W = x_np.shape
    seq_len = T * (H // 2) * (W // 2)
    rel = G.rel_l2

    # ---- 0. what does the official cross-attention do with context length? --
    src = open(f"{args.wan_repo}/wan/modules/model.py").read()
    for pat in (r".*context_lens.*", r".*k_lens.*"):
        hits = [l.strip() for l in src.splitlines() if re.match(pat, l.strip())][:4]
        for h in hits:
            print(f"[src] {h}")

    def off_forward(model, x_t, ctx_list, *, autocast=True, dtype=torch.bfloat16):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast):
            return model([x_t], t=torch.tensor([t750], device=dev, dtype=torch.float32),
                         context=ctx_list, seq_len=seq_len)[0].to(torch.float32).cpu().numpy()[None]

    # ---- official model, bf16 --------------------------------------------- #
    from wan.modules.model import WanModel
    off = WanModel.from_pretrained(args.ckpt_dir).to(dev).eval().requires_grad_(False)
    x32 = torch.from_numpy(x_np[0]).to(dev)
    up = torch.from_numpy(ctx_unpad).to(dev, torch.bfloat16)
    pad = torch.from_numpy(ctx_pad).to(dev, torch.bfloat16)

    base = off_forward(off, x32, [up])
    print(f"[sanity] official(bf16, unpadded ctx) vs golden : {rel(base, golden_bf16):.3e}")
    e_pad = off_forward(off, x32, [pad])
    print(f"[PAD]    official padded-512 ctx vs unpadded    : {rel(e_pad, base):.3e}")
    e_cast = off_forward(off, x32.to(torch.bfloat16), [up])
    print(f"[CAST]   official bf16-x vs fp32-x-autocast     : {rel(e_cast, base):.3e}")

    # attention backend share: swap official flash_attention for SDPA
    import wan.modules.attention as watt
    import wan.modules.model as wmodel
    orig_fa = watt.flash_attention

    def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None,
                       q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                       dtype=torch.bfloat16, version=None):
        # q/k/v: [B, L, H, C]; honor k_lens masking so semantics stay identical
        qh, kh, vh = (u.transpose(1, 2) for u in (q, k, v))       # [B, H, L, C]
        mask = None
        if k_lens is not None:
            L = kh.shape[2]
            idx = torch.arange(L, device=kh.device)[None]
            mask = (idx < k_lens.to(kh.device)[:, None])[:, None, None]   # [B,1,1,L]
        out = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2)

    watt.flash_attention = sdpa_attention
    for m in (watt, wmodel):
        if hasattr(m, "attention"):
            pass  # model.py may bind `flash_attention` directly; patched via watt
    if hasattr(wmodel, "flash_attention"):
        wmodel.flash_attention = sdpa_attention
    e_attn = off_forward(off, x32, [up])
    watt.flash_attention = orig_fa
    if hasattr(wmodel, "flash_attention"):
        wmodel.flash_attention = orig_fa
    print(f"[ATTN]   official sdpa vs flash                 : {rel(e_attn, base):.3e}")

    # ---- MATH: fp32, no autocast ------------------------------------------- #
    off_f32 = off.float()
    up32 = torch.from_numpy(ctx_unpad).to(dev, torch.float32)
    out_f32 = off_forward(off_f32, x32, [up32], autocast=False)
    G.save_golden(args.goldens, "dit_fp32", {"x": x_np, "context": ctx_pad,
                                             "context_len": np.int64(plen),
                                             "timestep": np.float32(t750), "out": out_f32},
                  {"note": "official WanModel fp32 forward, no autocast — exact-math golden"})
    print(f"[MATH]   saved dit_fp32 golden (std {out_f32.std():.4f})")
    del off, off_f32
    torch.cuda.empty_cache()

    # ---- diffusers port ----------------------------------------------------- #
    from diffusers import WanTransformer3DModel
    dif = WanTransformer3DModel.from_pretrained(args.diffusers_root, subfolder="transformer",
                                                torch_dtype=torch.bfloat16).to(dev).eval()

    def dif_forward(model, x_t, ctx_t):
        tt = torch.tensor([t750], device=dev, dtype=torch.float32)
        with torch.no_grad():
            return model(hidden_states=x_t, timestep=tt, encoder_hidden_states=ctx_t,
                         return_dict=False)[0].to(torch.float32).cpu().numpy()

    xb = torch.from_numpy(x_np).to(dev, torch.bfloat16)
    d_pad = dif_forward(dif, xb, pad[None])
    d_unpad = dif_forward(dif, xb, up[None])
    print(f"[port]   diffusers padded-512  vs golden        : {rel(d_pad, golden_bf16):.3e}")
    print(f"[port]   diffusers UNPADDED    vs golden        : {rel(d_unpad, golden_bf16):.3e}")

    import copy
    dif32 = copy.deepcopy(dif).float()
    # padded-512 context: the pad rows are attended, trained-in semantics
    # (see the [PAD]/[port] rows) — the fp32 math test must use them too.
    d_f32 = dif_forward(dif32, torch.from_numpy(x_np).to(dev, torch.float32),
                        torch.from_numpy(ctx_pad).to(dev, torch.float32)[None])
    print(f"[MATH]   diffusers fp32 vs official fp32        : {rel(d_f32, out_f32):.3e}")


if __name__ == "__main__":
    main()
