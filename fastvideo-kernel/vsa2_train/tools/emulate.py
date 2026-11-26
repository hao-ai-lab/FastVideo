"""A bf16 flash-style implementation of the training step, in torch, against which a solution's error is measured on
inputs beyond the five workloads (see edge.py).

It computes what reference.py defines, as a FlashAttention-style kernel rounds it: the fine branch's scores in fp32
from the bf16 inputs; P rounded to bf16 before the PV product and normalized by the fp32 row sum at the end; in the
backward, P recomputed from lse, D from the bf16 output less the gated coarse share, dS rounded to bf16 before the dQ
and dK products and P before the dV product; fp32 accumulation and statistics; the coarse branch in fp32; bf16
outputs. Its backward is written out by hand, so it is also an independent check of the reference's gradients.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reference as R  # noqa: E402


def bf16(x):
    return x.to(torch.bfloat16).float()


def batches(mask, n):
    """The reference's batching of query blocks by how many key blocks they keep: (query blocks, gathered key order,
    picked) per batch, for one head's [N, N] mask."""
    counts = mask.sum(-1)
    rows = torch.argsort(counts, stable=True)
    c = counts[rows].tolist()
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and (stop + 1 - start) * c[stop] <= R.FINE_BATCH_PAIRS:
            stop += 1
        part, width = rows[start:stop], c[stop - 1]
        start = stop
        order = torch.where(mask[part], torch.arange(n, device=mask.device), n).sort(-1).values[:, :width]
        picked = order < n
        yield part, torch.where(picked, order, 0), picked


@torch.no_grad()
def emulate(q, k, v, gate, dout, lay, selection, scale):
    """q, k, v, gate, dout [B, H, N, 128, D] bf16 on blocks, selection [B, H, N_v, N_v] bool -> out, lse, dq, dk, dv,
    dgate on blocks (bf16, except lse fp32)."""
    b_, h_, n, rows, d = q.shape
    valid = R.row_valid(lay, q.device)
    mask = R.block_mask(selection, lay)
    keep = valid[None, None, :, :, None]
    qf, kf, vf, gf, dof = (x.float() for x in (q, k, v, gate, dout))

    # The coarse branch in fp32, and its share of the gradients by autograd: it is small, and not where a kernel's
    # rounding lies.
    with torch.enable_grad():
        leaves = [x.clone().requires_grad_() for x in (qf, kf, vf)]
        coarse = R.coarse_branch(*leaves, valid, scale)  # [B, H, N, D]
        rows_c = coarse[:, :, :, None, :].expand(b_, h_, n, rows, d)
        dgate = torch.where(keep, rows_c * dof, 0.0).detach()
        shares = torch.autograd.grad((rows_c * gf * dof * keep).sum(), leaves)
    coarse = coarse.detach()

    fine = torch.zeros(b_, h_, n, rows, d, device=q.device)
    lse = torch.zeros(b_, h_, n, rows, device=q.device)
    for bi in range(b_):
        for hi in range(h_):
            for part, order, picked in batches(mask[bi, hi], n):
                c, width = len(part), order.shape[1]
                usable = (picked[:, :, None] & valid[order]).reshape(c, width * rows)
                keys = kf[bi, hi][order].reshape(c, width * rows, d)
                values = vf[bi, hi][order].reshape(c, width * rows, d)
                s = (qf[bi, hi, part] @ keys.transpose(-1, -2)) * scale
                s = s.masked_fill(~usable[:, None, :], float("-inf"))
                m = s.amax(-1, keepdim=True)
                p = torch.exp(s - m)
                total = p.sum(-1, keepdim=True)
                fine[bi, hi, part] = (bf16(p) @ values) / total
                lse[bi, hi, part] = (m + torch.log(total))[..., 0]
    out = bf16(torch.where(keep, fine + gf * coarse[:, :, :, None, :], 0.0))

    big_d = (dof * out).sum(-1) - (dof * gf * coarse[:, :, :, None, :]).sum(-1)  # [B, H, N, 128]
    dq, dk, dv = (torch.zeros(b_, h_, n, rows, d, device=q.device) for _ in range(3))
    for bi in range(b_):
        for hi in range(h_):
            for part, order, picked in batches(mask[bi, hi], n):
                c, width = len(part), order.shape[1]
                usable = (picked[:, :, None] & valid[order]).reshape(c, width * rows)
                qb, dob = qf[bi, hi, part], dof[bi, hi, part]
                keys = kf[bi, hi][order].reshape(c, width * rows, d)
                values = vf[bi, hi][order].reshape(c, width * rows, d)
                s = (qb @ keys.transpose(-1, -2)) * scale
                s = s.masked_fill(~usable[:, None, :], float("-inf"))
                p = torch.exp(s - lse[bi, hi, part][..., None])
                p = torch.where(valid[part][:, :, None], p, 0.0)
                ds = bf16(p * (dob @ values.transpose(-1, -2) - big_d[bi, hi, part][..., None]))
                dq[bi, hi, part] += (ds @ keys) * scale
                flat, take = order.reshape(-1), picked.reshape(-1)
                dkk = ((ds.transpose(-1, -2) @ qb) * scale).reshape(c * width, rows, d)
                dvv = (bf16(p).transpose(-1, -2) @ dob).reshape(c * width, rows, d)
                dk[bi, hi].index_add_(0, flat[take], dkk[take])
                dv[bi, hi].index_add_(0, flat[take], dvv[take])
    grads = [bf16(torch.where(keep, g + share, 0.0)) for g, share in zip((dq, dk, dv), shares)]
    return out, lse, *grads, bf16(dgate)
