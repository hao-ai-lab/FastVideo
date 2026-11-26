"""One training step of VSA2 attention in a video DiT's joint [text; video] attention, on a GB200 GPU.

A request is one attention layer of a video DiT with 32 heads of dimension 128 at 704x1248, during distillation
training, on one GPU of a sequence-
parallel group of 8: the GPU holds 4 heads of the whole sequence. The sequence is the text tokens (up to 1,536) and
the video latent of T x 22 x 39 tokens (T = 6 * seconds + 1 latent frames at 24 fps), cut into blocks of 128 rows as
VSA-H3 tiles it: the text in order (the last block padded with zero rows), then the video in tiles of 4 x 4 x 8
tokens, tiles in raster order of the tile grid and the tokens of a tile in raster order of the tile, the latent padded
up to whole tiles with zero rows. VSA2's router pools Q and K over windows of 2 x 2 x 2 tokens (16 per tile) and keeps,
per head, the N_v * topk video tile pairs it scores highest plus every diagonal pair; text queries attend densely and
every video query keeps every text block (see reference.py). topk is VSA-H3's compute_topk at the recipe's sparsity
0.9: ceil(0.1 * N_v).

Inputs, in run()'s order:
    q, k, v      [1, 4, S, 128] bf16, contiguous; S = N * 128 for the N = P + N_v blocks; padding rows zero
    gate         [1, 4, S, 128] bf16: to_gate_compress's output per row, head and channel; zero on text and padding rows
    dout         [1, 4, S, 128] bf16: the gradient of the loss with respect to out; zero on padding rows
    t, h, w      int: the video latent's extent in tokens
    text         int: the text tokens, L_text (P = ceil(L_text / 128) text blocks)
    topk         int: video tiles selected per video query tile, on average over the head
Outputs:
    out          [1, 4, S, 128] bf16: O = O_fine + gate * O_coarse; zero on padding rows
    selection    [1, 4, N_v, N_v] float32: 1 for each video tile pair the router selected, 0 otherwise
    lse          [1, 4, S] float32: each valid row's natural log-sum-exp of its scaled fine-branch scores
    dq, dk, dv   [1, 4, S, 128] bf16: the gradients of sum(out * dout); zero on padding rows
    dgate        [1, 4, S, 128] bf16: likewise for the gate (not judged on text rows)

The inputs are synthetic: in each head, queries and keys share a random field over the video tokens' positions whose
similarity decays with distance (random Fourier features of a Gaussian kernel), plus noise, so that the router keeps
mostly nearby tiles, as attention with rotary positions does; the text tokens share a field over their order; values,
gates and dout are noise. A kernel must be correct for any input.
"""

import math

import torch

OUTPUTS = ["out", "selection", "lse", "dq", "dk", "dv", "dgate"]

TILE = (4, 4, 8)
WINDOW = (2, 2, 2)
ROWS = 128
HEADS, HEAD_DIM = 4, 128
FRAME = (22, 39)  # tokens per latent frame at 704x1248, after the VAE and the (1, 2, 2) patching
SPARSITY = 0.9

# Each workload is one attention layer of the student at a clip length (latent frames T = 6 * seconds + 1) and a
# prompt length; s8 is the clip its recipe trains on.
WORKLOADS = {
    "s8": {"frames": 49, "text": 512},
    "s8t1536": {"frames": 49, "text": 1536},
    "s12": {"frames": 73, "text": 512},
    "s16": {"frames": 97, "text": 512},
    "s20": {"frames": 121, "text": 512},
}
# Tolerances, |x - ref| <= atol + rtol * |ref| on every judged element. A bf16 flash-style forward and backward (P and
# dS rounded to bf16 for the products, fp32 accumulation and statistics, bf16 outputs) was measured against this
# reference on every workload: out and dgate need atol 1.0e-3 at rtol 2e-2, dq, dk and dv 3.4e-3, and lse is exact
# to fp32. out and dgate get the attention output's tolerance, the gradients three times what they need, lse that of
# fp32 scores of bf16 inputs.
for _w in WORKLOADS.values():
    _w.update(atol=2e-3, rtol=2e-2, match=1.0)
    _w["outputs"] = {"lse": {"atol": 1e-3, "rtol": 1e-5}, **{g: {"atol": 1e-2} for g in ("dq", "dk", "dv")}}

# The field the queries and keys share: its features, its length scale (in units of the latent's extent along each
# axis, or of the text's length), and how strongly queries and keys follow it.
FIELD_FEATURES = 128
FIELD_SCALE = 0.12
FIELD_WEIGHT, NOISE_WEIGHT = 0.6, 1.0


def topk_of(video_blocks):
    """VSA-H3's compute_topk at SPARSITY: the tiles a query tile keeps, on average."""
    return max(1, min(math.ceil((1 - SPARSITY) * video_blocks), video_blocks))


def geometry(workload):
    """(T, H, W), the tile grid, P (text blocks), N_v (video tiles) and N."""
    entry = WORKLOADS[workload]
    thw = (entry["frames"], *FRAME)
    grid = tuple(math.ceil(n / b) for n, b in zip(thw, TILE))
    p = math.ceil(entry["text"] / ROWS)
    nv = math.prod(grid)
    return thw, grid, p, nv, p + nv


def _tile_positions(thw, grid, device):
    """Each video slot's (t, h, w), [N_v * 128, 3] long, and whether it is a real token."""
    nt, nh, nw = grid
    bt, bh, bw = TILE
    tile = torch.arange(nt * nh * nw, device=device)
    slot = torch.arange(ROWS, device=device)
    origin = torch.stack([tile // (nh * nw) * bt, tile // nw % nh * bh, tile % nw * bw], -1)
    local = torch.stack([slot // (bh * bw), slot // bw % bh, slot % bw], -1)
    pos = (origin[:, None, :] + local[None, :, :]).reshape(-1, 3)
    return pos, (pos < torch.tensor(thw, device=device)).all(-1)


def _field(x, gen, device):
    """A random field over positions x [M, A] in [0, 1]: [M, HEAD_DIM] fp32."""
    freq = torch.randn(FIELD_FEATURES, x.shape[1], generator=gen, device=device) / FIELD_SCALE
    phase = torch.rand(FIELD_FEATURES, generator=gen, device=device) * 2 * math.pi
    mix = torch.randn(FIELD_FEATURES, HEAD_DIM, generator=gen, device=device) * math.sqrt(2 / FIELD_FEATURES)
    return torch.cos(x @ freq.T + phase) @ mix


def make_inputs(workload, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    thw, grid, p, nv, n = geometry(workload)
    text = WORKLOADS[workload]["text"]
    seq = n * ROWS
    pos, video_valid = _tile_positions(thw, grid, device)
    valid = torch.cat([torch.arange(p * ROWS, device=device) < text, video_valid])
    x_video = pos.float() / torch.tensor(thw, device=device, dtype=torch.float32)
    x_text = (torch.arange(p * ROWS, device=device, dtype=torch.float32) / max(text, 1))[:, None]

    q = torch.empty(1, HEADS, seq, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k, v, gate, dout = (torch.empty_like(q) for _ in range(4))
    for head in range(HEADS):
        field = torch.cat([_field(x_text, gen, device), _field(x_video, gen, device)])
        for out in (q, k):
            noise = torch.randn(seq, HEAD_DIM, generator=gen, device=device)
            out[0, head] = (FIELD_WEIGHT * field + NOISE_WEIGHT * noise).to(torch.bfloat16)
        v[0, head] = torch.randn(seq, HEAD_DIM, generator=gen, device=device).to(torch.bfloat16)
        gate[0, head] = torch.tanh(torch.randn(seq, HEAD_DIM, generator=gen, device=device)).to(torch.bfloat16)
        dout[0, head] = torch.randn(seq, HEAD_DIM, generator=gen, device=device).to(torch.bfloat16)
    for x in (q, k, v, gate, dout):
        x[0, :, ~valid] = 0
    gate[0, :, : p * ROWS] = 0  # the text rows have no compression gate
    t, h, w = thw
    return [q, k, v, gate, dout, t, h, w, text, topk_of(nv)]


def make_outputs(inputs):
    q, text = inputs[0], inputs[8]
    p = math.ceil(text / ROWS)
    nv = q.shape[2] // ROWS - p
    b_, h_, s, _ = q.shape
    return [
        torch.zeros_like(q),
        torch.zeros(b_, h_, nv, nv, dtype=torch.float32, device=q.device),
        torch.zeros(b_, h_, s, dtype=torch.float32, device=q.device),
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros_like(q),
    ]


def bounds(workload):
    """FLOPs, exponentials and bytes of one call, whole and by stage, counting whole blocks of the kept pairs.

    - router: pooled Q K^T over every video window pair, one exponential each;
    - fwd: the fine branch's QK^T and PV over the kept block pairs (every text block's N, every video tile's text
      blocks, its N_v * topk / N_v selected tiles and its diagonal), one exponential per score;
    - bwd: the backward's five products per kept pair (QK^T again, dP = dO V^T, dV += P^T dO, dQ += dS K,
      dK += dS^T Q) and one exponential per score;
    - bytes: q, k, v, gate and dout read, out, lse, the selection and the four gradients written, once each.
    The stages are matched by kernel name: a router's kernels are named *router*, the forward's *fwd*, the
    backward's *bwd*."""
    thw, grid, p, nv, n = geometry(workload)
    g = math.prod(b // r for b, r in zip(TILE, WINDOW))
    topk = topk_of(nv)
    pairs = p * n + nv * (p + topk + 1)
    product = 2 * ROWS * ROWS * HEAD_DIM  # one 128 x 128 x 128 matrix product of a block pair
    router = {"flops": 2 * (nv * g) ** 2 * HEAD_DIM * HEADS, "exps": (nv * g) ** 2 * HEADS}
    fwd = {"flops": 2 * product * pairs * HEADS, "exps": ROWS * ROWS * pairs * HEADS}
    bwd = {"flops": 5 * product * pairs * HEADS, "exps": ROWS * ROWS * pairs * HEADS}
    seq = n * ROWS
    tensor = HEADS * seq * HEAD_DIM * 2
    nbytes = 10 * tensor + HEADS * seq * 4 + HEADS * nv * nv * 4
    return {
        "flops": router["flops"] + fwd["flops"] + bwd["flops"],
        "bytes": nbytes,
        "stages": [
            {"name": "router", "match": "router", "flops": router["flops"], "exps": router["exps"]},
            {"name": "fwd", "match": "fwd", "flops": fwd["flops"], "exps": fwd["exps"], "bytes": 5 * tensor},
            {"name": "bwd", "match": "bwd", "flops": bwd["flops"], "exps": bwd["exps"], "bytes": 10 * tensor},
        ],
    }
