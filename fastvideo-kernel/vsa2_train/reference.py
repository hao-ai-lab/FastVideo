"""What run() computes and how a solution's outputs are judged (tools/spec.py gives the inputs and outputs).

run() is one training step of VSA2 attention in a joint [text; video] transformer: the forward, then the backward of
the loss sum(out * dout). VSA2 is Zhang et al., "Improving Video Sparse Attention with Fine-grained Router and Sparse
Rebasing" (ICML 2026), section 3.2 and the pseudocode of its Figure 12; the joint layout is FastVideo's VSA-H3 backend
(fastvideo/attention/backends/video_sparse_attn_h3.py), whose 128-token tiles a video DiT's distillation student trains with. All
of it is computed here in fp32, from fp32 copies of the inputs, with autograd for the backward.

- **Blocks.** The sequence is cut into N blocks of 128 rows. The first P = ceil(L_text / 128) hold the text tokens
  in order (the last one padded with zero rows). The other N_v hold the video latent of T x H x W tokens in tiles of
  4 x 4 x 8 tokens: tiles in raster order of the tile grid, the tokens of a tile in raster order of the tile, and a
  latent the tile does not divide padded up to whole tiles with zero rows. Padding rows are never attended to.
- **Coarse branch** (eq. 1-4): mean-pooled Q, K and V per block (text blocks and video tiles alike), dense attention
  between all N blocks, each block's output broadcast back to its rows.
- **Router** (eq. 5-9), without gradients, over the video alone: Q and K mean-pooled over windows of 2 x 2 x 2
  tokens, G = 16 windows per tile; a softmax of each query window over every key window of the video; the
  probabilities pooled to tile pairs (the sum over the G x G window pairs divided by G x G); and a top-K over all
  N_v x N_v tile pairs of a head (per head, not per query tile): N_v * topk pairs, plus every diagonal pair.
- **Fine branch** (eq. 12): exact attention of each query row over the valid rows of the key blocks its block keeps.
  A text block keeps every block: text queries attend densely. A video tile keeps every text block (the text keys are
  exempt from the top-K, as VSA-H3's default "exempt" mode has them) and the video tiles the router selected for it.
- **Gated merge** (eq. 13-14): O = O_fine + gate * O_coarse, with the gate given per row, head and channel (the
  output of the layer's to_gate_compress); the joint attention's text rows have no gate, so the gate is zero there.
- **Backward.** The router's selection takes no gradient (the paper's router is gradient-free by design); gradients
  flow through the fine branch, the coarse branch (through its pooling and its softmax) and the gate.

What the paper and VSA-H3 leave open, and this reference fixes:

- A pooled vector is the mean over the valid rows of its block or window. A router window with no valid token is not
  a key of the router and contributes nothing as a query.
- The router's softmax runs over all key windows of the video, as the paper's two-pass kernel implies; text keys do
  not enter it, since they do not compete for the top-K.
- The pooled score of a tile pair divides by G x G even when some windows are invalid, so a tile with fewer valid
  tokens scores lower, as eq. 7's mean over a fixed G x G does.
- Ties in the top-K are broken by torch.topk; a check of another implementation must accept any selection that agrees
  with this one up to the scores' rounding (see `selection_agrees`).
- Every block holds at least one valid row: padding is shorter than a tile along each axis and the last text block
  holds the last text tokens. So every video tile keeps at least itself, and no softmax runs over an empty set.
- `lse` is the natural log-sum-exp of each valid row's scaled fine-branch scores, the statistic a forward saves for
  its backward.

How a solution is judged:
- `selection` is judged on every tile pair except those whose router score lies within MARGIN_RTOL of the head's
  threshold (the K-th largest score), NaN there.
- When the harness passes the solution's outputs and its selection is a valid top-K of these scores up to that
  rounding (every diagonal pair, between N_v * topk and N_v * topk + N_v pairs per head, and the same pairs as this
  reference away from the threshold), everything else is computed under the solution's own selection; otherwise under
  this reference's, so that a wrong selection also fails the other outputs.
- `out`, `dq`, `dk`, `dv` and `dgate` are judged on every row: valid rows against this reference, padding rows, which
  must be zero, too. `lse` is judged on the valid rows, and `dgate` on the video rows (the text rows' gate is zero).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

TILE = (4, 4, 8)
WINDOW = (2, 2, 2)
ROWS = 128  # rows per block: a video tile's tokens, or a run of text tokens

# Selected (query block, key block) pairs per batch of the fine branch: a memory bound, not a semantic one. Each
# batch is recomputed in the backward (Recompute) instead of keeping its scores.
FINE_BATCH_PAIRS = 4096
# Router logits per chunk of its softmax, over all batches and heads (2^28 fp32 values = 1 GiB): a memory bound.
ROUTER_CHUNK_ELEMENTS = 1 << 28
# The rounding band of the router's scores around a head's threshold: pairs inside it are not judged in `selection`,
# and a solution's selection may differ from this one there.
MARGIN_RTOL, MARGIN_ATOL = 1e-2, 1e-12


@dataclass(frozen=True)
class Layout:
    """A joint sequence: `text` text tokens, then a video latent of `thw` tokens in tiles of TILE."""

    text: int
    thw: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.text < 0 or len(self.thw) != 3 or any(int(d) <= 0 for d in self.thw):
            raise ValueError(f"expected text >= 0 and three positive latent sizes, got {self.text} and {self.thw}")

    @property
    def grid(self) -> tuple[int, int, int]:
        """Tiles along t, h and w."""
        return tuple(math.ceil(n / b) for n, b in zip(self.thw, TILE))

    @property
    def prefix_blocks(self) -> int:
        """P, the text blocks."""
        return math.ceil(self.text / ROWS)

    @property
    def video_blocks(self) -> int:
        """N_v, the video tiles."""
        return math.prod(self.grid)

    @property
    def blocks(self) -> int:
        """N = P + N_v."""
        return self.prefix_blocks + self.video_blocks

    @property
    def seq(self) -> int:
        """S, the rows of the sequence, padding included."""
        return self.blocks * ROWS

    @property
    def windows_per_tile(self) -> int:
        """G, router windows per video tile."""
        return math.prod(b // r for b, r in zip(TILE, WINDOW))


def tile_positions(lay: Layout, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Each video slot's (t, h, w), [N_v, 128, 3] long, and whether it is a real token, [N_v, 128] bool."""
    nt, nh, nw = lay.grid
    bt, bh, bw = TILE
    tile = torch.arange(nt * nh * nw, device=device)
    slot = torch.arange(ROWS, device=device)
    origin = torch.stack([tile // (nh * nw) * bt, tile // nw % nh * bh, tile % nw * bw], -1)
    local = torch.stack([slot // (bh * bw), slot // bw % bh, slot % bw], -1)
    pos = origin[:, None, :] + local[None, :, :]
    return pos, (pos < torch.tensor(lay.thw, device=device)).all(-1)


def row_valid(lay: Layout, device=None) -> torch.Tensor:
    """[N, 128] bool: whether each row of each block is a real token, not padding."""
    text = (torch.arange(lay.prefix_blocks * ROWS, device=device) < lay.text).reshape(lay.prefix_blocks, ROWS)
    return torch.cat([text, tile_positions(lay, device)[1]])


def window_of_slot(device=None) -> torch.Tensor:
    """[128] long: the router window (0..G-1) of each slot of a video tile, windows in raster order of the tile."""
    (bt, bh, bw), (rt, rh, rw) = TILE, WINDOW
    t = torch.arange(bt, device=device)[:, None, None] // rt
    h = torch.arange(bh, device=device)[None, :, None] // rh
    w = torch.arange(bw, device=device)[None, None, :] // rw
    return ((t * (bh // rh) + h) * (bw // rw) + w).reshape(-1)


def block_means(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Means over the valid rows of each block: x [..., N, 128, D], valid [N, 128] -> [..., N, D]."""
    weight = valid.to(x.dtype)
    return torch.einsum("...nrd,nr->...nd", x, weight) / weight.sum(-1).clamp(min=1)[:, None]


def window_means(x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Means over the valid tokens of each router window of each tile: x [..., N_v, 128, D], valid [N_v, 128] ->
    ([..., N_v, G, D], [N_v, G] bool: whether the window holds a valid token)."""
    g = math.prod(b // r for b, r in zip(TILE, WINDOW))
    onehot = F.one_hot(window_of_slot(x.device), g).to(x.dtype)  # [128, G]
    weight = valid.to(x.dtype)[:, :, None] * onehot[None]  # [N_v, 128, G]
    total = torch.einsum("...nrd,nrg->...ngd", x, weight)
    count = weight.sum(1)  # [N_v, G]
    return total / count.clamp(min=1)[..., None], count > 0


def coarse_branch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor, scale: float) -> torch.Tensor:
    """Eq. 1-3: dense attention between the blocks' mean-pooled Q, K and V; [..., N, 128, D] -> [..., N, D]."""
    qc, kc, vc = (block_means(t, valid) for t in (q, k, v))
    return torch.softmax(qc @ kc.transpose(-1, -2) * scale, dim=-1) @ vc


@torch.no_grad()
def router_scores(q: torch.Tensor, k: torch.Tensor, valid: torch.Tensor, scale: float) -> torch.Tensor:
    """Eq. 5-7: the pooled router score of every video tile pair; q, k [B, H, N_v, 128, D] of the video tiles,
    valid [N_v, 128] -> [B, H, N_v, N_v] fp32.

    Each valid query window takes a softmax over every valid key window of the video; a tile pair's score is the sum
    of its G x G window pairs' probabilities divided by G x G."""
    qr, live = window_means(q.float(), valid)  # [B, H, N_v, G, D], [N_v, G]
    kr, _ = window_means(k.float(), valid)
    b_, h_, n, g, d = qr.shape
    qr, kr = qr.reshape(b_, h_, n * g, d), kr.reshape(b_, h_, n * g, d)
    live = live.reshape(n * g)
    key_bias = torch.where(live, 0.0, float("-inf"))
    scores = torch.zeros(b_, h_, n, n, dtype=torch.float32, device=q.device)
    step = max(g, ROUTER_CHUNK_ELEMENTS // (b_ * h_ * n * g) // g * g)
    for start in range(0, n * g, step):
        stop = min(start + step, n * g)
        p = torch.softmax(qr[:, :, start:stop] @ kr.transpose(-1, -2) * scale + key_bias, dim=-1)
        p = p * live[start:stop, None]  # invalid query windows contribute nothing
        rows = (stop - start) // g
        scores[:, :, start // g : start // g + rows] = p.reshape(b_, h_, rows, g, n, g).sum(dim=(3, 5)) / (g * g)
    return scores


@torch.no_grad()
def select_blocks(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Eq. 8-9: per head, the top N_v * topk tile pairs, plus every diagonal pair; [B, H, N_v, N_v] scores ->
    [B, H, N_v, N_v] bool."""
    b_, h_, n, _ = scores.shape
    k = min(max(int(topk), 0) * n, n * n)
    mask = torch.zeros(b_, h_, n * n, dtype=torch.bool, device=scores.device)
    if k:
        mask.scatter_(-1, scores.reshape(b_, h_, n * n).topk(k, dim=-1).indices, True)
    mask = mask.reshape(b_, h_, n, n)
    mask[:, :, torch.arange(n), torch.arange(n)] = True
    return mask


def block_mask(selection: torch.Tensor, lay: Layout) -> torch.Tensor:
    """The key blocks each query block attends in the fine branch: [B, H, N_v, N_v] selection -> [B, H, N, N] bool.
    Text query blocks keep every block; video tiles keep every text block and the tiles selected for them."""
    b_, h_ = selection.shape[:2]
    p = lay.prefix_blocks
    mask = torch.ones(b_, h_, lay.blocks, lay.blocks, dtype=torch.bool, device=selection.device)
    mask[:, :, p:, p:] = selection
    return mask


class Recompute(torch.autograd.Function):
    """Runs `fn(*tensors)` without keeping its intermediates, and again in the backward to differentiate it: the memory
    of one batch of the fine branch instead of all of them. (torch.utils.checkpoint would do the same, but it imports
    torch's compiler, which the harness's own profile.py, beside this file in checks/, breaks.)"""

    @staticmethod
    def forward(ctx, fn, *tensors):
        ctx.fn = fn
        ctx.save_for_backward(*tensors)
        with torch.no_grad():
            return fn(*tensors)

    @staticmethod
    def backward(ctx, *grads):
        tensors = [t.detach().requires_grad_(t.requires_grad) for t in ctx.saved_tensors]
        with torch.enable_grad():
            outs = ctx.fn(*tensors)
        pairs = [(o, g) for o, g in zip(outs, grads) if g is not None]
        wanted = [t for t in tensors if t.requires_grad]
        got = iter(torch.autograd.grad([o for o, _ in pairs], wanted, [g for _, g in pairs], allow_unused=True))
        return (None, *[next(got) if t.requires_grad else None for t in tensors])


def _attend(qb: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, usable: torch.Tensor, scale: float):
    """Attention of a batch of query blocks over their gathered keys: qb [C, 128, D], keys and values [C, W, D],
    usable [C, W] bool -> (out [C, 128, D], lse [C, 128])."""
    s = (qb @ keys.transpose(-1, -2)) * scale
    s = s.masked_fill(~usable[:, None, :], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    return torch.exp(s - lse[..., None]) @ values, lse


def fine_branch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 12: each query row's attention over the valid rows of the key blocks its block keeps.

    q, k, v [B, H, N, 128, D], mask [B, H, N, N] -> (out [B, H, N, 128, D], lse [B, H, N, 128]), differentiable.
    Query blocks are batched by how many key blocks they keep; each batch is recomputed in the backward."""
    b_, h_, n, rows, d = q.shape
    counts = mask.sum(-1)  # [B, H, N]
    outs, lses = [], []
    for bi in range(b_):
        for hi in range(h_):
            order_rows = torch.argsort(counts[bi, hi], stable=True)
            c = counts[bi, hi, order_rows].tolist()
            qh, kh, vh = q[bi, hi], k[bi, hi], v[bi, hi]
            pieces_o, pieces_l, start = [], [], 0
            while start < n:
                # The counts ascend: a batch takes query blocks while all of them at its largest count fit.
                stop = start + 1
                while stop < n and (stop + 1 - start) * c[stop] <= FINE_BATCH_PAIRS:
                    stop += 1
                part = order_rows[start:stop]
                width = c[stop - 1]
                start = stop
                # Each row's kept blocks first, ascending; the rest of the row repeats block 0, masked out.
                kept = mask[bi, hi, part]  # [C, N]
                order = torch.where(kept, torch.arange(n, device=q.device), n).sort(-1).values[:, :width]
                picked = order < n
                order = torch.where(picked, order, 0)
                usable = (picked[:, :, None] & valid[order]).reshape(len(part), width * rows)

                def batch(qh, kh, vh, part=part, order=order, usable=usable, width=width):
                    keys = kh[order].reshape(len(part), width * rows, d)
                    values = vh[order].reshape(len(part), width * rows, d)
                    return _attend(qh[part], keys, values, usable, scale)

                o, lse = Recompute.apply(batch, qh, kh, vh)
                pieces_o.append(o)
                pieces_l.append(lse)
            inverse = torch.argsort(order_rows)
            outs.append(torch.cat(pieces_o)[inverse])
            lses.append(torch.cat(pieces_l)[inverse])
    return torch.stack(outs).reshape(b_, h_, n, rows, d), torch.stack(lses).reshape(b_, h_, n, rows)


def forward(q, k, v, gate, lay: Layout, selection: torch.Tensor, scale: float):
    """VSA2 on blocks under a given video selection: q, k, v, gate [B, H, N, 128, D] (fp32, padding rows zero) ->
    (out [B, H, N, 128, D] with zero padding rows, lse [B, H, N, 128]); differentiable in q, k, v and gate."""
    valid = row_valid(lay, q.device)
    fine, lse = fine_branch(q, k, v, block_mask(selection, lay), valid, scale)
    coarse = coarse_branch(q, k, v, valid, scale)
    out = fine + gate * coarse[:, :, :, None, :]
    return torch.where(valid[None, None, :, :, None], out, 0.0), lse


def selection_agrees(
    scores: torch.Tensor, mask: torch.Tensor, other: torch.Tensor, topk: int, rtol: float = 1e-3, atol: float = 1e-9
) -> tuple[bool, str]:
    """Whether `other`, a selection another implementation made, is a top-K of these scores up to their rounding.

    A valid selection is some top-K set of N_v * topk pairs per head, plus every diagonal pair. So `other` must hold
    every diagonal pair and between N_v * topk and N_v * topk + N_v pairs per head, and every pair on which it differs
    from `mask` must score within `rtol` (relative) or `atol` of the K-th largest score of its head."""
    if other.shape != mask.shape or other.dtype != torch.bool:
        return False, f"the selection is {tuple(other.shape)} {other.dtype}, expected {tuple(mask.shape)} bool"
    n = mask.shape[-1]
    if not bool(other[:, :, torch.arange(n), torch.arange(n)].all()):
        return False, "a diagonal tile pair is not selected"
    k = min(max(int(topk), 0) * n, n * n)
    count = other.sum((-1, -2))
    if bool((count < k).any() or (count > min(k + n, n * n)).any()):
        return False, f"a head selects {int(count.min())}..{int(count.max())} pairs, outside [{k}, {k + n}]"
    differs = other ^ mask
    if k == 0 or k == n * n:
        return not bool(differs.any()), "the selection differs where every pair or only the diagonal is kept"
    flat = scores.reshape(*scores.shape[:2], n * n)
    threshold = flat.topk(k, dim=-1).values[..., -1:].reshape(*scores.shape[:2], 1, 1)
    near = (scores - threshold).abs() <= rtol * threshold.abs() + atol
    bad = differs & ~near
    if bool(bad.any()):
        return False, f"{int(bad.sum())} pairs differ from the reference by more than the rounding of the scores"
    return True, f"{int(differs.sum())} pairs differ, all at the threshold"


def threshold_band(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """[B, H, N_v, N_v] bool: the off-diagonal pairs whose score is within the rounding band of their head's K-th
    largest score (empty when every pair or none is kept)."""
    b_, h_, n, _ = scores.shape
    k = min(max(int(topk), 0) * n, n * n)
    if k == 0 or k == n * n:
        return torch.zeros_like(scores, dtype=torch.bool)
    thr = scores.reshape(b_, h_, n * n).topk(k, dim=-1).values[..., -1:].reshape(b_, h_, 1, 1)
    near = (scores - thr).abs() <= MARGIN_RTOL * thr.abs() + MARGIN_ATOL
    near[:, :, torch.arange(n), torch.arange(n)] = False
    return near


def valid_selection(other: torch.Tensor, mask: torch.Tensor, near: torch.Tensor, topk: int) -> bool:
    """Whether a solution's selection (float, 1 for selected) is a valid top-K that agrees with `mask` away from the
    threshold band `near`."""
    n = mask.shape[-1]
    if other.shape != mask.shape:
        return False
    other = other > 0.5
    k = min(max(int(topk), 0) * n, n * n)
    count = other.sum((-1, -2))
    diagonal = bool(other[:, :, torch.arange(n), torch.arange(n)].all())
    counted = bool(((count >= k) & (count <= min(k + n, n * n))).all())
    return diagonal and counted and not bool(((other ^ mask) & ~near).any())


def run(q, k, v, gate, dout, t, h, w, text, topk, outputs=None):
    lay = Layout(int(text), (int(t), int(h), int(w)))
    b_, h_, s, d = q.shape
    n, p = lay.blocks, lay.prefix_blocks
    if s != lay.seq:
        raise ValueError(f"the sequence holds {s} rows, expected {n} blocks of {ROWS}")
    scale = 1.0 / math.sqrt(d)
    valid = row_valid(lay, q.device)

    # fp32 leaves on blocks; TF32 would round the products this definition computes exactly.
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        blocks = [x.float().reshape(b_, h_, n, ROWS, d).requires_grad_() for x in (q, k, v, gate)]
        qb, kb, vb, gb = blocks

        # The router, and the selection judged away from the threshold.
        scores = router_scores(qb[:, :, p:], kb[:, :, p:], valid[p:], scale)
        mask = select_blocks(scores, topk)
        near = threshold_band(scores, topk)
        selection = torch.where(near, float("nan"), mask.float())

        # What the rest is judged under: the solution's own selection when it is a valid top-K of these scores.
        use = mask
        if outputs is not None and len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
            other = outputs[1].to(q.device)
            if valid_selection(other, mask, near, topk):
                use = other > 0.5

        with torch.enable_grad():
            out, lse = forward(qb, kb, vb, gb, lay, use, scale)
            loss = (out * dout.float().reshape(b_, h_, n, ROWS, d)).sum()
            dq, dk, dv, dgate = torch.autograd.grad(loss, blocks)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32

    rows_valid = valid[None, None, :, :]
    lse = torch.where(rows_valid, lse.detach(), float("nan"))
    # The text rows' gate is zero by contract, so their gate gradient is not judged.
    text_rows = torch.zeros_like(valid)
    text_rows[:p] = valid[:p]
    dgate = torch.where(text_rows[None, None, :, :, None], float("nan"), dgate)
    shaped = [x.detach().reshape(b_, h_, s, d) for x in (out, dq, dk, dv)]
    out, dq, dk, dv = (x.to(q.dtype) for x in shaped)
    dgate = dgate.reshape(b_, h_, s, d).to(q.dtype)
    return out, selection, lse.reshape(b_, h_, s), dq, dk, dv, dgate
