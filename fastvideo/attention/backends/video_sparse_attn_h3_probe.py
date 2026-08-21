# SPDX-License-Identifier: Apache-2.0
"""Attention-mass probe for VSA-H3 selection quality.

Enabled via ``FASTVIDEO_H3_VSA_PROBE=<out_dir>`` on a sparsity-0 run (mask
selects everything, so the model follows its exact dense trajectory while
we measure what any top-k WOULD have captured). Per (step, layer) it
records, from the pooled tile scores:

- ``recall_mean``: pooled softmax mass of video->video attention captured
  by the top-k video tiles, as a function of k/V, per head;
- ``prefix_mass``: pooled mass video queries place on text/cond/audio tiles
  (decides exempt-vs-compete);
- ``true_recall@{frac}``: token-true validation on sampled query rows —
  exact attention row aggregated per tile, mass captured by the tiles the
  POOLED ranking would pick (catches pooling-proxy failure);
- ``perfect_recall@{frac}``: the selection ceiling — mass captured when
  ranking by the true tile masses themselves.

One .pt file per (step, layer, rank); aggregate offline.
"""

import os

import torch

_FRACS = (0.05, 0.10, 0.125, 0.25, 0.50, 0.75)
_TRUE_ROWS = 128  # sampled query rows per layer for the token-true check


def probe_enabled() -> str | None:
    return os.environ.get("FASTVIDEO_H3_VSA_PROBE") or None


@torch.no_grad()
def record_probe(
    out_dir: str,
    layer: int,
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    attn_metadata,
) -> None:
    """scores: [B, H, n, n] scaled pooled scores; query/key tiled BSHD."""
    step = int(attn_metadata.current_timestep)
    P = attn_metadata.num_prefix_tiles
    V = attn_metadata.num_video_tiles

    probs = torch.softmax(scores.float(), dim=-1)  # pooled attention, all tiles
    vid_q = probs[:, :, P:, :]  # video-query rows
    prefix_mass = vid_q[..., :P].sum(-1).mean(dim=(0, 2))  # [H]

    vid_vid = vid_q[..., P:]
    vid_vid = vid_vid / vid_vid.sum(-1, keepdim=True).clamp_min(1e-12)
    sorted_mass = vid_vid.sort(dim=-1, descending=True).values.cumsum(-1)  # [B,H,Vq,V]
    ks = [max(1, min(int(round(f * V)), V)) for f in _FRACS]
    recall = torch.stack([sorted_mass[..., k - 1] for k in ks])  # [F,B,H,Vq]
    recall_mean = recall.mean(dim=(1, 3))  # [F,H]

    # token-true check on sampled rows (exact row softmax, tile-aggregated)
    gen = torch.Generator(device="cpu").manual_seed(step * 1000 + layer)
    # sample among video rows in the PADDED/tiled domain that are non-pad
    from fastvideo.attention.backends.video_sparse_attn_h3 import token_tile_and_valid
    token_tile, token_valid = token_tile_and_valid(attn_metadata.variable_block_sizes)
    video_rows = torch.nonzero((token_tile >= P) & token_valid, as_tuple=False).flatten()
    idx = video_rows[torch.randint(0, video_rows.numel(), (_TRUE_ROWS, ), generator=gen).to(query.device)]

    q_s = query[:, idx].float()  # [B, R, H, D]
    logits = torch.einsum("brhd,bshd->bhrs", q_s, key.float()) / (query.shape[-1]**0.5)
    logits = logits.masked_fill(~token_valid.view(1, 1, 1, -1), float("-inf"))
    true_probs = torch.softmax(logits, dim=-1)
    n_tiles = attn_metadata.variable_block_sizes.numel()
    tile_mass = torch.zeros(*true_probs.shape[:3], n_tiles, device=query.device, dtype=true_probs.dtype)
    tile_mass.index_add_(3, token_tile, true_probs)  # [B,H,R,n]
    tm_vid = tile_mass[..., P:]
    tm_vid = tm_vid / tm_vid.sum(-1, keepdim=True).clamp_min(1e-12)
    # pooled ranking per sampled row's q-tile
    pooled_rank = scores[:, :, P:, P:].argsort(dim=-1, descending=True)  # [B,H,Vq,V]
    row_qtile = (token_tile[idx] - P).view(1, 1, -1, 1).expand(pooled_rank.shape[0], pooled_rank.shape[1], -1, 1)
    row_rank = pooled_rank.gather(2, row_qtile.expand(-1, -1, -1, pooled_rank.shape[-1]))  # [B,H,R,V]
    perfect_sorted = tm_vid.sort(dim=-1, descending=True).values.cumsum(-1)  # selection ceiling
    true_recall = {}
    perfect_recall = {}
    for f, k in zip(_FRACS, ks, strict=True):
        captured = tm_vid.gather(-1, row_rank[..., :k]).sum(-1)  # [B,H,R]
        true_recall[f] = captured.mean().item()
        perfect_recall[f] = perfect_sorted[..., k - 1].mean().item()

    os.makedirs(out_dir, exist_ok=True)
    payload = dict(step=step,
                   layer=layer,
                   P=P,
                   V=V,
                   fracs=_FRACS,
                   prefix_mass=prefix_mass.cpu(),
                   recall_mean=recall_mean.cpu(),
                   true_recall=true_recall,
                   perfect_recall=perfect_recall)
    # under SP each rank holds a distinct head subset; keep every rank's stats
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    torch.save(payload, os.path.join(out_dir, f"probe_step{step:03d}_layer{layer:03d}_r{rank}.pt"))
