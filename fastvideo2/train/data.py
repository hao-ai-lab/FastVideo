"""Parquet dataloading for training — fastvideo-main's map-style dataset
reproduced byte-for-byte at world size 1.

Order-defining facts (verbatim from main's ``parquet_dataset_map_style.py``):

  * files: ``os.walk(realpath(root))``, realpath each ``*.parquet`` (on an HF
    snapshot this resolves symlinks INTO ``blobs/``, so the sort key is the
    blob path — arbitrary-looking but deterministic for given content),
    then one global sort by path;
  * sampler: ``torch.randperm(total, generator=manual_seed(seed))``,
    truncated to a batch multiple (drop_last), chunked sequentially
    (world 1: every SP-group shard is the identity);
  * rows: fp32 ``np.frombuffer`` decode of ``*_bytes``/``*_shape`` (main
    ignores the ``_dtype`` column — so do we); ``text_embedding`` padded or
    cropped to ``text_padding_length`` with a 1/0 attention mask.

The trainer casts to bf16 on device afterwards (main's ``_get_next_batch``).
"""
from __future__ import annotations

import os
from typing import Any


def parquet_files_and_lengths(root: str) -> tuple[list[str], list[int]]:
    import pyarrow.parquet as pq
    root = os.path.realpath(os.path.expanduser(root))
    files: list[str] = []
    for r, _, fs in os.walk(root):
        files.extend(os.path.realpath(os.path.join(r, f))
                     for f in fs if f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    lengths = {f: pq.ParquetFile(f).metadata.num_rows for f in files}
    files.sort()
    return files, [lengths[f] for f in files]


def batch_indices(total: int, batch_size: int, seed: int) -> list[list[int]]:
    import torch
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:(total // batch_size) * batch_size]
    return [idx[i:i + batch_size].tolist()
            for i in range(0, len(idx), batch_size)]


def read_row(files: list[str], lengths: list[int], gidx: int) -> dict[str, Any]:
    import pyarrow.parquet as pq
    for f, n in zip(files, lengths):
        if gidx < n:
            break
        gidx -= n
    else:
        raise IndexError(f"row {gidx} out of bounds")
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        rows = pf.metadata.row_group(g).num_rows
        if gidx < rows:
            tbl = pf.read_row_group(g).to_pydict()
            return {k: v[gidx] for k, v in tbl.items()}
        gidx -= rows
    raise IndexError(f"row {gidx} out of bounds in {f}")


def _decode(row: dict, name: str) -> Any:
    import numpy as np
    import torch
    return torch.from_numpy(np.frombuffer(
        row[f"{name}_bytes"], dtype=np.float32
    ).reshape(row[f"{name}_shape"]).copy())


def load_batch(files: list[str], lengths: list[int], indices: list[int], *,
               text_padding_length: int = 512) -> dict[str, Any]:
    """-> latents [B,C,T,H,W] fp32, embeds [B,pad,D] fp32, mask [B,pad],
    captions [B] — main's collated batch before device/bf16."""
    import torch
    lat, emb, mask, captions = [], [], [], []
    for i in indices:
        row = read_row(files, lengths, i)
        lat.append(_decode(row, "vae_latent"))
        e = _decode(row, "text_embedding")
        n, d = e.shape
        if text_padding_length > n:
            e = torch.cat([e, torch.zeros(text_padding_length - n, d)], 0)
            m = torch.cat([torch.ones(n),
                           torch.zeros(text_padding_length - n)], 0)
        else:
            e, m = e[:text_padding_length], torch.ones(text_padding_length)
        emb.append(e)
        mask.append(m)
        captions.append(str(row.get("caption", "")))
    return {"latents": torch.stack(lat), "embeds": torch.stack(emb),
            "mask": torch.stack(mask), "captions": captions}
