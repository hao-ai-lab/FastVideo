# SPDX-License-Identifier: Apache-2.0
"""Convert openvid-wantrack parquets to bf16 (a COPY; never mutates the source).

Casts the big float tensor fields (vae_latent, first_frame_latent, text_embedding,
clip_feature, track_points, track_visibility) from float32 to bfloat16 and tags their
``*_dtype`` column ``"bfloat16"``. Integer / tiny fields (object_ids, track_weights) and all
metadata are copied unchanged. Training already downcasts these fields to bf16
(``wan.py`` / ``wantrack.py``), so the quality loss is negligible while the files ~halve.

Requires the loader change in ``fastvideo/dataset/utils.py`` that honors the ``*_dtype``
column (both decoders). Without it, the bf16 bytes would be misread as float32.

The output mirrors the source tree (shardNNN/combined_parquet_dataset/worker_N/*.parquet), so
point training ``data_path`` at ``--dst`` once you've converted what you want. Resumable:
already-written files are skipped. CPU only; run on a compute node (I/O + RAM heavy).

Examples:
  # first half of all parquets -> a bf16 sibling dir
  python data_pipeline/convert_parquets_to_bf16.py --fraction 0.5

  # a fixed number of files, dry-run first
  python data_pipeline/convert_parquets_to_bf16.py --limit 100 --dry-run
"""
from __future__ import annotations

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
import torch

DEFAULT_SRC = "/home/hal-shared/motionstream/data/openvid-wantrack-parquets"
DEFAULT_DST = "/home/hal-shared/motionstream/data/openvid-wantrack-parquets-bf16"

# Big float fields consumed at bf16 by training -> safe to store bf16.
DEFAULT_BF16_FIELDS = [
    "vae_latent", "first_frame_latent", "text_embedding", "clip_feature",
    "track_points", "track_visibility",
]
# Left untouched (integer labels / tiny): object_ids, track_weights, all metadata.

_SRC_STR_TO_TORCH = {
    "float32": torch.float32, "float16": torch.float16, "float64": torch.float64,
}


def _to_bf16_bytes(b: bytes, dtype_str: str) -> tuple[bytes, str]:
    """Re-encode a raw float tensor blob as bf16. Returns (bytes, dtype_label)."""
    if not b:                         # empty optional field -> leave as-is
        return b, (dtype_str or "")
    if dtype_str == "bfloat16":       # already converted
        return b, dtype_str
    src = _SRC_STR_TO_TORCH.get(dtype_str or "float32")
    if src is None:
        raise ValueError(f"cannot convert stored dtype {dtype_str!r} to bf16")
    t = torch.frombuffer(bytearray(b), dtype=src).to(torch.bfloat16)
    return t.view(torch.uint8).numpy().tobytes(), "bfloat16"


def convert_file(src_path: str, dst_path: str, fields: list[str]) -> tuple[int, int]:
    """Convert one parquet file. Returns (src_bytes, dst_bytes)."""
    tbl = pq.read_table(src_path)
    names = list(tbl.schema.names)
    cols: dict[str, object] = {n: tbl.column(n) for n in names}

    for fld in fields:
        bkey, dkey = f"{fld}_bytes", f"{fld}_dtype"
        if bkey not in cols or dkey not in cols:
            continue
        b_list = tbl.column(bkey).to_pylist()
        d_list = tbl.column(dkey).to_pylist()
        new_b, new_d = [], []
        for b, d in zip(b_list, d_list, strict=True):
            nb, nd = _to_bf16_bytes(b, d)
            new_b.append(nb)
            new_d.append(nd)
        cols[bkey] = pa.array(new_b, type=pa.binary())
        cols[dkey] = pa.array(new_d, type=pa.string())

    out = pa.table([cols[n] for n in names], schema=tbl.schema)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + ".tmp"
    pq.write_table(out, tmp)
    os.replace(tmp, dst_path)
    return os.path.getsize(src_path), os.path.getsize(dst_path)


def verify_file(src_path: str, dst_path: str, fields: list[str]) -> None:
    """Round-trip check: one bf16 field on row 0 must equal src fp32 -> bf16."""
    s = pq.ParquetFile(src_path).read_row_group(0).slice(0, 1).to_pylist()[0]
    d = pq.ParquetFile(dst_path).read_row_group(0).slice(0, 1).to_pylist()[0]
    for fld in fields:
        sb, db = s.get(f"{fld}_bytes"), d.get(f"{fld}_bytes")
        if not sb:
            continue
        assert d.get(f"{fld}_dtype") == "bfloat16", f"{fld}: dtype not tagged bfloat16"
        src_t = torch.frombuffer(bytearray(sb), dtype=_SRC_STR_TO_TORCH[s[f"{fld}_dtype"]]).to(torch.bfloat16)
        dst_t = torch.frombuffer(bytearray(db), dtype=torch.bfloat16)
        assert torch.equal(dst_t, src_t), f"{fld}: bf16 round-trip mismatch"
        return  # one field is enough
    return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--dst", default=DEFAULT_DST)
    p.add_argument("--fraction", type=float, default=0.5, help="fraction of the sorted parquet list to convert (default first 0.5)")
    p.add_argument("--limit", type=int, default=None, help="convert at most N files (overrides --fraction)")
    p.add_argument("--offset", type=int, default=0, help="skip the first N files of the sorted list (parallelize by running disjoint --offset/--limit ranges)")
    p.add_argument("--fields", default=",".join(DEFAULT_BF16_FIELDS), help="comma-separated fields to cast to bf16")
    p.add_argument("--overwrite", action="store_true", help="re-convert files already present in --dst")
    p.add_argument("--no-verify", action="store_true", help="skip the per-file round-trip check")
    p.add_argument("--dry-run", action="store_true", help="list what would be converted; write nothing")
    args = p.parse_args()

    src_root = os.path.realpath(args.src)
    dst_root = os.path.realpath(args.dst)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if os.path.commonpath([src_root, dst_root]) == src_root and dst_root != src_root:
        raise SystemExit(f"--dst {dst_root} is inside --src; choose a separate directory")
    if src_root == dst_root:
        raise SystemExit("refusing to convert in place; --dst must differ from --src")

    all_files = []
    for root, _, files in os.walk(src_root):
        for f in files:
            if f.endswith(".parquet"):
                all_files.append(os.path.join(root, f))
    all_files.sort()
    n_total = len(all_files)
    n_take = args.limit if args.limit is not None else int(args.fraction * n_total)
    selected = all_files[args.offset:args.offset + n_take]
    print(f"[bf16] {n_total} parquet(s) found; converting {len(selected)} "
          f"[offset {args.offset}, {'limit ' + str(args.limit) if args.limit is not None else f'fraction {args.fraction}'}]")
    print(f"[bf16] fields -> bf16: {fields}")
    print(f"[bf16] src={src_root}\n[bf16] dst={dst_root}")

    if args.dry_run:
        for f in selected[:5]:
            print("  would convert:", os.path.relpath(f, src_root))
        if len(selected) > 5:
            print(f"  ... and {len(selected) - 5} more")
        return

    src_tot = dst_tot = done = skipped = 0
    for i, sp in enumerate(selected, 1):
        rel = os.path.relpath(sp, src_root)
        dp = os.path.join(dst_root, rel)
        if os.path.exists(dp) and not args.overwrite:
            skipped += 1
            continue
        sb, db = convert_file(sp, dp, fields)
        if not args.no_verify:
            verify_file(sp, dp, fields)
        src_tot += sb
        dst_tot += db
        done += 1
        if done % 20 == 0 or i == len(selected):
            gb = 1024 ** 3
            print(f"[bf16] {i}/{len(selected)} | converted {done}, skipped {skipped} | "
                  f"{src_tot/gb:.1f}GB -> {dst_tot/gb:.1f}GB"
                  f"{f' ({dst_tot/src_tot*100:.0f}%)' if src_tot else ''}")

    print(f"[bf16] done: converted {done}, skipped {skipped}. Output tree at {dst_root}")
    print(f"[bf16] point training data_path at {dst_root} once you've converted enough shards.")


if __name__ == "__main__":
    main()
