# SPDX-License-Identifier: Apache-2.0
"""Verify the 720p i2v_track parquet set against the 480p reference.

Checks:
  1. row count, no duplicate ``file_name``
  2. clip-id set is EXACTLY the 480p set (and the source manifest)
  3. ``vae_latent_shape`` / ``first_frame_latent_shape`` == [16, 31, 90, 160] on every row
     (shape columns are tiny, so this is checked exhaustively, not sampled)
  4. schema field names + dtypes identical to the 480p set
  5. on --sample-tracks random clips: track_points / track_visibility / object_ids /
     track_weights bytes IDENTICAL to the 480p row for the same file_name

Latent-decode sanity is a separate tool: data_pipeline/check_720p_latents.py

Usage:
  python data_pipeline/verify_720p_parquet.py \
    --new  /mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset_720p \
    --ref  /mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset \
    --manifest /mnt/lustre/vlm-s4duan/openvid_1m/videos2caption.json \
    --expect-shape 16,31,90,160 --sample-tracks 8
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

LIGHT_COLS = ["file_name", "id", "vae_latent_shape", "first_frame_latent_shape", "width", "height", "num_frames", "fps"]
TRACK_FIELDS = ["track_points", "track_visibility", "object_ids", "track_weights"]


def _files(root: str) -> list[str]:
    return sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))


def _scan(files: list[str], workers: int = 32):
    """-> (rows, shape_counter, ffshape_counter, schema_repr). Reads only tiny columns."""
    rows: list[tuple[str, str]] = []
    shapes: Counter = Counter()
    ffshapes: Counter = Counter()
    schema_repr = None

    def one(f):
        t = pq.read_table(f, columns=LIGHT_COLS)
        return f, t

    with ThreadPoolExecutor(workers) as ex:
        for i, (f, t) in enumerate(ex.map(one, files)):
            if schema_repr is None:
                schema_repr = [(fl.name, str(fl.type)) for fl in pq.ParquetFile(f).schema_arrow]
            fn = t.column("file_name").to_pylist()
            ids = t.column("id").to_pylist()
            rows.extend(zip(fn, ids))
            shapes.update(tuple(s) for s in t.column("vae_latent_shape").to_pylist())
            ffshapes.update(tuple(s) for s in t.column("first_frame_latent_shape").to_pylist())
            if (i + 1) % 500 == 0:
                print(f"    ...scanned {i+1}/{len(files)} files, {len(rows)} rows", flush=True)
    return rows, shapes, ffshapes, schema_repr


def _find_rows(root: str, wanted: set[str]) -> dict[str, dict]:
    """Full rows (incl. binary cols) for the given file_names. Scans until all found."""
    out: dict[str, dict] = {}
    for f in _files(root):
        names = pq.read_table(f, columns=["file_name"]).column("file_name").to_pylist()
        hit = [i for i, n in enumerate(names) if n in wanted and n not in out]
        if not hit:
            continue
        t = pq.read_table(f)
        for i in hit:
            r = t.slice(i, 1).to_pylist()[0]
            out[r["file_name"]] = r
        if len(out) == len(wanted):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--expect-shape", default="16,31,90,160")
    ap.add_argument("--sample-tracks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    expect = tuple(int(x) for x in a.expect_shape.split(","))
    ok = True

    nf, rf = _files(a.new), _files(a.ref)
    print(f"[1] parquet files: new={len(nf)} ref={len(rf)}")
    print("[2] scanning NEW (720p) ...", flush=True)
    nrows, nshapes, nff, nschema = _scan(nf)
    print("[3] scanning REF (480p) ...", flush=True)
    rrows, rshapes, rff, rschema = _scan(rf)

    nnames = [x[0] for x in nrows]
    rnames = [x[0] for x in rrows]
    nset, rset = set(nnames), set(rnames)
    print(f"\n=== COUNTS ===\n  new rows: {len(nrows)}  unique file_name: {len(nset)}")
    print(f"  ref rows: {len(rrows)}  unique file_name: {len(rset)}")

    dup = [k for k, v in Counter(nnames).items() if v > 1]
    print(f"  duplicate file_name in new: {len(dup)}" + (f" e.g. {dup[:5]}" if dup else " OK"))
    ok &= not dup

    missing, extra = rset - nset, nset - rset
    print(f"\n=== CLIP-ID SET vs 480p ===\n  in 480p but MISSING from 720p: {len(missing)}")
    if missing:
        print(f"    {sorted(missing)[:50]}")
        with open("/mnt/lustre/vlm-s4duan/openvid_1m/_missing_720p.txt", "w") as fh:
            fh.write("\n".join(sorted(missing)))
        print("    (full list -> openvid_1m/_missing_720p.txt)")
    print(f"  in 720p but EXTRA vs 480p: {len(extra)}" + (f" {sorted(extra)[:20]}" if extra else ""))
    ok &= not missing and not extra

    if a.manifest:
        man = {it["path"].rsplit(".", 1)[0] for it in json.load(open(a.manifest))}
        print(f"  source manifest clips: {len(man)} | missing vs manifest: {len(man - nset)} | extra: {len(nset - man)}")
        ok &= not (man - nset)

    print(f"\n=== LATENT SHAPES (all rows) ===\n  new vae_latent_shape: {dict(nshapes)}")
    print(f"  new first_frame_latent_shape: {dict(nff)}")
    print(f"  ref vae_latent_shape: {dict(rshapes)}")
    good = set(nshapes) == {expect} and set(nff) == {expect}
    print(f"  all == {list(expect)}: {'YES' if good else 'NO'}")
    ok &= good

    print("\n=== SCHEMA ===")
    same_schema = nschema == rschema
    print(f"  new schema == ref schema (names+types): {'YES' if same_schema else 'NO'}")
    if not same_schema:
        print(f"    new: {nschema}\n    ref: {rschema}")
    ok &= same_schema

    if a.sample_tracks > 0:
        random.seed(a.seed)
        pick = set(random.sample(sorted(nset & rset), min(a.sample_tracks, len(nset & rset))))
        print(f"\n=== TRACK / TEXT IDENTITY vs 480p ({len(pick)} sampled clips) ===", flush=True)
        newr = _find_rows(a.new, pick)
        refr = _find_rows(a.ref, pick)
        for name in sorted(pick):
            n, r = newr.get(name), refr.get(name)
            if n is None or r is None:
                print(f"  {name}: NOT FOUND (new={n is not None} ref={r is not None})")
                ok = False
                continue
            res = {f: n[f + "_bytes"] == r[f + "_bytes"] for f in TRACK_FIELDS}
            res["text_embedding"] = n["text_embedding_bytes"] == r["text_embedding_bytes"]
            res["caption"] = n["caption"] == r["caption"]
            clipf = n["clip_feature_shape"] == r["clip_feature_shape"]
            bad = [k for k, v in res.items() if not v]
            print(f"  {name}: latent {tuple(n['vae_latent_shape'])} | "
                  f"tracks+text identical: {'YES' if not bad else 'NO ' + str(bad)} | "
                  f"clip_feature shape match: {clipf}")
            ok &= not bad and clipf

    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} ===")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
