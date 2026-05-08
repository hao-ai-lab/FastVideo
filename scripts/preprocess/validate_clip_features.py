"""Validate a CLIP-backfilled Wan latent parquet cache (TECH-118).

Reports:
    * Total rows and parquet files
    * Per-column-group: missing count, shape distribution, dtype distribution
    * Round-trip sanity for 5 random rows per group
    * (optional) done.log SHA verification on a 1% sample

Exit code 0 if all gates pass, 1 otherwise.

Usage:
    python scripts/preprocess/validate_clip_features.py \\
        --src /leonardo_scratch/large/userexternal/mshariat/latents_wan_v1/W21_480x832_49f_16fps_clip
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# For each column-group, what we expect on disk. `expect_empty=True` means
# every row's bytes column should be exactly b"" (post-PR1 trainer treats
# empty tensors as absent; we rely on that for the two columns we don't fill).
EXPECTED = {
    "clip_feature": {
        "shape": [13, 257, 1280],
        "dtype": "float16",
        "np_dtype": np.float16,
        "expect_empty": False,
    },
    "first_frame_latent": {
        "shape": [],
        "dtype": "",
        "np_dtype": None,
        "expect_empty": True,
    },
    "pil_image": {
        "shape": [],
        "dtype": "",
        "np_dtype": None,
        "expect_empty": True,
    },
}


def list_parquet_files(src: Path) -> list[Path]:
    return sorted(src.rglob("data_chunk_*.parquet"))


def validate_file(parquet_path: Path) -> dict:
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    out = {"path": str(parquet_path), "num_rows": n_rows, "groups": {}}
    for tensor_name in EXPECTED:
        bytes_col = f"{tensor_name}_bytes"
        shape_col = f"{tensor_name}_shape"
        dtype_col = f"{tensor_name}_dtype"
        if bytes_col not in table.schema.names:
            out["groups"][tensor_name] = {"missing_column": True}
            continue
        bytes_list = table[bytes_col].to_pylist()
        shape_list = table[shape_col].to_pylist()
        dtype_list = table[dtype_col].to_pylist()
        missing = sum(1 for b in bytes_list if not b)
        shape_counter = Counter(tuple(s) if s else () for s in shape_list)
        dtype_counter = Counter(d if d else "" for d in dtype_list)
        out["groups"][tensor_name] = {
            "missing": missing,
            "shapes": shape_counter,
            "dtypes": dtype_counter,
        }
    return out


def aggregate(per_file: list[dict]) -> dict:
    agg = {"num_files": len(per_file), "num_rows": 0, "groups": {}}
    for tn in EXPECTED:
        agg["groups"][tn] = {"missing": 0, "shapes": Counter(), "dtypes": Counter()}
    for f in per_file:
        agg["num_rows"] += f["num_rows"]
        for tn, g in f["groups"].items():
            if g.get("missing_column"):
                agg["groups"].setdefault(tn, {})["missing_column"] = True
                continue
            agg["groups"][tn]["missing"] += g["missing"]
            agg["groups"][tn]["shapes"].update(g["shapes"])
            agg["groups"][tn]["dtypes"].update(g["dtypes"])
    return agg


def round_trip_sample(parquet_path: Path, n: int = 5) -> dict:
    """Decode bytes/shape/dtype for n random rows; assert reshape works (or empty)."""
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    indices = random.sample(range(n_rows), min(n, n_rows))
    results: dict[str, list[bool]] = {tn: [] for tn in EXPECTED}
    for tn, exp in EXPECTED.items():
        bytes_col = table[f"{tn}_bytes"].to_pylist()
        shape_col = table[f"{tn}_shape"].to_pylist()
        dtype_col = table[f"{tn}_dtype"].to_pylist()
        for i in indices:
            if exp["expect_empty"]:
                results[tn].append(bytes_col[i] == b"")
                continue
            try:
                arr = np.frombuffer(bytes_col[i], dtype=dtype_col[i]).reshape(shape_col[i])
                results[tn].append(arr.shape == tuple(exp["shape"]))
            except Exception:
                results[tn].append(False)
    return {tn: all(v) for tn, v in results.items()}


def verify_done_log(src: Path, done_log: Path | None, sample_rate: float) -> tuple[bool, list[str]]:
    """Re-hash a `sample_rate` fraction of files in done.log and confirm SHA match.
    Returns (all_pass, mismatches)."""
    log = done_log or (src / "backfill_done.log")
    if not log.exists():
        return True, []
    entries = []
    with log.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                entries.append((parts[0], int(parts[1]), parts[2]))
    if not entries:
        return True, []
    n_check = max(1, int(len(entries) * sample_rate))
    sampled = random.sample(entries, n_check)
    mismatches = []
    for path_str, n_rows, sha in sampled:
        p = Path(path_str)
        if not p.exists():
            mismatches.append(f"missing file: {p}")
            continue
        table = pq.read_table(p)
        if table.num_rows != n_rows:
            mismatches.append(f"{p}: row count {table.num_rows} != {n_rows}")
            continue
        h = hashlib.sha256()
        for b in table["clip_feature_bytes"].to_pylist():
            h.update(b)
        if h.hexdigest() != sha:
            mismatches.append(f"{p}: sha mismatch")
    return len(mismatches) == 0, mismatches


def report(agg: dict, round_trip_ok: dict[str, bool], log_ok: bool, log_mismatches: list[str]) -> bool:
    """Print human-readable report. Returns True iff all gates pass."""
    print(f"\n=== Backfill validation ===")
    print(f"files: {agg['num_files']}  rows: {agg['num_rows']}")
    all_ok = True
    for tn, expected in EXPECTED.items():
        g = agg["groups"][tn]
        if g.get("missing_column"):
            print(f"  [{tn}] MISSING COLUMN entirely - FAIL")
            all_ok = False
            continue
        ok_rt = round_trip_ok.get(tn, False)
        if expected["expect_empty"]:
            # All bytes empty, all shapes [], all dtypes "". `g["missing"]` counts
            # empty-bytes rows; for these columns ALL rows should be empty.
            ok_missing = g["missing"] == agg["num_rows"]
            ok_shape = list(g["shapes"].keys()) in ([()], [])
            ok_dtype = list(g["dtypes"].keys()) in ([""], [])
            ok_all = ok_missing and ok_shape and ok_dtype and ok_rt
            flag = "OK (empty by design)" if ok_all else "FAIL"
        else:
            ok_missing = g["missing"] == 0
            ok_shape = (
                len(g["shapes"]) == 1
                and tuple(expected["shape"]) in g["shapes"]
            )
            ok_dtype = (
                len(g["dtypes"]) == 1
                and expected["dtype"] in g["dtypes"]
            )
            ok_all = ok_missing and ok_shape and ok_dtype and ok_rt
            flag = "OK" if ok_all else "FAIL"
        all_ok = all_ok and ok_all
        print(f"  [{tn}] {flag}: missing={g['missing']}, shapes={dict(g['shapes'])}, "
              f"dtypes={dict(g['dtypes'])}, round_trip={ok_rt}")
    print(f"  [done.log] {'OK' if log_ok else 'FAIL'}: {len(log_mismatches)} mismatches")
    for m in log_mismatches[:5]:
        print(f"    - {m}")
    if not log_ok:
        all_ok = False
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--done-log", type=Path, default=None)
    parser.add_argument("--sha-sample-rate", type=float, default=0.01,
                        help="Fraction of done.log entries to re-hash for SHA verification.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    files = list_parquet_files(args.src)
    if not files:
        print(f"No parquet files found under {args.src}", file=sys.stderr)
        return 1

    per_file = [validate_file(f) for f in files]
    agg = aggregate(per_file)

    # Round-trip sanity on the first file with rows
    rt_target = next((f for f in files if pq.ParquetFile(f).metadata.num_rows > 0), None)
    rt_ok = round_trip_sample(rt_target) if rt_target else {}

    log_ok, mismatches = verify_done_log(args.src, args.done_log, args.sha_sample_rate)

    return 0 if report(agg, rt_ok, log_ok, mismatches) else 1


if __name__ == "__main__":
    sys.exit(main())
