# SPDX-License-Identifier: Apache-2.0
"""Build the V2 QAD prompt manifests (train + held-out eval) from prompt sources.

Sources: parquet shards with a ``prompt`` column (the decoder latent generator's outputs; only that column is read)
and JSONL files with a ``prompt`` field (e.g. ``source.jsonl``, OmniRef captions). Prompts are deduplicated, then a
seeded shuffle assigns ``--eval-count`` held-out prompts and the rest to training. Each row gets a deterministic seed.
Embeddings are filled in by ``precompute_v2_text.py`` (rows carry the future embedding path).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path


def read_prompts(parquet_globs: list[str], jsonl_globs: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    found: dict[str, str] = {}
    for pattern in parquet_globs:
        for path in sorted(glob.glob(pattern)):
            if "prompt" in pq.read_schema(path).names:
                for prompt in pq.read_table(path, columns=["prompt"]).column("prompt").to_pylist():
                    found.setdefault(" ".join(prompt.split()), path)
    for pattern in jsonl_globs:
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                for line in handle:
                    if line.strip():
                        prompt = json.loads(line).get("prompt")
                        if isinstance(prompt, str) and prompt.strip():
                            found.setdefault(" ".join(prompt.split()), path)
    return sorted(found)


def make_rows(prompts: list[str], embed_dir: str, seed: int) -> list[dict]:
    rows = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        rows.append({
            "id": f"v2-{digest}",
            "source": "prompt",
            "prompt": prompt,
            "embedding": str(Path(embed_dir) / f"{digest}.pt"),
            "seed": int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="*", default=[], help="globs of parquet shards with a prompt column")
    parser.add_argument("--jsonl", nargs="*", default=[], help="globs of JSONL files with a prompt field")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-count", type=int, default=30)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    prompts = read_prompts(args.parquet, args.jsonl)
    random.Random(args.seed).shuffle(prompts)
    if len(prompts) <= args.eval_count:
        raise SystemExit(f"only {len(prompts)} prompts")
    out = Path(args.out_dir)
    embed_dir = out / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    eval_prompts, train_prompts = prompts[:args.eval_count], prompts[args.eval_count:]
    if args.max_train:
        train_prompts = train_prompts[:args.max_train]
    for name, subset in (("eval", eval_prompts), ("train", train_prompts)):
        rows = make_rows(subset, str(embed_dir), args.seed)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(json.dumps({"split": name, "rows": len(rows)}))


if __name__ == "__main__":
    main()
