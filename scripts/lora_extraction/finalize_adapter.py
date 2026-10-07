"""Turn a raw extracted adapter into one that is correct to publish.

An extractor that writes whole parameters next to its low-rank factors leaves three
problems behind, and all three are silent:

1. **Unchanged tensors get shipped.** A "delta" that is bit-identical to the base states
   that something changed, in a file whose only job is to record what changed. On the
   FastH3 adapters this is 193 of 326 whole tensors -- the majority.
2. **The whole tensors have no convention.** Written as a bare ``<param>.weight`` they
   are indistinguishable from a factor, so a loader either misreads them or, as
   FastVideo's did, drops them without a word.
3. **Nothing records what the file is.** Rank, base revision, and how to apply the thing
   live in the author's head.

This pass fixes all three against the base checkpoint: it drops what did not change,
renames what did to the ``.diff`` / ``.diff_b`` / ``.set_weight`` convention that
``fastvideo.models.loader.lora_patch`` and ComfyUI both read, and stamps provenance into
the safetensors header. It can also truncate rank, which is exact: the factors come from
an SVD with singular values in descending order, so keeping the leading ``r`` columns of
``lora_B`` and rows of ``lora_A`` is the rank-``r`` optimum of the same decomposition.

    python scripts/lora_extraction/finalize_adapter.py \\
        --adapter raw/rank-64/adapter_model.safetensors \\
        --base /models/MiniMax-H3/transformer \\
        --out publish/rank-64/adapter_model.safetensors \\
        --name FastVideo-FastH3-4-step-v1.1 --card
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOG = logging.getLogger("finalize_adapter")

DIFF_SUFFIX = ".diff"
DIFF_BIAS_SUFFIX = ".diff_b"
SET_WEIGHT_SUFFIX = ".set_weight"
LOW_RANK_MARKER = ".lora_"


class BaseWeights:
    """Key-addressed view of a base transformer that reads one tensor at a time.

    Only the parameters an adapter ships whole are ever compared -- a couple of hundred
    norms and biases -- so materializing all 62 GB to answer that would be absurd, and
    doing it once per rank variant more so. The map from key to shard comes from the
    safetensors headers, which are a few hundred KB.
    """

    def __init__(self, base: str) -> None:
        path = Path(base)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(base))
        if (path / "transformer").is_dir():
            path = path / "transformer"
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors under {path}")
        self._where: Dict[str, Path] = {}
        for shard in shards:
            with safe_open(shard, framework="pt") as handle:
                for key in handle.keys():
                    self._where[key] = shard
        LOG.info("base: %d tensors across %d shards under %s", len(self._where), len(shards), path)

    def __contains__(self, key: str) -> bool:
        return key in self._where

    def get(self, key: str) -> Optional[torch.Tensor]:
        shard = self._where.get(key)
        if shard is None:
            return None
        with safe_open(shard, framework="pt") as handle:
            return handle.get_tensor(key)


def truncate_rank(name: str, tensor: torch.Tensor, rank: Optional[int]) -> torch.Tensor:
    """Keep the leading ``rank`` components of an SVD-derived factor."""
    if rank is None:
        return tensor
    if name.endswith(("lora_A.weight", "lora_A")):
        return tensor[:rank].contiguous()  # (r, in)
    if name.endswith(("lora_B.weight", "lora_B")):
        return tensor[:, :rank].contiguous()  # (out, r)
    return tensor


def finalize(adapter: str,
             base: "str | BaseWeights",
             out: str,
             rank: Optional[int] = None,
             name: Optional[str] = None,
             min_delta: float = 0.0) -> dict:
    """Rewrite ``adapter`` into publishable form. Returns a summary dict.

    ``base`` may be a prebuilt :class:`BaseWeights` so a caller finalizing several rank
    variants of the same checkpoint pays for the shard index once.
    """
    base_sd = base if isinstance(base, BaseWeights) else BaseWeights(base)

    kept: Dict[str, torch.Tensor] = {}
    stats = {"low_rank": 0, "diff": 0, "diff_b": 0, "set_weight": 0, "dropped_identical": 0, "dropped_absent_ft": 0}
    ranks_seen: set = set()

    with safe_open(adapter, framework="pt") as handle:
        source_meta = handle.metadata() or {}
        for key in sorted(handle.keys()):
            tensor = handle.get_tensor(key)

            if LOW_RANK_MARKER in key:
                tensor = truncate_rank(key, tensor, rank)
                if ".lora_A" in key:
                    ranks_seen.add(int(tensor.shape[0]))
                kept[key] = tensor.contiguous()
                stats["low_rank"] += 1
                continue

            # A whole parameter. Its fate is decided entirely by the base checkpoint.
            base_tensor = base_sd.get(key)
            if base_tensor is None:
                if not key.endswith(".weight"):
                    LOG.warning("no .set_* spelling for non-weight parameter absent from base: %s", key)
                    stats["dropped_absent_ft"] += 1
                    continue
                kept[key[:-len(".weight")] + SET_WEIGHT_SUFFIX] = tensor.contiguous()
                stats["set_weight"] += 1
                continue

            if base_tensor.shape != tensor.shape:
                LOG.warning("shape mismatch, dropping %s: adapter %s vs base %s", key, tuple(tensor.shape),
                            tuple(base_tensor.shape))
                stats["dropped_absent_ft"] += 1
                continue

            delta = tensor.to(torch.float32) - base_tensor.to(torch.float32)
            if float(delta.abs().max()) <= min_delta:
                stats["dropped_identical"] += 1
                continue

            if key.endswith(".bias"):
                kept[key[:-len(".bias")] + DIFF_BIAS_SUFFIX] = delta.to(tensor.dtype).contiguous()
                stats["diff_b"] += 1
            elif key.endswith(".weight"):
                kept[key[:-len(".weight")] + DIFF_SUFFIX] = delta.to(tensor.dtype).contiguous()
                stats["diff"] += 1
            else:
                LOG.warning("no .diff spelling for %s", key)
                stats["dropped_absent_ft"] += 1

    metadata = {
        "format": "fastvideo-lora-v2",
        "base_model": source_meta.get("base_model", ""),
        "finetuned_model": source_meta.get("finetuned_model", name or ""),
        "base_revision": source_meta.get("base_revision", ""),
        "finetuned_revision": source_meta.get("finetuned_revision", ""),
        "rank": str(sorted(ranks_seen)[-1]) if ranks_seen else str(rank or ""),
        "application": ("W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned"),
        "low_rank_tensors": str(stats["low_rank"]),
        "diff_tensors": str(stats["diff"] + stats["diff_b"]),
        "set_weight_tensors": str(stats["set_weight"]),
        "dropped_unchanged": str(stats["dropped_identical"]),
        "finalized_from": Path(adapter).name,
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(out_path), metadata=metadata)
    out_path.chmod(0o644)

    size_gib = out_path.stat().st_size / 2**30
    LOG.info(
        "%s -> %s  (%.2f GiB)  low_rank=%d diff=%d diff_b=%d set_weight=%d  dropped: %d unchanged, %d unusable",
        Path(adapter).name, out_path, size_gib, stats["low_rank"], stats["diff"], stats["diff_b"],
        stats["set_weight"], stats["dropped_identical"], stats["dropped_absent_ft"])
    return {"out": str(out_path), "size_gib": size_gib, "metadata": metadata, **stats}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="Raw extracted adapter (.safetensors)")
    p.add_argument("--base", required=True, help="Base transformer directory or HF model id")
    p.add_argument("--out", required=True, help="Output adapter path")
    p.add_argument("--rank", type=int, default=None, help="Truncate factors to this rank (exact for SVD factors)")
    p.add_argument("--name", default=None, help="Fine-tuned model name, when the source file does not record one")
    p.add_argument("--min-delta",
                   type=float,
                   default=0.0,
                   help="Drop a whole-tensor delta whose max abs value is at or below this (default: exact zeros only)")
    p.add_argument("--summary-json", default=None, help="Write the summary dict here")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    summary = finalize(adapter=args.adapter,
                       base=args.base,
                       out=args.out,
                       rank=args.rank,
                       name=args.name,
                       min_delta=args.min_delta)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
