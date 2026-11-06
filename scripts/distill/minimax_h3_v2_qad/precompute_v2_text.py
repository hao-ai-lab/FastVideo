# SPDX-License-Identifier: Apache-2.0
"""Precompute FastH3 text embeddings (Qwen3-VL conditioner, bf16 V2 model) for the V2 QAD manifests.

Builds the V2 inference pipeline once (single GPU), calls its conditioning stage on each prompt and writes
``{"embeds": [L, D] bf16, "tags": [L] long}`` to every row's ``embedding`` path. Existing files are kept, so reruns and
sharding (``--shard i --num-shards n``) are safe.
"""
from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

import torch


def _build(config_path: str, model_path: str, shard: int):
    """One-rank external-launcher generator, configured for latent output only."""
    from fastvideo.api.compat import generator_config_to_fastvideo_args, normalize_generator_config
    from fastvideo.api.parser import load_raw_config
    from fastvideo.entrypoints.video_generator import VideoGenerator

    for name, value in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"), ("MASTER_ADDR", "127.0.0.1")):
        os.environ.setdefault(name, value)
    raw = load_raw_config(config_path)
    generator_raw = dict(raw["generator"], model_path=model_path)
    generator_raw["engine"] = dict(generator_raw["engine"], execution_backend="external_launcher")
    fastvideo_args = generator_config_to_fastvideo_args(normalize_generator_config(generator_raw))
    fastvideo_args.output_type = "latent"
    generator = VideoGenerator.from_fastvideo_args(fastvideo_args)
    return generator, generator.executor.worker.worker.pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="FastH3 run config (generator + request)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for path in args.manifests:
        rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    todo = [r for r in rows[args.shard::args.num_shards] if not Path(r["embedding"]).exists()]
    print(json.dumps({"rows": len(rows), "todo": len(todo)}), flush=True)
    if not todo:
        return
    os.environ.setdefault("MASTER_PORT", str(29800 + args.shard))
    generator, pipeline = _build(args.config, args.model_path, args.shard)
    pipeline.post_init()  # idempotent: builds the stages
    stage = next(s for s in pipeline._stage_name_mapping.values() if type(s).__name__ == "MiniMaxH3ConditioningStage")
    device = torch.device("cuda")
    with torch.no_grad():
        for done, row in enumerate(todo, 1):
            embeds, tags = stage._encode_fl2va(types.SimpleNamespace(prompt=row["prompt"], extra={}), device)
            target = Path(row["embedding"])
            tmp = target.with_suffix(".tmp")
            torch.save({"embeds": embeds[0].to(torch.bfloat16).cpu(), "tags": tags.cpu()}, tmp)
            tmp.replace(target)
            if done % 50 == 0:
                print(json.dumps({"done": done, "of": len(todo)}), flush=True)
    generator.shutdown()


if __name__ == "__main__":
    main()
