# SPDX-License-Identifier: Apache-2.0
"""Encode JSONL prompts for native-shape MiniMax H3 data-free T2AV training.

Only the tokenizer, processor and text encoder are loaded. Each JSONL record
contains id, prompt, width, height and num_frames; geometry is carried by the
canonical bucket directory, not by fabricated video or audio latent tensors.
See docs/training/minimax_h3_t2av_preprocess.md for the input contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.parquet_io import ParquetDatasetWriter, records_to_table
from fastvideo.dataset.dataloader.record_schema import text_only_record_creator
from fastvideo.dataset.dataloader.schema import pyarrow_schema_text_only
from fastvideo.dataset.shape_bucket import ExactVideoShapeBucket


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    geometry: ExactVideoShapeBucket


def load_prompt_records(path: Path, *, shard_index: int = 0, num_shards: int = 1) -> list[PromptRecord]:
    """Validate all rows before round-robin sharding; never normalize prompts."""
    from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_CANVAS_MULTIPLE, video_latent_num_frames

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards), with num_shards positive")
    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    fields = {"id", "prompt", "width", "height", "num_frames"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path}:{line_number}"
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(f"{where}: expected exactly {sorted(fields)}")
            for name in ("id", "prompt"):
                if not isinstance(row[name], str) or not row[name].strip():
                    raise ValueError(f"{where}: {name} must be a nonempty string")
            if row["id"] in seen_ids:
                raise ValueError(f"{where}: duplicate id {row['id']!r}")
            seen_ids.add(row["id"])
            for name in ("width", "height", "num_frames"):
                if type(row[name]) is not int or row[name] <= 0:
                    raise ValueError(f"{where}: {name} must be a positive integer")
            geometry = ExactVideoShapeBucket(row["width"], row["height"], row["num_frames"])
            if geometry.width % MINIMAX_H3_CANVAS_MULTIPLE or geometry.height % MINIMAX_H3_CANVAS_MULTIPLE:
                raise ValueError(f"{where}: width and height must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}")
            video_latent_num_frames(geometry.num_frames)
            records.append(PromptRecord(row["id"], row["prompt"], geometry))
    if not records:
        raise ValueError(f"No prompt records in {path}")
    selected = records[shard_index::num_shards]
    if not selected:
        raise ValueError("This shard has no prompt records; reduce num_shards")
    return selected


def encode_records(
    records: Sequence[PromptRecord],
    output_dir: Path,
    encode_prompt: Callable[[str], np.ndarray],
    *,
    samples_per_file: int = 16,
    max_tokens: int = 65536,
) -> int:
    """Write a fresh shard, then publish its training cache after all rows succeed.

    ``encode_prompt`` is injectable so the complete storage/loader contract is
    testable without loading Qwen or a GPU. A failed shard has no published
    cache; it must be removed explicitly before retrying.
    """
    if not records:
        raise ValueError("At least one prompt record is required")
    if type(samples_per_file) is not int or samples_per_file <= 0 or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("samples_per_file and max_tokens must be positive integers")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    buckets: dict[str, list[PromptRecord]] = defaultdict(list)
    for record in records:
        buckets[record.geometry.bucket_id].append(record)
    for bucket, bucket_records in sorted(buckets.items()):
        writer = ParquetDatasetWriter(str(output_dir / bucket), samples_per_file=samples_per_file)
        for record in bucket_records:
            embedding = np.asarray(encode_prompt(record.prompt))
            if embedding.ndim != 2 or embedding.shape[1] != 5120 or not 0 < embedding.shape[0] <= max_tokens:
                raise ValueError(f"Prompt {record.id!r} must encode to [1..{max_tokens}, 5120], got {embedding.shape}")
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)
            if not np.isfinite(embedding).all():
                raise ValueError(f"Prompt {record.id!r} has nonfinite text embeddings")
            if embedding.nbytes > (1 << 31) - 1:
                raise ValueError(f"Prompt {record.id!r} exceeds the Parquet binary value size limit")
            row = text_only_record_creator(record.id, embedding, record.prompt)
            # Preserve the input identifier exactly; the generic helper prefixes it.
            row["id"] = record.id
            writer.append_table(records_to_table([row], pyarrow_schema_text_only))
            writer.flush(num_workers=1)
        writer.flush(num_workers=1, write_remainder=True)

    files = tuple(str(path) for path in sorted(output_dir.rglob("*.parquet")))
    lengths = tuple(pq.ParquetFile(path).metadata.num_rows for path in files)
    if sum(lengths) != len(records):
        raise RuntimeError("Written Parquet row count disagrees with prompt count")
    cache_dir = output_dir / "map_style_cache"
    cache_dir.mkdir()
    temporary = cache_dir / "file_info.pkl.tmp"
    with temporary.open("xb") as handle:
        pickle.dump((files, lengths), handle)
    temporary.rename(cache_dir / "file_info.pkl")
    return len(records)


def _load_encoder(model_path: Path, shard_index: int) -> Callable[[str], np.ndarray]:
    import torch

    import fastvideo.envs as envs
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import PipelineComponentLoader
    from fastvideo.pipelines import ForwardBatch
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
    from fastvideo.utils import verify_model_config_and_directory

    envs.setdefault_external("MASTER_ADDR", "127.0.0.1")
    envs.setdefault_external("MASTER_PORT", str(29531 + shard_index))
    envs.setdefault_external("RANK", "0")
    envs.setdefault_external("WORLD_SIZE", "1")
    envs.setdefault_external("LOCAL_RANK", "0")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    model_index = verify_model_config_and_directory(str(model_path))
    args = FastVideoArgs(
        model_path=str(model_path),
        pipeline_config=MiniMaxH3PipelineConfig(),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=False,
    )

    def load(name: str) -> Any:
        return PipelineComponentLoader.load_module(
            module_name=name,
            component_model_path=str(model_path / name),
            transformers_or_diffusers=model_index[name][0],
            fastvideo_args=args,
        )

    stage = MiniMaxH3ConditioningStage(
        tokenizer=load("tokenizer"),
        processor=load("processor"),
        conditioner=load("text_encoder"),
    )

    def encode(prompt: str) -> np.ndarray:
        with torch.inference_mode():
            batch = ForwardBatch(data_type="video", prompt=prompt)
            batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
            batch = stage.forward(batch, args)
            if not batch.prompt_embeds or len(batch.prompt_embeds) != 1:
                raise RuntimeError("MiniMax H3 conditioning must return exactly one text embedding")
            return batch.prompt_embeds[0].squeeze(0).float().cpu().contiguous().numpy()

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Local MiniMax-H3 checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--samples-per-file", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=65536, help="Reject longer prompts; never truncate")
    args = parser.parse_args()
    records = load_prompt_records(args.prompts_file, shard_index=args.shard_index, num_shards=args.num_shards)
    output = args.output_dir / f"shard_{args.shard_index:05d}"
    if output.exists():
        raise FileExistsError(f"Output shard already exists: {output}; use a fresh output directory")
    if args.samples_per_file <= 0 or args.max_tokens <= 0:
        parser.error("--samples-per-file and --max-tokens must be positive")
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory is missing: {model_path}")
    encoder = _load_encoder(model_path, args.shard_index)
    count = encode_records(records, output, encoder, samples_per_file=args.samples_per_file, max_tokens=args.max_tokens)
    print(f"Encoded {count} prompts into {output}; set training.data.data_path to this shard directory")


if __name__ == "__main__":
    main()
