# SPDX-License-Identifier: Apache-2.0
"""Encode a prompt list into MiniMax H3 text-only conditioning rows (data-free DMD2).

Each line of ``--prompts-file`` is one complete H3 prompt document (for the
VidProM-H3 set: ``integrated_multimodal_description: ... overall_soundscape:
... non_diegetic_music: ...`` fed to the model verbatim as one string). Every
prompt is tokenized raw (no chat template, ``add_special_tokens=False``) and
encoded through ``MiniMaxH3ConditioningStage`` — the same Qwen3-VL layer-50
path that produced the validated t2va overfit rows — then written as
``pyarrow_schema_text_only`` records for ``rollout_mode: simulate`` training
(``training.data.preprocessed_data_type: text_only``).

Embeddings are stored as float32 to match the training collate, which decodes
``text_embedding_bytes`` with a hard-coded ``np.float32``
(``fastvideo/dataset/utils.py``). At ~300 tokens x 5120 dims that is ~6 MB per
prompt — budget ~380 GB for the full 63k VidProM set.

Sharding: ``--num-shards N`` splits the prompt list round-robin by line index;
run one process per GPU with distinct ``--shard-index``/``CUDA_VISIBLE_DEVICES``
(and a distinct ``MASTER_PORT`` — each process initializes a one-rank process
group). Each shard writes ``<output-dir>/shard_XX/``; the training dataloader
walks the directory tree, so pointing ``training.data.data_path`` at
``--output-dir`` picks up every shard. Restarting a shard resumes after the
rows already on disk (delete the shard directory to re-encode from scratch).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter, records_to_table)
from fastvideo.dataset.dataloader.record_schema import text_only_record_creator
from fastvideo.dataset.dataloader.schema import pyarrow_schema_text_only
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.utils import verify_model_config_and_directory


def _init_single_process_distributed(shard_index: int) -> None:
    """Initialize the one-rank process groups required by component loaders.

    Concurrent shards on one host must not share a rendezvous port, so the
    default port is offset by the shard index (an explicit ``MASTER_PORT``
    still wins).
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29531 + shard_index))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)


def _load_component(
    name: str,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> Any:
    """Load one checkpoint component through the inference component registry."""
    transformers_or_diffusers, _ = model_index[name][:2]
    return PipelineComponentLoader.load_module(
        module_name=name,
        component_model_path=str(model_path / name),
        transformers_or_diffusers=transformers_or_diffusers,
        fastvideo_args=fastvideo_args,
    )


def _load_shard_prompts(prompts_file: Path, shard_index: int, num_shards: int) -> list[tuple[int, str]]:
    """Return this shard's ``(global_line_index, prompt)`` pairs, round-robin by line."""
    entries: list[tuple[int, str]] = []
    with prompts_file.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            prompt = line.strip()
            if prompt:
                entries.append((index, prompt))
    return entries[shard_index::num_shards]


def _count_existing_rows(shard_dir: Path) -> int:
    """Count rows already written under a shard directory (resume offset)."""
    total = 0
    for parquet_path in sorted(shard_dir.rglob("*.parquet")):
        total += pq.ParquetFile(parquet_path).metadata.num_rows
    return total


def main(args: argparse.Namespace) -> None:
    _init_single_process_distributed(args.shard_index)

    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"MiniMax H3 model directory is missing at {model_path}")
    model_index = verify_model_config_and_directory(
        str(model_path),
        required_component_dirs=("tokenizer", "processor", "text_encoder"),
    )

    shard = _load_shard_prompts(args.prompts_file, args.shard_index, args.num_shards)
    shard_dir = args.output_dir / f"shard_{args.shard_index:02d}"
    already_done = _count_existing_rows(shard_dir) if shard_dir.is_dir() else 0
    if already_done >= len(shard):
        print(f"Shard {args.shard_index}/{args.num_shards}: all {len(shard)} rows already encoded")
        return
    todo = shard[already_done:]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(shard)} prompts total, "
          f"{already_done} already on disk, encoding {len(todo)} now -> {shard_dir}")

    fastvideo_args = FastVideoArgs(
        model_path=str(model_path),
        pipeline_config=MiniMaxH3PipelineConfig(),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        use_fsdp_inference=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    print("Loading MiniMax H3 tokenizer, processor, and Qwen3-VL encoder")
    tokenizer = _load_component("tokenizer", model_path, model_index, fastvideo_args)
    processor = _load_component("processor", model_path, model_index, fastvideo_args)
    conditioner = _load_component("text_encoder", model_path, model_index, fastvideo_args)
    stage = MiniMaxH3ConditioningStage(
        conditioner=conditioner,
        tokenizer=tokenizer,
        processor=processor,
    )

    writer = ParquetDatasetWriter(out_dir=str(shard_dir), samples_per_file=args.samples_per_file)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for done, (global_index, prompt) in enumerate(todo, start=1):
            batch = ForwardBatch(data_type="video", prompt=prompt)
            batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
            batch = stage.forward(batch, fastvideo_args)
            if not batch.prompt_embeds:
                raise RuntimeError(f"MiniMax H3 conditioning returned no embedding for line {global_index}")
            # float32 to match the training collate's np.frombuffer dtype.
            text_embedding = batch.prompt_embeds[0].squeeze(0).float().cpu().contiguous().numpy()
            records.append(
                text_only_record_creator(
                    text_name=f"vidprom_{global_index:05d}",
                    text_embedding=text_embedding,
                    caption=prompt,
                ))

            if len(records) >= args.flush_every or done == len(todo):
                writer.append_table(records_to_table(records, pyarrow_schema_text_only))
                records = []
                written = writer.flush(write_remainder=done == len(todo))
                rate = done / (time.monotonic() - started)
                remaining = (len(todo) - done) / rate if rate > 0 else float("inf")
                print(f"[shard {args.shard_index}] {done}/{len(todo)} encoded "
                      f"({rate:.2f} prompts/s, ~{remaining / 60:.0f} min left, flushed {written} rows)")

    print(f"Shard {args.shard_index} complete: {already_done + len(todo)}/{len(shard)} rows in {shard_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, required=True, help="one H3 prompt document per line")
    parser.add_argument("--model-path", type=Path, required=True, help="MiniMax-H3 checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="dataset root; shards write shard_XX/ under it")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--samples-per-file", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=256, help="rows buffered between parquet flushes")
    parser.add_argument("--limit", type=int, default=None, help="encode at most N prompts this run (smoke tests)")
    cli_args = parser.parse_args()
    if not 0 <= cli_args.shard_index < cli_args.num_shards:
        parser.error(f"--shard-index {cli_args.shard_index} must be in [0, {cli_args.num_shards})")
    main(cli_args)
