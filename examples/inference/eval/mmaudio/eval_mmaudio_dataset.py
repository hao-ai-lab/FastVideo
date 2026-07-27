# SPDX-License-Identifier: Apache-2.0
"""Distributed MMAudio inference over FastVideo preprocessing caches.

Launch with ``torchrun``. Each process owns one complete MMAudio pipeline and
processes a non-padded, rank-strided subset of the dataset. The output manifest
is directly consumable by ``fastvideo eval run --manifest ...``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from fastvideo.dataset.mmaudio_feature_dataset import build_mmaudio_feature_dataset
from fastvideo.distributed import cleanup_dist_env_and_memory
from fastvideo.pipelines.basic.mmaudio import MMAudioPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class _RankStrideSampler(Sampler[int]):
    """Assign every global index exactly once without tail padding."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class _IndexedDataset(Dataset):

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, dict[str, Any]]:
        return index, self.dataset[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=14159265)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "sample"


def _feature_shapes(transformer: torch.nn.Module) -> dict[str, int]:
    arch = transformer.config.arch_config
    return {
        "latent_seq_len": int(transformer.latent_seq_len),
        "latent_dim": int(transformer.latent_dim),
        "clip_seq_len": int(transformer.clip_seq_len),
        "clip_dim": int(arch.clip_dim),
        "sync_seq_len": int(transformer.sync_seq_len),
        "sync_dim": int(arch.sync_dim),
        "text_seq_len": int(arch.text_seq_len),
        "text_dim": int(arch.text_dim),
    }


def _build_pipeline(args: argparse.Namespace, world_size: int) -> MMAudioPipeline:
    # Cached features make the three conditioning encoders unnecessary. The
    # stages still exist, but consume direct tensors and never dereference
    # these sentinels.
    ignored_component = object()
    return MMAudioPipeline.from_pretrained(
        str(args.model_path),
        inference_mode=True,
        loaded_modules={
            "text_encoder": ignored_component,
            "tokenizer": ignored_component,
            "image_encoder": ignored_component,
            "image_encoder_2": ignored_component,
        },
        workload_type="v2a",
        num_gpus=world_size,
        tp_size=1,
        sp_size=1,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        enable_torch_compile=args.compile,
    )


def _forward_batch(
    sample: dict[str, Any],
    *,
    duration_seconds: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: str,
) -> ForwardBatch:
    return ForwardBatch(
        data_type="video",
        prompt="",
        negative_prompt=negative_prompt,
        audio_start_in_s=0.0,
        audio_end_in_s=duration_seconds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        num_videos_per_prompt=1,
        height=8,
        width=8,
        num_frames=1,
        save_video=False,
        return_frames=False,
        extra={
            "mmaudio_clip_features": sample["clip_features"].unsqueeze(0),
            "mmaudio_sync_features": sample["sync_features"].unsqueeze(0),
            "mmaudio_text_features": sample["text_features"].unsqueeze(0),
        },
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _merge_rank_outputs(output_dir: Path, world_size: int) -> None:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rank in range(world_size):
        manifest = output_dir / f"manifest_rank_{rank:05d}.jsonl"
        failure_path = output_dir / f"failures_rank_{rank:05d}.jsonl"
        if manifest.is_file():
            rows.extend(json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
        if failure_path.is_file():
            failures.extend(
                json.loads(line) for line in failure_path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows.sort(key=lambda row: int(row["index"]))
    failures.sort(key=lambda row: int(row["index"]))
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample ids found while merging inference manifests")
    _write_jsonl(output_dir / "eval_manifest.jsonl", rows)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    summary = {
        "num_succeeded": len(rows),
        "num_failed": len(failures),
        "world_size": world_size,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("MMAudio dataset inference requires CUDA")
    torch.cuda.set_device(local_rank)

    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _build_pipeline(args, world_size)
    transformer = pipeline.get_module("transformer")
    dataset = build_mmaudio_feature_dataset(
        args.feature_root,
        feature_shapes=_feature_shapes(transformer),
        include_metadata=True,
    )
    dataset_size = len(dataset)
    if args.max_samples > 0:
        dataset_size = min(dataset_size, args.max_samples)
    sampler = _RankStrideSampler(dataset_size, rank, world_size)
    loader = DataLoader(
        _IndexedDataset(dataset),
        batch_size=None,
        sampler=sampler,
        num_workers=max(0, args.num_workers),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    progress = tqdm(
        loader,
        total=len(sampler),
        desc=f"MMAudio inference rank {rank}",
        position=local_rank,
    )
    for index, sample in progress:
        index = int(index)
        sample_id = _safe_name(str(sample.get("sample_id", index)))
        source_path = str(sample.get("source_path", ""))
        caption = str(sample.get("caption", ""))
        output_path = audio_dir / f"{sample_id}.wav"
        sample_seed = args.seed + index
        try:
            if not source_path or not Path(source_path).is_file():
                raise FileNotFoundError(f"source video is missing for sample {sample_id}: {source_path}")
            if args.overwrite or not output_path.is_file():
                batch = _forward_batch(
                    sample,
                    duration_seconds=args.duration_seconds,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=sample_seed,
                    negative_prompt=args.negative_prompt,
                )
                result = pipeline.forward(batch, pipeline.fastvideo_args)
                audio = result.extra.get("audio")
                sample_rate = result.extra.get("audio_sample_rate")
                if not isinstance(audio, np.ndarray) or not isinstance(sample_rate, int):
                    raise RuntimeError("MMAudio pipeline did not return decoded audio")
                temporary = output_path.with_suffix(".tmp.wav")
                wavfile.write(temporary, sample_rate, audio.astype(np.float32))
                temporary.replace(output_path)
            manifest_rows.append({
                "id": sample_id,
                "index": index,
                "video": str(Path(source_path).resolve()),
                "audio": str(output_path.resolve()),
                "reference_audio_source": str(Path(source_path).resolve()),
                "text_prompt": caption,
                "seed": sample_seed,
            })
        except Exception as error:  # noqa: BLE001 - isolate corrupt benchmark samples
            failure_rows.append({
                "id": sample_id,
                "index": index,
                "source": source_path,
                "error_type": type(error).__name__,
                "error": str(error),
            })

    _write_jsonl(output_dir / f"manifest_rank_{rank:05d}.jsonl", manifest_rows)
    _write_jsonl(output_dir / f"failures_rank_{rank:05d}.jsonl", failure_rows)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        _merge_rank_outputs(output_dir, world_size)
        print(f"Merged eval manifest: {output_dir / 'eval_manifest.jsonl'}")
        print(f"Inference summary: {output_dir / 'summary.json'}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    pipeline.close()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
