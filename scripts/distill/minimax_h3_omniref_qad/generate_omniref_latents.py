# SPDX-License-Identifier: Apache-2.0
"""Generate FastH3 OmniRef (Ref2VA) DiT latents from precomputed conditioning rows.

Training and evaluation data for reference-conditioned FastH3 students. Each input
row is one OmniRef Ref2VA training parquet row (``minimax_h3_ref2va`` record
schema): the Qwen3-VL presentation (``text_embedding`` + ``text_token_tags``),
the clean normalized reference latent rows (``reference_video_latent_rows`` /
``reference_audio_latent_rows``) and their ordered geometry. Nothing is
re-encoded: no text encoder, no reference media, no VAE encode.

Per row this script does what ``MiniMaxH3LatentPreparationStage`` does after its
encoders (0.999 noise augmentation of the reference rows with the request
generator, then target video/audio noise, packed with
``build_ref2va_packed_sequence``), runs the pipeline's own
``MiniMaxH3DenoisingStage`` (PDD fused blocks, VSA-H3 with the checkpoint's
reference keep rate), and keeps the target video rows exactly as
``MiniMaxH3VideoDecodingStage`` would decode them. The saved latent is
``[24, T, H, W]`` float32 in the VAE's normalized space,
``(raw - latents_mean) / latents_std``: the DiT's own space, so this equals
decode-stage ``denormalize_latents`` followed by normalization.

Output rows match ``generate_h3_latents.py`` (``id``, ``vae_latent_bytes/shape/
dtype``, ``generator``, ``case``, ``width``, ``height``, ``num_frames``, ``seed``)
and are written atomically, ``FLUSH_EVERY`` rows per file. Re-running skips ids
already written. Rows are chosen deterministically: per (case, resolution) the
manifest entries up to ``--max-frames`` are shuffled with ``--seed`` and the first
``--per-group`` are taken, interleaved across groups so a partial run stays
balanced.

One process per GPU (each its own single-rank external launcher), e.g.::

    CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 python generate_omniref_latents.py \\
        --model-path /models/fasth3_omniref_composed --manifest /data/omniref/MANIFEST.jsonl \\
        --output-dir /data/gen-latents/omniref/data \\
        --per-group 300 --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

FLUSH_EVERY = 8
GENERATOR_NAME = "omniref_step3500"
CASES = ("first_frame", "first_last_frame", "storyboard", "continue_scene", "continue_shot")
RESOLUTIONS = ("480p", "768p")
BUCKET_FRAMES = re.compile(r"bucket=\d+x\d+-(\d+)f")
ROW_COLUMNS = [
    "id",
    "vae_latent_shape",
    "audio_latent_shape",
    "text_embedding_bytes",
    "text_embedding_shape",
    "text_token_tags_bytes",
    "text_token_tags_shape",
    "reference_video_latent_rows_bytes",
    "reference_video_latent_rows_shape",
    "reference_audio_latent_rows_bytes",
    "reference_audio_latent_rows_shape",
    "reference_media_types",
    "reference_has_audio",
    "reference_num_latent_frames",
    "reference_latent_heights",
    "reference_latent_widths",
    "reference_num_audio_latents",
    "width",
    "height",
    "num_frames",
]


# --------------------------------------------------------------------------- plan
def clip_seed(base_seed: int, clip_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{clip_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def load_manifest(path: str, cases: list[str], max_frames: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            entry = json.loads(line)
            match = BUCKET_FRAMES.search(entry["parquet"])
            if match is None or entry["case"] not in cases or entry["resolution"] not in RESOLUTIONS:
                continue
            frames = int(match.group(1))
            if frames <= max_frames:
                groups[(entry["case"], entry["resolution"])].append({**entry, "frames": frames})
    return groups


def select_clips(groups: dict[tuple[str, str], list[dict[str, Any]]],
                 seed: int,
                 per_group: int,
                 exclude_sources: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Per group: seeded shuffle, drop excluded source clips, take ``per_group``; interleave the groups."""
    picked = []
    for key in sorted(groups):
        entries = sorted(groups[key], key=lambda entry: entry["id"])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(entries)
        entries = [entry for entry in entries if entry["id"].split(":", 1)[-1] not in exclude_sources]
        picked.append(entries[:per_group])
    plan = []
    for index in range(max((len(entries) for entries in picked), default=0)):
        for entries in picked:
            if index < len(entries):
                entry = entries[index]
                source_id = entry["id"].split(":", 1)[-1]
                clip_id = f"omniref-{entry['case']}-{source_id}-{entry['resolution']}"
                plan.append({
                    "id": clip_id,
                    "source": source_id,
                    "case": entry["case"],
                    "resolution": entry["resolution"],
                    "parquet": entry["parquet"],
                    "seed": clip_seed(seed, clip_id)
                })
    return plan


def clip_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The full deterministic plan; shard ``s`` takes every ``num_shards``-th entry from ``s``."""
    groups = load_manifest(args.manifest, args.cases, args.max_frames)
    if args.eval_dir is None:
        return select_clips(groups, args.seed, args.per_group)[args.shard::args.num_shards]
    # A held-out set: no source clip (at either resolution) of the main plan.
    main_plan = select_clips(groups, args.exclude_seed, args.exclude_per_group)
    excluded = frozenset(clip["source"] for clip in main_plan)
    return select_clips(groups, args.seed, args.per_group, excluded)[args.shard::args.num_shards]


def existing_ids(data_dir: Path) -> set[str]:
    import pyarrow.parquet as pq

    done: set[str] = set()
    for path in data_dir.glob("*.parquet"):
        done.update(pq.read_table(path, columns=["id"]).column("id").to_pylist())
    return done


def write_part(data_dir: Path, shard: int, rows: list[dict[str, Any]]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = data_dir / f"omniref-shard{shard:03d}-{int(time.time() * 1000)}.parquet"
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- rows
def _tensor(row: dict[str, Any], name: str) -> torch.Tensor:
    shape = [int(value) for value in row[f"{name}_shape"]]
    array = np.frombuffer(row[f"{name}_bytes"], dtype=np.float32)
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f"{name}: {array.size} values do not fill shape {shape}")
    return torch.from_numpy(array.reshape(shape).copy())


def read_row(parquet: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=ROW_COLUMNS).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"{parquet}: expected one row, got {len(rows)}")
    return rows[0]


def prepared_references(row: dict[str, Any]) -> list[Any]:
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference

    fields = ("reference_media_types", "reference_has_audio", "reference_num_latent_frames",
              "reference_latent_heights", "reference_latent_widths", "reference_num_audio_latents")
    return [
        MiniMaxH3PreparedReference(media_type=media_type,
                                   has_audio=bool(has_audio),
                                   num_latent_frames=int(frames),
                                   latent_height=int(height),
                                   latent_width=int(width),
                                   num_audio_latents=int(audio))
        for media_type, has_audio, frames, height, width, audio in zip(*(row[name] for name in fields), strict=True)
    ]


# --------------------------------------------------------------------------- generation
class OmniRefLatentGenerator:
    """Single-rank (or SPMD) pipeline driver that injects precomputed Ref2VA conditioning."""

    def __init__(self, args: argparse.Namespace) -> None:
        from fastvideo import VideoGenerator
        from fastvideo.api import (ComponentConfig, EngineConfig, GeneratorConfig, OffloadConfig, ParallelismConfig,
                                   PipelineSelection)

        world_size = int(os.environ.setdefault("WORLD_SIZE", "1"))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        config = GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=world_size,
                execution_backend="external_launcher",
                use_fsdp_inference=world_size > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=world_size),
                offload=OffloadConfig(dit=False,
                                      dit_layerwise=False,
                                      text_encoder=True,
                                      vae=True,
                                      pin_cpu_memory=False,
                                      lazy_module_load=True),
            ),
            pipeline=PipelineSelection(
                workload_type="i2v",
                components=ComponentConfig(override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline"),
            ),
        )
        self.generator = VideoGenerator.from_config(config)
        worker = self.generator.executor.worker.worker
        self.pipeline = worker.pipeline
        self.fastvideo_args = worker.fastvideo_args
        self.is_output_rank = bool(self.fastvideo_args.is_output_rank)
        if not self.pipeline._denoise_stages_ready:
            self.pipeline._add_denoise_stages(ref2va=True)
        self.denoise = self.pipeline._stage_name_mapping["denoising_stage"]
        self.scheduler = self.pipeline.get_module("scheduler")

    def _geometry(self, row: dict[str, Any]) -> tuple[int, int, int, int]:
        from fastvideo.pipelines.basic.minimax_h3.packing import audio_latent_num_frames, video_latent_num_frames

        arch = self.fastvideo_args.pipeline_config.vae_config.arch_config
        ratio = int(arch.spatial_compression_ratio)
        num_frames, height, width = int(row["num_frames"]), int(row["height"]), int(row["width"])
        geometry = (video_latent_num_frames(num_frames), height // ratio, width // ratio)
        expected = tuple(int(value) for value in row["vae_latent_shape"][1:])
        if geometry != expected or int(row["vae_latent_shape"][0]) != int(arch.latent_channels):
            raise ValueError(f"target latent geometry {geometry} disagrees with the row's {row['vae_latent_shape']}")
        num_audio = audio_latent_num_frames(num_frames)
        if num_audio != int(row["audio_latent_shape"][-1]):
            raise ValueError(f"{num_audio} audio latents disagree with the row's {row['audio_latent_shape']}")
        return (*geometry, num_audio)

    def _batch(self, row: dict[str, Any], seed: int) -> Any:
        """The ForwardBatch ``MiniMaxH3LatentPreparationStage`` would hand to denoising."""
        from diffusers.utils.torch_utils import randn_tensor

        from fastvideo.distributed import get_local_torch_device
        from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_AUDIO_CHANNELS,
                                                                  MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                                                  build_ref2va_packed_sequence, h3_dit_patch_size,
                                                                  keyframe_condition_noise, patchify_video_latents)
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
        from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

        device = get_local_torch_device()
        args = self.fastvideo_args
        patch = h3_dit_patch_size(args)
        channels = int(args.pipeline_config.vae_config.arch_config.latent_channels)
        audio_channels = int(args.pipeline_config.audio_vae_config.arch_config.latent_channels)
        frames, height, width, num_audio = self._geometry(row)
        references = prepared_references(row)
        tags = _tensor(row, "text_token_tags").round().to(torch.long)
        embeds = _tensor(row, "text_embedding").to(device=device, dtype=torch.bfloat16)[None]
        layout = build_ref2va_packed_sequence(tags, references, frames, height, width, num_audio, patch)

        generator = torch.Generator("cpu").manual_seed(seed)
        shapes = tuple((ref.num_latent_frames, ref.latent_height, ref.latent_width) for ref in references
                       if ref.media_type != "audio")
        noise = keyframe_condition_noise(shapes, patch, channels, generator=generator, device=device)
        clean = _tensor(row, "reference_video_latent_rows").to(device)
        condition_video = self.scheduler.scale_noise(clean, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
        video_noise = randn_tensor((1, channels, frames, height, width),
                                   generator=generator,
                                   device=device,
                                   dtype=torch.float32)
        audio_rows = randn_tensor((num_audio * MINIMAX_H3_AUDIO_CHANNELS, audio_channels),
                                  generator=generator,
                                  device=device,
                                  dtype=torch.float32)
        condition_audio = _tensor(row, "reference_audio_latent_rows").to(device)
        video_rows = torch.cat((condition_video, patchify_video_latents(video_noise, patch)))
        audio_rows = torch.cat((condition_audio, audio_rows))
        if video_rows.shape[0] != layout.video_indices.numel() or audio_rows.shape[0] != layout.audio_indices.numel():
            raise ValueError("packed rows do not match the Ref2VA layout")

        batch = ForwardBatch(data_type="video",
                             prompt="",
                             seed=seed,
                             num_frames=int(row["num_frames"]),
                             height=int(row["height"]),
                             width=int(row["width"]),
                             num_inference_steps=len(args.pipeline_config.pdd_step_indices) - 1,
                             guidance_scale=1.0,
                             VSA_sparsity=float(args.VSA_sparsity))
        batch.generator = generator
        batch.prompt_embeds = [embeds]
        batch.latents = video_rows
        batch.audio_latents = audio_rows
        batch.raw_latent_shape = (1, channels, frames, height, width)
        batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout
        return batch

    @torch.no_grad()
    def generate(self, row: dict[str, Any], seed: int) -> torch.Tensor:
        """Normalized target-video latent ``[24, T, H, W]`` (float32, CPU)."""
        from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size, unpatchify_video_tokens
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY

        batch = self.denoise.forward(self._batch(row, seed), self.fastvideo_args)
        layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
        _, channels, frames, height, width = batch.raw_latent_shape
        latents = unpatchify_video_tokens(batch.latents[layout.num_condition_video_rows:], frames, height, width,
                                          channels, h3_dit_patch_size(self.fastvideo_args))
        return latents[0].float().cpu().contiguous()

    @torch.no_grad()
    def decode(self, normalized: torch.Tensor) -> np.ndarray:
        """Decode one normalized ``[24, T, H, W]`` latent to uint8 ``[T, H, W, 3]`` with the original VAE."""
        from fastvideo.distributed import get_local_torch_device

        from fastvideo.models import pinned_offload

        device = get_local_torch_device()
        vae = self.pipeline.get_module("vae")
        pinned_offload.load(vae, device, pin=False)
        try:
            latents = vae.denormalize_latents(normalized[None].to(device=device, dtype=torch.float32))
            output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                vae.decode_to_pixels(latents, output)
        finally:
            pinned_offload.unload(vae)
        return output[0].mul(255).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()

    def shutdown(self) -> None:
        self.generator.shutdown()


# --------------------------------------------------------------------------- preview
def first_reference_latent(row: dict[str, Any], patch: tuple[int, int, int]) -> torch.Tensor | None:
    from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens

    references = [ref for ref in prepared_references(row) if ref.media_type != "audio"]
    if not references:
        return None
    ref = references[0]
    count = ref.num_latent_frames * (ref.latent_height // patch[1]) * (ref.latent_width // patch[2])
    rows = _tensor(row, "reference_video_latent_rows")[:count]
    return unpatchify_video_tokens(rows, ref.num_latent_frames, ref.latent_height, ref.latent_width, 24, patch)[0]


def write_preview(driver: OmniRefLatentGenerator, clip: dict[str, Any], row: dict[str, Any], saved: bytes,
                  shape: list[int], preview_dir: Path) -> None:
    """Decode the serialized latent, the dataset target and the first reference for a visual check."""
    import imageio.v2 as imageio
    import pyarrow.parquet as pq

    from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size

    preview_dir.mkdir(parents=True, exist_ok=True)
    generated = driver.decode(torch.from_numpy(np.frombuffer(saved, dtype=np.float32).reshape(shape).copy()))
    target_row = pq.read_table(clip["parquet"], columns=["vae_latent_bytes", "vae_latent_shape"]).to_pylist()[0]
    target_latent = _tensor(target_row, "vae_latent")
    target = driver.decode(target_latent)
    stem = preview_dir / clip["id"]
    imageio.mimsave(f"{stem}.mp4", list(np.concatenate((generated, target), axis=2)), fps=24, format="mp4")
    reference = first_reference_latent(row, h3_dit_patch_size(driver.fastvideo_args))
    if reference is not None:
        frames = driver.decode(reference)
        imageio.mimsave(f"{stem}.reference.mp4", list(frames), fps=24, format="mp4")
    stats = {
        "id": clip["id"],
        "generated_mean": float(np.frombuffer(saved, dtype=np.float32).mean()),
        "generated_std": float(np.frombuffer(saved, dtype=np.float32).std()),
        "target_mean": float(target_latent.mean()),
        "target_std": float(target_latent.std()),
    }
    print(json.dumps({"preview": str(stem), **stats}), flush=True)


# --------------------------------------------------------------------------- main
def append_eval_manifest(path: Path, entry: dict[str, Any]) -> None:
    """Append one row under an exclusive lock, rereading so shard processes don't drop each other's rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current = json.loads(path.read_text()) if path.is_file() else []
            if entry["id"] not in {item["id"] for item in current}:
                current.append(entry)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(current, indent=1))
            tmp.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_eval(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    """Write raw (denormalized) ``[1, 24, T, H, W]`` latents, as the decode stage decodes them, plus a manifest."""
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    done = {item["id"] for item in manifest}
    # The model's own vae/config.json: the pipeline config's VAE arch in this process keeps its dataclass
    # defaults (zeros/ones) until the VAE itself loads.
    vae_config = json.loads((Path(args.model_path) / "vae" / "config.json").read_text())
    mean = torch.tensor(vae_config["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae_config["latents_std"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    driver = OmniRefLatentGenerator(args)
    for clip in plan:
        if clip["id"] in done:
            continue
        start = time.perf_counter()
        row = read_row(clip["parquet"])
        raw = (driver.generate(row, clip["seed"])[None] * std + mean).contiguous()
        if not bool(torch.isfinite(raw).all()):
            print(json.dumps({"id": clip["id"], "error": "non-finite latent"}), flush=True)
            continue
        name = f"{clip['case']}_{clip['resolution']}_{clip['id']}.pt"
        if driver.is_output_rank:
            tmp = eval_dir / f"{name}.tmp"
            torch.save(raw, tmp)
            tmp.replace(eval_dir / name)
            append_eval_manifest(
                manifest_path, {
                    "file": name,
                    **{key: clip[key] for key in ("id", "source", "case", "resolution", "parquet", "seed")},
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "num_frames": int(row["num_frames"]),
                    "shape": list(raw.shape),
                    "dtype": "float32",
                    "space": "raw (denormalized; decode-stage input)",
                    "generator": args.generator_name,
                })
        print(json.dumps({"id": clip["id"], "shape": list(raw.shape), "seconds": round(time.perf_counter() - start, 1)}),
              flush=True)
    driver.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="composed FastH3 OmniRef PDD model directory")
    parser.add_argument("--manifest", required=True, help="OmniRef MANIFEST.jsonl (case, id, parquet, resolution)")
    parser.add_argument("--output-dir", default=None, help="directory receiving the parquet parts")
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--per-group", type=int, default=300, help="clips per (case, resolution)")
    parser.add_argument("--max-frames", type=int, default=243)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new clips (0: no limit)")
    parser.add_argument("--generator-name", default=GENERATOR_NAME)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--preview-dir", default=None, help="decode the first --preview clips here as MP4")
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--eval-dir",
                        default=None,
                        help="held-out mode: write raw [1, 24, T, H, W] .pt latents and manifest.json here, from "
                        "source clips outside the main plan (--exclude-seed / --exclude-per-group)")
    parser.add_argument("--exclude-seed", type=int, default=20261007)
    parser.add_argument("--exclude-per-group", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = clip_plan(args)
    if args.eval_dir is not None:
        print(json.dumps({"eval_clips": len(plan)}), flush=True)
        run_eval(args, plan)
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --eval-dir is given")
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    done = existing_ids(data_dir)
    todo = [clip for clip in plan if clip["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(json.dumps({"shard": args.shard, "planned": len(plan), "todo": len(todo)}), flush=True)
    if not todo:
        return
    driver = OmniRefLatentGenerator(args)
    pending: list[dict[str, Any]] = []
    previews = 0
    for clip in todo:
        start = time.perf_counter()
        try:
            row = read_row(clip["parquet"])
            latent = driver.generate(row, clip["seed"])
            if not bool(torch.isfinite(latent).all()):
                raise ValueError("non-finite latent")
        except Exception as error:  # noqa: BLE001 - record and continue; one bad row must not stop the shard
            print(json.dumps({"id": clip["id"], "error": repr(error)[:500]}), flush=True)
            continue
        record = {
            "id": clip["id"],
            "vae_latent_bytes": latent.numpy().tobytes(),
            "vae_latent_shape": list(latent.shape),
            "vae_latent_dtype": "float32",
            "generator": args.generator_name,
            "case": clip["case"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "num_frames": int(row["num_frames"]),
            "seed": int(clip["seed"]),
        }
        print(json.dumps({
            "id": clip["id"],
            "shape": record["vae_latent_shape"],
            "seconds": round(time.perf_counter() - start, 1)
        }),
              flush=True)
        if not driver.is_output_rank:
            continue
        pending.append(record)
        if args.preview_dir and previews < args.preview:
            write_preview(driver, clip, row, record["vae_latent_bytes"], record["vae_latent_shape"],
                          Path(args.preview_dir))
            previews += 1
        if len(pending) >= FLUSH_EVERY:
            write_part(data_dir, args.shard, pending)
            pending = []
    if pending and driver.is_output_rank:
        write_part(data_dir, args.shard, pending)
    driver.shutdown()


if __name__ == "__main__":
    main()
