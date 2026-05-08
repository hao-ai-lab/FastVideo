"""Backfill CLIP image features into existing Wan latent parquet caches (TECH-118).

Reads each parquet shard, decodes the source video at the same sampled/letterboxed
frames the original vae_latent was built from, runs Wan-Fun's CLIP-H encoder on
K=13 frames (latent endpoints 0,4,8,...,48), and writes a new parquet with three
extra column groups:

    clip_feature_*       [K, 257, 1280] fp16 - real values
    first_frame_latent_* empty bytes (post-PR1 trainer treats empty as absent;
                         random-context mode rebuilds image_latents from
                         vae_latent and ignores this column)
    pil_image_*          empty bytes (trainer never consumes pil_image in fwd)

The cache is intended for random-context I2V training (max_condition_latents
>= 1). In legacy mode (max_condition_latents=0) the trainer explicitly raises
ValueError("requires first_frame_latent") since this column is empty -
preferred over a cryptic shape-mismatch crash from a stub.

Throughput design (post-Smoke #1: 80 s/file → target 10-15 s/file):

  * Read mp4s from /leonardo_scratch (Phase 0b staging) — pass --video-root.
  * N=8 CPU prefetch threads decode + letterbox + sample 13 frames per row,
    then push (parquet_path, table, [N_valid*13, 3, 480, 832] uint8) onto a
    bounded queue (maxsize=2). The GPU consumer pulls and runs the CLIP
    forward, hiding decode/preproc behind GPU compute.
  * The HF `CLIPImageProcessor` is loaded with use_fast=True (Rust-backed
    `CLIPImageProcessorFast`); ~3-5x faster than the slow Python path that
    Smoke #1 used. Frame-0 oracle gates the "minor differences" risk.
  * CLIP forward micro-batch defaults to 256 (was 64).

Writes are atomic (.tmp + os.replace). A done.log records (path, num_rows,
sha256_of_clip_bytes_concat) per file for resumability.

Usage (single GPU smoke):
    python scripts/preprocess/backfill_clip_features.py \\
        --src /leonardo_scratch/large/userexternal/mshariat/latents_wan_v1/W21_480x832_49f_16fps_clip \\
        --video-root /leonardo_scratch/large/userexternal/mshariat/clips_wan_v1/master_77f_25fps_720p/clips \\
        --shards shard_00 \\
        --max-files 15 \\
        --model-path /leonardo_work/AIFAC_P02_082/.cache/Wan2.1-I2V-14B-480P-Diffusers
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision.io
from transformers import AutoImageProcessor

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.dataset.transform import LetterboxResizeVideo
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.preprocess.preprocess_pipeline_i2v import PreprocessPipeline_I2V

logger = init_logger(__name__)

# Constants (tied to W21_480x832_49f_16fps cache spec - see README at
# /leonardo_work/AIFAC_P02_082/data/latents_wan_v1/W21_480x832_49f_16fps/README.md)
TRAIN_FPS = 16
NUM_FRAMES = 49
TARGET_HW = (480, 832)
K_LATENT_ENDPOINTS = 13
LATENT_FRAME_INDICES = [4 * i for i in range(K_LATENT_ENDPOINTS)]  # [0,4,...,48]

# CLIP output dims used by Wan-Fun I2V (CLIP-H/14, penultimate hidden state)
CLIP_TOKENS = 257
CLIP_HIDDEN = 1280


class ClipOnlyI2VPipeline(PreprocessPipeline_I2V):
    """Strip PreprocessPipeline_I2V down to just image_encoder.

    Avoids loading T5 (40 GB) and VAE (3 GB) we don't need for CLIP backfill.
    The image_processor that the pipeline auto-loads is the SLOW path; we
    instantiate the fast one separately in main() and bypass `get_module`.
    """

    _required_config_modules = ["image_encoder", "image_processor"]

    def create_pipeline_stages(self, fastvideo_args):
        # Skip TextEncodingStage; we don't have a tokenizer/text_encoder.
        pass

    def initialize_pipeline(self, fastvideo_args):
        pass


def build_clip_pipeline(model_path: str) -> ClipOnlyI2VPipeline:
    """Load just the image_encoder from the Wan-Fun I2V model."""
    pipeline_config = PipelineConfig.from_pretrained(model_path)
    pipeline_config.vae_config = WanVAEConfig(load_encoder=False, load_decoder=False)
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pipeline_config=pipeline_config,
    )
    pipeline = ClipOnlyI2VPipeline(model_path, fastvideo_args)
    return pipeline


def build_fast_image_processor(model_path: Path):
    """Load HF CLIPImageProcessorFast directly (skip the FastVideo loader, which
    may yield the slow Python processor). The fast processor accepts uint8 torch
    tensors directly, so we don't need PIL conversion."""
    processor_dir = model_path / "image_processor"
    proc = AutoImageProcessor.from_pretrained(processor_dir, use_fast=True)
    return proc


def resolve_video_path(video_root: Path, file_name: str) -> Path:
    """Map a parquet `file_name` to its source mp4 via the `<root>/<id[:2]>/<id>.mp4` layout."""
    return video_root / file_name[:2] / f"{file_name}.mp4"


def decode_and_letterbox(
    video_path: Path,
    src_fps: float,
    src_num_frames_metadata: int,
    transform: LetterboxResizeVideo,
) -> torch.Tensor:
    """Decode mp4, resample 25fps -> 16fps with PINNED indices (no random crop),
    letterbox to (832, 480), return uint8 [49, 3, 480, 832]."""
    video, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    n_decoded = video.shape[0]
    # FrameSamplingStage uses metadata-derived num_frames, which can disagree with
    # the actual decoded frame count by 1-2 frames on tail-overshoot clips. Mirror
    # the original cache: arange on metadata-num_frames, then truncate to whatever
    # was actually decoded.
    n_for_resample = src_num_frames_metadata
    frame_indices = np.arange(0, n_for_resample, src_fps / TRAIN_FPS).astype(int)[:NUM_FRAMES]
    frame_indices = np.clip(frame_indices, 0, n_decoded - 1)
    if len(frame_indices) < NUM_FRAMES:
        last = int(frame_indices[-1]) if len(frame_indices) else 0
        pad = np.full(NUM_FRAMES - len(frame_indices), last, dtype=np.int64)
        frame_indices = np.concatenate([frame_indices, pad])

    sampled = video[frame_indices]  # [49, C, H, W] uint8
    out = transform(sampled)  # [49, 3, 480, 832] uint8
    return out


@dataclasses.dataclass
class WorkItem:
    parquet_path: Path
    table: pa.Table | None
    frames: torch.Tensor | None  # [N_valid * K, 3, 480, 832] uint8, or None
    valid_indices: list[int]
    skipped: list[tuple]
    error: str | None  # producer-side error, or "no-valid-frames"


def producer_worker(
    in_q: queue.Queue,
    out_q: queue.Queue,
    video_root: Path,
    transform: LetterboxResizeVideo,
) -> None:
    """Consume parquet paths from in_q; produce WorkItems on out_q."""
    while True:
        path = in_q.get()
        try:
            if path is None:
                return
            try:
                table = pq.read_table(path)
                file_names = table["file_name"].to_pylist()
                fps_col = table["fps"].to_pylist()
                duration_col = table["duration_sec"].to_pylist()

                all_frames: list[torch.Tensor] = []
                valid_indices: list[int] = []
                skipped: list[tuple] = []
                for i, name in enumerate(file_names):
                    video_path = resolve_video_path(video_root, name)
                    if not video_path.exists():
                        skipped.append((i, name, "missing"))
                        continue
                    try:
                        src_fps = float(fps_col[i])
                        src_num_frames = int(math.ceil(src_fps * float(duration_col[i])))
                        sampled_uint8 = decode_and_letterbox(
                            video_path, src_fps, src_num_frames, transform
                        )
                        picked = sampled_uint8[LATENT_FRAME_INDICES]  # [13, 3, 480, 832]
                        all_frames.append(picked)
                        valid_indices.append(i)
                    except Exception as e:  # pragma: no cover
                        skipped.append((i, name, f"decode-error: {e}"))

                if not all_frames:
                    out_q.put(WorkItem(path, table, None, valid_indices, skipped, "no-valid-frames"))
                else:
                    flat = torch.cat(all_frames, dim=0).contiguous()
                    out_q.put(WorkItem(path, table, flat, valid_indices, skipped, None))
            except Exception as e:  # pragma: no cover
                tb = traceback.format_exc()
                out_q.put(WorkItem(path, None, None, [], [], f"producer-error: {e}\n{tb}"))
        finally:
            in_q.task_done()


def encode_clip_features(
    image_processor,
    image_encoder,
    frames_uint8: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """frames_uint8: [N, 3, H, W] uint8 in [0, 255].
    Returns: [N, 257, 1280] fp16 on CPU.
    """
    device = next(image_encoder.parameters()).device
    outputs = []
    for start in range(0, frames_uint8.shape[0], batch_size):
        chunk = frames_uint8[start : start + batch_size]
        # CLIPImageProcessorFast accepts uint8 torch tensors directly (no PIL).
        proc = image_processor(images=chunk, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device)
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            out = image_encoder(pixel_values=pixel_values)
            feats = out.last_hidden_state  # [B, 257, 1280]
        outputs.append(feats.to(torch.float16).cpu())
    return torch.cat(outputs, dim=0)


def build_and_write_parquet(
    parquet_path: Path,
    table: pa.Table,
    feats: torch.Tensor | None,
    valid_indices: list[int],
    skipped: list[tuple],
) -> tuple[int, str]:
    """Append clip_feature/ffl/pil columns and write atomically.

    `feats` is [N_valid, K, 257, 1280] fp16 on CPU, or None if no rows had valid
    videos (in which case the parquet is rewritten with all clip_feature columns
    empty — better than orphaning state, but the validator will flag this).
    """
    n_rows = table.num_rows
    valid_set = set(valid_indices)
    valid_iter = iter(range(len(valid_indices)))

    clip_bytes_col: list[bytes] = []
    clip_shape_col: list[list[int]] = []
    clip_dtype_col: list[str] = []
    ffl_bytes_col: list[bytes] = []
    ffl_shape_col: list[list[int]] = []
    ffl_dtype_col: list[str] = []
    pil_bytes_col: list[bytes] = []
    pil_shape_col: list[list[int]] = []
    pil_dtype_col: list[str] = []

    for i in range(n_rows):
        # Trainer treats numel()==0 as absent (post-PR1 _get_non_empty_batch_tensor).
        ffl_bytes_col.append(b"")
        ffl_shape_col.append([])
        ffl_dtype_col.append("")
        pil_bytes_col.append(b"")
        pil_shape_col.append([])
        pil_dtype_col.append("")
        if feats is not None and i in valid_set:
            v_idx = next(valid_iter)
            row_feat = feats[v_idx].contiguous().numpy()  # [13, 257, 1280] fp16
            clip_bytes_col.append(row_feat.tobytes())
            clip_shape_col.append([K_LATENT_ENDPOINTS, CLIP_TOKENS, CLIP_HIDDEN])
            clip_dtype_col.append("float16")
        else:
            clip_bytes_col.append(b"")
            clip_shape_col.append([])
            clip_dtype_col.append("")

    h = hashlib.sha256()
    for b in clip_bytes_col:
        h.update(b)
    sha = h.hexdigest()

    # Drop any pre-existing CLIP/ffl/pil columns so re-runs are idempotent
    # (without this, append_column on an already-rewritten parquet produces
    # duplicate columns, which pyarrow reads in single-file mode but cannot
    # unify across a dataset glob).
    existing = set(table.schema.names)
    for col in (
        "clip_feature_bytes", "clip_feature_shape", "clip_feature_dtype",
        "first_frame_latent_bytes", "first_frame_latent_shape", "first_frame_latent_dtype",
        "pil_image_bytes", "pil_image_shape", "pil_image_dtype",
    ):
        if col in existing:
            table = table.drop_columns([col])

    new_table = (
        table
        .append_column("clip_feature_bytes", pa.array(clip_bytes_col, type=pa.binary()))
        .append_column("clip_feature_shape", pa.array(clip_shape_col, type=pa.list_(pa.int64())))
        .append_column("clip_feature_dtype", pa.array(clip_dtype_col, type=pa.string()))
        .append_column("first_frame_latent_bytes", pa.array(ffl_bytes_col, type=pa.binary()))
        .append_column("first_frame_latent_shape", pa.array(ffl_shape_col, type=pa.list_(pa.int64())))
        .append_column("first_frame_latent_dtype", pa.array(ffl_dtype_col, type=pa.string()))
        .append_column("pil_image_bytes", pa.array(pil_bytes_col, type=pa.binary()))
        .append_column("pil_image_shape", pa.array(pil_shape_col, type=pa.list_(pa.int64())))
        .append_column("pil_image_dtype", pa.array(pil_dtype_col, type=pa.string()))
    )

    tmp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(new_table, tmp, compression="snappy")
    os.replace(tmp, parquet_path)

    if skipped:
        logger.warning(
            "Backfill: %d rows skipped in %s (first 3: %s)",
            len(skipped), parquet_path.name, skipped[:3],
        )
    return n_rows, sha


def _chunk_sort_key(p: Path) -> tuple:
    """Natural-sort by trailing chunk index so `data_chunk_2` < `data_chunk_10`.
    Falls back to lex sort within the same parent dir for any unexpected name."""
    name = p.stem  # e.g. "data_chunk_42"
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (str(p.parent), int(parts[1]))
    return (str(p.parent), -1, name)


def list_parquet_files(src: Path, shards: list[str] | None) -> list[Path]:
    if shards:
        roots = [src / s for s in shards]
    else:
        roots = sorted([p for p in src.iterdir() if p.is_dir() and p.name.startswith("shard_")])
    files = []
    for r in roots:
        files.extend(sorted(r.rglob("data_chunk_*.parquet"), key=_chunk_sort_key))
    return files


def load_done_log(path: Path) -> dict[str, tuple[int, str]]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                out[parts[0]] = (int(parts[1]), parts[2])
    return out


def append_done_log(path: Path, parquet_path: Path, n_rows: int, sha: str) -> None:
    with path.open("a") as f:
        f.write(f"{parquet_path}\t{n_rows}\t{sha}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True,
                        help="Destination root containing shard_NN/ subdirs (the snapshot on scratch).")
    parser.add_argument("--video-root", type=Path, required=True,
                        help="Source mp4 root (e.g. .../clips_wan_v1/master_77f_25fps_720p/clips/)")
    parser.add_argument("--model-path", type=Path,
                        default=Path("/leonardo_work/AIFAC_P02_082/.cache/Wan2.1-I2V-14B-480P-Diffusers"))
    parser.add_argument("--shards", nargs="*", default=None,
                        help="Subset of shard names to process (default: all under --src).")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most this many parquet files (debug / smoke run).")
    parser.add_argument("--micro-batch", type=int, default=256,
                        help="CLIP forward micro-batch (frames per fwd pass). Default: 256.")
    parser.add_argument("--video-prefetch-workers", type=int, default=8,
                        help="CPU threads decoding+letterboxing videos. 0 disables (serial).")
    parser.add_argument("--prefetch-queue-size", type=int, default=2,
                        help="Bounded out-queue size; caps RAM at ~queue_size * per-tile.")
    parser.add_argument("--done-log", type=Path, default=None,
                        help="Audit log (default: <src>/backfill_done.log).")
    args = parser.parse_args()

    src: Path = args.src
    if not src.exists():
        sys.exit(f"--src does not exist: {src}")
    if not args.video_root.exists():
        sys.exit(f"--video-root does not exist: {args.video_root}")
    done_log = args.done_log or (src / "backfill_done.log")

    maybe_init_distributed_environment_and_model_parallel(1, 1)

    logger.info("Loading CLIP encoder from %s ...", args.model_path)
    pipeline = build_clip_pipeline(str(args.model_path))
    image_encoder = pipeline.get_module("image_encoder").to("cuda").eval()
    image_processor = build_fast_image_processor(args.model_path)
    logger.info(
        "Loaded image_encoder dtype=%s, image_processor=%s (use_fast=True)",
        next(image_encoder.parameters()).dtype,
        type(image_processor).__name__,
    )

    transform = LetterboxResizeVideo(size=TARGET_HW)

    files = list_parquet_files(src, args.shards)
    if args.max_files:
        files = files[: args.max_files]

    done = load_done_log(done_log)
    work_files = [f for f in files if str(f) not in done]
    logger.info(
        "Backfill plan: %d total, %d already done, %d to process. done_log=%s",
        len(files), len(files) - len(work_files), len(work_files), done_log,
    )
    if not work_files:
        logger.info("Nothing to do.")
        return

    n_workers = max(0, int(args.video_prefetch_workers))
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue(maxsize=max(1, int(args.prefetch_queue_size)))

    if n_workers >= 1:
        workers = [
            threading.Thread(
                target=producer_worker,
                args=(in_q, out_q, args.video_root, transform),
                daemon=True,
                name=f"prefetch-{i}",
            )
            for i in range(n_workers)
        ]
        for w in workers:
            w.start()
        for f in work_files:
            in_q.put(f)
        for _ in workers:
            in_q.put(None)
    else:
        # workers=0: simulate a single inline worker so the consumer loop is unified.
        workers = []
        for f in work_files:
            in_q.put(f)

    n_remaining = len(work_files)
    n_done = 0
    n_failed = 0
    t0 = time.time()
    while n_remaining > 0:
        if n_workers >= 1:
            item = out_q.get()
        else:
            # Inline path: pop one parquet, run producer logic in main thread.
            path = in_q.get()
            try:
                table = pq.read_table(path)
                file_names = table["file_name"].to_pylist()
                fps_col = table["fps"].to_pylist()
                duration_col = table["duration_sec"].to_pylist()
                all_frames: list[torch.Tensor] = []
                valid_indices: list[int] = []
                skipped: list[tuple] = []
                for i, name in enumerate(file_names):
                    vp = resolve_video_path(args.video_root, name)
                    if not vp.exists():
                        skipped.append((i, name, "missing"))
                        continue
                    try:
                        sampled = decode_and_letterbox(
                            vp, float(fps_col[i]),
                            int(math.ceil(float(fps_col[i]) * float(duration_col[i]))),
                            transform,
                        )
                        all_frames.append(sampled[LATENT_FRAME_INDICES])
                        valid_indices.append(i)
                    except Exception as e:  # pragma: no cover
                        skipped.append((i, name, f"decode-error: {e}"))
                if not all_frames:
                    item = WorkItem(path, table, None, valid_indices, skipped, "no-valid-frames")
                else:
                    flat = torch.cat(all_frames, dim=0).contiguous()
                    item = WorkItem(path, table, flat, valid_indices, skipped, None)
            except Exception as e:  # pragma: no cover
                item = WorkItem(path, None, None, [], [], f"inline-error: {e}\n{traceback.format_exc()}")

        n_remaining -= 1

        if item.error and item.error != "no-valid-frames":
            logger.error("[FAIL] %s: %s", item.parquet_path, item.error)
            n_failed += 1
            continue

        try:
            if item.frames is not None:
                feats_flat = encode_clip_features(
                    image_processor, image_encoder, item.frames, args.micro_batch,
                )
                feats = feats_flat.view(
                    len(item.valid_indices), K_LATENT_ENDPOINTS, CLIP_TOKENS, CLIP_HIDDEN,
                )
            else:
                feats = None
            n_rows, sha = build_and_write_parquet(
                item.parquet_path, item.table, feats, item.valid_indices, item.skipped,
            )
            append_done_log(done_log, item.parquet_path, n_rows, sha)
            n_done += 1
            elapsed = time.time() - t0
            files_per_s = n_done / max(elapsed, 1e-6)
            sec_per_file = elapsed / max(n_done, 1)
            logger.info(
                "[done %d/%d] %s rows=%d valid=%d sha=%s elapsed=%.1fs (%.2f s/file, %.2f files/s)",
                n_done, len(work_files), item.parquet_path.name, n_rows,
                len(item.valid_indices), sha[:12], elapsed, sec_per_file, files_per_s,
            )
        except Exception as e:  # pragma: no cover
            logger.error("[WRITE-FAIL] %s: %s\n%s", item.parquet_path, e, traceback.format_exc())
            n_failed += 1
            continue

    if n_workers >= 1:
        for w in workers:
            w.join(timeout=30)

    elapsed = time.time() - t0
    logger.info(
        "Backfill done: %d ok, %d failed, %.1fs total, %.2f s/file avg",
        n_done, n_failed, elapsed, elapsed / max(n_done, 1),
    )


if __name__ == "__main__":
    main()
