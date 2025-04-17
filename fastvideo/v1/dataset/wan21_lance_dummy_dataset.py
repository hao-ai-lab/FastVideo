# wan21_lance_dummy_dataset.py
"""
End-to-end **dummy** dataset generator + loader that mirrors Wan 2.1 specs
==========================================================================

Two sub-commands (run from the shell):

    python wan21_lance_dummy_dataset.py generate  --rows 200 --max-T 16
    python wan21_lance_dummy_dataset.py benchmark --workers 4 --batches 500

* **generate** → streams realistic dummy samples into a Lance dataset placed on
  the fastest writable NVMe mount (falls back to ./lance_dataset).
* **benchmark** → iterates the dataset through a Torch `DataLoader` to sanity-check I/O.

Tensor and metadata specs (matching Wan 2.1 video-DiT training):

* Latent tensor  `(16 x T x 90 x 160)` **float32**
* Text embedding `(512 x 4096)`  **bfloat16** (stored as raw `uint16` buffer)
* Attention mask `(512,)`        `uint8`
* Frames per second 16 → `duration_sec = T / 16`

The default run (200 rows, T = 16) writes ≈ 3.6 GB and finishes in ≈ 40 s on a PCIe-4 NVMe SSD.
"""

# ─────────────────────────── Standard library ────────────────────────────
import argparse
import logging
import math
import os
import pathlib
import random
import time
from pathlib import Path
from typing import List, Tuple

# ─────────────────────────── 3-rd-party ───────────────────────────────────
import numpy as np
import pyarrow as pa
import lance
import lance.torch.data as ltd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset
from blkinfo import BlkDiskInfo

mp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

# ───────────────────────── Wan 2.1 constants ─────────────────────────────
VAE_C, VAE_H, VAE_W = 16, 90, 160
SEQ_LEN, EMB_DIM = 512, 4096
FPS = 16.0
MAx_T_DEFAULT = 16

LATENT_DTYPE = np.float32
EMB_BUF_DTYPE = np.uint16  # raw lossless storage for bfloat16
EMB_DTYPE_STR = "bfloat16"
MASK_DTYPE = np.uint8

# ───────────────────────────── Arrow schema ──────────────────────────────
pyarrow_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("vae_latent_bytes", pa.binary()),
        pa.field("vae_latent_shape", pa.list_(pa.int64())),
        pa.field("vae_latent_dtype", pa.string()),
        pa.field("text_embedding_bytes", pa.binary()),
        pa.field("text_embedding_shape", pa.list_(pa.int64())),
        pa.field("text_embedding_dtype", pa.string()),
        pa.field("text_attention_mask_bytes", pa.binary()),
        pa.field("text_attention_mask_shape", pa.list_(pa.int64())),
        pa.field("text_attention_mask_dtype", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("media_type", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("num_frames", pa.int32()),
        pa.field("duration_sec", pa.float32()),
        pa.field("fps", pa.float32()),
    ]
)

# ────────────────────────────── Logging ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("wan21_cli")


# ───────────────────── Storage helper (NVMe autodetect) ──────────────────
def _writable_mounts(node):
    """Yield every writable mountpoint in this blkinfo node (recursive)."""
    mp = node.get("mountpoint") or ""
    if mp and os.access(mp, os.W_OK):
        yield mp
    for child in node.get("children", []):
        yield from _writable_mounts(child)


def detect_fast_storage(dir_name: str = "lance_dataset") -> Path:
    disks = BlkDiskInfo().get_disks({"tran": "nvme"})
    for disk in disks:
        for mnt in _writable_mounts(disk):
            p = Path(mnt) / dir_name
            p.mkdir(parents=True, exist_ok=True)
            log.info("Using NVMe mount %s (device %s)", p, disk["name"])
            return p

    # Fallback (no NVMe writable mount)
    p = Path.cwd() / dir_name
    p.mkdir(parents=True, exist_ok=True)
    log.warning("No writable NVMe mount found; using %s", p)
    return p


# ───────────────────── RecordBatch factory ───────────────────────────────
def make_record_batch(start_idx: int, rows: int, max_T: int) -> pa.RecordBatch:
    """Create one Arrow RecordBatch of dummy Wan 2.1 samples."""
    ids, lat_b, lat_s, lat_d = [], [], [], []
    emb_b, emb_s, emb_d = [], [], []
    msk_b, msk_s = [], []
    fnames, captions = [], []
    m_type, m_w, m_h, m_T = [], [], [], []
    m_dur, m_fps = [], []

    for idx in range(start_idx, start_idx + rows):
        T = random.randint(max_T // 2, max_T)

        # Latents
        latent = np.random.randn(VAE_C, T, VAE_H, VAE_W).astype(LATENT_DTYPE)

        # bfloat16 embeddings → uint16 buffer
        emb_bf16 = torch.empty(
            (SEQ_LEN, EMB_DIM), dtype=torch.bfloat16
        ).uniform_(-1, 1)
        emb_uint16 = emb_bf16.view(torch.int16).cpu().numpy().view(np.uint16)

        # Attention mask
        mask = np.zeros(SEQ_LEN, dtype=MASK_DTYPE)
        mask[: random.randint(SEQ_LEN // 4, SEQ_LEN)] = 1

        # Append columns
        ids.append(f"item_{idx:06d}")

        lat_b.append(latent.tobytes())
        lat_s.append(list(latent.shape))
        lat_d.append("float32")
        emb_b.append(emb_uint16.tobytes())
        emb_s.append(list(emb_uint16.shape))
        emb_d.append(EMB_DTYPE_STR)
        msk_b.append(mask.tobytes())
        msk_s.append([SEQ_LEN])

        fnames.append(f"clip_{idx:06d}.mp4")
        captions.append(f"Dummy clip {idx}")
        m_type.append("video")
        m_w.append(VAE_W * 8)
        m_h.append(VAE_H * 8)
        m_T.append(T)
        m_fps.append(FPS)
        m_dur.append(T / FPS)

    arrays = [
        pa.array(ids),
        pa.array(lat_b, pa.binary()),
        pa.array(lat_s, pa.list_(pa.int32())),
        pa.array(lat_d),
        pa.array(emb_b, pa.binary()),
        pa.array(emb_s, pa.list_(pa.int32())),
        pa.array(emb_d),
        pa.array(msk_b, pa.binary()),
        pa.array(msk_s, pa.list_(pa.int32())),
        pa.array(["bool"] * rows),  # attention_mask dtype
        pa.array(fnames),
        pa.array(captions),
        pa.array(m_type),
        pa.array(m_w, pa.int32()),
        pa.array(m_h, pa.int32()),
        pa.array(m_T, pa.int32()),
        pa.array(m_dur, pa.float32()),
        pa.array(m_fps, pa.float32()),
    ]
    return pa.RecordBatch.from_arrays(arrays, [f.name for f in pyarrow_schema])


# ─────────────────────────── Dataset writer  ─────────────────────────────
def generate_dataset(
    path: pathlib.Path, total_rows: int, max_T: int, fragment_mb: int = 250
):
    """
    Stream-write `total_rows` samples with fragment ≈ `fragment_mb` MB
    using the modern `lance.write_dataset` API.
    """
    est_row_bytes = (
        VAE_C * max_T * VAE_H * VAE_W * LATENT_DTYPE().nbytes
        + SEQ_LEN * EMB_DIM * EMB_BUF_DTYPE().nbytes
        + SEQ_LEN
    )
    rows_per_file = max(
        1, math.floor((fragment_mb * 1_000_000) / est_row_bytes)
    )
    batch_rows = min(2_000, rows_per_file)

    log.info("Estimated row size: %.2f MB", est_row_bytes / 1_048_576)
    log.info(
        "Fragment: %d rows (~%d MB)  Batch: %d rows",
        rows_per_file,
        fragment_mb,
        batch_rows,
    )

    def record_batch_iter():
        written = 0
        while written < total_rows:
            n = min(batch_rows, total_rows - written)
            yield make_record_batch(written, n, max_T)
            written += n

    t0 = time.time()
    mode = "overwrite" if path.exists() else "create"
    lance.write_dataset(
        record_batch_iter(),
        str(path),
        schema=pyarrow_schema,
        mode=mode,
        max_rows_per_file=rows_per_file,  # ≈ rows per fragment
    )
    log.info(
        "Generated %d rows in %.1f s (≈%.2f GB)",
        total_rows,
        time.time() - t0,
        total_rows * est_row_bytes / 1_073_741_824,
    )


# ───────────────────────────  Loader class  ──────────────────────────────
class Wan21Dataset(IterableDataset):
    """Zero-copy Lance → Torch stream, slicing the last `slice_T` frames."""

    def __init__(
        self,
        path: pathlib.Path,
        slice_T: int,
        reader_bs: int = 64,
        workers: int = 0,
    ):
        self.path = str(path)
        self.slice_T = slice_T
        self.reader_bs = reader_bs
        self.readahead = max(4, workers * 2)

    @staticmethod
    def _arrow_to_torch(rb: pa.RecordBatch, **_):
        out = {"lat": [], "emb": [], "msk": []}
        for lb, ls, eb, es, mb, ms in zip(
            rb.column("vae_latent_bytes"),
            rb.column("vae_latent_shape"),
            rb.column("text_embedding_bytes"),
            rb.column("text_embedding_shape"),
            rb.column("text_attention_mask_bytes"),
            rb.column("text_attention_mask_shape"),
        ):
            # Create a copy of the numpy arrays to make them writable
            lat = (
                np.frombuffer(lb.as_buffer(), dtype=LATENT_DTYPE)
                .reshape(ls.as_py())
                .copy()
            )
            emb_uint16 = (
                np.frombuffer(eb.as_buffer(), dtype=EMB_BUF_DTYPE)
                .reshape(es.as_py())
                .copy()
            )
            msk = (
                np.frombuffer(mb.as_buffer(), dtype=MASK_DTYPE)
                .reshape(ms.as_py())
                .copy()
            )
            out["lat"].append(torch.from_numpy(lat))
            out["emb"].append(torch.from_numpy(emb_uint16).view(torch.bfloat16))
            out["msk"].append(torch.from_numpy(msk).bool())
        return out

    def __iter__(self):
        reader = ltd.LanceDataset(
            self.path,
            columns=[
                "vae_latent_bytes",
                "vae_latent_shape",
                "text_embedding_bytes",
                "text_embedding_shape",
                "text_attention_mask_bytes",
                "text_attention_mask_shape",
            ],
            batch_size=self.reader_bs,
            batch_readahead=self.readahead,
            to_tensor_fn=self._arrow_to_torch,
        )
        for rb in reader:
            for lat, emb, msk in zip(rb["lat"], rb["emb"], rb["msk"]):
                if lat.shape[1] < self.slice_T:
                    continue
                yield lat[:, -self.slice_T :], emb, msk


# ───────────────────────────── Collate fn ────────────────────────────────
def pad_and_stack(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-size latents, then stack into a mini-batch."""
    lat, emb, msk = zip(*batch)

    # Determine padded extents
    max_T = max(x.shape[1] for x in lat)
    max_H = max(x.shape[2] for x in lat)
    max_W = max(x.shape[3] for x in lat)

    # 1. Pad & stack latents -------------------------------------------------
    lat_padded = torch.stack(
        [
            torch.nn.functional.pad(
                x,
                (
                    0,
                    max_W - x.shape[3],  #  W‑dim
                    0,
                    max_H - x.shape[2],  #  H‑dim
                    0,
                    max_T - x.shape[1],
                ),  #  T‑dim
            )
            for x in lat
        ]
    )

    # 2. Build visibility mask without index_put_ ---------------------------
    lat_mask = torch.zeros((len(lat), max_T, max_H, max_W), dtype=torch.bool)
    for i, x in enumerate(lat):
        lat_mask[i, : x.shape[1], : x.shape[2], : x.shape[3]] = True

    # 3. Stack embeddings & attention masks (already uniform) ---------------
    return lat_padded, torch.stack(emb), lat_mask, torch.stack(msk)


# ─────────────────────────────── CLI ─────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Wan 2.1 dummy Lance dataset CLI")
    sp = ap.add_subparsers(dest="cmd", required=True)

    gen = sp.add_parser("generate", help="Generate dummy dataset")
    gen.add_argument("--rows", type=int, default=200)
    gen.add_argument("--max-T", type=int, default=MAx_T_DEFAULT)

    bench = sp.add_parser("benchmark", help="Benchmark loader throughput")
    bench.add_argument("--slice-T", type=int, default=8)
    bench.add_argument("--reader-bs", type=int, default=64)
    bench.add_argument("--loader-bs", type=int, default=4)
    bench.add_argument("--workers", type=int, default=2)
    bench.add_argument("--batches", type=int, default=500)

    args = ap.parse_args()
    root = detect_fast_storage()
    ds_path = root / "dummy_video_data.lance"

    if args.cmd == "generate":
        generate_dataset(ds_path, args.rows, args.max_T)
        return

    if not ds_path.exists():
        ap.error(f"Dataset not found ({ds_path}); run 'generate' first.")

    dataset = Wan21Dataset(
        ds_path,
        slice_T=args.slice_T,
        reader_bs=args.reader_bs,
        workers=args.workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.loader_bs,
        collate_fn=pad_and_stack,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    t0, rows = time.time(), 0
    for idx, _ in enumerate(loader):
        rows += args.loader_bs
        if idx + 1 >= args.batches:
            break
    elapsed = time.time() - t0
    log.info(
        "Benchmark: %d batches (≈%d rows) in %.2f s → %.1f rows/s",
        idx + 1,
        rows,
        elapsed,
        rows / elapsed,
    )


if __name__ == "__main__":
    main()
