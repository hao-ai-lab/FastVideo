---
license: apache-2.0
task_categories:
- text-to-video
- image-to-video
tags:
- video-generation
- point-tracking
- motionstream
- wantrack
- fastvideo
---

# OpenVid-WanTrack Processed (v2, 720p, **bf16**)

FastVideo preprocessing parquets for training the TrackWan point-track-conditioned I2V model on
the OpenVid-derived WanTrack set. Each row is one 121-frame clip with its VAE latents, text and
image conditioning, and dense CoTracker3 tracks — everything the trainer memory-maps, so no video
decoding happens at train time.

**This is the `bfloat16` variant** of `…/openvid-wantrack-processed` (v2, 720p): the large float
tensor fields are stored in **bf16** instead of float32, so the dataset is roughly **half the size**.
Everything else (clips, ids, shapes, layout) is identical.

## ⚠️ Precision — read before loading

- The big tensors — `vae_latent`, `first_frame_latent`, `clip_feature`, `text_embedding`,
  `track_points`, `track_visibility` — are **`bfloat16`**. Each field's `_dtype` column says so.
- `object_ids` and `track_weights` are kept **`float32`** (small integer/label fields).
- **You must honor the per-field `_dtype` when decoding.** numpy has **no** bfloat16, so
  `np.frombuffer(bytes, "bfloat16")` fails — decode via `torch.frombuffer` (see Loading below).
- **Quality is unaffected for training:** the TrackWan trainer already downcasts these fields to
  bf16 before use, so storing bf16 just pre-applies the exact rounding the model does anyway.
- The FastVideo trainer's loader honors `_dtype`, so pointing `data_path` at this set "just works".

## Layout

```
shard000/combined_parquet_dataset/worker_*/data_chunk_*.parquet
shard001/combined_parquet_dataset/worker_*/data_chunk_*.parquet
...
shard259/...                      # shard259 is a 110-clip remainder; all others are 1000
```

~259,110 clips across 260 shards. Clip ids join 1:1 with `openvid-wantrack-clips` (videos),
`openvid-wantrack-tracks-v2` (raw npz tracks), and OpenVid-1M captions.

## Row schema (`pyarrow_schema_i2v_track`, 33 columns)

Scalars: `id, file_name, caption, media_type, width, height, num_frames, duration_sec, fps`.

Tensors — each stored as a triplet `<name>_bytes` (raw buffer), `_shape` (list<int64>), `_dtype`:

| tensor | shape (720p) | dtype | description |
|--------|--------------|-------|-------------|
| `vae_latent` | `[16, 31, 90, 160]` | **bfloat16** | WanVAE latent of the clip (training target) |
| `first_frame_latent` | `[16, 31, 90, 160]` | **bfloat16** | I2V conditioning: VAE-encode of `[frame0, zeros...]` |
| `clip_feature` | `[257, 1280]` | **bfloat16** | CLIP image embedding of frame 0 |
| `text_embedding` | `[L, 4096]` | **bfloat16** | T5 caption embedding (variable length `L`, padding stripped) |
| `track_points` | `[121, 2500, 2]` | **bfloat16** | CoTracker tracks, **normalized [0,1]** |
| `track_visibility` | `[121, 2500]` | **bfloat16** | per-frame visibility |
| `object_ids` | `[2500]` | float32 | FastSAM object id per track (-1 = background) |
| `track_weights` | `[2500]` | float32 | low-rank motion weight in [0,1] |

`num_frames=31` for the latents (VAE 4x temporal compression: `(121-1)/4+1`); `track_points`
stay at native `121`. Text embedding length varies per row (padding removed), so read the
per-row `_shape`.

## Config

- Video: 1280x720, 121 frames, 24 fps
- VAE: FastVideo WanVAE (latents encoded in fp32, **stored as bf16**), `use_feature_cache=True`
- CLIP: frame-0 image embedding; T5: caption text embedding
- Tracks: CoTracker3, 50x50 grid (2500 points), FastSAM segmentation

## Loading

numpy cannot represent bfloat16, so decode through torch, honoring each field's `_dtype`:

```python
import glob, torch, pyarrow.parquet as pq

_STR2T = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

def decode(row, name):
    dt = _STR2T[row[f"{name}_dtype"]]
    # bytearray() -> writable buffer that doesn't alias the parquet row
    return torch.frombuffer(bytearray(row[f"{name}_bytes"]), dtype=dt).reshape(row[f"{name}_shape"])

files = glob.glob("**/*.parquet", recursive=True)          # all shards
row = pq.read_table(files[0]).slice(0, 1).to_pylist()[0]
lat = decode(row, "vae_latent")        # torch.bfloat16, shape [16, 31, 90, 160]
tracks = decode(row, "track_points")   # torch.bfloat16, normalized [0,1]
```

The FastVideo trainer discovers all parquets under the dataset root via `os.walk` and its loader
honors the `_dtype` column, so point `data_path` at the directory containing the `shard*/` folders.
