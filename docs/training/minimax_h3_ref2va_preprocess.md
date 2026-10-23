# Preprocess MiniMax-H3 reference-conditioned training data

MiniMax-H3 DMD/PDD training consumes precomputed target video/audio latents,
Qwen3-VL embeddings and ordered reference conditions. The packed training
adapter uses `pyarrow_schema_ref2va`: **clean** normalized visual reference rows,
clean audio rows, and the reference geometry needed to reconstruct the packed
sequence. The adapter applies visual-reference noise augmentation during batch
preparation using the stored `reference_noise_seed`.

This differs from the existing Ref2VA SFT `minimax_h3_ref2va_precomputed_v1`
format, which stores already augmented `ref_visual_anchor` tensors. Keep using
`preprocess_minimax_h3_ref2va` for that format. Do not rename or directly combine
shards from these two schemas.

The SFT preprocessor keeps each target clip's own geometry: the canvas follows
the clip's aspect ratio at the inference short edge (`resolve_canvas_size`), and
the frame count is the longest `17 * n + 5` that fits the clip, capped at 362
(about 15 seconds; a longer clip trains on its start). Clips under 124 frames at
24 fps fail. Target audio is cut to the samples covered by that frame count's
whole audio latents, so the record matches what inference generates. To train
on those lengths, leave `training.data.num_latent_t` and `num_frames` unset in
the training config. When they are set, every row is cropped to that length;
for example, the 124-frame overfit values crop a 10-second clip to its first
5.17 seconds while its caption still describes all 10.

## Input

Provide your own local MiniMax-H3 checkpoint with `vae`, `audio_vae`, `tokenizer`,
`processor`, `text_encoder` and `transformer_ref` components. Encoding runs on one
CUDA device and releases each large component before loading the next. The
trainer subsequently reads the encoded tensors without loading the encoders.

The input accepts JSON or JSONL using the public raw Ref2VA schema. Paths are
relative to the manifest. Each row needs a target video with synchronized audio,
a caption, and at least one image or video reference. This example contains no
bundled media:

```json
{
  "schema_version": "minimax_h3_ref2va_raw_v1",
  "id": "example-1",
  "target": {"video": "media/target.mp4"},
  "caption": "The person walks through the scene.",
  "references": [
    {"type": "image", "image": "media/person.png"},
    {"type": "audio", "audio": "media/voice.wav"}
  ],
  "target_geometry": {"width": 832, "height": 480, "num_frames": 124}
}
```

References remain in their declared order. Supported entries are `image`
(with `image`), silent `video` (with `video` and optional `fps`), `audio` (with
`audio` and optional `sample_rate`), and `video_audio` (with both `video` and
`audio`, plus optional rates). Images/videos follow the existing inference
reference sizing and presentation rules. The target canvas is independent of
the reference geometry.

`target_geometry` is optional and overrides the command's defaults for that row.
Width and height must be positive multiples of 32. The frame count must be
`17 * n + 5`, for a nonnegative integer `n`. Targets are resampled to 24 fps,
trimmed to that count and resized to the explicitly requested canvas using
Lanczos interpolation. Target stereo audio is resampled to 32 kHz and trimmed
to the same duration. Short video or audio inputs fail instead of being extended.
Choose canvases with the desired aspect ratio; resizing does not preserve a
mismatched source aspect ratio.

## Encode

```bash
python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_packed_ref2va \
  --manifest data/my_ref2va/train.jsonl \
  --model-path data/models/MiniMax-H3 \
  --output-dir data/my_ref2va/encoded \
  --width 1344 --height 768 --num-frames 124 \
  --max-presentation-tokens 65536
```

Add `--validate-manifest-only` to check paths, reference declarations and geometry
without loading models. Actual encoding requires CUDA and the normal MiniMax-H3
preprocessing dependencies. This single-process command does not launch a
training job or manage a cluster scheduler.

The command preserves the complete multimodal presentation. The presentation
limit rejects an oversized sample; it never truncates text or reference tokens.
Each row is checked for target shape, finite tensors and successful Parquet
roundtrip. All output is staged and published after every row passes; a nonempty
output directory is refused. Use a fresh directory for a changed dataset.

Output has one row per shard under directories such as
`bucket=832x480-124f/data_00000000.parquet`. Preserve these bucket names: native
sampling uses them to schedule matching target shapes across data-parallel
workers. Metadata keeps the original order and geometry of every reference,
text/vision token tags and the fixed reference noise seed. Video targets are
`[24, T, H/16, W/16]`; audio targets are `[2, 32, T_audio]`; clean reference rows
are `[N_visual, 96]` and `[N_audio, 32]`. Reference audio rows include both stereo
channels in channel-major order.

## Connect to training

For the packed reference-conditioned DMD/PDD adapter, set:

```yaml
training:
  data:
    data_path: data/my_ref2va/encoded
    preprocessed_data_type: ref2va
    train_batch_size: 1
    native_shape_bucketing: true
    max_presentation_tokens: 65536
    require_existing_parquet_cache: false
```

The first normal dataset initialization builds its file index under
`map_style_cache`. For later runs, `require_existing_parquet_cache: true` requires
a valid existing index and fails instead of rescanning. This index is a pickle
file: use only an index you trust and regenerate it after moving or changing the
dataset. Index paths must remain inside the dataset root.

Native sampling groups exact target shapes into global microbatches. Rare shape
buckets repeat rows within that bucket to fill a global microbatch; the sampler
reports the repeats. This preserves each bucket's presence but can change its
relative frequency when buckets are small. SP ranks share the same local rows.

Optional `fixed_sequence_capacities` selects the smallest fitting padded joint
sequence length from positive multiples of 256. The length includes target
video/audio, text and reference tokens. Oversized rows are rejected during
metadata admission; capacities never crop a document. Capacity padding is
separate from cross-sample packing, which the Ref2VA adapter does not support.
See [reference-conditioned DMPDD training](omniref_dmpdd.md) for model roles,
optimization settings and the training command.
