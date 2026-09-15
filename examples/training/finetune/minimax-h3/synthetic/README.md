# MiniMax H3 Ref2VA synthetic training fixture

This directory demonstrates the raw manifest accepted by the MiniMax H3
Ref2VA preprocessor. The checked-in media is deterministic, repository-owned
synthetic content: a geometric animation, a still geometric reference, and
mathematically generated tones. It contains no people or third-party media.

## Raw manifest contract

The preprocessor accepts JSON, JSONL, or a JSON object containing a `data`
array. Every sample has this shape:

```json
{
  "schema_version": "minimax_h3_ref2va_raw_v1",
  "id": "unique-sample-id",
  "target": {"video": "relative/or/absolute/target.mp4"},
  "caption": "The target description.",
  "references": [
    {"type": "image", "image": "reference.png"},
    {"type": "video", "video": "silent-reference.mp4"},
    {"type": "video_audio", "video": "reference-with-sound.mp4"},
    {"type": "audio", "audio": "reference.wav"}
  ]
}
```

Relative media paths resolve from the manifest directory. `video` is always a
silent visual reference even when its container has an audio stream;
`video_audio` explicitly uses both streams. A sample supports at most 12 total
references: 9 images, 3 videos, and 3 standalone audio references. A non-empty
reference list must include at least one image or video. FL2VA training is not
part of this workflow.

Preprocessing encodes target video/audio latents, dynamic Qwen3-VL hidden
states and tags, ordered visual anchors at clean-time `0.999`, and clean audio
anchors. It writes one validated Parquet row per shard. The video VAE, audio
VAE, and Qwen3-VL stack are loaded sequentially on one GPU; plan for the full
MiniMax H3 component checkpoint and substantial preprocessing time even for a
small manifest.

## Fixture generation and provenance

`generate_fixture.py` is the complete source for all three checked-in media
files. It uses fixed formulas and codec settings; it requires NumPy, Pillow,
and PyAV from the FastVideo environment. The visual/audio content is source
reproducible, but MP4/AAC container bytes are not claimed to be bit-identical
across PyAV, FFmpeg, or libx264 versions. `PROVENANCE.json` records the exact
generation library versions used for the checked-in bytes. Regenerate or
verify them with:

```bash
python examples/training/finetune/minimax-h3/synthetic/generate_fixture.py
python examples/training/finetune/minimax-h3/synthetic/generate_fixture.py --verify
```

`--verify` checks the checked-in bytes against the recorded hashes; regeneration
updates both the media and provenance for the active codec environment.
`PROVENANCE.json` records the exact generation parameters and SHA-256 digest of
each media file. The source and generated artifacts are licensed under the
repository's Apache-2.0 license. They have no external source, third-party
rights, likeness, voice, consent, or redistribution dependency.

## Configurable setup

The setup script verifies the checked-in fixture before manifest validation
and accepts all machine-specific inputs through environment variables:

```bash
MODEL_DIR=/path/to/MiniMax-H3 \
MANIFEST=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/preprocessed-ref2va \
PYTHON_BIN=/path/to/python \
CUDA_VISIBLE_DEVICES=0 \
bash examples/train/setup_minimax_h3_ref2va_synthetic_single_sample.sh
```

`ENV_FILE=/path/to/optional.env` may be supplied when the selected checkpoint
requires environment-based authentication. It is never required for local
inputs. Existing non-empty output is refused by default. Set
`REPLACE_EXISTING=1` to stage and validate a replacement; the previous dataset
is moved to a sibling `.backup-*` directory rather than deleted.

The example training configs describe a 64-rank topology with SP=8 and HSDP
8x8. Adjust paths and distributed dimensions to the actual cluster. A modular
training checkpoint is DCP state, not an inference directory. Export a
checkpoint with:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint /path/to/checkpoint-400 \
  --output-dir /path/to/minimax-h3-export \
  --verify
```

Full Ref2VA tuning replaces `transformer_ref/`. H3 LoRA export merges the
adapter into native transformer weights and writes the same component used by
the public T2VA or Ref2VA inference pipeline. Standalone H3 adapter export and
runtime adapter selection are not supported by this workflow.

Export gathers the complete transformer state on CPU while the live training
model still exists. `--verify` releases that training graph before strict
reload, but it does not make the gather streaming. For the roughly 62 GiB H3
transformer, use a host with comfortably more than the combined live-model,
gathered-state, and runtime working set; a 121 GiB unified-memory GB10 is not a
validated full-export target.
