# Preprocess prompts for MiniMax H3 T2AV distillation

Data-free DMD2 synthesizes video and audio latents during training. Its dataset
contains prompt embeddings and the requested output geometry. No reference
images, reference audio, input videos, or VAE encoding are required.

Prepare a UTF-8 JSONL file. Each nonempty line must contain exactly these fields:

```json
{"id":"example-1","prompt":"A waterfall flows into a lake. Water rushes and birds call.","width":1344,"height":768,"num_frames":124}
{"id":"example-2","prompt":"A violinist performs on a quiet stage. A solo violin melody is heard.","width":768,"height":1344,"num_frames":243}
```

IDs must be unique nonempty strings. Prompts are passed verbatim through the
existing MiniMax H3 Qwen3-VL conditioning stage, without a chat template or
added special tokens. Embedded newlines and whitespace are preserved. Width
and height must be positive multiples of 32; frame counts must be positive
and have the form `17*n + 5` (for example 124, 243, or 345). Frame counts use
H3's 24 fps video clock; the model derives synchronized audio length from it.
The CLI rejects invalid geometry rather than resizing or rounding it.

Download the MiniMax-H3 checkpoint first, then run one process on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_text_only \
  --model-path /path/to/MiniMax-H3 \
  --prompts-file ./data/prompts.jsonl \
  --output-dir ./data/h3-prompts \
  --samples-per-file 16 \
  --max-tokens 65536
```

Only the tokenizer, processor and text encoder are loaded, once per process.
The resulting float32 embeddings have shape `[tokens, 5120]`; nonfinite or
oversized embeddings fail preprocessing. `--max-tokens` is a rejection limit,
not a crop length. Match it to the training model's presentation-token limit.
Large prompts increase both preprocessing storage and training memory needs.

The output layout is:

```text
h3-prompts/shard_00000/
  bucket=1344x768-124f/worker_0/data_chunk_0.parquet
  bucket=768x1344-243f/worker_0/data_chunk_0.parquet
  map_style_cache/file_info.pkl
```

Each Parquet uses the existing `text_only` schema: `id`, `caption`, and
`text_embedding_bytes`, `text_embedding_shape`, `text_embedding_dtype`.
The canonical bucket directory supplies the native output dimensions. Keep
those directory names when moving a dataset. The generated cache contains
absolute paths, so regenerate it after moving data by doing one initial load
with `require_existing_parquet_cache: false`.

Configure the training loader with:

```yaml
training:
  data:
    data_path: ./data/h3-prompts/shard_00000
    preprocessed_data_type: text_only
    native_shape_bucketing: true
    require_existing_parquet_cache: true
    max_presentation_tokens: 65536
    train_batch_size: 1
    training_cfg_rate: 0.0
```

Use `rollout_mode: simulate` in the DMD2 method. Native-shape sampling keeps
all data-parallel groups on the same geometry at each microstep; sequence-
parallel ranks receive the same row. A bucket too small to fill a global
microbatch repeats rows within that bucket, and the loader reports those
repeats. It does not merge independent prompts into one sequence.

For parallel preprocessing, launch separate single-GPU processes with the
same input and output paths, distinct `--shard-index` values and a shared
`--num-shards`. Prompt records are divided round-robin before grouping by
geometry. Each process has its own output shard and default rendezvous port;
an explicit `MASTER_PORT` must also be unique. Pass the resulting shard
directories as a YAML list under `data_path`.

Existing output shards are never overwritten or resumed by row count. A cache
is published only after all rows in a shard succeed. After a failed run, use
a fresh output directory or explicitly remove the failed shard before retrying.
Only load caches produced by a trusted preprocessing run: the existing cache
format is Python pickle. Strict cache mode validates indexed paths and counts
without scanning for new files or repairing the index.
