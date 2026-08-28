# OpenAI-compatible serving examples

The REST serving engine is model-agnostic. Any model supported by
`VideoGenerator` can use the same `/v1/models`, `/v1/videos`, and `/v1/images`
surface; the two configs here are FastH3 validation profiles.

Launch the full FastH3 checkpoint:

```bash
fastvideo serve --config examples/serving/openai_fasth3.yaml
```

Launch the dense FastH3 LoRA on the base MiniMax-H3 checkpoint:

```bash
adapter_path="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors)"

fastvideo serve --config examples/serving/openai_fasth3_lora.yaml \
  --generator.pipeline.components.lora_path "$adapter_path"
```

FastH3 adapters are hybrid startup patches: alongside low-rank factors they
may contain dense deltas and a VSA compression-gate replacement. They must be
selected when the server starts. A request may carry the vLLM-Omni `lora`
selector, but its name, path, and scale must match that startup adapter. A VSA
adapter also needs `attention_backend: VIDEO_SPARSE_ATTN_H3`, `VSA_sparsity`,
and `VSA_tile_size` like the full-checkpoint config.

Submit and poll an asynchronous job:

```bash
job_id="$(curl -sS http://localhost:8000/v1/videos \
  -H 'content-type: application/json' \
  -d '{"model":"fasth3","prompt":"A fox runs through fresh snow."}' \
  | jq -r .id)"

curl -sS "http://localhost:8000/v1/videos/$job_id"
curl -o result.mp4 "http://localhost:8000/v1/videos/$job_id/content"
```

For a blocking call, `POST /v1/videos/sync` returns the MP4 body directly.
