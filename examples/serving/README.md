# OpenAI-compatible serving examples

The REST serving engine is model-agnostic. Any model supported by
`VideoGenerator` can use the same `/v1/models`, `/v1/videos`, and `/v1/images`
surface. The two FastH3 configs here are source-backed examples, not recorded
serving benchmarks. Both configure four CUDA GPUs without a GPU model or VRAM claim.

Use the [H3 server cookbook](https://haoailab.com/FastVideo/cookbook/openai-api/)
for the guided install, server, and client workflow.

Launch the full FastH3 checkpoint:

```bash
fastvideo serve --config examples/serving/openai_fasth3.yaml --server.host 127.0.0.1
```

Launch the dense FastH3 LoRA on the base MiniMax-H3 checkpoint:

```bash
adapter_path="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors)"

fastvideo serve --config examples/serving/openai_fasth3_lora.yaml \
  --server.host 127.0.0.1 \
  --generator.pipeline.components.lora_path "$adapter_path"
```

FastH3 adapters are hybrid startup patches: alongside low-rank factors they
may contain dense deltas and a VSA compression-gate replacement. They must be
selected when the server starts. A request may carry the vLLM-Omni `lora`
selector, but its name, path, and scale must match that startup adapter. A VSA
adapter also needs `attention_backend: VIDEO_SPARSE_ATTN_H3`, `VSA_sparsity`,
and `VSA_tile_size` like the full-checkpoint config.

After startup, open `http://127.0.0.1:8000/playground/` to generate and review
H3 videos in your browser. CUDA prompts reuse the loaded model. The playground
and API clients share the same server; both can run locally.

For native Apple Silicon MLX, use
`python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3.yaml`
after preparing the weights and editing their paths in that configuration.
The same playground and clients work with MLX. Its pipeline still loads and
releases components between phases to limit unified-memory use. See the
cookbook for setup and the supported text-to-video/audio request fields.

Or submit, poll, and download with the OpenAI Python client:

```bash
python -m pip install openai==3.6.0
python examples/serving/clients/video.py
```

The `clients/` directory also contains cURL and OpenAI JavaScript examples.
They default to model `fasth3`; set `FASTVIDEO_MODEL` to the advertised alias
when using the LoRA config. Set `FASTVIDEO_BASE_URL` for another endpoint.
These clients call your FastVideo server, not OpenAI's cloud. The SDK key
`local` is a placeholder, not authentication. Keep the server on loopback or
use an authenticated proxy for remote access.

For a blocking call, `POST /v1/videos/sync` returns the MP4 body directly.
