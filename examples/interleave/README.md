# FastVideo Interleave Examples

This directory contains two InterleaveThinker-oriented entrypoints:

- `flux2_klein_interleave_serve.yaml`: run FastVideo as an
  InterleaveThinker-compatible generator service.
- `interleave_run.yaml`: run the native planner -> generator -> critic loop
  through `fastvideo interleave-run`.
- `interleave_single_prompt.py`: run the native FastVideo interleave
  orchestrator with a fallback single-prompt planner and accept-all critic.

## Compatibility Service

This service starts a FastVideo-backed image endpoint compatible with
InterleaveThinker-style generator calls.

InterleaveThinker reward code posts JSON to `/edit`:

```json
{
  "image": "<optional base64 png>",
  "prompt": "make the sky sunset orange",
  "num_inference_step": 4,
  "guidance_scale": 1.0,
  "width": 1024,
  "height": 1024
}
```

FastVideo returns:

```json
{
  "success": true,
  "edited_image": "<base64 png>",
  "file_path": "/abs/path/to/output.png"
}
```

Run:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  fastvideo interleave-serve --config examples/interleave/flux2_klein_interleave_serve.yaml
```

Then point InterleaveThinker at:

```bash
export EDIT_API_ENDPOINT=http://<host>:8011
export EDIT_MODEL_NAME=klein
```

The same app also exposes `/generate` and `/v1/interleave/edit` aliases.

## Native Interleave Run CLI

Run the native orchestrator from a config file:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  fastvideo interleave-run --config examples/interleave/interleave_run.yaml
```

Override common per-run fields from the CLI:

```bash
fastvideo interleave-run \
  --config examples/interleave/interleave_run.yaml \
  --prompt "create a clean product photo of a red ceramic mug" \
  --output-dir outputs/interleave_mug \
  --trace-path outputs/interleave_mug/trace.json
```

The example defaults to `planner.kind: single_prompt` and
`critic.kind: accept_all` so it can smoke-test the FastVideo image backend
without loading planner/critic checkpoints. To run the real InterleaveThinker
loop, switch those blocks to `interleave_thinker` and set:

```yaml
planner:
  kind: interleave_thinker
  init_from: InterleaveThinker/InterleaveThinker-Planner-8B
  processor_from: Qwen/Qwen3-VL-8B-Instruct
  trainable: false
  torch_dtype: bf16
  device_map: cuda:0
  attn_implementation: sdpa
  max_new_tokens: 2048

critic:
  kind: interleave_thinker
  init_from: InterleaveThinker/Critic-SFT-8B
  processor_from: Qwen/Qwen3-VL-8B-Instruct
  trainable: false
  torch_dtype: bf16
  device_map: cuda:0
  attn_implementation: sdpa
  max_new_tokens: 512
```

Loading the planner, critic, and image generator in one process can be memory
intensive. On tighter GPUs, run the `/edit` compatibility service separately or
load only the planner/critic in the native runner while the generator is served
out-of-process.

## Nano Banana / Gemini API Backends

The InterleaveThinker RL config can use closed-source Google models through
lazy wrappers:

- `fastvideo.train.methods.rl.rewards.GeminiNanoBananaEditScorer`
  generates edits with Nano Banana and scores them with Gemini.
- `fastvideo.entrypoints.interleave.generator.NanoBananaImageGeneratorBackend`
  implements the same image backend protocol as the local FastVideo generator.

Install the optional SDK and provide a key only when using these API backends:

```bash
uv pip install -e ".[interleave-api]"
export GEMINI_API_KEY=...
```

Supported Nano Banana aliases are:

- `nano-banana` -> `gemini-2.5-flash-image`
- `nano-banana-pro` -> `gemini-3-pro-image`
- `nano-banana-2` -> `gemini-3.1-flash-image`

## Native Single-Prompt Trace

Run:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  python examples/interleave/interleave_single_prompt.py \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "a brushed steel espresso machine on a marble counter, morning window light" \
    --output-dir outputs/interleave_single_prompt
```

This writes an image plus `trace.json`. The trace records planner/generator/
critic attempts and omits base64 image payloads by default.
