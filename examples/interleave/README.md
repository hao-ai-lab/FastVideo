# FastVideo Interleave Examples

This directory contains two InterleaveThinker-oriented entrypoints:

- `flux2_klein_interleave_serve.yaml`: run FastVideo as an
  InterleaveThinker-compatible generator service.
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
