# FastVideo Interleave Service

This example starts a FastVideo-backed image service compatible with
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
