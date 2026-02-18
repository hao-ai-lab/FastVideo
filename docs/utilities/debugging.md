# Debugging

This page collects practical debugging steps for FastVideo inference issues.

## Collect Environment Info

From the repository root, run:

```bash
python collect_env.py
```

Attach the output when filing a GitHub issue.

## Increase Logging

FastVideo logging level is controlled by environment variables:

```bash
FASTVIDEO_LOGGING_LEVEL=DEBUG \
FASTVIDEO_STAGE_LOGGING=1 \
python your_script.py
```

Useful variables:

- `FASTVIDEO_LOGGING_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `FASTVIDEO_STAGE_LOGGING`: print per-stage timings during pipeline execution
- `FASTVIDEO_ATTENTION_BACKEND`: force an attention backend (for example
  `TORCH_SDPA` or `FLASH_ATTN`)

## Common Failure Modes

### Out-of-memory

Try, in order:

1. Reduce `height`, `width`, `num_frames`, or `num_inference_steps`.
2. Enable offloading flags such as `dit_layerwise_offload` (single GPU) or
   `use_fsdp_inference` (multi-GPU).
3. Enable `vae_cpu_offload`, `image_encoder_cpu_offload`, and
   `text_encoder_cpu_offload`.

See [Inference Offloading](../inference/offloading.md) for recommended
combinations.

### Attention backend import errors

If forcing a backend fails, verify optional dependencies are installed:

- `FLASH_ATTN`: `flash-attn`
- `SLIDING_TILE_ATTN` and `VIDEO_SPARSE_ATTN`: `fastvideo-kernel`
- `SAGE_ATTN` / `SAGE_ATTN_THREE`: SageAttention packages

As a fallback, use:

```bash
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

### Configuration parsing errors

When using `--config`, keep keys aligned with CLI argument names (underscores or
hyphens are both accepted). For nested config values, use nested objects
(`vae_config`, `dit_config`) rather than dotted keys.

## Issue Template

When opening an issue, include:

- exact command or Python snippet,
- model ID/path,
- full traceback,
- `collect_env.py` output,
- whether the problem reproduces with `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`.
