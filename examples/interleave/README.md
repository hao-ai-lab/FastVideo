# FastVideo Interleave Examples

This directory contains a small Python example for the reusable Interleave
orchestration helpers. It does not add FastVideo CLI commands or HTTP routes.

## Single-Prompt Trace

Run:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  python examples/interleave/interleave_single_prompt.py \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "a brushed steel espresso machine on a marble counter, morning window light" \
    --output-dir outputs/interleave_single_prompt
```

The script uses `VideoGenerator` directly with a fallback single-prompt planner
and accept-all critic. It writes an image plus `trace.json`; the trace records
planner/generator/critic attempts and omits base64 image payloads by default.

## Planner And Critic Follow-Up

This inference PR ships the reusable workflow layer and the fallback
single-prompt planner. The Qwen3-VL InterleaveThinker planner/critic model
wrappers and training configs are intentionally deferred to a follow-up
training PR.

## Optional Gemini Backends

The workflow helper can use closed-source Google image models through lazy
wrappers:

- `fastvideo.workflow.interleave_thinker.generator.NanoBananaImageGeneratorBackend`
  implements the same image backend protocol as the local FastVideo generator.

Install the optional SDK and provide a key only when using these API backends:

```bash
uv pip install -e ".[eval-judge]"
export GEMINI_API_KEY=...
```

Supported Nano Banana aliases are:

- `nano-banana` -> `gemini-2.5-flash-image`
- `nano-banana-pro` -> `gemini-3-pro-image`
- `nano-banana-2` -> `gemini-3.1-flash-image`
